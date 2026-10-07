"""Standalone iterative-search entrypoint (interface v0.1.1).

Runs ONE search policy over a train split of the canonical table and writes
its archive into `--log-dir`; intended to be launched through
`scripts/run_experiment.py` with `configs/experiments/search_*.json`, which
records config / data hash / environment / stdout / DONE marker.

The search layer only *predicts*: it never queries real labels, never marks
anything PARKED, and never touches the existing 96-candidate hand-off batch.

Usage (plain):
  python scripts/loop/search_run.py --policy multi_start_beam --seed 1 \
      --max-unique-predictions 256 --n-init 400 \
      --output runs/search/metrics.csv --log-dir runs/search/archive
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    from .interface import TARGET_COLUMN, normalize
    from .search import SEARCH_INTERFACE_VERSION, SearchContext, SurrogateScorer
    from .search_archive import SearchArchive, sha256_file
    from .search_policies import make_policy
    from .surrogates import make_surrogate
except ImportError:  # plain-script invocation (run_experiment.py entrypoint)
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.loop.interface import TARGET_COLUMN, normalize
    from scripts.loop.search import SEARCH_INTERFACE_VERSION, SearchContext, SurrogateScorer
    from scripts.loop.search_archive import SearchArchive, sha256_file
    from scripts.loop.search_policies import make_policy
    from scripts.loop.surrogates import make_surrogate


def parse_range(value: str) -> tuple[int, int]:
    lo, hi = value.split(":")
    return int(lo), int(hi)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="FAD iterative search | interface v0.1.1")
    p.add_argument("--policy", required=True,
                   choices=["stratified_random", "matched_seed_random_walk",
                            "multi_start_hill_climb", "multi_start_beam", "mutation_only_ga",
                            "iterated_local_search", "simulated_annealing", "mutation_only_adalead"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--model-seed", type=int, default=None,
                   help="surrogate seed; defaults to --seed. Kept separate so a "
                        "retrain-consistency arm can use the SAME split with a "
                        "different model seed")
    p.add_argument("--max-unique-predictions", type=int, default=256)
    p.add_argument("--max-generations", type=int, default=8)
    p.add_argument("--allowed-n-from-wt", default="4:13", type=parse_range)
    p.add_argument("--n-init", type=int, default=400)
    p.add_argument("--surrogate", default="xgb", choices=["xgb", "fusion", "rf"])
    p.add_argument("--n-members", type=int, default=5)
    p.add_argument("--screen-k", type=int, default=100)
    p.add_argument("--data-dir", default="data/proxy2000_v2")
    # policy hyperparameters (defaults are written explicitly into the config,
    # never left implicit — plan §6)
    p.add_argument("--beam-width", type=int, default=32)
    p.add_argument("--offspring-per-parent", type=int, default=32)
    p.add_argument("--one-hop-fraction", type=float, default=0.8)
    p.add_argument("--restart-fraction", type=float, default=0.1)
    p.add_argument("--diversity-pool-factor", type=int, default=4)
    p.add_argument("--diversity-distance", type=int, default=2)
    p.add_argument("--max-stagnant-generations", type=int, default=None)
    p.add_argument("--n-measured-seeds", type=int, default=8)
    p.add_argument("--n-random-seeds", type=int, default=8)
    p.add_argument("--min-mutual-distance", type=int, default=2)
    p.add_argument("--population-size", type=int, default=32)
    p.add_argument("--offspring-size", type=int, default=128)
    p.add_argument("--tournament-size", type=int, default=2)
    p.add_argument("--elite-count", type=int, default=2)
    p.add_argument("--ils-perturb-sites", type=int, default=2)
    p.add_argument("--sa-temperature-scale", type=float, default=0.25)
    p.add_argument("--sa-final-temperature-ratio", type=float, default=0.02)
    p.add_argument("--adalead-tolerance-scale", type=float, default=0.05)
    p.add_argument("--adalead-parent-cap", type=int, default=32)
    p.add_argument("--adalead-rollouts", type=int, default=16)
    p.add_argument("--output", default="runs/search/metrics.csv")
    p.add_argument("--log-dir", default="runs/search/archive")
    args = p.parse_args(argv)
    if args.max_stagnant_generations is None:
        args.max_stagnant_generations = 4 if args.policy == 'mutation_only_ga' else 2
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    data_path = PROJECT_ROOT / args.data_dir / "fad_proxy2000_v2_full.csv"
    df = pd.read_csv(data_path)
    df["Sequence"] = df["Sequence"].map(normalize)
    if df["Sequence"].duplicated().any():
        raise ValueError("canonical data contains duplicate sequences")
    values = df[TARGET_COLUMN].astype(float).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("canonical gbsa contains non-finite values")

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(df))
    idx_train = perm[: args.n_init]
    measured_seqs = df["Sequence"].values[idx_train].tolist()
    measured_vals = values[idx_train].tolist()
    model_seed = args.model_seed if args.model_seed is not None else args.seed

    # The whole canonical table is forbidden (measured rows stay eligible as
    # *parents* but can never be re-exported).  Holdout labels are never read.
    forbidden = frozenset(df["Sequence"])

    surrogate = make_surrogate(args.surrogate, seed=model_seed,
                               n_members=args.n_members, screen_k=args.screen_k)
    surrogate.fit(measured_seqs, np.asarray(measured_vals, dtype=float))

    policy_cfg = {
        "beam_width": args.beam_width,
        "offspring_per_parent": args.offspring_per_parent,
        "one_hop_fraction": args.one_hop_fraction,
        "restart_fraction": args.restart_fraction,
        "diversity_pool_factor": args.diversity_pool_factor,
        "diversity_distance": args.diversity_distance,
        "max_stagnant_generations": args.max_stagnant_generations,
        "n_measured_seeds": args.n_measured_seeds,
        "n_random_seeds": args.n_random_seeds,
        "min_mutual_distance": args.min_mutual_distance,
        "population_size": args.population_size,
        "offspring_size": args.offspring_size,
        "tournament_size": args.tournament_size,
        "elite_count": args.elite_count,
        "ils_perturb_sites": args.ils_perturb_sites,
        "sa_temperature_scale": args.sa_temperature_scale,
        "sa_final_temperature_ratio": args.sa_final_temperature_ratio,
        "adalead_tolerance_scale": args.adalead_tolerance_scale,
        "adalead_parent_cap": args.adalead_parent_cap,
        "adalead_rollouts": args.adalead_rollouts,
    }
    policy = make_policy(args.policy, **policy_cfg)

    search_id = (f"search_{args.policy}_{args.seed}_{args.max_unique_predictions}"
                 f"_m{model_seed}")
    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = PROJECT_ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    context = SearchContext(
        search_id=search_id,
        outer_round=1,
        measured_sequences=tuple(measured_seqs),
        measured_gbsa=tuple(float(v) for v in measured_vals),
        forbidden_sequences=forbidden,
        seed=args.seed,
        allowed_n_from_wt=tuple(args.allowed_n_from_wt),
        max_unique_predictions=args.max_unique_predictions,
        max_generations=args.max_generations,
        data_sha256=sha256_file(data_path).upper(),
        archive_dir=str(log_dir),
    )

    t0 = time.perf_counter()
    archive = SearchArchive(log_dir)
    cache = archive.load_cache()
    scorer = SurrogateScorer(
        surrogate, context, cache=cache,
        ood_train_sequences=measured_seqs,
        model_id=str(getattr(surrogate, "model_id", None) or "unset"),
        feature_version=str(getattr(surrogate, "feature_version", None) or "unset"))
    result = policy.search(context, scorer)
    seconds = time.perf_counter() - t0
    # Write weights only after the archive has accepted the context/model.
    # A rejected resume must not replace the original fitted model artifact.
    surrogate.save(log_dir / 'fitted_surrogate.pkl')

    row = {
        "search_id": search_id,
        "policy": args.policy,
        "seed": args.seed,
        "model_seed": model_seed,
        "outer_round": 1,
        "n_unique_predicted": result.n_unique_predicted,
        "n_candidates": len(result.ranked_candidates),
        "best_pred_gbsa": (float(result.ranked_candidates[0].pred_gbsa)
                           if result.ranked_candidates else ""),
        "best_candidate_id": (result.ranked_candidates[0].candidate_id
                              if result.ranked_candidates else ""),
        "stop_reason": result.stop_reason,
        "n_groups": len(result.groups),
        "group_roles": ",".join(g.group_role for g in result.groups),
        "model_id": scorer.model_id,
        "feature_version": scorer.feature_version,
        "budget": args.max_unique_predictions,
        "seconds": round(seconds, 3),
        "archive_dir": str(log_dir),
    }
    out = Path(args.output)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out, index=False)
    print(json.dumps({
        "search_interface_version": SEARCH_INTERFACE_VERSION,
        "policy": args.policy,
        "seed": args.seed,
        "n_unique_predicted": result.n_unique_predicted,
        "n_candidates": len(result.ranked_candidates),
        "best_pred_gbsa": row["best_pred_gbsa"],
        "best_candidate_id": row["best_candidate_id"],
        "stop_reason": result.stop_reason,
        "n_groups": len(result.groups),
        "group_roles": row["group_roles"],
        "model_id": scorer.model_id,
        "seconds": round(seconds, 3),
    }, ensure_ascii=False, indent=2))
    print(f"[OK] 1 row -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

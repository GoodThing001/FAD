"""Search-policy comparison (plan §5 阶段 B, repaired matched-seed arm).

Runs one search per (seed, policy) on the SAME train split, surrogate snapshot
and hard unique-prediction budget, then reports paired comparisons against
`stratified_random` on the diagnostics the plan lists:

  best predicted gbsa (lower = the policy's strongest predicted candidate),
  top-50 candidate diversity (mean pairwise 13-site Hamming),
  top-50 OOD fraction and mean distance to the training set,
  feasibility (scored / proposed) and rejection breakdown,
  wall time, stop reasons, group counts.

Supplementary arms:
  cross-model ranking stability  the main arm's top-50 candidates re-scored by
                                two surrogates retrained on the SAME split with
                                different model seeds -> Spearman pairs;
  retrain consistency (arm B)   `--consistency-seeds` policies re-run with a
                                different model seed on the same split ->
                                top-50 candidate Jaccard overlap.

All numbers are PROXY-table computational diagnostics: `best_pred_gbsa` comes
from the surrogate, so a lower value is an observation about the search, NOT a
discovery claim; real GBSA discovery requires 阶段 C with real measurements.

Outputs under --out-dir: runs.csv / summary.json / completeness.json /
README.txt. Sub-runs whose metrics.csv already exists are reused verbatim —
re-run a changed driver in a fresh --out-dir.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.loop.interface import MUT_POS, WT_141, normalize, TARGET_COLUMN  # noqa: E402
from scripts.loop.surrogates import make_surrogate  # noqa: E402

POLICIES = ("stratified_random", "matched_seed_random_walk",
            "multi_start_hill_climb", "multi_start_beam", "mutation_only_ga")
DEFAULT_POLICIES = "stratified_random,multi_start_hill_climb,multi_start_beam"
MATCHED_POLICIES = ("matched_seed_random_walk", "multi_start_hill_climb",
                    "multi_start_beam", "mutation_only_ga")
TOP_K = 50
CROSS_MODEL_SEEDS = (1000, 2000)   # offsets added to the split seed for re-fits


def spearman(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2:
        return float("nan")
    return float(pearsonr(rankdata(a), rankdata(b))[0])


def bootstrap_ci(values, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mutation_distance(a: str, b: str) -> int:
    a, b = normalize(a), normalize(b)
    return sum(1 for p in MUT_POS if a[p - 1] != b[p - 1])


def top50_diversity(cands: pd.DataFrame) -> float:
    """Mean pairwise 13-site Hamming over the top-50 ranked candidates."""
    seqs = cands["sequence"].map(normalize).tolist()[:TOP_K]
    if len(seqs) < 2:
        return float("nan")
    ds = [mutation_distance(seqs[i], seqs[j])
          for i in range(len(seqs)) for j in range(i + 1, len(seqs))]
    return float(np.mean(ds))


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return float("nan")
    return len(set_a & set_b) / len(set_a | set_b)


def split_train(a, seed: int) -> tuple[list[str], np.ndarray]:
    """The driver's own copy of the search_run split (same RandomState call)."""
    df = pd.read_csv(PROJECT_ROOT / a.data_dir / "fad_proxy2000_v2_full.csv")
    df["Sequence"] = df["Sequence"].map(normalize)
    values = df[TARGET_COLUMN].astype(float).to_numpy()
    perm = np.random.RandomState(seed).permutation(len(df))
    idx = perm[: a.n_init]
    return df["Sequence"].to_numpy()[idx].tolist(), values[idx]


def build_cmd(a, seed: int, model_seed: int, policy: str, run_dir: Path) -> list[str]:
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts/loop/search_run.py"),
        "--policy", policy,
        "--seed", str(seed), "--model-seed", str(model_seed),
        "--max-unique-predictions", str(a.budget),
        "--max-generations", str(a.max_generations),
        "--allowed-n-from-wt", a.allowed_n_from_wt,
        "--n-init", str(a.n_init),
        "--surrogate", a.surrogate, "--n-members", str(a.n_members),
        "--screen-k", str(a.screen_k),
        "--data-dir", a.data_dir,
        "--beam-width", str(a.beam_width),
        "--offspring-per-parent", str(a.offspring_per_parent),
        "--one-hop-fraction", str(a.one_hop_fraction),
        "--restart-fraction", str(a.restart_fraction),
        "--diversity-pool-factor", str(a.diversity_pool_factor),
        "--diversity-distance", str(a.diversity_distance),
        "--max-stagnant-generations", str(a.max_stagnant_generations),
        "--n-measured-seeds", str(a.n_measured_seeds),
        "--n-random-seeds", str(a.n_random_seeds),
        "--min-mutual-distance", str(a.min_mutual_distance),
        "--population-size", str(a.population_size),
        "--offspring-size", str(a.offspring_size),
        "--tournament-size", str(a.tournament_size),
        "--elite-count", str(a.elite_count),
        "--output", str(run_dir / "metrics.csv"),
        "--log-dir", str(run_dir / "checkpoints"),
    ]
    return cmd


def run_subrun(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    return proc.returncode, proc.stdout + proc.stderr


def collect_arm(a, seeds, policies, *, arm: str, model_seed_offset: int,
                runs_dir: Path) -> tuple[list[dict], list[dict]]:
    """Run (or reuse) one arm of sub-runs; returns (rows, failures)."""
    rows: list[dict] = []
    failures: list[dict] = []
    for seed in seeds:
        for policy in policies:
            run_dir = runs_dir / f"seed{seed}_{policy}_{arm}"
            if not (run_dir / "metrics.csv").is_file():
                run_dir.mkdir(parents=True, exist_ok=True)
                cmd = build_cmd(a, seed, seed + model_seed_offset, policy, run_dir)
                rc, log = run_subrun(cmd)
                if rc != 0:
                    (run_dir / "stderr.log").write_text(log, encoding="utf-8")
                    print(f"[fail] seed={seed} policy={policy} arm={arm} rc={rc}")
                    failures.append({"seed": seed, "policy": policy, "arm": arm,
                                     "reason": f"sub-run rc={rc}"})
                    continue
            rows.append(read_subrun(a, seed, policy, arm, model_seed_offset,
                                    run_dir, runs_dir))
        print(f"[{arm}] seed {seed} done ({len(rows)} rows, {len(failures)} failures)")
    return rows, failures


def read_subrun(a, seed: int, policy: str, arm: str, model_seed_offset: int,
                run_dir: Path, runs_dir: Path) -> dict:
    metrics = pd.read_csv(run_dir / "metrics.csv").iloc[0]
    ckpt = run_dir / "checkpoints"
    summary = json.loads((ckpt / "search_summary.json").read_text("utf-8"))
    cands = pd.read_csv(ckpt / "search_candidates.csv")
    top = cands.head(TOP_K)
    rej = summary.get("rejection_stats", {})
    proposed = int(rej.get("proposed", 0))
    rejected = sum(int(rej.get(k, 0)) for k in
                   ("rejected_forbidden", "rejected_duplicate", "rejected_constraint"))
    row = {
        "seed": seed, "policy": policy, "arm": arm,
        "model_seed": int(metrics["model_seed"]),
        "n_unique_predicted": int(summary["n_unique_predicted"]),
        "stop_reason": summary["stop_reason"],
        "best_pred_gbsa": float(summary["best_pred_gbsa"]),
        "best_candidate_id": summary["best_candidate_id"],
        "n_candidates": int(summary["n_candidates"]),
        "n_groups": int(summary["n_groups"]),
        "seconds_total": float(summary["seconds_total"]),
        "model_id": summary["model_id"],
        "feas_rate": (max(0, int(metrics["n_unique_predicted"]) -
                          sum(1 for line in (ckpt / 'search_evaluated.jsonl').read_text('utf-8').splitlines()
                              if json.loads(line)['operator'] == 'seed')) / proposed
                      if proposed else float("nan")),
        "rejection_rate": (rejected / proposed if proposed else float("nan")),
        "diversity_top50": top50_diversity(top),
        "ood_fraction_top50": float(top["ood_flag"].astype(bool).mean()),
        "mean_min_train_hamming_top50": float(pd.to_numeric(
            top["nearest_train_hamming"], errors="coerce").mean()),
    }
    row["top50_ids"] = tuple(sorted(top["candidate_id"].tolist()))
    if arm == "A" and not a.skip_cross_model:
        # cross-model ranking stability: re-score the top-50 with two
        # surrogates retrained on the SAME split, different model seeds
        seqs = top["sequence"].map(normalize).tolist()
        base = pd.to_numeric(top["pred_gbsa"], errors="coerce").to_numpy(float)
        train_seqs, train_vals = split_train(a, seed)
        for i, offset in enumerate(CROSS_MODEL_SEEDS, start=1):
            model = make_surrogate(a.surrogate, seed=seed + offset,
                                   n_members=a.n_members, screen_k=a.screen_k)
            model.fit(train_seqs, train_vals)
            pred = model.predict(seqs).mu
            row[f"spearman_model_a_vs_{i}"] = spearman(base, pred)
    return row


def check_completeness(df: pd.DataFrame, seeds, consistency_seeds, policies) -> dict:
    problems: list[str] = []
    expected = (len(seeds) + len(consistency_seeds)) * len(policies)
    if len(df) != expected:
        problems.append(f"row count {len(df)} != expected {expected}")
    if df.duplicated(subset=["seed", "policy", "arm"]).any():
        problems.append("duplicated (seed, policy, arm) rows")
    for seed in list(seeds) + list(consistency_seeds):
        for policy in policies:
            n_a = ((df["seed"] == seed) & (df["policy"] == policy)
                   & (df["arm"] == "A")).sum()
            if n_a != 1:
                problems.append(f"missing seed={seed} policy={policy} arm=A ({n_a})")
    for seed in consistency_seeds:
        for policy in policies:
            n_b = ((df["seed"] == seed) & (df["policy"] == policy)
                   & (df["arm"] == "B")).sum()
            if n_b != 1:
                problems.append(f"missing seed={seed} policy={policy} arm=B ({n_b})")
    for column in ("best_pred_gbsa", "n_unique_predicted", "diversity_top50"):
        if column not in df.columns or df[column].isna().all():
            problems.append(f"missing column {column}")
    return {"ok": not problems, "problems": problems,
            "expected_rows": expected, "observed_rows": int(len(df))}


def paired_delta(df: pd.DataFrame, policy: str, metric: str,
                 better: str, baseline: str = "stratified_random") -> dict:
    """Paired (policy - random) per-seed deltas with bootstrap CI and W/T/L.

    `better` declares the winning direction for `metric`:
    'lower' (d<0 wins) or 'higher' (d>0 wins)."""
    base = df[(df["policy"] == baseline) & (df["arm"] == "A")]
    other = df[(df["policy"] == policy) & (df["arm"] == "A")]
    common = sorted(set(base["seed"]) & set(other["seed"]))
    base = base.set_index("seed")
    other = other.set_index("seed")
    d = (other.loc[common, metric] - base.loc[common, metric]).to_numpy(float)
    lo, hi = bootstrap_ci(d)
    wins = int(np.nansum(d < 0 if better == "lower" else d > 0))
    ties = int(np.nansum(np.abs(d) <= 1e-12))
    losses = int(np.nansum(d > 0 if better == "lower" else d < 0))
    return {"n_pairs": len(common), "delta_mean": float(np.nanmean(d)),
            "delta_ci95": [lo, hi], "delta_median": float(np.nanmedian(d)),
            "wins": wins, "ties": ties, "losses": losses}


def aligned_prediction_comparison(runs_dir: Path, seeds, policies) -> tuple[pd.DataFrame, dict]:
    """Align search results by actual ordered model calls and verify shared seeds."""
    matched = [p for p in MATCHED_POLICIES if p in policies]
    if len(matched) < 2 or "matched_seed_random_walk" not in matched:
        return pd.DataFrame(), {}
    out: list[dict] = []
    for seed in seeds:
        archives = {}
        signatures = {}
        identities = {}
        for policy in matched:
            path = runs_dir / f"seed{seed}_{policy}_A" / "checkpoints" \
                / "search_evaluated.jsonl"
            rows = [json.loads(line) for line in path.read_text(
                encoding="utf-8").splitlines() if line.strip()]
            summary_path = path.parent / "search_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            snapshot = json.loads((path.parent / "search_snapshot.json")
                                  .read_text(encoding="utf-8"))
            identities[policy] = {k: snapshot[k] for k in (
                "data_sha256", "measured_sha256", "forbidden_sha256",
                "allowed_n_from_wt", "max_unique_predictions",
                "model_snapshot")}
            if len(rows) != int(summary["n_unique_predicted"]):
                raise ValueError(f"{seed} {policy}: archive length != actual prediction count")
            signatures[policy] = [(r["candidate_id"], r["sequence"],
                                   r["parent_evidence"]) for r in rows
                                  if r["operator"] == "seed"]
            archives[policy] = rows
        if any(signatures[p] != signatures[matched[0]] for p in matched[1:]):
            raise ValueError(f"seed {seed}: initial seed IDs/order differ between policies")
        if any(identities[p] != identities[matched[0]] for p in matched[1:]):
            raise ValueError(f"seed {seed}: split/model/constraints differ between policies")
        common_n = min(len(archives[p]) for p in matched)
        for policy in matched:
            prefix = archives[policy][:common_n]
            exportable = [r for r in prefix if not
                          (r["operator"] == "seed" and
                           r["parent_evidence"] == "measured")]
            if not exportable:
                raise ValueError(f"{seed} {policy}: no exportable candidate at {common_n}")
            best = min(exportable, key=lambda r: (float(r["pred_gbsa"]),
                                                  r["candidate_id"]))
            out.append({"seed": seed, "policy": policy,
                        "initial_seed_count": len(signatures[policy]),
                        "common_actual_predictions": common_n,
                        "actual_predictions": len(archives[policy]),
                        "best_pred_gbsa_at_common_n": float(best["pred_gbsa"]),
                        "best_candidate_id_at_common_n": best["candidate_id"]})
    aligned = pd.DataFrame(out)
    paired = {p: paired_delta(aligned.assign(arm="A"), p,
                             "best_pred_gbsa_at_common_n", "lower",
                             baseline="matched_seed_random_walk")
              for p in matched if p != "matched_seed_random_walk"}
    return aligned, {"baseline": "matched_seed_random_walk",
                     "initial_seeds_identical": True,
                     "alignment": "per seed, minimum actual unique predictions among matched policies",
                     "n_seeds": len(seeds),
                     "common_n_min": int(aligned["common_actual_predictions"].min()),
                     "common_n_median": float(aligned["common_actual_predictions"].median()),
                     "common_n_max": int(aligned["common_actual_predictions"].max()),
                     "mean_best_at_common_n": aligned.groupby("policy")[
                         "best_pred_gbsa_at_common_n"].mean().to_dict(),
                     "paired_vs_matched_random": paired}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", default=DEFAULT_POLICIES)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--seed-start", type=int, default=201)
    ap.add_argument("--consistency-seeds", type=int, default=10,
                    help="first N seeds that also run arm B (model seed +1000)")
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--n-init", type=int, default=300)
    ap.add_argument("--surrogate", default="xgb")
    ap.add_argument("--n-members", type=int, default=5)
    ap.add_argument("--screen-k", type=int, default=100)
    ap.add_argument("--max-generations", type=int, default=8)
    ap.add_argument("--allowed-n-from-wt", default="4:13")
    ap.add_argument("--beam-width", type=int, default=32)
    ap.add_argument("--offspring-per-parent", type=int, default=32)
    ap.add_argument("--one-hop-fraction", type=float, default=0.8)
    ap.add_argument("--restart-fraction", type=float, default=0.1)
    ap.add_argument("--diversity-pool-factor", type=int, default=4)
    ap.add_argument("--diversity-distance", type=int, default=2)
    ap.add_argument("--max-stagnant-generations", type=int, default=2)
    ap.add_argument("--n-measured-seeds", type=int, default=8)
    ap.add_argument("--n-random-seeds", type=int, default=8)
    ap.add_argument("--min-mutual-distance", type=int, default=2)
    ap.add_argument("--population-size", type=int, default=32)
    ap.add_argument("--offspring-size", type=int, default=128)
    ap.add_argument("--tournament-size", type=int, default=2)
    ap.add_argument("--elite-count", type=int, default=2)
    ap.add_argument("--data-dir", default="data/proxy2000_v2")
    ap.add_argument("--skip-cross-model", action="store_true",
                    help="omit the supplementary re-fit ranking stability arm")
    ap.add_argument("--out-dir", default="evidence/search_compare")
    ap.add_argument("--output", default=None,
                    help="per-policy summary metrics CSV (set by run_experiment.py)")
    a = ap.parse_args()

    policies = [s.strip() for s in a.policies.split(",") if s.strip()]
    for p in policies:
        if p not in POLICIES:
            raise ValueError(f"unknown policy {p!r}; available: {list(POLICIES)}")
    if "stratified_random" not in policies and "matched_seed_random_walk" not in policies:
        raise ValueError("comparison needs a random baseline")

    out_dir = PROJECT_ROOT / a.out_dir
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.txt").write_text(
        "Search-policy comparison (plan 阶段 B).\n"
        "Every policy sees the same train split, surrogate snapshot and hard\n"
        "unique-prediction CEILING. Only matched_seed_random_walk, hill and beam\n"
        "share initial seeds; aligned_runs.csv uses actual prediction prefixes.\n"
        "best_pred_gbsa comes from the SURROGATE, so it\n"
        "is a computational diagnostic, not a discovery claim; real GBSA\n"
        "discovery needs 阶段 C with real measurements. Arm B (retrain\n"
        "consistency) re-runs the first --consistency-seeds seeds with model\n"
        f"seed +1000. argv: {' '.join(sys.argv[1:])}\n", encoding="utf-8")

    seeds = list(range(a.seed_start, a.seed_start + a.seeds))
    consistency_seeds = seeds[: min(a.consistency_seeds, len(seeds))]
    rows_a, failures = collect_arm(a, seeds, policies, arm="A",
                                   model_seed_offset=0, runs_dir=runs_dir)
    rows_b = []
    if consistency_seeds:
        rows_b, failures_b = collect_arm(a, consistency_seeds, policies, arm="B",
                                         model_seed_offset=CROSS_MODEL_SEEDS[0],
                                         runs_dir=runs_dir)
        failures += failures_b

    df = pd.DataFrame(rows_a + rows_b)
    df.to_csv(out_dir / "runs.csv", index=False)

    # consistency: Jaccard of top-50 candidate sets between arm A and arm B
    consistency_rows = []
    for seed in consistency_seeds:
        for policy in policies:
            row_a = df[(df["seed"] == seed) & (df["policy"] == policy)
                       & (df["arm"] == "A")]
            row_b = df[(df["seed"] == seed) & (df["policy"] == policy)
                       & (df["arm"] == "B")]
            if row_a.empty or row_b.empty or "top50_ids" not in df.columns:
                continue
            consistency_rows.append({
                "seed": seed, "policy": policy,
                "jaccard_top50": jaccard(set(row_a.iloc[0]["top50_ids"]),
                                         set(row_b.iloc[0]["top50_ids"])),
            })
    cdf = pd.DataFrame(consistency_rows)
    if not cdf.empty:
        cdf.to_csv(out_dir / "consistency.csv", index=False)

    completeness = check_completeness(df, seeds, consistency_seeds, policies)
    if failures:
        completeness["sub_run_failures"] = failures
        completeness["ok"] = False
    (out_dir / "completeness.json").write_text(
        json.dumps(completeness, ensure_ascii=False, indent=2), encoding="utf-8")

    arm_a = df[df["arm"] == "A"]
    summary: dict = {"config": vars(a), "policies": policies,
                     "n_seeds": len(seeds), "consistency_seeds": len(consistency_seeds),
                     "budget": a.budget, "target": TARGET_COLUMN,
                     "direction": "minimize (lower is stronger)"}
    means = arm_a.groupby("policy").agg(
        mean_best_pred=("best_pred_gbsa", "mean"),
        sd_best_pred=("best_pred_gbsa", "std"),
        mean_n_unique_predicted=("n_unique_predicted", "mean"),
        mean_seconds=("seconds_total", "mean"),
        mean_diversity_top50=("diversity_top50", "mean"),
        mean_ood_fraction_top50=("ood_fraction_top50", "mean"),
        mean_hamming_top50=("mean_min_train_hamming_top50", "mean"),
        mean_feas_rate=("feas_rate", "mean"),
        mean_n_groups=("n_groups", "mean"),
    ).reset_index()
    summary["per_policy"] = means.to_dict("records")
    summary["stop_reason_counts"] = (arm_a.groupby(["policy", "stop_reason"])
                                     .size().reset_index(name="n").to_dict("records"))

    aligned, matched_summary = aligned_prediction_comparison(
        runs_dir, seeds, policies)
    if not aligned.empty:
        aligned.to_csv(out_dir / "aligned_runs.csv", index=False)
        summary["matched_seed_actual_prediction_comparison"] = matched_summary

    paired = {}
    baseline = ("stratified_random" if "stratified_random" in policies
                else "matched_seed_random_walk")
    for policy in policies:
        if policy == baseline:
            continue
        paired[policy] = {
            "best_pred": paired_delta(df, policy, "best_pred_gbsa", "lower", baseline),
            "diversity": paired_delta(df, policy, "diversity_top50", "higher", baseline),
            "ood_fraction": paired_delta(df, policy, "ood_fraction_top50", "lower", baseline),
            "hamming": paired_delta(df, policy, "mean_min_train_hamming_top50",
                                    "lower", baseline),
        }
    summary["paired_vs_random"] = paired

    cross = {}
    for policy in policies if not a.skip_cross_model else []:
        sub = arm_a[arm_a["policy"] == policy]
        entry = {}
        for i in (1, 2):
            col = f"spearman_model_a_vs_{i}"
            vals = pd.to_numeric(sub[col], errors="coerce").to_numpy(float)
            lo, hi = bootstrap_ci(vals)
            entry[col] = {"mean": float(np.nanmean(vals)), "ci95": [lo, hi],
                          "n": int(np.isfinite(vals).sum())}
        cross[policy] = entry
    summary["cross_model_ranking_stability"] = cross

    if not cdf.empty:
        cons = {}
        for policy in policies:
            vals = cdf[cdf["policy"] == policy]["jaccard_top50"].to_numpy(float)
            lo, hi = bootstrap_ci(vals)
            cons[policy] = {"mean_jaccard_top50": float(np.nanmean(vals)),
                            "ci95": [lo, hi], "n": int(np.isfinite(vals).sum())}
        summary["retrain_consistency"] = cons
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False,
                                                     indent=2), encoding="utf-8")

    metrics_path = Path(a.output) if a.output else out_dir / "metrics.csv"
    if not metrics_path.is_absolute():
        metrics_path = PROJECT_ROOT / metrics_path
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    final_rows = []
    for policy in policies:
        sub = arm_a[arm_a["policy"] == policy]
        p = paired.get(policy, {})
        cr = cross.get(policy, {})
        final_rows.append({
            "policy": policy, "n_seeds": int(len(sub)),
            "mean_best_pred_gbsa": float(sub["best_pred_gbsa"].mean()),
            "mean_n_unique_predicted": float(sub["n_unique_predicted"].mean()),
            "mean_diversity_top50": float(sub["diversity_top50"].mean()),
            "mean_ood_fraction_top50": float(sub["ood_fraction_top50"].mean()),
            "mean_seconds_total": float(sub["seconds_total"].mean()),
            "delta_best_pred_vs_random": p.get("best_pred", {}).get("delta_mean", 0.0),
            "delta_best_pred_ci_low": p.get("best_pred", {}).get("delta_ci95", [0, 0])[0],
            "delta_best_pred_ci_high": p.get("best_pred", {}).get("delta_ci95", [0, 0])[1],
            "best_pred_wins": p.get("best_pred", {}).get("wins", 0),
            "best_pred_ties": p.get("best_pred", {}).get("ties", 0),
            "best_pred_losses": p.get("best_pred", {}).get("losses", 0),
            "delta_diversity_top50": p.get("diversity", {}).get("delta_mean", 0.0),
            "delta_ood_fraction_top50": p.get("ood_fraction", {}).get("delta_mean", 0.0),
            "delta_hamming_top50": p.get("hamming", {}).get("delta_mean", 0.0),
            "mean_spearman_model_a_vs_1": cr.get("spearman_model_a_vs_1", {}).get("mean", float("nan")),
            "mean_spearman_model_a_vs_2": cr.get("spearman_model_a_vs_2", {}).get("mean", float("nan")),
        })
    pd.DataFrame(final_rows).to_csv(metrics_path, index=False)
    print(json.dumps(paired, ensure_ascii=False, indent=2))
    if not completeness["ok"]:
        print(f"[FAILED] incomplete comparison -> {out_dir.relative_to(PROJECT_ROOT)}")
        return 1
    print(f"[ok] {out_dir.relative_to(PROJECT_ROOT)} | metrics -> {metrics_path} | "
          f"rows={completeness['observed_rows']}/{completeness['expected_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

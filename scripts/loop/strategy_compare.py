"""Equal-budget strategy comparison for the closed loop (plan §8.1, package D).

Runs `scripts/loop/loop.py` once per (seed, strategy) on the *same* split and the
same candidate pool, then reports paired comparisons against `random`.

What the numbers mean
---------------------
`--mode retrospective` uses the proxy table as the measurement oracle: the loop may
only reveal `gbsa` for candidates inside its own pool, the holdout is never queried,
and labels come from the same table the library was measured in.  The resulting
curves therefore answer "which strategy finds strong proxy binders fastest on this
dataset", NOT "which strategy will find strong molecules in the lab".

Outputs (under --out-dir):
  runs.csv                 one row per (seed, strategy, round), including the
                           pre-registered additional metrics (top-5 median of newly
                           queried labels, top-10% pool recall, cumulative curves)
  summary.json             per-round means; paired deltas vs random with bootstrap CI,
                           median/Q25/Q75, win rates, worst-10% seeds, top-5 median and
                           top-10% recall deltas, and the aggregated OOD strata
  completeness.json        row-count/pairing-key checks; the driver exits non-zero if
                           any seed/strategy/round is missing or a sub-run failed
  ood_strata_aggregate.csv per (strategy, n_from_wt): pooled OOD rate, mean distance
                           to the pre-acquisition training set, revealed-count, mean
                           prediction error, and how often the stratum's best candidate
                           was selected
  README.txt               the caveat above, verbatim, plus the resolved command line

Usage:
  python scripts/loop/strategy_compare.py --seeds 30 --out-dir evidence/loop_20260930/strategy_compare
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.loop.interface import TARGET_COLUMN, normalize  # noqa: E402
from scripts.loop.selection import STRATEGIES  # noqa: E402


def check_completeness(df: pd.DataFrame, seeds, strategies, a) -> dict:
    """§5.3/§7: a comparison with a missing seed/strategy/round is not a result."""
    problems: list[str] = []
    seed_list = list(seeds)
    expected_rows = len(seed_list) * len(strategies) * a.rounds
    if len(df) != expected_rows:
        problems.append(f"row count {len(df)} != expected {expected_rows} "
                        f"({len(seed_list)} seeds x {len(strategies)} strategies "
                        f"x {a.rounds} rounds)")
    key = ["seed", "strategy", "round"]
    if df.duplicated(subset=key).any():
        problems.append("duplicated (seed, strategy, round) rows")
    for seed in seed_list:
        for strategy in strategies:
            sub = df[(df["seed"] == seed) & (df["strategy"] == strategy)]
            rounds = sorted(sub["round"].astype(int).tolist())
            if rounds != list(range(1, a.rounds + 1)):
                problems.append(f"seed={seed} strategy={strategy} rounds={rounds}")
    if df.empty:
        problems.append("no rows at all")
    else:
        required = ["best_selected", "regret", "pool_best", "n_new_unique"]
        for column in required:
            if column not in df.columns:
                problems.append(f"missing column {column}")
            elif df[column].isna().all():
                problems.append(f"column {column} is entirely missing values")
        if not np.isfinite(df["pool_best"].to_numpy(float)).all():
            problems.append("non-finite pool_best")
        if (df["regret"].dropna() < -1e-9).any():
            problems.append("negative regret (discovery metric includes shared init labels?)")
    sub_runs = df[["seed", "strategy"]].drop_duplicates() if not df.empty else pd.DataFrame()
    return {"ok": not problems, "problems": problems,
            "expected_runs": len(seed_list) * len(strategies),
            # distinct (seed, strategy) sub-runs; observed_rows below counts the
            # per-round rows each sub-run contributes
            "observed_runs": int(len(sub_runs)),
            "expected_rows": expected_rows, "observed_rows": int(len(df)),
            "seeds": seed_list, "strategies": strategies, "rounds": a.rounds}


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def build_cmd(a, seed: int, strategy: str, run_dir: Path) -> list[str]:
    return [
        sys.executable, str(PROJECT_ROOT / "scripts/loop/loop.py"),
        "--mode", a.mode, "--rounds", str(a.rounds), "--budget", str(a.budget),
        "--n-init", str(a.n_init), "--n-holdout", str(a.n_holdout),
        "--pool-size", str(a.pool_size), "--surrogate", a.surrogate,
        "--n-members", str(a.n_members), "--strategy", strategy, "--seed", str(seed),
        "--data-dir", a.data_dir,
        "--output", str(run_dir / "metrics.csv"),
        "--log-dir", str(run_dir / "checkpoints"),
    ]


def pool_truth(a, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(sequences, values) of the retrospective pool, exactly as the loop builds it.

    The loop's split: `perm = RandomState(seed).permutation(n)`; init = perm[:n_init],
    holdout = perm[n_init:n_init+n_holdout], pool = perm[n_init+n_holdout : +n_pool].
    This function reproduces the pool so the driver can compute recall against the
    pool's true top-10% without ever reading a sub-run's holdout.
    """
    df = pd.read_csv(PROJECT_ROOT / a.data_dir / "fad_proxy2000_v2_full.csv")
    df["Sequence"] = df["Sequence"].map(normalize)
    values = df[TARGET_COLUMN].astype(float).to_numpy()
    perm = np.random.RandomState(seed).permutation(len(df))
    n_pool = min(a.pool_size, len(df) - a.n_init - a.n_holdout)
    idx = perm[a.n_init + a.n_holdout:a.n_init + a.n_holdout + n_pool]
    return df["Sequence"].to_numpy()[idx], values[idx]


def pool_best(a, seed: int) -> float:
    """Best (lowest) gbsa in the retrospective pool — the driver's own split copy."""
    _, values = pool_truth(a, seed)
    return float(values.min())


def top10_recall(new_queried: set[str], pool_seqs: np.ndarray, pool_values: np.ndarray) -> float:
    """Fraction of the pool's true best-10% (lowest gbsa) that was queried.

    `new_queried` must contain only sequences the strategy itself queried (init
    labels never come from the pool, so no subtraction is needed).
    """
    if len(pool_values) == 0:
        return float("nan")
    k = max(1, int(round(0.1 * len(pool_values))))
    top_idx = set(np.argsort(pool_values)[:k])
    hits = sum(1 for i in top_idx if pool_seqs[i] in new_queried)
    return hits / k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="retrospective", choices=["retrospective"])
    ap.add_argument("--strategies",
                    default="random,greedy,greedy_diverse,maxmin_coverage")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--n-init", type=int, default=300)
    ap.add_argument("--n-holdout", type=int, default=200)
    ap.add_argument("--pool-size", type=int, default=600)
    ap.add_argument("--surrogate", default="xgb")
    ap.add_argument("--n-members", type=int, default=5)
    ap.add_argument("--data-dir", default="data/proxy2000_v2")
    ap.add_argument("--out-dir", default="evidence/loop_20260930/strategy_compare")
    ap.add_argument("--output", default=None,
                    help="per-strategy final-round metrics CSV (set by run_experiment.py)")
    a = ap.parse_args()

    strategies = [s.strip() for s in a.strategies.split(",") if s.strip()]
    for s in strategies:
        if s not in STRATEGIES:
            raise ValueError(f"unknown strategy {s!r}; available: {sorted(STRATEGIES)}")

    out_dir = PROJECT_ROOT / a.out_dir
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.txt").write_text(
        "Retrospective proxy-oracle strategy comparison.\n"
        "Labels come from the proxy table through TableOracle: only pool candidates can\n"
        "be revealed, the holdout is never queried, and every strategy sees the same\n"
        "split, pool, budget and seed. These numbers rank strategies on the PROXY target\n"
        "only; they are not wet-lab evidence.\n"
        f"argv: {' '.join(sys.argv[1:])}\n", encoding="utf-8")

    rows: list[dict] = []
    failures: list[dict] = []
    seeds = range(a.seed_start, a.seed_start + a.seeds)
    for seed in seeds:
        best_in_pool = pool_best(a, seed)
        for strategy in strategies:
            run_dir = runs_dir / f"seed{seed}_{strategy}"
            run_dir.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(build_cmd(a, seed, strategy, run_dir),
                                  cwd=str(PROJECT_ROOT), capture_output=True, text=True)
            if proc.returncode != 0:
                (run_dir / "stderr.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
                print(f"[fail] seed={seed} strategy={strategy} rc={proc.returncode}")
                failures.append({"seed": seed, "strategy": strategy,
                                 "reason": f"sub-run rc={proc.returncode}"})
                continue
            metrics = pd.read_csv(run_dir / "metrics.csv")
            state = json.loads((run_dir / "checkpoints" / "state.json").read_text(encoding="utf-8"))
            # labeled_values is in reveal order (init first, then one block per round),
            # so the cumulative best is taken over the first n_labeled_after entries.
            values = np.asarray(state["labeled_values"], dtype=float)
            pool_seqs, pool_values = pool_truth(a, seed)
            for _, r in metrics.iterrows():
                k = int(r["round"])
                n_after = int(r["n_labeled_after"])
                best_so_far = float(np.min(values[:n_after])) if n_after else float("nan")
                # Discovery metric: best label among the candidates *this strategy
                # queried* (init labels are shared by every arm, so they cannot
                # separate strategies and would let regret go negative).
                new_block = values[a.n_init:n_after]
                best_new = float(np.min(new_block)) if new_block.size else float("nan")
                top5_cum = (float(np.median(np.sort(new_block)[:5]))
                            if new_block.size else float("nan"))
                recall_cum = top10_recall(
                    set(state["labeled_sequences"][a.n_init:n_after]),
                    pool_seqs, pool_values)
                batch_file = run_dir / "checkpoints" / f"round_{k:03d}_batch.csv"
                best_pred = (float(pd.read_csv(batch_file)["pred_gbsa"].min())
                             if batch_file.is_file() else float("nan"))
                rows.append({
                    "seed": seed, "strategy": strategy, "round": k,
                    "n_labeled_after": n_after,
                    "n_new_unique": int(r["n_new_unique"]),
                    "best_labeled": best_so_far,
                    "best_selected": best_new,
                    "pool_best": best_in_pool,
                    "regret": (best_new - best_in_pool) if np.isfinite(best_new) else float("nan"),
                    "best_pred_in_batch": best_pred,
                    "top5_median_new_cumulative": top5_cum,
                    "top10_recall_cumulative": recall_cum,
                    "holdout_spearman_pre": r.get("holdout_spearman_pre", np.nan),
                    "holdout_spearman_post": r.get("holdout_spearman_post", np.nan),
                    "ood_fraction": r.get("ood_fraction", np.nan),
                    "status": r["status"],
                })
        print(f"[seed {seed}] done ({len(rows)} rows so far)")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "runs.csv", index=False)

    completeness = check_completeness(df, seeds, strategies, a)
    if failures:
        completeness["sub_run_failures"] = failures
        completeness["ok"] = False
    (out_dir / "completeness.json").write_text(
        json.dumps(completeness, ensure_ascii=False, indent=2), encoding="utf-8")
    if not completeness["ok"]:
        print("[INCOMPLETE] " + json.dumps(completeness["problems"], ensure_ascii=False))

    summary: dict = {"config": vars(a), "strategies": strategies,
                     "n_seeds": a.seeds, "rounds": a.rounds, "budget": a.budget,
                     "target": TARGET_COLUMN, "direction": "minimize (lower is stronger)"}
    per_round = df.groupby(["strategy", "round"]).agg(
        mean_best_selected=("best_selected", "mean"), sd_best_selected=("best_selected", "std"),
        mean_regret=("regret", "mean"),
        mean_holdout_post=("holdout_spearman_post", "mean")).reset_index()
    summary["per_round"] = per_round.to_dict("records")

    paired = paired_summary(df, strategies, a)
    summary["paired_vs_random_final_round"] = paired
    strata_agg = aggregate_ood_strata(out_dir, runs_dir, seeds, strategies)
    if strata_agg is not None:
        summary["ood_strata_aggregate"] = strata_agg.to_dict("records")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    metrics_path = Path(a.output) if a.output else out_dir / "metrics.csv"
    if not metrics_path.is_absolute():
        metrics_path = PROJECT_ROOT / metrics_path
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    last = df[df["round"] == df["round"].max()]
    final_rows = []
    for strategy in strategies:
        sub = last[last["strategy"] == strategy]
        p = paired.get(strategy, {})
        final_rows.append({
            "strategy": strategy, "n_seeds": int(len(sub)),
            "final_round": int(a.rounds),
            "mean_best_selected": float(sub["best_selected"].mean()),
            "mean_regret": float(sub["regret"].mean()),
            "mean_top5_median_new": float(sub["top5_median_new_cumulative"].mean()),
            "mean_top10_recall": float(sub["top10_recall_cumulative"].mean()),
            "mean_holdout_spearman_post": float(pd.to_numeric(
                sub["holdout_spearman_post"], errors="coerce").mean()),
            "delta_regret_vs_random": p.get("delta_regret_mean", 0.0),
            "delta_regret_ci_low": p.get("delta_regret_ci95", [0, 0])[0],
            "delta_regret_ci_high": p.get("delta_regret_ci95", [0, 0])[1],
            "regret_wins": p.get("regret_wins", int(len(sub))),
            "regret_ties": p.get("regret_ties", 0),
            "regret_losses": p.get("regret_losses", 0),
            "worst10pct_delta_regret_mean": p.get("worst10pct_delta_regret_mean", 0.0),
            "best10pct_delta_regret_mean": p.get("best10pct_delta_regret_mean", 0.0),
            "best10pct_share_of_mean": p.get("best10pct_share_of_mean", 0.0),
            "delta_top5_median_new": p.get("delta_top5_median_new_mean", 0.0),
            "delta_top10_recall": p.get("delta_top10_recall_mean", 0.0),
        })
    pd.DataFrame(final_rows).to_csv(metrics_path, index=False)
    print(json.dumps(paired, ensure_ascii=False, indent=2))
    if not completeness["ok"]:
        print(f"[FAILED] incomplete comparison -> {out_dir.relative_to(PROJECT_ROOT)}")
        return 1
    print(f"[ok] {out_dir.relative_to(PROJECT_ROOT)} | metrics -> {metrics_path} | "
          f"rows={completeness['observed_rows']}/{completeness['expected_rows']}")
    return 0

def tail_stats(d: np.ndarray, common, n_tail: int) -> dict:
    """Tail decomposition of paired deltas (d < 0 = strategy better than random).

    worst10pct = the 10% seeds with the LARGEST deltas (the strategy did worst
    relative to random there); best10pct = the 10% with the most negative deltas.
    """
    asc = np.argsort(d)
    best = asc[:n_tail]
    worst = asc[::-1][:n_tail]
    total = float(np.nansum(d))
    return {
        "worst10pct_seeds": [int(common[i]) for i in worst],
        "worst10pct_delta_regret_mean": float(np.nanmean(d[worst])),
        "best10pct_seeds": [int(common[i]) for i in best],
        "best10pct_delta_regret_mean": float(np.nanmean(d[best])),
        "best10pct_share_of_mean": float(np.nansum(d[best]) / total) if total else float("nan"),
    }


def paired_summary(df: pd.DataFrame, strategies, a) -> dict:
    """Per-strategy paired comparison against `random` on the final round."""
    last = df[df["round"] == df["round"].max()]
    base = last[last["strategy"] == "random"].set_index("seed")
    paired = {}
    for strategy in strategies:
        if strategy == "random":
            continue
        other = last[last["strategy"] == strategy].set_index("seed")
        common = sorted(set(base.index) & set(other.index))
        if not common:
            continue
        d_regret = (other.loc[common, "regret"] - base.loc[common, "regret"]).to_numpy(float)
        d_spear = (other.loc[common, "holdout_spearman_post"]
                   - base.loc[common, "holdout_spearman_post"]).to_numpy(float)
        d_top5 = (other.loc[common, "top5_median_new_cumulative"]
                  - base.loc[common, "top5_median_new_cumulative"]).to_numpy(float)
        d_recall = (other.loc[common, "top10_recall_cumulative"]
                    - base.loc[common, "top10_recall_cumulative"]).to_numpy(float)
        lo_r, hi_r = bootstrap_ci(d_regret)
        lo_s, hi_s = bootstrap_ci(d_spear)
        lo_t, hi_t = bootstrap_ci(d_top5)
        lo_c, hi_c = bootstrap_ci(d_recall)
        tails = tail_stats(d_regret, common, max(1, int(round(0.1 * len(common)))))
        paired[strategy] = {
            "n_pairs": len(common),
            "delta_regret_mean": float(np.nanmean(d_regret)),
            "delta_regret_ci95": [lo_r, hi_r],
            "delta_regret_median": float(np.nanmedian(d_regret)),
            "delta_regret_q25": float(np.nanpercentile(d_regret, 25)),
            "delta_regret_q75": float(np.nanpercentile(d_regret, 75)),
            "regret_wins": int(np.nansum(d_regret < 0)),
            "regret_ties": int(np.nansum(np.abs(d_regret) <= 1e-12)),
            "regret_losses": int(np.nansum(d_regret > 0)),
            **tails,
            "delta_holdout_spearman_mean": float(np.nanmean(d_spear)),
            "delta_holdout_spearman_ci95": [lo_s, hi_s],
            "spearman_wins": int(np.nansum(d_spear > 0)),
            "delta_top5_median_new_mean": float(np.nanmean(d_top5)),
            "delta_top5_median_new_ci95": [lo_t, hi_t],
            "top5_wins": int(np.nansum(d_top5 < 0)),
            "delta_top10_recall_mean": float(np.nanmean(d_recall)),
            "delta_top10_recall_ci95": [lo_c, hi_c],
            "recall_wins": int(np.nansum(d_recall > 0)),
        }
    return paired


def aggregate_ood_strata(out_dir: Path, runs_dir: Path, seeds, strategies) -> dict | None:
    """§5.3.4 stress layer: aggregate per-round OOD strata across all sub-runs."""
    strata_rows = []
    for seed in seeds:
        for strategy in strategies:
            run_dir = runs_dir / f"seed{seed}_{strategy}"
            for strata_file in sorted((run_dir / "checkpoints").glob("round_*_ood_strata.csv")):
                for _, s in pd.read_csv(strata_file).iterrows():
                    strata_rows.append({"seed": seed, "strategy": strategy, **s.to_dict()})
    if not strata_rows:
        return None
    sdf = pd.DataFrame(strata_rows)
    sdf["pool_ood_rate"] = (sdf["pool_ood_count"].astype(int)
                            / sdf["pool_count"].astype(int))
    for column in ("mean_min_train_hamming", "mean_min_train_hamming_selected",
                   "mae_pred_vs_measured"):
        sdf[column] = pd.to_numeric(sdf[column], errors="coerce")
    agg = sdf.groupby(["strategy", "n_from_wt"]).agg(
        n_rounds=("run_round", "count"),
        mean_pool_ood_rate=("pool_ood_rate", "mean"),
        mean_min_train_hamming=("mean_min_train_hamming", "mean"),
        mean_min_train_hamming_selected=("mean_min_train_hamming_selected", "mean"),
        total_revealed=("revealed_count", "sum"),
        mean_mae=("mae_pred_vs_measured", "mean"),
        selected_best_rate=("selected_best_in_stratum", "mean"),
    ).reset_index()
    agg.to_csv(out_dir / "ood_strata_aggregate.csv", index=False)
    return agg


if __name__ == "__main__":
    raise SystemExit(main())

"""Recompute a comparison package's derived summaries from its `runs.csv`.

The 2026-10-01 audit found that the shipped `summary.json` mislabeled the
`worst10pct` tail (it pointed at the seeds where the strategy improved MOST) and
that `completeness.json.observed_runs` counted round-rows instead of sub-runs.
`runs.csv` is untouched and the primary statistics (mean/CI/median/wins, top-5,
top-10% recall, holdout deltas) are deterministic functions of it, so they are
recomputed identically; only the tail labels, the win/tie/loss breakdown and the
`observed_runs` semantic change. This tool never re-runs experiments.

Usage:
  python tools/recompute_compare_summary.py evidence/loop_20260930/strategy_compare_100seed
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.loop.interface import TARGET_COLUMN  # noqa: E402
from scripts.loop.strategy_compare import (bootstrap_ci, check_completeness,  # noqa: E402
                                           paired_summary)


class _A:
    def __init__(self, rounds: int):
        self.rounds = rounds


def main() -> int:
    out_dir = Path(sys.argv[1])
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    summary_old = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    cfg = summary_old["config"]
    strategies = summary_old["strategies"]
    seeds = range(cfg["seed_start"], cfg["seed_start"] + cfg["seeds"])

    df = pd.read_csv(out_dir / "runs.csv")
    if "top5_median_new_cumulative" not in df.columns:
        print(f"[skip] {out_dir.relative_to(ROOT)} predates the tail/top5 fields; "
              "its summary has no mislabeled worst10pct and needs no recomputation")
        return 0
    a = _A(cfg["rounds"])
    completeness = check_completeness(df, seeds, strategies, a)

    summary: dict = {"config": cfg, "strategies": strategies,
                     "n_seeds": cfg["seeds"], "rounds": cfg["rounds"],
                     "budget": cfg["budget"], "target": TARGET_COLUMN,
                     "direction": "minimize (lower is stronger)"}
    per_round = df.groupby(["strategy", "round"]).agg(
        mean_best_selected=("best_selected", "mean"), sd_best_selected=("best_selected", "std"),
        mean_regret=("regret", "mean"),
        mean_holdout_post=("holdout_spearman_post", "mean")).reset_index()
    summary["per_round"] = per_round.to_dict("records")
    summary["paired_vs_random_final_round"] = paired_summary(df, strategies, a)
    strata_path = out_dir / "ood_strata_aggregate.csv"
    if strata_path.is_file():
        summary["ood_strata_aggregate"] = pd.read_csv(strata_path).to_dict("records")

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    (out_dir / "completeness.json").write_text(
        json.dumps(completeness, ensure_ascii=False, indent=2), encoding="utf-8")

    last = df[df["round"] == df["round"].max()]
    paired = summary["paired_vs_random_final_round"]
    final_rows = []
    for strategy in strategies:
        sub = last[last["strategy"] == strategy]
        p = paired.get(strategy, {})
        final_rows.append({
            "strategy": strategy, "n_seeds": int(len(sub)),
            "final_round": int(cfg["rounds"]),
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
    pd.DataFrame(final_rows).to_csv(out_dir / "metrics_recomputed.csv", index=False)

    (out_dir / "RECOMPUTED.md").write_text(
        "Derived summaries in this directory were recomputed from `runs.csv` on "
        "2026-10-01 with the corrected tail labeling (worst10pct = the seeds where "
        "the strategy did WORST relative to random; best10pct = the biggest "
        "improvements) and the corrected `completeness.json.observed_runs` semantic "
        "(distinct seed x strategy sub-runs, not round rows).\n"
        "`runs.csv` and the server run directory are untouched; the primary "
        "statistics (mean/CI/median/quartiles/wins, top-5, top-10% recall, holdout "
        "deltas) are deterministic functions of `runs.csv` and recompute identically. "
        "`metrics_recomputed.csv` is the corrected per-strategy roll-up.\n",
        encoding="utf-8")
    print(f"[ok] recomputed {out_dir.relative_to(ROOT)} "
          f"(observed_runs={completeness['observed_runs']})")
    for strategy, p in paired.items():
        print(f"  {strategy}: wins/ties/losses = {p['regret_wins']}/{p['regret_ties']}/"
              f"{p['regret_losses']}, worst10={p['worst10pct_delta_regret_mean']:.4f}, "
              f"best10={p['best10pct_delta_regret_mean']:.4f} "
              f"(share {p['best10pct_share_of_mean']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

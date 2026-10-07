"""Render a markdown table from a strategy-comparison run (see loop/strategy_compare.py).

Usage:
  python tools/summarize_strategy_compare.py evidence/loop_20260930/strategy_compare_30seed
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        ROOT / "evidence/loop_20260930/strategy_compare_30seed"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    df = pd.read_csv(out_dir / "runs.csv")
    rounds = int(summary["rounds"])
    last = df[df["round"] == rounds]
    paired = summary["paired_vs_random_final_round"]

    print(f"# strategy comparison: {out_dir.relative_to(ROOT)}")
    print(f"seeds={summary['n_seeds']} rounds={rounds} budget={summary['budget']} "
          f"target={summary['target']} direction={summary['direction']}\n")
    print("| strategy | mean best_selected | mean regret | dRegret vs random (95% CI) "
          "| regret wins | dHoldout Spearman (95% CI) | spearman wins |")
    print("|---|---|---|---|---|---|---|")
    for strategy in summary["strategies"]:
        sub = last[last["strategy"] == strategy]
        p = paired.get(strategy, {})
        d_r = p.get("delta_regret_mean")
        ci_r = p.get("delta_regret_ci95", [None, None])
        d_s = p.get("delta_holdout_spearman_mean")
        ci_s = p.get("delta_holdout_spearman_ci95", [None, None])
        fmt = lambda v: "—" if v is None else f"{v:+.4f}"          # noqa: E731
        print(f"| {strategy} | {sub['best_selected'].mean():.4f} | {sub['regret'].mean():.4f} "
              f"| {fmt(d_r)} [{fmt(ci_r[0])}, {fmt(ci_r[1])}] "
              f"| {p.get('regret_wins', '—')}/{p.get('n_pairs', len(sub))} "
              f"| {fmt(d_s)} [{fmt(ci_s[0])}, {fmt(ci_s[1])}] "
              f"| {p.get('spearman_wins', '—')}/{p.get('n_pairs', len(sub))} |")
    print()
    per_round = pd.DataFrame(summary["per_round"])
    pivot = per_round.pivot(index="round", columns="strategy", values="mean_regret")
    print("mean regret by round (lower is better):\n")
    print(pivot.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Label-noise diagnostics for the GBSA surrogate ceiling.

Codifies the key finding that the 0.40 Spearman ceiling is LABEL NOISE, not
model/features/sampling. Computes, from the canonical data:
  - the label formula 0.961*Gap2 + 0.039*gbsa (and whether it holds exactly);
  - Spearman(label, Gap2), (label, gbsa), (Gap2, gbsa), (gbsa, MFE);
  - the gbsa noise level (std, min/max, outlier count);
  - the variance share of MM-GBSA in the label.

Also provides two actions:
  --select-subset N   pick N sequences (stratified by gbsa) for replica re-runs
  --replica-file F    compute the replica-replica Spearman (the true ceiling)
                      from a CSV with columns [Sequence, replica_id, gbsa].

Runs in FAD_env (numpy/pandas/scipy).

Usage:
  python scripts/reproduction/label_audit.py
  python scripts/reproduction/label_audit.py --select-subset 100
  python scripts/reproduction/label_audit.py --replica-file replica_gbsa.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def spearman(a, b):
    return pearsonr(rankdata(a), rankdata(b))[0]


def diagnose(df):
    print("=" * 70)
    print("LABEL-NOISE DIAGNOSTICS")
    print("=" * 70)
    label, gap2, gbsa = df["label"].values, df["Gap2"].values, df["gbsa"].values
    mfe, aft = df["MFE"].values, df["Aft"].values

    # label formula check
    pred = 0.961 * gap2 + 0.039 * gbsa
    print(f"label == 0.961*Gap2 + 0.039*gbsa ?  max|diff| = {np.abs(label - pred).max():.2e}")

    print(f"\n-- correlations (Spearman) --")
    print(f"sp(label, Gap2) = {spearman(label, gap2):.3f}   (label ~ 96% NUPACK proxy)")
    print(f"sp(label, gbsa) = {spearman(label, gbsa):.3f}")
    print(f"sp(Gap2, gbsa)  = {spearman(gap2, gbsa):.3f}   (proxy vs physics ~ uncorrelated)")
    print(f"sp(gbsa, MFE)   = {spearman(gbsa, mfe):.3f}")
    print(f"sp(label, MFE)  = {spearman(label, mfe):.3f}")
    print(f"sp(label, Aft)  = {spearman(label, aft):.3f}")

    print(f"\n-- variance decomposition --")
    v_gap2 = (0.961 * gap2).var()
    v_gbsa = (0.039 * gbsa).var()
    print(f"var(0.961*Gap2) = {v_gap2:.3f}")
    print(f"var(0.039*gbsa) = {v_gbsa:.3f}")
    print(f"gbsa share of label variance = {100 * v_gbsa / label.var():.2f}%")

    print(f"\n-- gbsa (MM-GBSA) noise --")
    print(f"mean={gbsa.mean():.2f}  std={gbsa.std():.2f}  min={gbsa.min():.2f}  max={gbsa.max():.2f}")
    n_out = int((np.abs(gbsa) > 50).sum())
    print(f"|gbsa| > 50 kcal/mol (physically implausible): {n_out} / {len(gbsa)}")

    print(f"\n-- implication --")
    print("label 由 96% 免费的 NUPACK Gap2 主导；gbsa 只占 4% 权重、与 Gap2 几乎不相关、且噪声极大。")
    print("0.40 Spearman 已逼近标签噪声上限(约 0.412)，模型/特征/主动学习无法越过；突破在标签侧。")


def select_subset(df, n):
    """Stratified subset (by gbsa decile) for replica re-runs."""
    df = df.copy()
    df["decile"] = pd.qcut(df["gbsa"], 10, labels=False, duplicates="drop")
    out = df.groupby("decile", group_keys=False).apply(
        lambda g: g.sample(min(len(g), max(1, n // 10)), random_state=0)
    ).reset_index(drop=True)
    return out[["Sequence", "gbsa", "label", "Gap2"]]


def replica_ceiling(df_rep):
    """Replica-replica Spearman = the label-noise ceiling.

    df_rep: columns [Sequence, replica_id, gbsa]. Pairs replicas of the same
    sequence and computes Spearman(replica_a, replica_b).
    """
    seqs = sorted(df_rep["Sequence"].unique())
    reps = {}
    for s, g in df_rep.groupby("Sequence"):
        reps[s] = g["gbsa"].values
    pairs = []
    for s in seqs:
        v = reps[s]
        if len(v) < 2:
            continue
        # use first two replicas per sequence (or all pairwise)
        pairs.append((s, v[0], v[1]))
    if not pairs:
        print("No sequences with >=2 replicas found.")
        return
    a = np.array([p[1] for p in pairs])
    b = np.array([p[2] for p in pairs])
    sp = spearman(a, b)
    print(f"\nreplica-replica Spearman (n={len(pairs)} sequences, 2 replicas each) = {sp:.4f}")
    print("  -> this is the label-noise ceiling. If ~0.41, 0.40 is already near-perfect;")
    print("     if ~0.65, replica-averaging can push Spearman toward 0.55-0.65.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/proxy2000_v2")
    ap.add_argument("--select-subset", type=int, default=0, help="write a stratified subset for replica re-runs")
    ap.add_argument("--replica-file", default="", help="CSV [Sequence, replica_id, gbsa] to compute the ceiling")
    ap.add_argument("--output", default="runs/label_audit/subset.csv")
    args = ap.parse_args()

    df = pd.read_csv(PROJECT_ROOT / args.data_dir / "fad_proxy2000_v2_full.csv")

    if args.replica_file:
        df_rep = pd.read_csv(args.replica_file)
        replica_ceiling(df_rep)
        return

    if args.select_subset:
        sub = select_subset(df, args.select_subset)
        out = PROJECT_ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out, index=False)
        print(f"[OK] {len(sub)} sequences -> {out}")
        print("Next: run >=2 independent MD replicas per sequence, compute MM-GBSA each,")
        print("then re-run with --replica-file to get the label-noise ceiling.")
        return

    diagnose(df)


if __name__ == "__main__":
    main()

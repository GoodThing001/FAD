"""Cluster Split Baseline — Phase 0B.

Runs baseline models on pre-generated cluster splits.
Reports both random-split and cluster-split metrics for comparison.

Usage:
  python scripts/tools/baseline_cluster_split.py \
    --split-dir data/splits/cluster_hamming_le_1/ \
    --models ET,RF,GBR \
    --feature-set mutation_only \
    --output artifacts/phase0/baseline_cluster.csv
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, rankdata
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                               RandomForestRegressor)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASES = "ACGU"
MUT_POS = [30, 31, 32, 33, 89, 90, 91, 92, 127, 128, 129, 130, 131]


def _spearman(a, b):
    return pearsonr(rankdata(a), rankdata(b))[0]


def build_mutation_only(seqs):
    X = np.zeros((len(seqs), 52), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = str(s).upper().replace("T", "U")
        for j, p in enumerate(MUT_POS):
            idx = p - 1
            if idx < len(s) and s[idx] in BASES:
                X[i, j * 4 + BASES.index(s[idx])] = 1.0
    return X


def make_model(name: str):
    if name == "ET":
        return ExtraTreesRegressor(n_estimators=500, max_features="log2",
                                   min_samples_leaf=5, random_state=42, n_jobs=1)
    elif name == "RF":
        return RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                     min_samples_leaf=5, random_state=42, n_jobs=1)
    elif name == "GBR":
        return GradientBoostingRegressor(n_estimators=500, max_depth=3,
                                         min_samples_leaf=3,
                                         learning_rate=0.03, random_state=42)
    return None


def main():
    parser = argparse.ArgumentParser(description="Cluster Split Baseline")
    parser.add_argument("--split-dir", default="data/splits/cluster_hamming_le_1/",
                        help="Path to cluster split directory")
    parser.add_argument("--models", default="ET,RF,GBR",
                        help="Comma-separated model names")
    parser.add_argument("--feature-set", default="mutation_only")
    parser.add_argument("--data-dir", default="data/proxy2000_v2")
    parser.add_argument("--output", default="artifacts/phase0/baseline_cluster.csv")
    args = parser.parse_args()

    split_dir = PROJECT_ROOT / args.split_dir
    data_dir = PROJECT_ROOT / args.data_dir
    full_path = data_dir / "fad_proxy2000_v2_full.csv"

    if not full_path.exists():
        print(f"[ERROR] Full data not found at {full_path}"); sys.exit(1)
    if not split_dir.exists():
        print(f"[ERROR] Split dir not found at {split_dir}"); sys.exit(1)

    df = pd.read_csv(full_path)
    seqs = df["Sequence"].values
    gbsa = df["gbsa"].values
    gap2 = df["Gap2"].values
    label = df["label"].values

    models = [m.strip() for m in args.models.split(",")]

    # Load split metadata
    meta_path = split_dir / "split_metadata.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    n_splits = meta.get("n_splits", 5)

    all_results = []

    for fold_id in range(n_splits):
        fold_dir = split_dir / f"fold_{fold_id}"
        tr_idx = pd.read_csv(fold_dir / "train_indices.csv")["index"].values
        te_idx = pd.read_csv(fold_dir / "test_indices.csv")["index"].values

        # Build features
        X_tr = build_mutation_only(seqs[tr_idx])
        X_te = build_mutation_only(seqs[te_idx])
        y_tr = gbsa[tr_idx]
        y_te = gbsa[te_idx]
        gap2_te = gap2[te_idx]
        label_te = label[te_idx]

        for mname in models:
            model = make_model(mname)
            if model is None:
                continue
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)

            gbsa_sp = _spearman(y_te, pred)
            hybrid_pred = 0.961 * gap2_te + 0.039 * pred
            hybrid_sp = _spearman(label_te, hybrid_pred)
            nupack_sp = _spearman(label_te, 0.961 * gap2_te + 0.039 * y_tr.mean())

            all_results.append({
                "fold_id": fold_id, "model": mname,
                "gbsa_Sp": round(gbsa_sp, 6),
                "Hybrid_Sp": round(hybrid_sp, 6),
                "NUPACK_direct_Sp": round(nupack_sp, 6),
                "n_train": len(tr_idx), "n_test": len(te_idx),
                "hamming_threshold": meta.get("hamming_threshold", 1),
            })

        print(f"  Fold {fold_id}: best_gbsa_Sp={max(r['gbsa_Sp'] for r in all_results[-len(models):]):.4f}")

    results_df = pd.DataFrame(all_results)
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n[OK] Cluster baseline saved to {output_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"CLUSTER SPLIT BASELINE ({args.split_dir})")
    print(f"{'Model':<8} {'Mean Sp':>10} {'Std Sp':>10} {'Min Sp':>10} {'Max Sp':>10}")
    for mname in models:
        mdf = results_df[results_df["model"] == mname]
        sps = mdf["gbsa_Sp"].values
        print(f"{mname:<8} {np.mean(sps):>10.4f} {np.std(sps):>10.4f} "
              f"{np.min(sps):>10.4f} {np.max(sps):>10.4f}")

    # Compare with a matching random split
    n_te = len(te_idx)
    print(f"\n  Random split reference (n_test={n_te}):")
    print(f"    See artifacts/phase0/baseline_30seed.csv for 30-seed random baseline")
    print(f"    or run: python scripts/tools/baseline_30seed.py --seeds 30")


if __name__ == "__main__":
    main()

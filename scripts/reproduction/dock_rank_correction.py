"""dock RANK-based correction of gbsa's ranking (targeted, not global).

The user's idea: use dock's ORDERING to correct gbsa's ORDERING, only where the
two disagree strongly (gbsa is likely noisy there), not global fusion (which
failed: dock and gbsa correlate only 0.289).

Tests (30-seed XGB on 303d + NUPACK):
  1. baseline: Spearman(pred, gbsa) over ALL test sequences
  2. diagnostic: Spearman over only LOW-disagreement test sequences
     (|rank(gbsa)-rank(dock)| small) -> is gbsa noise concentrated in disagreement?
  3. correction: for HIGH-disagreement TRAINING sequences, replace gbsa with the
     dock-implied gbsa (dock rescaled), retrain, eval on ORIGINAL gbsa

Runs in FAD_env.

Usage:
  python -m scripts.reproduction.dock_rank_correction --outer-seeds 30
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from scipy.stats import rankdata
from sklearn.model_selection import train_test_split

from .active_learning_strong import build_seq, load_nupack_useful, _spearman
from .rank_objective import make_xgb

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer-seeds", type=int, default=30)
    ap.add_argument("--data-dir", default="data/proxy2000_v2")
    ap.add_argument("--output", default="runs/dock_rank_correction/results.json")
    args = ap.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    df = pd.read_csv(data_dir / "fad_proxy2000_v2_full.csv")
    seqs = [str(s) for s in df["Sequence"]]
    gbsa = df["gbsa"].to_numpy(dtype=np.float32)
    dock = df["dock"].to_numpy(dtype=np.float32)
    X = np.hstack([build_seq(seqs),
                   load_nupack_useful(data_dir / "nupack_features_full.csv", seqs)[0]])

    seeds = list(range(1, args.outer_seeds + 1))
    all_sps, agree_sps, corr_sps = [], [], []

    for seed in seeds:
        tr, te = train_test_split(range(len(X)), test_size=0.2, random_state=seed)
        lo, hi = np.percentile(gbsa[tr], 2.5), np.percentile(gbsa[tr], 97.5)
        y_tr = np.clip(gbsa[tr], lo, hi).astype(np.float32)

        # baseline
        m = make_xgb(); m.fit(X[tr], y_tr)
        pred = m.predict(X[te])
        all_sps.append(_spearman(gbsa[te], pred))

        # diagnostic: only low-disagreement test sequences (top 70% agreement).
        # Ranks computed WITHIN the test fold only (no cross-fold statistics).
        rg_te = rankdata(gbsa[te]) / len(te)
        rd_te = rankdata(dock[te]) / len(te)
        disagree_te = np.abs(rg_te - rd_te)
        thresh = np.quantile(disagree_te, 0.7)
        agree_mask = disagree_te <= thresh
        agree_sps.append(_spearman(gbsa[te][agree_mask], pred[agree_mask]))

        # correction: for HIGH-disagreement training sequences, use dock-implied gbsa.
        # Ranks computed WITHIN the training fold only (no test-label leakage).
        y_corr = y_tr.copy()
        dock_tr = dock[tr]
        dock_scaled = dock_tr * (y_tr.std() / dock_tr.std())
        rg_tr = rankdata(gbsa[tr]) / len(tr)
        rd_tr = rankdata(dock_tr) / len(tr)
        disagree_tr = np.abs(rg_tr - rd_tr)
        high = disagree_tr > np.quantile(disagree_tr, 0.85)  # top 15% disagreement
        y_corr[high] = dock_scaled[high]
        m = make_xgb(); m.fit(X[tr], y_corr)
        corr_sps.append(_spearman(gbsa[te], m.predict(X[te])))

    res = {
        "baseline_all": float(np.mean(all_sps)),
        "baseline_low_disagreement_test": float(np.mean(agree_sps)),
        "dock_rank_corrected": float(np.mean(corr_sps)),
    }
    print(f"baseline (all test seqs)          : {res['baseline_all']:.4f}")
    print(f"baseline (low-disagreement test)  : {res['baseline_low_disagreement_test']:.4f}  "
          f"<- if much higher, gbsa noise is in disagreement seqs")
    print(f"dock rank-corrected (train)       : {res['dock_rank_corrected']:.4f}")

    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print(f"[OK] -> {out}")


if __name__ == "__main__":
    main()

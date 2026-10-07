"""k-NN Hamming label smoothing: denoise gbsa labels by averaging neighbors.

The gbsa label is noisy (std 10.4, 25 outliers >50 kcal/mol). Since the 2000
sequences differ only at 13 mutation positions, sequences at small Hamming
distance should have similar true gbsa — averaging their (independent) label
noise reduces it. Trains XGB on smoothed labels, evaluates Spearman on the
ORIGINAL labels (the metric of record).

Also reports the Hamming-distance neighbor distribution (to check whether
near neighbors actually exist).

Runs in FAD_env (xgboost).

Usage:
  python -m scripts.reproduction.label_smoothing --outer-seeds 30
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

try:
    from .active_learning_strong import build_seq, load_nupack_useful, _spearman
    from .rank_objective import make_xgb
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from active_learning_strong import build_seq, load_nupack_useful, _spearman
    from rank_objective import make_xgb

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MUT_POS = [30, 31, 32, 33, 89, 90, 91, 92, 127, 128, 129, 130, 131]


def hamming_matrix(seqs):
    """(n, n) Hamming distance over the 13 mutation positions (0..13)."""
    n = len(seqs)
    arr = np.zeros((n, 13), dtype=np.int8)
    for i, s in enumerate(seqs):
        s = str(s).upper().replace("T", "U")
        for j, p in enumerate(MUT_POS):
            arr[i, j] = ord(s[p - 1])
    # pairwise hamming via broadcasting
    D = np.zeros((n, n), dtype=np.int8)
    for j in range(13):
        col = arr[:, j][:, None]
        D += (col != col.T).astype(np.int8)
    return D


def smooth_labels(D, y, k, alpha):
    """Smoothed label: (1-alpha)*y + alpha * mean(y of k nearest neighbors)."""
    n = len(y)
    ys = np.zeros(n)
    for i in range(n):
        # exclude self (D[i,i]=0)
        order = np.argsort(D[i])
        order = order[1:]  # drop self
        knn = order[:k]
        ys[i] = (1 - alpha) * y[i] + alpha * y[knn].mean()
    return ys.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer-seeds", type=int, default=30)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--data-dir", default="data/proxy2000_v2")
    ap.add_argument("--output", default="runs/label_smoothing/results.json")
    args = ap.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    df = pd.read_csv(data_dir / "fad_proxy2000_v2_full.csv")
    seqs = [str(s) for s in df["Sequence"]]
    y = df["gbsa"].to_numpy(dtype=np.float32)
    X = np.hstack([build_seq(seqs),
                   load_nupack_useful(data_dir / "nupack_features_full.csv", seqs)[0]])
    print(f"features: {X.shape[1]}d")

    # Hamming neighbor distribution
    D = hamming_matrix(seqs)
    for d in range(1, 7):
        n_d = int((D == d).sum() // 2)  # symmetric pairs
        print(f"  Hamming distance {d}: {n_d} pairs")

    seeds = list(range(1, args.outer_seeds + 1))
    sp_orig, sp_smooth = [], []
    for seed in seeds:
        tr, te = train_test_split(range(len(X)), test_size=0.2, random_state=seed)
        lo, hi = np.percentile(y[tr], 2.5), np.percentile(y[tr], 97.5)
        y_tr = np.clip(y[tr], lo, hi)

        # NO LEAKAGE: smooth labels using only TRAINING-set neighbors
        tr_seqs = [seqs[i] for i in tr]
        D_tr = hamming_matrix(tr_seqs)
        y_tr_s = np.clip(smooth_labels(D_tr, y[tr], args.k, args.alpha), lo, hi)

        m = make_xgb(); m.fit(X[tr], y_tr)
        sp_orig.append(_spearman(y[te], m.predict(X[te])))
        m = make_xgb(); m.fit(X[tr], y_tr_s)
        sp_smooth.append(_spearman(y[te], m.predict(X[te])))

    mo, so = float(np.mean(sp_orig)), float(np.std(sp_orig))
    ms, ss = float(np.mean(sp_smooth)), float(np.std(sp_smooth))
    paired = np.asarray(sp_smooth) - np.asarray(sp_orig)
    rng = np.random.default_rng(20260924)
    boot = rng.choice(paired, size=(20000, len(paired)), replace=True).mean(axis=1)
    print(f"\n{'='*60}")
    print(f"XGB on original labels : {mo:.4f} +/- {so:.4f}")
    print(f"XGB on smoothed labels (k={args.k}, alpha={args.alpha}) : {ms:.4f} +/- {ss:.4f}")
    print(f"Δ = {ms - mo:+.4f}  paired CI [{np.percentile(boot, 2.5):.4f}, "
          f"{np.percentile(boot, 97.5):.4f}]  wins {int((paired > 0).sum())}/{len(paired)}")

    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {"original": {"mean": mo, "std": so},
               "smoothed": {"mean": ms, "std": ss, "k": args.k, "alpha": args.alpha},
               "paired_delta_mean": float(paired.mean()),
               "paired_bootstrap_ci95": [float(np.percentile(boot, 2.5)),
                                         float(np.percentile(boot, 97.5))],
               "wins": int((paired > 0).sum()), "n_seeds": len(paired),
               "per_seed": {"original": sp_orig, "smoothed": sp_smooth}}
    if out.suffix.lower() == ".csv":
        # run_experiment.py compatibility: write per-seed metrics.csv rows,
        # keep the JSON summary alongside under the same stem.
        pd.DataFrame({"seed": seeds, "original": sp_orig,
                      "smoothed": sp_smooth}).to_csv(out, index=False)
        json_out = out.parent / (out.stem + "_summary.json")
        json.dump(summary, open(json_out, "w"), indent=2)
    else:
        json.dump(summary, open(out, "w"), indent=2)
    print(f"[OK] -> {out}")


if __name__ == "__main__":
    main()

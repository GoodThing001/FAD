"""Sparse epistasis regression (compressed-sensing route, Ho & Hsu 2015).

L1-regularized linear model on first-order one-hot (52d) + all pairwise
position interactions (78 pairs x 16 = 1248d) = 1300d, fitted per seed with
inner 3-fold CV alpha selection. 13 positions x 2000 samples is the regime
where compressed sensing applies. Outputs:
  - per-seed Spearman vs XGB-495d and RF-52d baselines
  - cross-seed stability of the selected features (top |coef|)

Runs on Linux only (Lasso BLAS crash on Windows is a known platform issue).

Usage (server, FAD_env):
  python -m scripts.reproduction.eval_sparse_epistasis --outer-seeds 30
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
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

BASES = "ACGU"
MUT_POS = [30, 31, 32, 33, 89, 90, 91, 92, 127, 128, 129, 130, 131]


def build_first_order(seqs):
    X = np.zeros((len(seqs), 52), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = str(s).upper().replace("T", "U")
        for j, p in enumerate(MUT_POS):
            idx = p - 1
            if s[idx] in BASES:
                X[i, j * 4 + BASES.index(s[idx])] = 1.0
    names = [f"p{p}_{b}" for p in MUT_POS for b in BASES]
    return X, names


def build_pairwise(seqs):
    """78 pairs x 16 base-combination indicators (one-hot of both positions)."""
    n = len(seqs)
    pairs = [(MUT_POS[a], MUT_POS[b]) for a in range(len(MUT_POS))
             for b in range(a + 1, len(MUT_POS))]
    X = np.zeros((n, len(pairs) * 16), dtype=np.float32)
    names = []
    for pi, (pa, pb) in enumerate(pairs):
        for bi, combo in enumerate([x + y for x in BASES for y in BASES]):
            names.append(f"p{pa}x{pb}_{combo}")
        for i, s in enumerate(seqs):
            s = str(s).upper().replace("T", "U")
            ca, cb = s[pa - 1], s[pb - 1]
            if ca in BASES and cb in BASES:
                X[i, pi * 16 + BASES.index(ca) * 4 + BASES.index(cb)] = 1.0
    return X, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer-seeds", type=int, default=30)
    ap.add_argument("--data-dir", default="data/proxy2000_v2")
    ap.add_argument("--output", default="runs/eval_sparse_epistasis/results.json")
    args = ap.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    df = pd.read_csv(data_dir / "fad_proxy2000_v2_full.csv")
    seqs = [str(s) for s in df["Sequence"]]
    y = df["gbsa"].to_numpy(dtype=np.float64)
    X_495 = np.hstack([build_seq(seqs),
                       load_nupack_useful(data_dir / "nupack_features_full.csv", seqs)[0]])
    X1, names1 = build_first_order(seqs)
    X2, names2 = build_pairwise(seqs)
    names = names1 + names2
    X_sparse = np.hstack([X1, X2]).astype(np.float32)
    print(f"sparse design: {X_sparse.shape[1]}d (52 first-order + "
          f"{len(pairs := []) or 1248} pairwise)")

    lasso_sps, xgb_sps, rf52_sps = [], [], []
    import collections
    coef_freq = collections.Counter()
    seeds = list(range(1, args.outer_seeds + 1))
    for seed in seeds:
        tr, te = train_test_split(range(len(X_sparse)), test_size=0.2, random_state=seed)
        lo, hi = np.percentile(y[tr], 2.5), np.percentile(y[tr], 97.5)
        y_tr = np.clip(y[tr], lo, hi).astype(np.float32)

        # inner 3-fold alpha selection, then refit (LassoCV does this)
        m = LassoCV(cv=3, n_alphas=30, max_iter=20000, random_state=seed, n_jobs=1)
        m.fit(X_sparse[tr], y_tr)
        lasso_sps.append(_spearman(y[te], m.predict(X_sparse[te])))
        nz = np.where(np.abs(m.coef_) > 1e-6)[0]
        for j in nz:
            coef_freq[names[j]] += 1

        mx = make_xgb()
        mx.fit(X_495[tr], y_tr)
        xgb_sps.append(_spearman(y[te], mx.predict(X_495[te])))

        mr = RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                   min_samples_leaf=5, random_state=42, n_jobs=1)
        mr.fit(X1[tr], y_tr)
        rf52_sps.append(_spearman(y[te], mr.predict(X1[te])))

        print(f"  seed {seed}/{args.outer_seeds} done "
              f"(lasso n_selected={len(nz)})", flush=True)

    res = {
        "lasso_epistasis": {"mean": float(np.mean(lasso_sps)),
                            "std": float(np.std(lasso_sps)), "per_seed": lasso_sps},
        "xgb_495d": {"mean": float(np.mean(xgb_sps)),
                     "std": float(np.std(xgb_sps)), "per_seed": xgb_sps},
        "rf_52d": {"mean": float(np.mean(rf52_sps)),
                   "std": float(np.std(rf52_sps)), "per_seed": rf52_sps},
        "top_selected_features": sorted(coef_freq.items(), key=lambda kv: -kv[1])[:30],
    }
    print(f"\nlasso(1300d): {res['lasso_epistasis']['mean']:.4f} | "
          f"xgb(495d): {res['xgb_495d']['mean']:.4f} | "
          f"rf(52d): {res['rf_52d']['mean']:.4f}")
    print("top-15 stable features:", res["top_selected_features"][:15])

    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print(f"[OK] -> {out}")


if __name__ == "__main__":
    main()

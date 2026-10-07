"""Negative controls (execution plan v8 Phase 4 leftovers, 10 seeds each).

  1. shuffled labels : permute gbsa on the training fold -> expect Spearman ≈ 0
                       (sanity: the model cannot learn from pure noise).
  2. permuted features: independently permute each feature column across the
                       training rows -> expect a large drop vs the real 495d
                       baseline (~0.35), showing the signal is in the feature
                       structure, not marginal feature distributions.

Usage (server, FAD_env):
  python -m scripts.reproduction.eval_negative_controls --seeds 10
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--data-dir", default="data/proxy2000_v2")
    ap.add_argument("--output", default="runs/eval_negative_controls/results.json")
    args = ap.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    df = pd.read_csv(data_dir / "fad_proxy2000_v2_full.csv")
    seqs = [str(s) for s in df["Sequence"]]
    y = df["gbsa"].to_numpy(dtype=np.float64)
    X = np.hstack([build_seq(seqs),
                   load_nupack_useful(data_dir / "nupack_features_full.csv", seqs)[0]])
    print(f"features: {X.shape[1]}d")

    real_sp, shuffled_sp, permuted_sp = [], [], []
    seeds = list(range(1, args.seeds + 1))
    for seed in seeds:
        tr, te = train_test_split(range(len(X)), test_size=0.2, random_state=seed)
        lo, hi = np.percentile(y[tr], 2.5), np.percentile(y[tr], 97.5)
        y_tr = np.clip(y[tr], lo, hi).astype(np.float32)

        m = make_xgb()
        m.fit(X[tr], y_tr)
        real_sp.append(_spearman(y[te], m.predict(X[te])))

        rng = np.random.RandomState(seed)
        m = make_xgb()
        m.fit(X[tr], rng.permutation(y_tr).astype(np.float32))
        shuffled_sp.append(_spearman(y[te], m.predict(X[te])))

        Xp = X.copy()
        for j in range(X.shape[1]):
            Xp[tr, j] = rng.permutation(X[tr, j])
        m = make_xgb()
        m.fit(Xp[tr], y_tr)
        permuted_sp.append(_spearman(y[te], m.predict(Xp[te])))

        print(f"  seed {seed}/{args.seeds}: real={real_sp[-1]:.4f} "
              f"shuffled={shuffled_sp[-1]:.4f} permuted={permuted_sp[-1]:.4f}", flush=True)

    res = {
        "real_labels": {"mean": float(np.mean(real_sp)), "std": float(np.std(real_sp)),
                        "per_seed": real_sp},
        "shuffled_labels": {"mean": float(np.mean(shuffled_sp)),
                            "std": float(np.std(shuffled_sp)), "per_seed": shuffled_sp},
        "permuted_features": {"mean": float(np.mean(permuted_sp)),
                              "std": float(np.std(permuted_sp)),
                              "per_seed": permuted_sp},
    }
    print(f"\nreal: {res['real_labels']['mean']:.4f} | "
          f"shuffled: {res['shuffled_labels']['mean']:.4f} (expect ~0) | "
          f"permuted: {res['permuted_features']['mean']:.4f} (expect << real)")

    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print(f"[OK] -> {out}")


if __name__ == "__main__":
    main()

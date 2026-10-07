"""Label-smoothing variants: LS vs NLS (negative smoothing) vs raw (30 seeds).

Wei et al. (ICML 2022): at HIGH noise rates standard label smoothing
over-smooths and loses its benefit; negative smoothing (pushing the label away
from the neighbor average) is then better. gbsa noise (std 10.4, 25 outliers
>50) may be in that regime. Paired 30-seed XGB comparison on 495d.

Variants (all train-fold internal, NO leakage; evaluation vs original gbsa):
  raw        : winsorized gbsa
  LS(k5,a0.5): (1-a)*y + a*mean(kNN Hamming labels)      [existing +0.019]
  NLS(a0.25) : (1+a)*y - a*mean(kNN Hamming labels)
  NLS(a0.5)  : (1+a)*y - a*mean(kNN Hamming labels)

Usage (server, FAD_env):
  python -m scripts.reproduction.eval_smoothing_variants --outer-seeds 30
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
    from .label_smoothing import hamming_matrix
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from active_learning_strong import build_seq, load_nupack_useful, _spearman
    from rank_objective import make_xgb
    from label_smoothing import hamming_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def smooth_labels(D, y, k, alpha):
    """(1-alpha)*y + alpha*mean(kNN labels); NLS uses alpha<0."""
    n = len(y)
    ys = np.zeros(n, dtype=np.float64)
    for i in range(n):
        order = np.argsort(D[i])[1:]          # drop self
        knn = order[:k]
        ys[i] = (1 - alpha) * y[i] + alpha * y[knn].mean()
    return ys.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer-seeds", type=int, default=30)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--data-dir", default="data/proxy2000_v2")
    ap.add_argument("--output", default="runs/eval_smoothing_variants/results.json")
    args = ap.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    df = pd.read_csv(data_dir / "fad_proxy2000_v2_full.csv")
    seqs = [str(s) for s in df["Sequence"]]
    y = df["gbsa"].to_numpy(dtype=np.float64)
    X = np.hstack([build_seq(seqs),
                   load_nupack_useful(data_dir / "nupack_features_full.csv", seqs)[0]])
    print(f"features: {X.shape[1]}d")

    variants = ["raw", "LS_k5_a0.5", "NLS_a0.25", "NLS_a0.5"]
    per_seed = {v: [] for v in variants}
    seeds = list(range(1, args.outer_seeds + 1))
    for seed in seeds:
        tr, te = train_test_split(range(len(X)), test_size=0.2, random_state=seed)
        lo, hi = np.percentile(y[tr], 2.5), np.percentile(y[tr], 97.5)
        y_tr = np.clip(y[tr], lo, hi)
        tr_seqs = [seqs[i] for i in tr]
        D_tr = hamming_matrix(tr_seqs)

        targets = {
            "raw": y_tr.astype(np.float32),
            "LS_k5_a0.5": np.clip(smooth_labels(D_tr, y_tr, args.k, 0.5), lo, hi),
            "NLS_a0.25": np.clip(smooth_labels(D_tr, y_tr, args.k, -0.25), lo, hi),
            "NLS_a0.5": np.clip(smooth_labels(D_tr, y_tr, args.k, -0.5), lo, hi),
        }
        for name in variants:
            m = make_xgb()
            m.fit(X[tr], targets[name])
            per_seed[name].append(_spearman(y[te], m.predict(X[te])))
        print(f"  seed {seed}/{args.outer_seeds} done", flush=True)

    res = {}
    print(f"\n{'='*60}")
    print(f"{'variant':<12} {'mean':>8} {'std':>8} {'delta_vs_raw':>12} {'wins':>8}")
    raw_arr = np.array(per_seed["raw"])
    for name in variants:
        arr = np.array(per_seed[name])
        delta = arr - raw_arr
        rng = np.random.default_rng(20260924)
        boot = rng.choice(delta, size=(20000, len(delta)), replace=True).mean(axis=1)
        res[name] = {
            "mean": float(arr.mean()), "std": float(arr.std()),
            "paired_delta_vs_raw": float(delta.mean()),
            "paired_ci95": [float(np.percentile(boot, 2.5)),
                            float(np.percentile(boot, 97.5))],
            "wins": int((delta > 0).sum()), "n_seeds": len(delta),
        }
        print(f"{name:<12} {arr.mean():>8.4f} {arr.std():>8.4f} "
              f"{delta.mean():>+12.4f} {int((delta > 0).sum()):>4}/{len(delta)}")

    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"per_seed": per_seed, "summary": res}, open(out, "w"), indent=2)
    print(f"[OK] -> {out}")


if __name__ == "__main__":
    main()

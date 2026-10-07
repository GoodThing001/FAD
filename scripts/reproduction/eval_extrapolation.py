"""Strong extrapolation protocol: mutation-count stratified holdout.

The existing cluster split (Hamming<=1 groups) is weak: all 2000 mutation
patterns are unique, so it degenerates to a random split. A stronger OOD test:
leave out whole mutation-count strata.

  count = Hamming distance to WT over the 13 mutation positions.
  strata: low 1-3 | mid 4-7 | high 8-13

Configs (XGB, 495d, winsor in-fold):
  high_out : train low+mid, test high   (extrapolate to heavily mutated)
  low_out  : train mid+high, test low   (extrapolate to near-WT)
  control  : 30-seed random 80/20 (matched test sizes reported)

Usage (server, FAD_env):
  python -m scripts.reproduction.eval_extrapolation --outer-seeds 30
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
WT = ("UAUCCUUCGGGGCAGGGUGGAAAUCCCGACCGGCGGUAGUAAAGCACAUUUGCUUUAGAGCCC"
      "GUGACCCGUGUGCAUAAGCACGCGGUGGAUUCAGUUUAAGCUGAAGCCGACAGUGAAAGUCUGGA"
      "UGGGAGAAGGAUG")


def mutation_counts(seqs):
    counts = []
    for s in seqs:
        s = str(s).upper().replace("T", "U")
        counts.append(sum(1 for p in MUT_POS if s[p - 1] != WT[p - 1]))
    return np.array(counts)


def fit_eval(X, y, tr, te):
    lo, hi = np.percentile(y[tr], 2.5), np.percentile(y[tr], 97.5)
    y_tr = np.clip(y[tr], lo, hi).astype(np.float32)
    m = make_xgb()
    m.fit(X[tr], y_tr)
    return _spearman(y[te], m.predict(X[te]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer-seeds", type=int, default=30)
    ap.add_argument("--data-dir", default="data/proxy2000_v2")
    ap.add_argument("--output", default="runs/eval_extrapolation/results.json")
    args = ap.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    df = pd.read_csv(data_dir / "fad_proxy2000_v2_full.csv")
    seqs = [str(s) for s in df["Sequence"]]
    y = df["gbsa"].to_numpy(dtype=np.float64)
    X = np.hstack([build_seq(seqs),
                   load_nupack_useful(data_dir / "nupack_features_full.csv", seqs)[0]])

    counts = mutation_counts(seqs)
    # tertile strata of the mutation-count distribution (the naive low/mid/high
    # thresholds were degenerate: 1 / 125 / 1874)
    lo_t, hi_t = np.quantile(counts, [1 / 3, 2 / 3])
    low = np.where(counts <= lo_t)[0]
    mid = np.where((counts > lo_t) & (counts <= hi_t))[0]
    high = np.where(counts > hi_t)[0]
    print(f"count tertiles: <= {lo_t:.0f} ({len(low)}), "
          f"{lo_t:.0f}-{hi_t:.0f} ({len(mid)}), > {hi_t:.0f} ({len(high)})")

    res = {}

    # deterministic leave-one-tercile-out configs (with train/test size guards)
    for name, tr, te in [
        ("high_out", np.concatenate([low, mid]), high),
        ("mid_out", np.concatenate([low, high]), mid),
        ("low_out", np.concatenate([mid, high]), low),
    ]:
        if len(te) < 10 or len(tr) < 100:
            print(f"  {name}: skipped (train={len(tr)}, test={len(te)})")
            continue
        sp = fit_eval(X, y, tr, te)
        res[name] = {"spearman": float(sp), "n_train": int(len(tr)),
                     "n_test": int(len(te)),
                     "test_count_range": [int(counts[te].min()), int(counts[te].max())]}
        print(f"  {name}: n_train={len(tr)} n_test={len(te)} "
              f"test_counts[{counts[te].min()},{counts[te].max()}] sp={sp:.4f}")

    # random 80/20 control (30 seeds)
    sps = []
    for seed in range(1, args.outer_seeds + 1):
        tr, te = train_test_split(range(len(X)), test_size=0.2, random_state=seed)
        sps.append(fit_eval(X, y, tr, te))
    res["random_control"] = {"spearman_mean": float(np.mean(sps)),
                             "spearman_std": float(np.std(sps)),
                             "per_seed": sps}
    print(f"  random control (30 seeds): {np.mean(sps):.4f} +/- {np.std(sps):.4f}")

    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print(f"[OK] -> {out}")


if __name__ == "__main__":
    main()

"""Evaluate marginal gain of ensemble-defect / WT-structure-probability features.

Paired 30-seed XGB comparison on gbsa:
  base  = 303d seq + 192d NUPACK useful blocks (495d)
  +defect, +wt_prob, +both  vs  base
Features aligned by Sequence. Runs in FAD_env.

Usage:
  python -m scripts.reproduction.eval_defect_features \
      --defect-csv runs/audit_ensemble_defect/ensemble_defect.csv \
      --outer-seeds 30 --output runs/eval_defect_features/results.json
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
    ap.add_argument("--outer-seeds", type=int, default=30)
    ap.add_argument("--defect-csv", default="runs/audit_ensemble_defect/ensemble_defect.csv")
    ap.add_argument("--data-dir", default="data/proxy2000_v2")
    ap.add_argument("--output", default="runs/eval_defect_features/results.json")
    args = ap.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    df = pd.read_csv(data_dir / "fad_proxy2000_v2_full.csv")
    seqs = [str(s) for s in df["Sequence"]]
    y = df["gbsa"].to_numpy(dtype=np.float64)
    X_base = np.hstack([build_seq(seqs),
                        load_nupack_useful(data_dir / "nupack_features_full.csv", seqs)[0]])

    feat = pd.read_csv(PROJECT_ROOT / args.defect_csv)
    feat["Sequence"] = feat["Sequence"].astype(str)
    feat = feat.set_index("Sequence").loc[seqs].reset_index()
    defect = feat["ensemble_defect"].to_numpy(dtype=np.float64)
    wt_prob = feat["wt_struct_prob"].to_numpy(dtype=np.float64)
    print(f"defect stats: mean={defect.mean():.2f} std={defect.std():.2f} "
          f"min={defect.min():.2f} max={defect.max():.2f}")
    print(f"wt_prob stats: mean={wt_prob.mean():.4f} "
          f"frac_zero={(wt_prob == 0).mean():.3f}")
    print(f"spearman(defect, gbsa)={_spearman(defect, y):.4f}  "
          f"spearman(wt_prob, gbsa)={_spearman(wt_prob, y):.4f}")

    variants = {
        "base": X_base,
        "base+defect": np.hstack([X_base, defect[:, None]]),
        "base+wt_prob": np.hstack([X_base, wt_prob[:, None]]),
        "base+both": np.hstack([X_base, defect[:, None], wt_prob[:, None]]),
    }
    per_seed = {k: [] for k in variants}
    seeds = list(range(1, args.outer_seeds + 1))
    for seed in seeds:
        tr, te = train_test_split(range(len(X_base)), test_size=0.2, random_state=seed)
        lo, hi = np.percentile(y[tr], 2.5), np.percentile(y[tr], 97.5)
        y_tr = np.clip(y[tr], lo, hi).astype(np.float32)
        for name, Xv in variants.items():
            m = make_xgb()
            m.fit(Xv[tr], y_tr)
            per_seed[name].append(_spearman(y[te], m.predict(Xv[te])))
        print(f"  seed {seed}/{args.outer_seeds} done", flush=True)

    res = {}
    print(f"\n{'='*60}")
    print(f"{'variant':<14} {'mean':>8} {'std':>8} {'delta_vs_base':>12}")
    base_arr = np.array(per_seed["base"])
    for name in variants:
        arr = np.array(per_seed[name])
        delta = arr - base_arr
        rng = np.random.default_rng(20260924)
        boot = rng.choice(delta, size=(20000, len(delta)), replace=True).mean(axis=1)
        res[name] = {
            "mean": float(arr.mean()), "std": float(arr.std()),
            "paired_delta_vs_base": float(delta.mean()),
            "paired_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "wins": int((delta > 0).sum()), "n_seeds": len(delta),
        }
        print(f"{name:<14} {arr.mean():>8.4f} {arr.std():>8.4f} "
              f"{delta.mean():>+12.4f}  wins {int((delta > 0).sum())}/{len(delta)}")

    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"per_seed": per_seed, "summary": res}, open(out, "w"), indent=2)
    print(f"[OK] -> {out}")


if __name__ == "__main__":
    main()

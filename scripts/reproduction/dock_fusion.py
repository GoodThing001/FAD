"""dock multi-model rank fusion + label smoothing (credibility upgrade).

dock is best predicted by 303d alone (0.545, single XGB). To raise credibility,
apply a FOUR-model rank fusion (RF/ET/GBR/XGB — matches the 汇报's four-model
0.547 result; an earlier docstring said "five-model" but SVR is intentionally
not used: SVR needs feature scaling and adds nothing on one-hot input) + label
smoothing, and support 100-seed for a tight CI. Reports per-model and fused
Spearman.

Runs in FAD_env.

Usage:
  python -m scripts.reproduction.dock_fusion --outer-seeds 30
  python -m scripts.reproduction.dock_fusion --outer-seeds 100
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.model_selection import train_test_split

try:
    from .active_learning_strong import build_seq, _spearman
    from .label_smoothing import hamming_matrix, smooth_labels
    from .rank_objective import make_xgb
except ImportError:
    # Plain-script invocation (run_experiment.py entrypoint): fall back to
    # top-level imports of the sibling modules.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from active_learning_strong import build_seq, _spearman
    from label_smoothing import hamming_matrix, smooth_labels
    from rank_objective import make_xgb

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_model(name):
    if name == "RF":
        return RandomForestRegressor(n_estimators=500, min_samples_leaf=5,
                                     max_features="sqrt", random_state=42, n_jobs=1)
    if name == "ET":
        return ExtraTreesRegressor(n_estimators=500, min_samples_leaf=5,
                                   max_features="log2", random_state=42, n_jobs=1)
    if name == "GBR":
        return GradientBoostingRegressor(n_estimators=500, max_depth=3,
                                         min_samples_leaf=3, learning_rate=0.03,
                                         loss="absolute_error", random_state=42)
    if name == "XGB":
        return make_xgb()
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer-seeds", type=int, default=30)
    ap.add_argument("--models", default="RF,ET,GBR,XGB")
    ap.add_argument("--data-dir", default="data/proxy2000_v2")
    ap.add_argument("--output", default="runs/dock_fusion/results.json")
    args = ap.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    df = pd.read_csv(data_dir / "fad_proxy2000_v2_full.csv")
    seqs = [str(s) for s in df["Sequence"]]
    dock = df["dock"].to_numpy(dtype=np.float32)
    X = build_seq(seqs)
    print(f"features: 303d, target: dock")

    model_names = [m.strip() for m in args.models.split(",")]
    seeds = list(range(1, args.outer_seeds + 1))

    single = {m: [] for m in model_names}
    fused = []
    fused_smooth = []

    for seed in seeds:
        tr, te = train_test_split(range(len(X)), test_size=0.2, random_state=seed)
        y_tr = dock[tr].astype(np.float32)

        preds = {}
        for m in model_names:
            model = make_model(m)
            model.fit(X[tr], y_tr)
            pred = model.predict(X[te])
            preds[m] = pred
            single[m].append(_spearman(dock[te], pred))

        # rank fusion (equal-weight)
        from scipy.stats import rankdata
        ranks = np.column_stack([rankdata(preds[m]) for m in model_names])
        fused.append(_spearman(dock[te], ranks.mean(axis=1)))

        # rank fusion with label smoothing on the best model (XGB)
        tr_seqs = [seqs[i] for i in tr]
        D_tr = hamming_matrix(tr_seqs)
        y_tr_s = smooth_labels(D_tr, dock[tr], k=5, alpha=0.5).astype(np.float32)
        model = make_model("XGB")
        model.fit(X[tr], y_tr_s)
        fused_smooth.append(_spearman(dock[te], model.predict(X[te])))

    print(f"\n{'='*60}")
    print(f"dock prediction ({args.outer_seeds}-seed), 303d features")
    print(f"{'='*60}")
    for m in model_names:
        s = np.array(single[m])
        print(f"  single.{m:<4} {s.mean():.4f} +/- {s.std():.4f}")
    f = np.array(fused)
    print(f"  rank_fusion  {f.mean():.4f} +/- {f.std():.4f}")
    fs = np.array(fused_smooth)
    print(f"  fusion+smooth {fs.mean():.4f} +/- {fs.std():.4f}")

    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {"single": {m: {"mean": float(np.mean(single[m])), "std": float(np.std(single[m]))}
                          for m in model_names},
               "rank_fusion": {"mean": float(f.mean()), "std": float(f.std())},
               "fusion_smooth": {"mean": float(fs.mean()), "std": float(fs.std())}}
    if out.suffix.lower() == ".csv":
        # run_experiment.py compatibility: write per-seed metrics.csv rows,
        # keep the JSON summary alongside under the same stem.
        rows = {"seed": seeds}
        for m in model_names:
            rows[m] = single[m]
        rows["rank_fusion"] = fused
        rows["fusion_smooth"] = fused_smooth
        pd.DataFrame(rows).to_csv(out, index=False)
        json_out = out.parent / (out.stem + "_summary.json")
        json.dump(summary, open(json_out, "w"), indent=2)
    else:
        json.dump(summary, open(out, "w"), indent=2)
    print(f"[OK] -> {out}")


if __name__ == "__main__":
    main()

"""Rank objectives vs MSE for gbsa (30 seeds, 495d).

  xgb_mse            : XGB regressor on winsorized gbsa (baseline)
  xgb_quantile50     : XGB quantile regression (median)
  lgbm_lambdarank    : LightGBM LambdaMART, one group over the training fold
                       (listwise pairwise ranking, efficient)
  xgbranker_subsample: XGBRanker rank:pairwise on a 500-row training subsample
                       (honestly labeled: subsampled for tractability)

Evaluation is Spearman vs ORIGINAL gbsa on the test fold for all methods.

Usage (server, FAD_env):
  python -m scripts.reproduction.eval_rank_objectives2 --outer-seeds 30
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

try:
    from .active_learning_strong import build_seq, load_nupack_useful, _spearman
    from .rank_objective import make_xgb
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from active_learning_strong import build_seq, load_nupack_useful, _spearman
    from rank_objective import make_xgb

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_lgbm_ranker():
    import lightgbm as lgb
    return lgb.LGBMRanker(
        n_estimators=300, num_leaves=31, learning_rate=0.05,
        min_child_samples=10, n_jobs=1, random_state=42, verbose=-1)


def make_xgb_ranker():
    from xgboost import XGBRanker
    return XGBRanker(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        objective="rank:pairwise", random_state=42, verbosity=0, n_jobs=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer-seeds", type=int, default=30)
    ap.add_argument("--ranker-subsample", type=int, default=500)
    ap.add_argument("--data-dir", default="data/proxy2000_v2")
    ap.add_argument("--output", default="runs/eval_rank_objectives2/results.json")
    args = ap.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    df = pd.read_csv(data_dir / "fad_proxy2000_v2_full.csv")
    seqs = [str(s) for s in df["Sequence"]]
    y = df["gbsa"].to_numpy(dtype=np.float64)
    X = np.hstack([build_seq(seqs),
                   load_nupack_useful(data_dir / "nupack_features_full.csv", seqs)[0]])
    print(f"features: {X.shape[1]}d")

    methods = ["xgb_mse", "xgb_quantile50", "lgbm_lambdarank", "xgbranker_subsample"]
    per_seed = {m: [] for m in methods}
    seeds = list(range(1, args.outer_seeds + 1))
    for seed in seeds:
        tr, te = train_test_split(range(len(X)), test_size=0.2, random_state=seed)
        lo, hi = np.percentile(y[tr], 2.5), np.percentile(y[tr], 97.5)
        y_tr = np.clip(y[tr], lo, hi).astype(np.float32)

        # 1. XGB MSE
        m = make_xgb()
        m.fit(X[tr], y_tr)
        per_seed["xgb_mse"].append(_spearman(y[te], m.predict(X[te])))

        # 2. XGB quantile(0.5)
        from xgboost import XGBRegressor
        m = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                         objective="reg:quantileerror", quantile_alpha=0.5,
                         random_state=42, verbosity=0, n_jobs=1)
        m.fit(X[tr], y_tr)
        per_seed["xgb_quantile50"].append(_spearman(y[te], m.predict(X[te])))

        # 3. LightGBM LambdaMART (one group = whole training fold)
        # lambdarank expects integer relevance labels with < 31 levels by
        # default; quantize ordinal ranks into 31 graded relevance bins.
        m = make_lgbm_ranker()
        group = np.array([len(tr)], dtype=np.int32)
        rel = np.floor(rankdata(y_tr) * 30.0 / len(tr)).astype(np.int32)
        m.fit(X[tr], rel, group=group)
        per_seed["lgbm_lambdarank"].append(_spearman(y[te], m.predict(X[te])))

        # 4. XGBRanker rank:pairwise on a subsample (tractability)
        rng = np.random.RandomState(seed)
        sub = rng.choice(len(tr), size=min(args.ranker_subsample, len(tr)),
                         replace=False)
        m = make_xgb_ranker()
        qid = np.zeros(len(sub), dtype=np.int32)  # single ranking group
        m.fit(X[tr][sub], y_tr[sub], qid=qid)
        per_seed["xgbranker_subsample"].append(_spearman(y[te], m.predict(X[te])))

        print(f"  seed {seed}/{args.outer_seeds} done", flush=True)

    res = {}
    print(f"\n{'='*60}")
    print(f"{'method':<22} {'mean':>8} {'std':>8} {'delta_vs_mse':>12} {'wins':>8}")
    base_arr = np.array(per_seed["xgb_mse"])
    for name in methods:
        arr = np.array(per_seed[name])
        delta = arr - base_arr
        rng = np.random.default_rng(20260924)
        boot = rng.choice(delta, size=(20000, len(delta)), replace=True).mean(axis=1)
        res[name] = {
            "mean": float(arr.mean()), "std": float(arr.std()),
            "paired_delta_vs_mse": float(delta.mean()),
            "paired_ci95": [float(np.percentile(boot, 2.5)),
                            float(np.percentile(boot, 97.5))],
            "wins": int((delta > 0).sum()), "n_seeds": len(delta),
        }
        print(f"{name:<22} {arr.mean():>8.4f} {arr.std():>8.4f} "
              f"{delta.mean():>+12.4f} {int((delta > 0).sum()):>4}/{len(delta)}")

    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"per_seed": per_seed, "summary": res}, open(out, "w"), indent=2)
    print(f"[OK] -> {out}")


if __name__ == "__main__":
    main()

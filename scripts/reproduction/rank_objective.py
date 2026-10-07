"""Rank-consistent + robust objectives: squeeze the last ~0.01-0.02 of Spearman.

The 0.40 Spearman ceiling is label noise, so these do NOT break the ceiling.
But the research (breakthrough-directions) found cheap objective changes that
claw back the final 1-2 Spearman points by directly optimizing the rank rather
than MSE. Tests, on 303d + NUPACK 192d, 30-seed:

  1. XGB regressor (MSE)              -- baseline
  2. XGB regressor on rank-normal y   -- rank-transform target
  3. XGBRanker (rank:pairwise)        -- direct pairwise ranking objective
  4. GBR with Huber loss              -- robust to gbsa outliers
  5. XGB with quantile(0.5) loss      -- median regression

Runs in FAD_env (xgboost).

Usage:
  python -m scripts.reproduction.rank_objective --outer-seeds 30
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from scipy.stats import norm, rankdata
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

try:
    from .active_learning_strong import build_seq, load_nupack_useful, _spearman
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from active_learning_strong import build_seq, load_nupack_useful, _spearman

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.03,
                        random_state=42, verbosity=0, n_jobs=1)


def make_xgb_ranker():
    from xgboost import XGBRanker
    return XGBRanker(objective="rank:pairwise", n_estimators=500, max_depth=3,
                     learning_rate=0.03, random_state=42, verbosity=0, n_jobs=1)


def rank_normal(y):
    n = len(y)
    q = (rankdata(y) - 0.5) / n
    q = np.clip(q, 1e-4, 1 - 1e-4)
    return norm.ppf(q).astype(np.float32)


def eval_one(X, y, seeds, mode):
    sps = []
    for seed in seeds:
        tr, te = train_test_split(range(len(X)), test_size=0.2, random_state=seed)
        y_tr_raw, y_te_raw = y[tr], y[te]
        lo, hi = np.percentile(y_tr_raw, 2.5), np.percentile(y_tr_raw, 97.5)
        y_tr = np.clip(y_tr_raw, lo, hi).astype(np.float32)

        if mode == "xgb_mse":
            m = make_xgb()
            m.fit(X[tr], y_tr)
            pred = m.predict(X[te])
        elif mode == "xgb_rank_normal":
            m = make_xgb()
            m.fit(X[tr], rank_normal(y_tr))
            pred = m.predict(X[te])
        elif mode == "xgb_ranker":
            m = make_xgb_ranker()
            m.fit(X[tr], y_tr, group=[len(tr)])
            pred = m.predict(X[te])
        elif mode == "gbr_huber":
            m = GradientBoostingRegressor(n_estimators=500, max_depth=3,
                                          learning_rate=0.03, loss="huber",
                                          random_state=42)
            m.fit(X[tr], y_tr)
            pred = m.predict(X[te])
        elif mode == "xgb_quantile":
            from xgboost import XGBRegressor
            m = XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.03,
                             objective="reg:quantileerror", quantile_alpha=0.5,
                             random_state=42, verbosity=0, n_jobs=1)
            m.fit(X[tr], y_tr)
            pred = m.predict(X[te])
        else:
            raise ValueError(mode)
        sps.append(_spearman(y_te_raw, pred))
    return float(np.mean(sps)), float(np.std(sps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer-seeds", type=int, default=30)
    ap.add_argument("--data-dir", default="data/proxy2000_v2")
    ap.add_argument("--output", default="runs/rank_objective/results.json")
    args = ap.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    df = pd.read_csv(data_dir / "fad_proxy2000_v2_full.csv")
    seqs = [str(s) for s in df["Sequence"]]
    y = df["gbsa"].to_numpy(dtype=np.float32)
    X_seq = build_seq(seqs)
    X_nup, _ = load_nupack_useful(data_dir / "nupack_features_full.csv", seqs)
    X = np.hstack([X_seq, X_nup])
    print(f"features: {X.shape[1]}d, seeds: {args.outer_seeds}")

    modes = ["xgb_mse", "xgb_rank_normal", "xgb_ranker", "gbr_huber", "xgb_quantile"]
    results = {}
    for mode in modes:
        mean, std = eval_one(X, y, list(range(1, args.outer_seeds + 1)), mode)
        results[mode] = {"mean": mean, "std": std}
        print(f"  {mode:<20} {mean:.4f} +/- {std:.4f}", flush=True)

    base = results["xgb_mse"]["mean"]
    print(f"\n{'='*60}")
    print(f"baseline (xgb_mse)     : {base:.4f}")
    for mode in modes[1:]:
        d = results[mode]["mean"] - base
        print(f"{mode:<20} {results[mode]['mean']:.4f}  (Δ {d:+.4f})")

    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print(f"[OK] -> {out}")


if __name__ == "__main__":
    main()

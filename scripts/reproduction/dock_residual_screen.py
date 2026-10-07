"""Exploratory paired screen: does dock add residual signal for GBSA?

The objective and test metric are always the original gbsa column. This script
never uses dock in the baseline GBSA model. The observed-dock arm is an upper
bound requiring a real dock value for every candidate at inference; the
predicted-dock arm only uses sequence features at inference. Both residual
maps are fitted on outer-training rows using inner out-of-fold GBSA predictions.
Results are exploratory and are not a replacement for the v8 release model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.reproduction.eval_fusion_smoothing import build

DATA = ROOT / "data/proxy2000_v2/fad_proxy2000_v2_full.csv"
REGISTRY = ROOT / "data/registry.csv"


def verify_data() -> str:
    digest = hashlib.sha256(DATA.read_bytes()).hexdigest()
    with REGISTRY.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = next(row["sha256"] for row in rows
                    if row["dataset_id"] == "proxy2000_v2_full")
    if digest.lower() != expected.lower():
        raise ValueError(f"canonical data SHA256 mismatch: {digest} != {expected}")
    return digest


def model(seed: int, n_estimators: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=n_estimators, max_depth=3,
                                         learning_rate=0.05,
                                         l2_regularization=1.0,
                                         random_state=seed)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(pearsonr(rankdata(a), rankdata(b)).statistic)


def top_recall(y: np.ndarray, pred: np.ndarray, fraction: float = 0.1) -> float:
    k = max(1, int(round(len(y) * fraction)))
    true_best = set(np.argsort(y)[:k])
    predicted_best = set(np.argsort(pred)[:k])
    return len(true_best & predicted_best) / k


def one_seed(X: np.ndarray, gbsa: np.ndarray, dock: np.ndarray,
             seed: int, n_estimators: int, inner_folds: int) -> dict:
    tr, te = train_test_split(np.arange(len(gbsa)), test_size=0.2,
                              random_state=seed)
    Xtr, Xte = X[tr], X[te]
    gtr, gte = gbsa[tr], gbsa[te]
    dtr, dte = dock[tr], dock[te]

    oof_g = np.empty(len(tr), dtype=float)
    oof_d = np.empty(len(tr), dtype=float)
    folds = KFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    for fold, (a, b) in enumerate(folds.split(Xtr)):
        lo, hi = np.percentile(gtr[a], [2.5, 97.5])
        gm = model(seed * 100 + fold, n_estimators)
        dm = model(seed * 100 + fold + 50, n_estimators)
        gm.fit(Xtr[a], np.clip(gtr[a], lo, hi))
        dm.fit(Xtr[a], dtr[a])
        oof_g[b] = gm.predict(Xtr[b])
        oof_d[b] = dm.predict(Xtr[b])

    lo, hi = np.percentile(gtr, [2.5, 97.5])
    clipped_gtr = np.clip(gtr, lo, hi)
    gm = model(seed * 100 + 90, n_estimators)
    dm = model(seed * 100 + 91, n_estimators)
    gm.fit(Xtr, clipped_gtr)
    dm.fit(Xtr, dtr)
    base = gm.predict(Xte).astype(float)
    pred_dock = dm.predict(Xte).astype(float)

    # Both residual fits use only outer-training GBSA labels. The observed-dock
    # arm may inspect dock on outer test sequences, so it is an availability
    # upper bound, not a deployable sequence-only model.
    resid = clipped_gtr - oof_g
    seq_map = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    obs_map = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    seq_map.fit(oof_d.reshape(-1, 1), resid)
    obs_map.fit(dtr.reshape(-1, 1), resid)
    corrected_seq = base + seq_map.predict(pred_dock.reshape(-1, 1))
    corrected_obs = base + obs_map.predict(dte.reshape(-1, 1))

    row = {"seed": seed, "n_train": len(tr), "n_test": len(te),
           "baseline_sp": spearman(gte, base),
           "predicted_dock_resid_sp": spearman(gte, corrected_seq),
           "observed_dock_resid_sp": spearman(gte, corrected_obs),
           "baseline_top10_recall": top_recall(gte, base),
           "predicted_dock_top10_recall": top_recall(gte, corrected_seq),
           "observed_dock_top10_recall": top_recall(gte, corrected_obs),
           "baseline_best20": float(gte[np.argsort(base)[:20]].min()),
           "predicted_dock_best20": float(gte[np.argsort(corrected_seq)[:20]].min()),
           "observed_dock_best20": float(gte[np.argsort(corrected_obs)[:20]].min())}
    row["predicted_dock_delta_sp"] = (row["predicted_dock_resid_sp"] -
                                       row["baseline_sp"])
    row["observed_dock_delta_sp"] = (row["observed_dock_resid_sp"] -
                                      row["baseline_sp"])
    # Supplementary, predeclared availability-only prefilter arms. Never choose
    # a retention fraction using these test GBSA labels. Every test candidate
    # needs observed docking; all retained candidates still rank by base GBSA.
    true_top = set(np.argsort(gte)[:40])
    for pct in (25, 50, 75):
        kept = np.argsort(dte)[:int(round(len(te) * pct / 100))]
        chosen = kept[np.argsort(base[kept])[:20]]
        row[f"prefilter_{pct}_true_top10_retained"] = len(true_top & set(kept)) / len(true_top)
        row[f"prefilter_{pct}_best20"] = float(gte[chosen].min())
        row[f"prefilter_{pct}_best20_delta"] = row[f"prefilter_{pct}_best20"] - row['baseline_best20']
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--output", default="runs/dock_residual_screen/metrics.csv")
    args = parser.parse_args()
    digest = verify_data()
    data = pd.read_csv(DATA)
    X = build(data["Sequence"].tolist())
    gbsa = data["gbsa"].to_numpy(dtype=float)
    dock = data["dock"].to_numpy(dtype=float)
    rows = []
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"data_sha256={digest} target=gbsa features=seq303", flush=True)
    for seed in range(args.seed_start, args.seed_start + args.outer_seeds):
        row = one_seed(X, gbsa, dock, seed, args.n_estimators, args.inner_folds)
        rows.append(row)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"seed={seed} baseline={row['baseline_sp']:.4f} "
              f"pred_dock_delta={row['predicted_dock_delta_sp']:+.4f} "
              f"observed_dock_delta={row['observed_dock_delta_sp']:+.4f}",
              flush=True)
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    rng = np.random.default_rng(20260930)
    summary = {"seed_start": args.seed_start, "n_seeds":len(rows),
               "data_sha256":digest, "target":"gbsa",
               "confirmation_scope":"existing table; observed dock requires candidate-time availability"}
    for arm in ("predicted_dock", "observed_dock"):
        delta = np.array([r[f"{arm}_delta_sp"] for r in rows])
        means = np.mean(rng.choice(delta, (10000, len(delta)), replace=True), axis=1)
        summary[arm] = {"mean_delta_sp": float(delta.mean()),
                        "paired_95ci": np.quantile(means, [0.025, 0.975]).tolist(),
                        "wins": int((delta > 0).sum()),
                        "n_seeds": len(delta)}
    summary['prefilter'] = {}
    for pct in (25, 50, 75):
        delta=np.array([r[f'prefilter_{pct}_best20_delta'] for r in rows])
        means=np.mean(rng.choice(delta,(10000,len(delta)),replace=True),axis=1)
        summary['prefilter'][str(pct)] = {
            'retention_fraction':pct/100,
            'mean_true_top10_retained':float(np.mean([r[f'prefilter_{pct}_true_top10_retained'] for r in rows])),
            'mean_best20_delta':float(delta.mean()),
            'paired_95ci':np.quantile(means,[.025,.975]).tolist(),
            'wins':int((delta < -1e-10).sum()),'ties':int((abs(delta)<=1e-10).sum()),
            'losses':int((delta>1e-10).sum()),
            'cost_scope':'observed docking required for all 400 heldout candidates; physical cost unavailable',
            'status':'supplementary fixed retention fractions, not tuned on confirmation results'}
    (out.parent / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

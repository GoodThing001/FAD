"""End-to-end candidate ranking demo (deployment-style, no new labels).

HISTORICAL DEMO ONLY (2026-09-30): its three-rank gbsa/dock/MFE objective is
not the current GBSA-only optimization policy; the weighted label coefficients
have not been validated. Do not use this output as the mainline candidate list.

Trains the release five-model rank fusion (303d, winsor_2.5, screen-k=100) on
ALL 2000 labeled sequences (deployment fit, no split), plus the four-model
dock rank fusion (dock is a second TARGET — its rank enters the final list,
never the gbsa model's inputs). Ranks the 31,000 verified NUPACK candidates:

  gbsa_rank  = rank of the gbsa fusion prediction (expensive-label correction)
  dock_rank  = rank of the dock fusion prediction (clean second target)
  mfe_rank   = rank of full_sequence_mfe (free NUPACK physics prior)
  combined   = mean(gbsa_rank, dock_rank, mfe_rank)

The final label formula (0.961 Gap2 + 0.039 gbsa) needs Aft/Gap2, which the
batches do not carry yet — noted as pending.

Output: runs/candidate_demo/top200.csv (top-200 candidates) + summary JSON.

Usage (server, FAD_env):
  python -m scripts.reproduction.rank_candidates_demo --top-k 200
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
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from .eval_fusion_smoothing import build, get_model
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_fusion_smoothing import build, get_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_dock_model(name):
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
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.03,
                            random_state=42, verbosity=0, n_jobs=1)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--batch-dir", default="data/candidates")
    ap.add_argument("--data-dir", default="data/proxy2000_v2")
    ap.add_argument("--output", default="runs/candidate_demo")
    args = ap.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    full_df = pd.read_csv(data_dir / "fad_proxy2000_v2_full.csv")
    seqs_train = [str(s) for s in full_df["Sequence"]]
    y = full_df["gbsa"].to_numpy(dtype=np.float64)
    X_tr = build(seqs_train)

    # deployment fit: winsor on all labels, screen-k=100, then five models
    lo, hi = np.percentile(y, 2.5), np.percentile(y, 97.5)
    y_w = np.clip(y, lo, hi).astype(np.float32)
    et_s = ExtraTreesRegressor(n_estimators=200, max_features="log2",
                               min_samples_leaf=5, random_state=42, n_jobs=1)
    et_s.fit(X_tr, y_w)
    top = np.argsort(-et_s.feature_importances_)[:100]
    X_tr = X_tr[:, top]
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    models = {m: get_model(m) for m in ["RF", "ET", "GBR", "XGB", "SVR"]}
    for m in models.values():
        m.fit(X_tr_s, y_w)
    print("release fusion fitted on 2000 labeled sequences (303d, k=100)")

    # dock second-target fusion (dock rank only — never a gbsa input)
    dock = full_df["dock"].to_numpy(dtype=np.float64)
    dock_models = {m: make_dock_model(m) for m in ["RF", "ET", "GBR", "XGB"]}
    for m in dock_models.values():
        m.fit(X_tr, dock.astype(np.float32))
    print("dock four-model fusion fitted (second target)")

    # load candidates
    batch_dir = PROJECT_ROOT / args.batch_dir
    frames = []
    for p in sorted(batch_dir.glob("candidate_pool_nupack_236_batch_*.csv")):
        frames.append(pd.read_csv(p))
    cand = pd.concat(frames, ignore_index=True)
    cand = cand.drop_duplicates(subset="sequence").reset_index(drop=True)
    print(f"candidates: {len(cand)} unique sequences")

    seqs_c = [str(s).upper().replace("T", "U") for s in cand["sequence"]]
    X_c = build(seqs_c)[:, top]
    X_c_s = sc.transform(X_c)
    preds = {mname: rankdata(m.predict(X_c_s)) for mname, m in models.items()}
    fused = np.mean(list(preds.values()), axis=0)
    dock_preds = {mname: rankdata(m.predict(X_c)) for mname, m in dock_models.items()}
    dock_fused = np.mean(list(dock_preds.values()), axis=0)
    mfe = cand["full_sequence_mfe"].to_numpy(dtype=np.float64)
    mfe_rank = rankdata(mfe)
    combined = rankdata(fused) + rankdata(dock_fused) + rankdata(mfe_rank)

    order = np.argsort(combined)
    topk = order[: args.top_k]
    out_dir = PROJECT_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df = cand.iloc[topk][["sequence", "full_sequence",
                              "full_sequence_mfe", "nupack_mfe_structure"]].copy()
    out_df.insert(0, "rank", np.arange(1, len(topk) + 1))
    out_df["gbsa_fusion_rank"] = rankdata(fused)[topk].astype(int)
    out_df["dock_fusion_rank"] = rankdata(dock_fused)[topk].astype(int)
    out_df["mfe_rank"] = mfe_rank[topk].astype(int)
    out_df.to_csv(out_dir / "top200.csv", index=False)

    summary = {
        "n_labeled_train": len(seqs_train),
        "n_candidates": int(len(cand)),
        "n_unique_candidates": int(cand["sequence"].nunique()),
        "top_k": args.top_k,
        "mfe_range": [float(np.nanmin(mfe)), float(np.nanmax(mfe))],
        "fusion_pred_range": [float(fused.min()), float(fused.max())],
        "dock_fusion_range": [float(dock_fused.min()), float(dock_fused.max())],
        "overlap_with_train": int(len(set(seqs_c) & set(seqs_train))),
        "ranking_components": "mean(rank(gbsa_fusion), rank(dock_fusion), rank(mfe))",
    }
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[OK] -> {out_dir / 'top200.csv'}")


if __name__ == "__main__":
    main()

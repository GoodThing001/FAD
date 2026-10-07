"""v8 fusion x label smoothing combo (30 seeds, paired).

Question: does k-NN Hamming label smoothing (validated +0.019 on XGB-495d)
stack with the release five-model rank fusion (0.375, 100 seeds)?

Pipeline = phase3_fusion_v8 exactly (303d, screen-k=100, inner 5-fold OOF,
equal-weight rank fusion, winsor_2.5), with labels optionally k-NN-smoothed
INSIDE the training fold before winsor (no leakage).

Configs (same seeds):
  raw_fusion    : release pipeline
  smooth_fusion : same + kNN smoothing (k=5, alpha=0.5) on training labels

Usage (server, FAD_env):
  python -m scripts.reproduction.eval_fusion_smoothing --outer-seeds 30
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from scipy.stats import pearsonr, rankdata
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from .label_smoothing import hamming_matrix
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from label_smoothing import hamming_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASES = "ACGU"
MUT_POS = [30, 31, 32, 33, 89, 90, 91, 92, 127, 128, 129, 130, 131]
WT_141 = ("UAUCCUUCGGGGCAGGGUGGAAAUCCCGACCGGCGGUAGUAAAGCACAUUUGCUUUAGAGCCC"
          "GUGACCCGUGUGCAUAAGCACGCGGUGGAUUCAGUUUAAGCUGAAGCCGACAGUGAAAGUCUGGA"
          "UGGGAGAAGGAUG")


def _spearman(a, b):
    return pearsonr(rankdata(a), rankdata(b))[0]


def build(seqs):
    Xm = np.zeros((len(seqs), 52), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = str(s).upper().replace("T", "U")
        for j, p in enumerate(MUT_POS):
            idx = p - 1
            if idx < len(s) and s[idx] in BASES:
                Xm[i, j * 4 + BASES.index(s[idx])] = 1.0
    ROLES = {30: "stem", 31: "stem", 32: "stem", 33: "junction",
             89: "loop", 90: "loop", 91: "loop", 92: "junction",
             127: "stem", 128: "stem", 129: "stem", 130: "junction", 131: "loop"}
    PAIRS = {30: 33, 31: 32, 32: 31, 33: 30,
             90: 92, 92: 90, 127: 131, 131: 127, 128: 130, 130: 128}
    Xs = np.zeros((len(seqs), 43), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = str(s).upper().replace("T", "U"); mc = sm = lm = 0
        for j, p in enumerate(MUT_POS):
            idx = p - 1; wt = WT_141[idx]; mb = s[idx]
            is_mut = 1.0 if mb != wt else 0.0; role = ROLES.get(p, "unknown")
            Xs[i, j * 3] = 1.0 if role == "stem" else 0.0
            Xs[i, j * 3 + 1] = 1.0 if role == "loop" else 0.0
            Xs[i, j * 3 + 2] = is_mut * (1.0 if role == "stem" else 0.0)
            mc += is_mut
            if is_mut:
                if role == "stem": sm += 1
                elif role == "loop": lm += 1
        Xs[i, -4] = mc; Xs[i, -3] = sm; Xs[i, -2] = lm; broken = 0
        for p in MUT_POS:
            part = PAIRS.get(p, 0)
            if part > 0 and s[p - 1] != WT_141[p - 1]:
                if s[part - 1] == WT_141[part - 1]: broken += 1
        Xs[i, -1] = float(broken)
    all_d = [a + b for a in BASES for b in BASES]
    Xw = np.zeros((len(seqs), 208), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = str(s).upper().replace("T", "U"); n = len(s)
        for j, p in enumerate(MUT_POS):
            idx = p - 1; lo = max(0, idx - 5); hi = min(n, idx + 6)
            sub = s[lo:hi]
            for t in range(len(sub) - 1):
                d = sub[t:t + 2]
                if d[0] in BASES and d[1] in BASES: Xw[i, j * 16 + all_d.index(d)] += 1
            total = max(1, len(sub) - 1)
            Xw[i, j * 16:j * 16 + 16] /= total
    return np.hstack([Xm, Xs, Xw])


def get_model(name):
    if name == "RF":
        return RandomForestRegressor(n_estimators=1747, min_samples_leaf=14,
                                     max_features="sqrt", random_state=42, n_jobs=1)
    if name == "ET":
        return ExtraTreesRegressor(n_estimators=1596, min_samples_leaf=7,
                                   max_features="log2", random_state=42, n_jobs=1)
    if name == "GBR":
        return GradientBoostingRegressor(n_estimators=500, max_depth=3,
                                         min_samples_leaf=3, learning_rate=0.03,
                                         loss="absolute_error", random_state=42)
    if name == "XGB":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.03,
                            random_state=42, verbosity=0, n_jobs=1)
    if name == "SVR":
        return SVR(kernel="rbf", C=0.57, gamma="scale", epsilon=0.047)
    raise ValueError(name)


def smooth_labels(D, y, k, alpha):
    n = len(y)
    ys = np.zeros(n, dtype=np.float64)
    for i in range(n):
        order = np.argsort(D[i])[1:]
        knn = order[:k]
        ys[i] = (1 - alpha) * y[i] + alpha * y[knn].mean()
    return ys


def run_seed(X, y, gap2_te, label_te, seqs, seed, inner_folds, screen_k, smooth):
    tr_idx, te_idx = train_test_split(range(len(X)), test_size=0.2, random_state=seed)
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr_raw = y[tr_idx]
    y_te_raw = y[te_idx]

    if smooth:
        D_tr = hamming_matrix([seqs[i] for i in tr_idx])
        y_tr_raw = smooth_labels(D_tr, y_tr_raw, k=5, alpha=0.5)

    lo, hi = np.percentile(y_tr_raw, 2.5), np.percentile(y_tr_raw, 97.5)
    y_tr = np.clip(y_tr_raw, lo, hi).astype(np.float32)

    if screen_k < X_tr.shape[1]:
        et_s = ExtraTreesRegressor(n_estimators=200, max_features="log2",
                                   min_samples_leaf=5, random_state=42, n_jobs=1)
        et_s.fit(X_tr, y_tr)
        top = np.argsort(-et_s.feature_importances_)[:screen_k]
        X_tr, X_te = X_tr[:, top], X_te[:, top]

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    model_names = ["RF", "ET", "GBR", "XGB", "SVR"]
    oof_preds = {m: np.full(len(tr_idx), np.nan, dtype=np.float32) for m in model_names}
    test_preds = {}
    kf = KFold(n_splits=inner_folds, shuffle=True, random_state=42)
    for itr, iva in kf.split(X_tr_s):
        for mname in model_names:
            model = get_model(mname)
            model.fit(X_tr_s[itr], y_tr[itr])
            oof_preds[mname][iva] = model.predict(X_tr_s[iva])

    single_results = {}
    for mname in model_names:
        model = get_model(mname)
        model.fit(X_tr_s, y_tr)
        pred = model.predict(X_te_s)
        test_preds[mname] = pred
        single_results[mname] = _spearman(y_te_raw, pred)

    ranks = np.column_stack([rankdata(test_preds[m]) for m in model_names])
    fused_eq = ranks.mean(axis=1)
    sp_eq = _spearman(y_te_raw, fused_eq)
    best_single = max(single_results.values())
    return sp_eq, best_single


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer-seeds", type=int, default=30)
    ap.add_argument("--inner-folds", type=int, default=5)
    ap.add_argument("--screen-k", type=int, default=100)
    ap.add_argument("--data-dir", default="data/proxy2000_v2")
    ap.add_argument("--output", default="runs/eval_fusion_smoothing/results.json")
    args = ap.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    full_df = pd.read_csv(data_dir / "fad_proxy2000_v2_full.csv")
    seqs = [str(s) for s in full_df["Sequence"]]
    X_all = build(full_df["Sequence"].values)
    y_all = full_df["gbsa"].values

    per_seed = {"raw_fusion": [], "smooth_fusion": [],
                "raw_best_single": [], "smooth_best_single": []}
    seeds = list(range(1, args.outer_seeds + 1))
    for seed in seeds:
        sp_raw, bs_raw = run_seed(X_all, y_all, None, None, seqs, seed,
                                  args.inner_folds, args.screen_k, smooth=False)
        sp_sm, bs_sm = run_seed(X_all, y_all, None, None, seqs, seed,
                                args.inner_folds, args.screen_k, smooth=True)
        per_seed["raw_fusion"].append(sp_raw)
        per_seed["smooth_fusion"].append(sp_sm)
        per_seed["raw_best_single"].append(bs_raw)
        per_seed["smooth_best_single"].append(bs_sm)
        print(f"  seed {seed}/{args.outer_seeds}: raw_fusion={sp_raw:.4f} "
              f"smooth_fusion={sp_sm:.4f}", flush=True)

    res = {}
    print(f"\n{'='*60}")
    for name in ["raw_fusion", "smooth_fusion"]:
        arr = np.array(per_seed[name])
        res[name] = {"mean": float(arr.mean()), "std": float(arr.std()),
                     "per_seed": arr.tolist()}
        print(f"{name:<20} {arr.mean():.4f} +/- {arr.std():.4f}")
    d = np.array(per_seed["smooth_fusion"]) - np.array(per_seed["raw_fusion"])
    rng = np.random.default_rng(20260924)
    boot = rng.choice(d, size=(20000, len(d)), replace=True).mean(axis=1)
    res["paired_delta"] = {"mean": float(d.mean()),
                           "ci95": [float(np.percentile(boot, 2.5)),
                                    float(np.percentile(boot, 97.5))],
                           "wins": int((d > 0).sum()), "n_seeds": len(d)}
    print(f"paired smooth-raw: {d.mean():+.4f} CI "
          f"[{res['paired_delta']['ci95'][0]:.4f},{res['paired_delta']['ci95'][1]:.4f}] "
          f"wins {res['paired_delta']['wins']}/{len(d)}")

    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"per_seed": per_seed, "summary": res}, open(out, "w"), indent=2)
    print(f"[OK] -> {out}")


if __name__ == "__main__":
    main()

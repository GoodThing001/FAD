"""Phase 3 v8 — OOF Rank Fusion with Optimized Hyperparams.

Uses Phase 1+2 confirmed best config:
  - Target: winsor_2.5
  - Loss: MAE
  - Features: mut_struct_wd (303d, no ED)
  - Models: RF(1747,leaf14), ET(1596,leaf7), GBR(default), XGB(default), SVR(C=0.57)

Generates strict OOF predictions, then evaluates rank fusion.

Usage:
  python scripts/tools/phase3_fusion_v8.py \
    --outer-seeds 30 --inner-folds 5 \
    --n-jobs 4 --log-dir logs/phase3_v8 \
    --output artifacts/phase3_v8/fusion_results.csv
"""

from __future__ import annotations

import argparse, json, sys, time, warnings
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASES = "ACGU"
MUT_POS = [30, 31, 32, 33, 89, 90, 91, 92, 127, 128, 129, 130, 131]
WT_141 = ("UAUCCUUCGGGGCAGGGUGGAAAUCCCGACCGGCGGUAGUAAAGCACAUUUGCUUUAGAGCCC"
          "GUGACCCGUGUGCAUAAGCACGCGGUGGAUUCAGUUUAAGCUGAAGCCGACAGUGAAAGUCUGGA"
          "UGGGAGAAGGAUG")


def _spearman(a, b):
    return pearsonr(rankdata(a), rankdata(b))[0]


# ── Features ────────────────────────────────────────────────────────
def build(seqs):
    # mut
    Xm = np.zeros((len(seqs), 52), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = str(s).upper().replace("T", "U")
        for j, p in enumerate(MUT_POS):
            idx = p - 1
            if idx < len(s) and s[idx] in BASES:
                Xm[i, j * 4 + BASES.index(s[idx])] = 1.0
    # struct
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
    # wd
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


# ── Optimized models ────────────────────────────────────────────────
OPTIMIZED_MODELS = {
    "RF": RandomForestRegressor(n_estimators=1747, min_samples_leaf=14,
                                 max_features="sqrt", random_state=42, n_jobs=1),
    "ET": ExtraTreesRegressor(n_estimators=1596, min_samples_leaf=7,
                               max_features="log2", random_state=42, n_jobs=1),
    "GBR": GradientBoostingRegressor(n_estimators=500, max_depth=3,
                                      min_samples_leaf=3, learning_rate=0.03,
                                      loss="absolute_error", random_state=42),
    "XGB": None,  # lazy import
    "SVR": SVR(kernel="rbf", C=0.57, gamma="scale", epsilon=0.047),
}


def get_model(name):
    if name == "XGB":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.03,
                                random_state=42, verbosity=0, n_jobs=1)
        except ImportError: return None
    return OPTIMIZED_MODELS.get(name)


# ── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 3 v8 — OOF Rank Fusion")
    parser.add_argument("--outer-seeds", type=int, default=30)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--models", default="RF,ET,GBR,XGB,SVR")
    parser.add_argument("--screen-k", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--output", default="artifacts/phase3_v8/fusion_results.csv")
    parser.add_argument("--data-dir", default="data/proxy2000_v2")
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(",")]
    seeds = list(range(1, args.outer_seeds + 1))

    # Logging
    log_dir = None
    if args.log_dir:
        from datetime import datetime
        log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "run_meta.json", "w") as f:
            json.dump({"started": datetime.now().isoformat(), "command": " ".join(sys.argv),
                       "host": __import__("socket").gethostname()}, f, indent=2)

    def ckpt(seed, results, elapsed):
        if not log_dir: return
        bests = {}
        for r in results:
            s = r["seed"]
            if s not in bests or r["gbsa_Sp"] > bests[s]: bests[s] = r["gbsa_Sp"]
        with open(log_dir / "progress.json", "w") as f:
            json.dump({"current_seed": seed, "elapsed_seconds": round(elapsed, 1),
                       "best_per_seed": bests, "total": len(results)}, f, indent=2)
        running = log_dir / "all_results.csv"
        pd.DataFrame([r for r in results if r["seed"] == seed]).to_csv(
            running, mode="a", index=False, header=not running.exists())

    # Load data
    data_dir = PROJECT_ROOT / args.data_dir
    full_df = pd.read_csv(data_dir / "fad_proxy2000_v2_full.csv")
    X_all = build(full_df["Sequence"].values)
    y_all = full_df["gbsa"].values
    gap2_all = full_df["Gap2"].values
    label_all = full_df["label"].values
    print(f"Data: {len(full_df)} sequences, features: {X_all.shape[1]}d")

    all_results = []; t0 = time.time()

    for si, seed in enumerate(seeds):
        tr_idx, te_idx = train_test_split(range(len(full_df)), test_size=0.2, random_state=seed)
        X_tr, X_te = X_all[tr_idx], X_all[te_idx]
        y_tr_raw, y_te_raw = y_all[tr_idx], y_all[te_idx]
        gap2_te = gap2_all[te_idx]; label_te = label_all[te_idx]

        # winsor_2.5 transform
        lo, hi = np.percentile(y_tr_raw, 2.5), np.percentile(y_tr_raw, 97.5)
        y_tr = np.clip(y_tr_raw, lo, hi).astype(np.float32)

        # Screen top-K
        if args.screen_k < X_tr.shape[1]:
            from sklearn.ensemble import ExtraTreesRegressor as _ET
            et_s = _ET(n_estimators=200, max_features="log2", min_samples_leaf=5,
                       random_state=42, n_jobs=1)
            et_s.fit(X_tr, y_tr)
            top = np.argsort(-et_s.feature_importances_)[:args.screen_k]
            X_tr, X_te = X_tr[:, top], X_te[:, top]

        sc = StandardScaler(); X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)

        # Generate OOF via inner CV
        kf = KFold(n_splits=args.inner_folds, shuffle=True, random_state=42)
        oof_preds = {m: np.full(len(tr_idx), np.nan, dtype=np.float32) for m in model_names}
        test_preds = {}

        for itr, iva in kf.split(X_tr_s):
            for mname in model_names:
                model = get_model(mname)
                if model is None: continue
                model.fit(X_tr_s[itr], y_tr[itr])
                oof_preds[mname][iva] = model.predict(X_tr_s[iva])

        # Final fit on all training data
        single_results = {}
        for mname in model_names:
            model = get_model(mname)
            if model is None: continue
            model.fit(X_tr_s, y_tr)
            pred = model.predict(X_te_s)
            test_preds[mname] = pred
            single_results[mname] = _spearman(y_te_raw, pred)

        # ── Rank fusion ──
        # Equal-weight rank average
        ranks = np.column_stack([rankdata(test_preds[m]) for m in model_names if m in test_preds])
        fused_eq = ranks.mean(axis=1)
        sp_eq = _spearman(y_te_raw, fused_eq)

        # Constrained weighted (optimize on OOF)
        oof_ranks = np.column_stack([rankdata(oof_preds[m]) for m in model_names
                                      if m in oof_preds and not np.any(np.isnan(oof_preds[m]))])
        valid_models = [m for m in model_names if m in oof_preds and not np.any(np.isnan(oof_preds[m]))]

        from scipy.optimize import minimize
        n_m = len(valid_models)
        w0 = np.ones(n_m) / n_m

        def obj(w):
            return -_spearman(y_tr_raw, oof_ranks @ w) + 0.05 * np.sum(w**2)

        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * n_m
        res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-9})
        w_opt = np.maximum(res.x, 0); w_opt /= w_opt.sum()

        fused_w = ranks[:, :n_m] @ w_opt
        sp_w = _spearman(y_te_raw, fused_w)
        best_single = max(single_results.values())

        seed_best = max(sp_eq, sp_w, best_single)

        # Record
        for mname, sp_val in single_results.items():
            all_results.append({"seed": seed, "method": "single", "model": mname,
                                "gbsa_Sp": round(sp_val, 6)})
        all_results.append({"seed": seed, "method": "equal_weight_rank", "model": "fusion",
                           "gbsa_Sp": round(sp_eq, 6), "n_members": n_m})
        all_results.append({"seed": seed, "method": "constrained_weight", "model": "fusion",
                           "gbsa_Sp": round(sp_w, 6), "n_members": n_m,
                           "weights": json.dumps({m: round(float(w_opt[i]), 4)
                                                  for i, m in enumerate(valid_models)})})

        elapsed = time.time() - t0
        print(f"  seed={seed:2d} [{si+1}/{len(seeds)}]  best_single={best_single:.4f}  "
              f"eq_fusion={sp_eq:.4f}  w_fusion={sp_w:.4f}  best={seed_best:.4f}  {elapsed:.0f}s")
        ckpt(seed, all_results, elapsed)

    # Save
    results_df = pd.DataFrame(all_results)
    out = PROJECT_ROOT / args.output; out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out, index=False)
    print(f"\n[OK] {len(results_df)} rows → {out}")

    # Summary
    print(f"\n{'='*80}")
    print("PHASE 3 v8 — OOF RANK FUSION RESULTS")
    print(f"{'='*80}")

    # Single models
    for mname in model_names:
        sdf = results_df[(results_df.method == "single") & (results_df.model == mname)]
        if len(sdf) > 0:
            sps = sdf.gbsa_Sp.values
            print(f"{'single.'+mname:<20} {np.mean(sps):.4f} +/- {np.std(sps):.4f}  [{np.min(sps):.4f},{np.max(sps):.4f}]")

    # Fusion methods
    for method in ["equal_weight_rank", "constrained_weight"]:
        fdf = results_df[results_df.method == method]
        if len(fdf) == 0: continue
        bdf = results_df[results_df.method == "single"]
        best_per_seed = bdf.groupby("seed").gbsa_Sp.max()

        f_means = []; deltas = []; wins = 0; n_total = 0
        for _, row in fdf.iterrows():
            s = row["seed"]
            if s in best_per_seed.index:
                delta = row["gbsa_Sp"] - best_per_seed[s]
                f_means.append(row["gbsa_Sp"])
                deltas.append(delta)
                if delta > 0: wins += 1
                n_total += 1

        if n_total > 0:
            print(f"{method:<20} {np.mean(f_means):.4f} +/- {np.std(f_means):.4f}  "
                  f"vs_best_single={np.mean(deltas):+.4f}  wins={wins}/{n_total}")


if __name__ == "__main__":
    main()

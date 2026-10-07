"""Feature-block weighted selection with feedback optimization.

Implements the "加权 + 反馈 + 优化" idea: instead of hard-selecting NUPACK
blocks (keep the 7 useful, drop the rest), assign a learnable weight to each of
the 18 NUPACK blocks and optimize the weights via out-of-fold (OOF) feedback
(Spearman). A block whose marginal delta is negative but which carries latent
complementary signal can then receive a small positive weight rather than being
discarded.

Method (per outer seed):
  1. split train/test (80/20)
  2. baseline 303d + each NUPACK block -> OOF rank predictions (inner 5-fold)
  3. optimize weights w (baseline + blocks) on OOF ranks to maximize Spearman
     (constraints w >= 0, sum w = 1, L2 penalty)
  4. apply learned weights to test ranks -> evaluate Spearman

Runs in FAD_env (xgboost/sklearn). Block definitions match eval_nupack_blocks.py.

Usage (on server, FAD_env):
  python scripts/reproduction/feature_weighting.py \
    --outer-seeds 10 --model XGB --screen-k 100 --n-jobs 8
"""

from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from scipy.optimize import minimize
from scipy.stats import pearsonr, rankdata
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASES = "ACGU"
MUT_POS = [30, 31, 32, 33, 89, 90, 91, 92, 127, 128, 129, 130, 131]
WT_141 = ("UAUCCUUCGGGGCAGGGUGGAAAUCCCGACCGGCGGUAGUAAAGCACAUUUGCUUUAGAGCCC"
          "GUGACCCGUGUGCAUAAGCACGCGGUGGAUUCAGUUUAAGCUGAAGCCGACAGUGAAAGUCUGGA"
          "UGGGAGAAGGAUG")

BLOCKS = {
    "pos_pair_prob": [r"nupack_pos\d+_pair_prob$", r"nupack_pos\d+_likely_paired$",
                      r"nupack_c\d+_mean_pair_prob$", r"nupack_mean_pair_prob$",
                      r"nupack_std_pair_prob$", r"nupack_frac_paired_gt_0_5$"],
    "pos_entropy": [r"nupack_pos\d+_entropy$"],
    "win_stats": [r"nupack_pos\d+_win\d+_(mean|std)$"],
    "cluster_pair": [r"nupack_inter_cluster_\d+_\d+$", r"nupack_intra_cluster_\d+$",
                     r"nupack_mut_long_range$"],
    "pair_dist": [r"nupack_pos\d+_pair_dist$"],
    "seg_entropy": [r"nupack_seg_entropy\d+_(mean|std|min|max)$", r"nupack_pos\d+_seg3_entropy$"],
    "ensemble": [r"nupack_ensemble_entropy$", r"nupack_ensemble_entropy_norm$",
                 r"nupack_participation_ratio$", r"nupack_n_pairs_gt_0_1$"],
    "contact_pca": [r"nupack_contact_"],
    "expected_acc": [r"nupack_expected_accuracy_", r"nupack_frac_accurate_",
                     r"nupack_cl\d+_expected_acc$"],
    "mfe_basic": [r"nupack_mfe$", r"nupack_mfe_n_stems$", r"nupack_mfe_stem_(max|mean)_len$",
                  r"nupack_mfe_n_loops$", r"nupack_mfe_loop_max_len$", r"nupack_mfe_n_unpaired$",
                  r"nupack_mfe_frac_paired$", r"nupack_mfe_pos\d+_is_paired$", r"nupack_mfe_c\d+_paired$"],
    "squarna": [r"nupack_sq_"],
    "mfe_detailed": [r"nupack_mfe_n_bulges$", r"nupack_mfe_bulge_(max|mean|total)$",
                     r"nupack_mfe_n_internal_loops$", r"nupack_mfe_iloop_(max|asym_max|asym_mean)$",
                     r"nupack_mfe_n_helices$", r"nupack_mfe_n_hairpins$",
                     r"nupack_mfe_hairpin_(max|mean)$", r"nupack_mfe_c\d+_(bulges|iloops)$"],
    "ufold_blocks": [r"nupack_block_"],
    "dist_moments": [r"nupack_triu_", r"nupack_density_", r"nupack_dist_", r"nupack_frac_above_"],
    "gradient": [r"nupack_pos\d+_grad_", r"nupack_pos\d+_partner_entropy$",
                 r"nupack_pos\d+_best_partner_dist$"],
    "energy_land": [r"nupack_n_very_stable_pairs$", r"nupack_n_moderate_pairs$",
                    r"nupack_stable_pair_ratio$", r"nupack_eig_ipr$", r"nupack_eig_n_eff_modes$"],
    "motif_positional": [r"nupack_motif_pos\d+_is_"],
    "motif_global": [r"nupack_motif_global_", r"nupack_motif_apt_"],
}


def _spearman(a, b):
    return pearsonr(rankdata(a), rankdata(b))[0]


def build_seq(seqs):
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
    PAIRS = {30: 33, 31: 32, 32: 31, 33: 30, 90: 92, 92: 90, 127: 131, 131: 127, 128: 130, 130: 128}
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
            if part > 0 and s[p - 1] != WT_141[p - 1] and s[part - 1] == WT_141[part - 1]:
                broken += 1
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
                if d[0] in BASES and d[1] in BASES:
                    Xw[i, j * 16 + all_d.index(d)] += 1
            total = max(1, len(sub) - 1)
            Xw[i, j * 16:j * 16 + 16] /= total
    return np.hstack([Xm, Xs, Xw])


def make_model(name):
    if name == "XGB":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.03,
                            random_state=42, verbosity=0, n_jobs=1)
    if name == "GBR":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(n_estimators=500, max_depth=3,
                                         min_samples_leaf=3, learning_rate=0.03,
                                         loss="absolute_error", random_state=42)
    if name == "ET":
        return ExtraTreesRegressor(n_estimators=500, max_features="log2",
                                   min_samples_leaf=5, random_state=42, n_jobs=1)
    raise ValueError(name)


def screen_topk(X_tr, y_tr, X_te, k):
    if k >= X_tr.shape[1]:
        return X_tr, X_te
    et = ExtraTreesRegressor(n_estimators=200, max_features="log2",
                             min_samples_leaf=5, random_state=42, n_jobs=1)
    et.fit(X_tr, y_tr)
    top = np.argsort(-et.feature_importances_)[:k]
    return X_tr[:, top], X_te[:, top]


def fit_predict(model_name, X_tr, y_tr, X_te):
    """Fit a model on (X_tr, y_tr) and predict X_te (no scaling inside)."""
    m = make_model(model_name)
    m.fit(X_tr, y_tr)
    return m.predict(X_te)


def fit_member(job):
    """Fit one member (baseline or baseline+block): return (name, oof_rank, test_pred)."""
    name, Xm_tr, Xm_te, y_tr, screen_k, inner_folds, model = job
    kf = KFold(n_splits=inner_folds, shuffle=True, random_state=42)
    oof = np.full(len(y_tr), np.nan, dtype=np.float32)
    for itr, iva in kf.split(Xm_tr):
        Xc_tr, Xc_va = screen_topk(Xm_tr[itr], y_tr[itr], Xm_tr[iva], screen_k)
        sc = StandardScaler()
        Xc_tr_s = sc.fit_transform(Xc_tr); Xc_va_s = sc.transform(Xc_va)
        oof[iva] = fit_predict(model, Xc_tr_s, y_tr[itr], Xc_va_s)
    Xc_tr, Xc_te = screen_topk(Xm_tr, y_tr, Xm_te, screen_k)
    sc = StandardScaler()
    Xc_tr_s = sc.fit_transform(Xc_tr); Xc_te_s = sc.transform(Xc_te)
    test_pred = fit_predict(model, Xc_tr_s, y_tr, Xc_te_s)
    return name, rankdata(oof), test_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer-seeds", type=int, default=10)
    ap.add_argument("--model", default="XGB")
    ap.add_argument("--screen-k", type=int, default=100)
    ap.add_argument("--inner-folds", type=int, default=5)
    ap.add_argument("--l2", type=float, default=0.05)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--data-dir", default="data/proxy2000_v2")
    ap.add_argument("--output", default="runs/feature_weighting/results.json")
    ap.add_argument("--log-dir", default="runs/feature_weighting")
    args = ap.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    full_df = pd.read_csv(data_dir / "fad_proxy2000_v2_full.csv")
    X_seq = build_seq(full_df["Sequence"].values)
    y_all = full_df["gbsa"].values

    nup_df = pd.read_csv(data_dir / "nupack_features_full.csv")
    nup_df["Sequence"] = nup_df["Sequence"].astype(str)
    canon = [str(s) for s in full_df["Sequence"]]
    nup_df = nup_df.set_index("Sequence").loc[canon]
    nup_cols = [c for c in nup_df.columns if c.startswith("nupack_")]
    X_nup = nup_df[nup_cols].to_numpy(dtype=np.float32)

    # group into blocks
    block_idx = {}
    used = set()
    for bname, pats in BLOCKS.items():
        idx = []
        for i, c in enumerate(nup_cols):
            if i in used:
                continue
            for p in pats:
                if re.search(p, c):
                    idx.append(i); used.add(i); break
        if idx:
            block_idx[bname] = idx
    unassigned = [i for i in range(len(nup_cols)) if i not in used]
    print(f"NUPACK {len(nup_cols)}d -> {len(block_idx)} blocks (+{len(unassigned)} unassigned)")
    for b, idx in block_idx.items():
        print(f"  {b:<18} {len(idx):>4}d")

    block_names = list(block_idx.keys())
    seeds = list(range(1, args.outer_seeds + 1))
    all_res = []
    t0 = time.time()

    for seed in seeds:
        tr_idx, te_idx = train_test_split(range(len(full_df)), test_size=0.2, random_state=seed)
        y_tr_raw, y_te_raw = y_all[tr_idx], y_all[te_idx]
        lo, hi = np.percentile(y_tr_raw, 2.5), np.percentile(y_tr_raw, 97.5)
        y_tr = np.clip(y_tr_raw, lo, hi).astype(np.float32)

        # Build feature matrices: baseline + baseline+block
        mats = {}  # name -> (X_tr, X_te) raw (unscaled)
        Xs_tr, Xs_te = X_seq[tr_idx], X_seq[te_idx]
        mats["baseline"] = (Xs_tr, Xs_te)
        for b, idx in block_idx.items():
            Xb_tr = np.hstack([Xs_tr, X_nup[tr_idx][:, idx]])
            Xb_te = np.hstack([Xs_te, X_nup[te_idx][:, idx]])
            mats[b] = (Xb_tr, Xb_te)

        # OOF + test predictions for each member (baseline + blocks), in parallel
        names = ["baseline"] + block_names
        jobs = [(n, mats[n][0], mats[n][1], y_tr, args.screen_k, args.inner_folds, args.model)
                for n in names]
        results = Parallel(n_jobs=args.n_jobs)(delayed(fit_member)(j) for j in jobs)
        oof_ranks = {r[0]: r[1] for r in results}
        test_preds = {r[0]: r[2] for r in results}

        # ── weight optimization (feedback) ──
        R = np.column_stack([oof_ranks[n] for n in names])  # (n_train, n_members)
        n_m = len(names)
        w0 = np.ones(n_m) / n_m

        def obj(w):
            return -_spearman(y_tr_raw, R @ w) + args.l2 * np.sum(w ** 2)

        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bnds = [(0.0, 1.0)] * n_m
        res = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons,
                       options={"maxiter": 1000, "ftol": 1e-9})
        w_opt = np.maximum(res.x, 0); w_opt /= w_opt.sum()

        # test evaluation
        R_te = np.column_stack([rankdata(test_preds[n]) for n in names])
        fused_sp = _spearman(y_te_raw, R_te @ w_opt)
        base_sp = _spearman(y_te_raw, rankdata(test_preds["baseline"]))
        # hard selection reference: only the 7 "useful" blocks (+baseline), equal weight
        useful = ["baseline", "contact_pca", "cluster_pair", "energy_land", "pos_entropy",
                  "motif_global", "pos_pair_prob", "motif_positional"]
        R_u = np.column_stack([rankdata(test_preds[n]) for n in useful if n in names])
        hard_sp = _spearman(y_te_raw, R_u.mean(axis=1))

        weights = {n: round(float(w_opt[i]), 4) for i, n in enumerate(names)}
        all_res.append({"seed": seed, "weighted_fusion": round(fused_sp, 6),
                        "baseline": round(base_sp, 6), "hard_useful7": round(hard_sp, 6),
                        "weights": weights})

        top_blocks = sorted(weights.items(), key=lambda kv: -kv[1])[:5]
        print(f"  seed={seed:2d}  base={base_sp:.4f}  weighted={fused_sp:.4f}  "
              f"hard7={hard_sp:.4f}  top={[(b, w) for b, w in top_blocks]}  {time.time()-t0:.0f}s")

        # checkpoint
        out_dir = PROJECT_ROOT / args.log_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "partial.json", "w") as f:
            json.dump(all_res, f, indent=2)

    # summary
    wf = np.array([r["weighted_fusion"] for r in all_res])
    bf = np.array([r["baseline"] for r in all_res])
    hf = np.array([r["hard_useful7"] for r in all_res])
    print(f"\n{'='*70}")
    print(f"weighted fusion : {wf.mean():.4f} +/- {wf.std():.4f}")
    print(f"baseline (303d)  : {bf.mean():.4f} +/- {bf.std():.4f}")
    print(f"hard useful7     : {hf.mean():.4f} +/- {hf.std():.4f}")
    print(f"weighted vs base : {(wf - bf).mean():+.4f}  (win {int((wf > bf).sum())}/{len(wf)})")
    print(f"weighted vs hard7: {(wf - hf).mean():+.4f}  (win {int((wf > hf).sum())}/{len(wf)})")
    # average learned weights across seeds
    agg = {n: float(np.mean([r["weights"].get(n, 0.0) for r in all_res]))
           for n in names}
    print("\nmean learned weights (sorted):")
    for n, w in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"  {n:<18} {w:.4f}")

    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"results": all_res, "mean_weights": agg,
               "weighted_mean": float(wf.mean()), "baseline_mean": float(bf.mean()),
               "hard_useful7_mean": float(hf.mean())}, open(out, "w"), indent=2)
    print(f"\n[OK] -> {out}")


if __name__ == "__main__":
    main()

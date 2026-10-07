"""Active-learning closed loop with a STRONG surrogate (XGB + NUPACK).

HISTORICAL / NOT VALID FOR FORMAL AL COMPARISON (audit 2026-09-30): the
minimization promising score is sign-reversed; train and query arrays are
standardized independently; useful NUPACK blocks were selected on full-data
target information. Keep for provenance and use the repaired loop for new claims.

Replaces run_active.py's weak MLP surrogate with the strong XGB + NUPACK stack
(the ~0.40-level features), and uses bootstrap bagging for predictive uncertainty
(std across ~15 bootstrapped XGB members). Acquisition = the 汇报 scheme
(uncertainty + coverage/novelty + promising + random control). Retrospective:
hide labels, simulate the expensive GBSA oracle, compare AL vs random learning
curves on a fixed holdout.

Runs in FAD_env (xgboost).

Usage (server, FAD_env):
  python scripts/reproduction/active_learning_strong.py \
    --n-init 400 --n-holdout 400 --n-batch 100 --n-rounds 10 --n-members 15
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, rankdata
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASES = "ACGU"
MUT_POS = [30, 31, 32, 33, 89, 90, 91, 92, 127, 128, 129, 130, 131]
WT_141 = ("UAUCCUUCGGGGCAGGGUGGAAAUCCCGACCGGCGGUAGUAAAGCACAUUUGCUUUAGAGCCC"
          "GUGACCCGUGUGCAUAAGCACGCGGUGGAUUCAGUUUAAGCUGAAGCCGACAGUGAAAGUCUGGA"
          "UGGGAGAAGGAUG")

USEFUL = {
    "contact_pca": [r"nupack_contact_"],
    "cluster_pair": [r"nupack_inter_cluster_\d+_\d+$", r"nupack_intra_cluster_\d+$",
                     r"nupack_mut_long_range$"],
    "energy_land": [r"nupack_n_very_stable_pairs$", r"nupack_n_moderate_pairs$",
                    r"nupack_stable_pair_ratio$", r"nupack_eig_ipr$", r"nupack_eig_n_eff_modes$"],
    "pos_entropy": [r"nupack_pos\d+_entropy$"],
    "motif_global": [r"nupack_motif_global_", r"nupack_motif_apt_"],
    "pos_pair_prob": [r"nupack_pos\d+_pair_prob$", r"nupack_pos\d+_likely_paired$",
                      r"nupack_c\d+_mean_pair_prob$", r"nupack_mean_pair_prob$",
                      r"nupack_std_pair_prob$", r"nupack_frac_paired_gt_0_5$"],
    "motif_positional": [r"nupack_motif_pos\d+_is_"],
}


def _spearman(a, b):
    return pearsonr(rankdata(a), rankdata(b))[0]


def standardize(X):
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - X.mean(axis=0)) / sd


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


def load_nupack_useful(csv_path, seqs):
    df = pd.read_csv(csv_path)
    df["Sequence"] = df["Sequence"].astype(str)
    cols = [c for c in df.columns if c.startswith("nupack_")]
    sel = sorted(set(i for pats in USEFUL.values() for i, c in enumerate(cols)
                     if any(re.search(p, c) for p in pats)))
    df = df.set_index("Sequence").loc[[str(s) for s in seqs]]
    return df[[cols[i] for i in sel]].to_numpy(dtype=np.float32), [cols[i] for i in sel]


def make_xgb(seed):
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=seed, verbosity=0, n_jobs=1)


def fit_bag_predict(X_tr, y_tr, X_q, n_members, seed0):
    """Bootstrap bagging: n_members XGB on bootstrap samples -> (mu, sigma) on X_q."""
    n = len(y_tr)
    rng = np.random.RandomState(seed0)
    preds = []
    for m in range(n_members):
        idx = rng.choice(n, size=n, replace=True)
        model = make_xgb(seed0 + m)
        model.fit(X_tr[idx], y_tr[idx])
        preds.append(model.predict(X_q))
    P = np.vstack(preds)
    return P.mean(axis=0), P.std(axis=0)


def select_batch(mu, sigma, pool_phy, labeled_phy, n_batch, minimize, seed):
    rng = np.random.RandomState(seed)
    pool_std = standardize(pool_phy)
    labeled_std = standardize(labeled_phy) if len(labeled_phy) else pool_std[:0]
    nov = cdist(pool_std, labeled_std).min(axis=1) if len(labeled_phy) else np.zeros(len(pool_phy))
    unc = sigma
    prom = -mu if minimize else mu
    selected = set()

    def take(order, k):
        got = []
        for i in order:
            if i not in selected:
                selected.add(int(i)); got.append(int(i))
                if len(got) >= k:
                    break
        return got

    n_unc, n_cov, n_prom, n_rand = 45, 25, 20, 10
    tot = n_unc + n_cov + n_prom + n_rand
    if tot != n_batch:
        n_unc = int(round(n_batch * n_unc / tot)); n_cov = int(round(n_batch * n_cov / tot))
        n_prom = int(round(n_batch * n_prom / tot)); n_rand = n_batch - n_unc - n_cov - n_prom
    take(np.argsort(-unc), n_unc)
    take(np.argsort(-nov), n_cov)
    take(np.argsort(prom), n_prom)
    rest = [i for i in range(len(mu)) if i not in selected]
    for i in rng.choice(rest, size=min(n_rand, len(rest)), replace=False):
        selected.add(int(i))
    while len(selected) < n_batch:
        for i in np.argsort(-unc):
            if len(selected) >= n_batch:
                break
            selected.add(int(i))
    return sorted(selected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-init", type=int, default=400)
    ap.add_argument("--n-holdout", type=int, default=400)
    ap.add_argument("--n-batch", type=int, default=100)
    ap.add_argument("--n-rounds", type=int, default=10)
    ap.add_argument("--n-members", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="data/proxy2000_v2")
    ap.add_argument("--output", default="runs/active_strong/results.json")
    ap.add_argument("--log-dir", default="runs/active_strong")
    args = ap.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    df = pd.read_csv(data_dir / "fad_proxy2000_v2_full.csv")
    seqs = [str(s) for s in df["Sequence"]]
    y = df["gbsa"].to_numpy(dtype=np.float32)
    X_seq = build_seq(seqs)
    X_nup, _ = load_nupack_useful(data_dir / "nupack_features_full.csv", seqs)
    X = np.hstack([X_seq, X_nup])
    print(f"features: {X.shape[1]}d (303 seq + {X_nup.shape[1]} nupack)")

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(seqs))
    hold = perm[: args.n_holdout]
    rest = perm[args.n_holdout:]
    init = rest[: args.n_init]
    pool0 = rest[args.n_init:]
    print(f"holdout={len(hold)} init={len(init)} pool={len(pool0)}")

    def run_loop(strategy, seed_off):
        labeled = init.copy()
        pool = pool0.copy()
        curve = []
        for rnd in range(args.n_rounds):
            X_lab = standardize(X[labeled])
            y_lab = y[labeled]
            mu_hold, _ = fit_bag_predict(X_lab, y_lab, standardize(X[hold]),
                                         args.n_members, args.seed + seed_off + rnd)
            sp = _spearman(y[hold], mu_hold)
            curve.append(sp)

            mu_pool, sig_pool = fit_bag_predict(X_lab, y_lab, standardize(X[pool]),
                                                args.n_members, args.seed + seed_off + rnd)
            if strategy == "active":
                sel = select_batch(mu_pool, sig_pool, X[pool], X[labeled],
                                   args.n_batch, True, args.seed + rnd)
            else:
                sel = np.random.RandomState(args.seed + rnd + seed_off).choice(
                    len(pool), size=min(args.n_batch, len(pool)), replace=False).tolist()
            labeled = np.concatenate([labeled, pool[sel]])
            pool = np.delete(pool, sel)
            print(f"  {strategy} rnd {rnd+1}/{args.n_rounds}: sp={sp:.4f} labeled={len(labeled)}",
                  flush=True)
        return curve

    print("== active learning loop (strong XGB surrogate) ==")
    curve_al = run_loop("active", seed_off=1000)
    print("== random control loop ==")
    curve_rand = run_loop("random", seed_off=2000)

    out = {"curve_active": curve_al, "curve_random": curve_rand,
           "n_init": args.n_init, "n_batch": args.n_batch, "n_members": args.n_members}
    outp = PROJECT_ROOT / args.output
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(outp, "w"), indent=2)
    print(f"\nactive: {['%.4f' % v for v in curve_al]}")
    print(f"random: {['%.4f' % v for v in curve_rand]}")
    print(f"done -> {outp}")


if __name__ == "__main__":
    main()

"""New physics features from literature survey — evaluate marginal gain over 303d + NUPACK.

Implements the highest-ROI new features identified by the literature survey:
  1. Turner nearest-neighbor dinucleotide stacking energy localized to the 13
     mutation sites (52d: 5' and 3' stacking dinucleotides at each site, 2×13×2,
     plus WT-delta forms — the docstring of an earlier draft said 26d, which was
     wrong). This is the "repair the failed energy feature" case from
     docs/source_evidence: the old Turner 'energy' feature (70d, 37C) had real
     signal (indep Sp 0.258) but was killed by temperature mismatch + redundancy;
     a localized, low-dim version re-adds that physics.
  2. (planned) distance-to-ligand SE(3)-invariant features from PDB 3K1V.

Evaluates with XGB: baseline (303d + NUPACK hard7) vs + new features, 30-seed.
Runs in FAD_env (xgboost). Note: the STACK table below is an APPROXIMATION of
Turner values (documented limitation; cite accordingly).

Usage (server, FAD_env):
  python scripts/reproduction/new_features.py --outer-seeds 30
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

from scipy.stats import pearsonr, rankdata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASES = "ACGU"
MUT_POS = [30, 31, 32, 33, 89, 90, 91, 92, 127, 128, 129, 130, 131]
WT_141 = ("UAUCCUUCGGGGCAGGGUGGAAAUCCCGACCGGCGGUAGUAAAGCACAUUUGCUUUAGAGCCC"
          "GUGACCCGUGUGCAUAAGCACGCGGUGGAUUCAGUUUAAGCUGAAGCCGACAGUGAAAGUCUGGA"
          "UGGGAGAAGGAUG")

# RNA dinucleotide stacking free energy (kcal/mol), approximate Turner/PseDNC
# nearest-neighbor values. Captures the correct ordering (GC/CG stacks most
# stable, AU/UU least); used as a relative physicochemical scale.
STACK = {
    "AA": -0.93, "AU": -1.10, "AC": -1.44, "AG": -1.81,
    "UA": -1.33, "UU": -0.93, "UC": -1.39, "UG": -1.45,
    "CA": -1.45, "CU": -1.39, "CC": -1.84, "CG": -2.17,
    "GA": -1.30, "GU": -1.39, "GC": -2.24, "GG": -1.84,
}

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


def build_seq(seqs):
    """303d hand-crafted (mut/struct/wd), same as feature_weighting.py."""
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


def build_stacking(seqs):
    """Turner NN dinucleotide stacking energy at the 13 sites.

    For each site i: stack5' = STACK[seq[i-1]+seq[i]], stack3' = STACK[seq[i]+seq[i+1]],
    plus delta vs WT. -> 13 x 4 = 52d.
    """
    n = len(seqs)
    X = np.zeros((n, 52), dtype=np.float32)
    for i, s0 in enumerate(seqs):
        s = str(s0).upper().replace("T", "U")
        for j, p in enumerate(MUT_POS):
            idx = p - 1
            d5 = (s[idx - 1] + s[idx]) if idx - 1 >= 0 else "AA"
            d3 = (s[idx] + s[idx + 1]) if idx + 1 < len(s) else "AA"
            wt5 = (WT_141[idx - 1] + WT_141[idx]) if idx - 1 >= 0 else "AA"
            wt3 = (WT_141[idx] + WT_141[idx + 1]) if idx + 1 < len(WT_141) else "AA"
            e5, e3 = STACK.get(d5, 0.0), STACK.get(d3, 0.0)
            X[i, j * 4] = e5
            X[i, j * 4 + 1] = e3
            X[i, j * 4 + 2] = e5 - STACK.get(wt5, 0.0)
            X[i, j * 4 + 3] = e3 - STACK.get(wt3, 0.0)
    return X


def make_xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.03,
                        random_state=42, verbosity=0, n_jobs=1)


def eval_features(X, y, seeds):
    """30-seed XGB Spearman for a feature matrix X."""
    sps = []
    for seed in seeds:
        tr, te = train_test_split(range(len(X)), test_size=0.2, random_state=seed)
        y_tr_raw = y[tr]
        lo, hi = np.percentile(y_tr_raw, 2.5), np.percentile(y_tr_raw, 97.5)
        y_tr = np.clip(y_tr_raw, lo, hi)
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr]); X_te = sc.transform(X[te])
        m = make_xgb()
        m.fit(X_tr, y_tr)
        sps.append(_spearman(y[te], m.predict(X_te)))
    return float(np.mean(sps)), float(np.std(sps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer-seeds", type=int, default=30)
    ap.add_argument("--data-dir", default="data/proxy2000_v2")
    ap.add_argument("--output", default="runs/new_features/results.json")
    args = ap.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    full_df = pd.read_csv(data_dir / "fad_proxy2000_v2_full.csv")
    seqs = [str(s) for s in full_df["Sequence"]]
    y = full_df["gbsa"].to_numpy(dtype=np.float32)
    X_seq = build_seq(seqs)
    X_stack = build_stacking(seqs)

    # NUPACK useful blocks
    nup_df = pd.read_csv(data_dir / "nupack_features_full.csv")
    nup_df["Sequence"] = nup_df["Sequence"].astype(str)
    cols = [c for c in nup_df.columns if c.startswith("nupack_")]
    sel = sorted(set(i for pats in USEFUL.values() for i, c in enumerate(cols)
                     if any(re.search(p, c) for p in pats)))
    X_nup = nup_df[[cols[i] for i in sel]].to_numpy(dtype=np.float32)
    print(f"seq 303d, nupack {X_nup.shape[1]}d, stacking 52d")

    seeds = list(range(1, args.outer_seeds + 1))

    base = np.hstack([X_seq, X_nup])
    s_base, _ = eval_features(base, y, seeds)
    s_stack, _ = eval_features(np.hstack([base, X_stack]), y, seeds)
    s_stackonly, _ = eval_features(X_stack, y, seeds)

    print(f"\n{'='*60}")
    print(f"303d + NUPACK          : {s_base:.4f}")
    print(f"303d + NUPACK + Turner : {s_stack:.4f}  (Δ {s_stack - s_base:+.4f})")
    print(f"Turner stacking alone  : {s_stackonly:.4f}")

    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"base": s_base, "base_plus_turner": s_stack, "turner_alone": s_stackonly,
               "delta": s_stack - s_base}, open(out, "w"), indent=2)
    print(f"[OK] -> {out}")


if __name__ == "__main__":
    main()

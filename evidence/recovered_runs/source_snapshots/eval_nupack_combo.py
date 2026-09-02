"""NUPACK Useful Sub-block Combination Validation — v11.

Validates whether the 7 useful NUPACK sub-blocks (192d), when COMBINED
with sequence features (303d), can beat the pure-sequence baseline (0.375).

7 useful blocks (from v10b):
  contact_pca (15) + cluster_pair (7) + energy_land (5) + pos_entropy (13)
  + motif_global (42) + pos_pair_prob (32) + motif_positional (78) = 192d

Protocol: 30 seeds, XGB/GBR, winsor_2.5 target, MAE loss, top-100 screening.
Compare: seq 303d (baseline) vs seq 303d + NUPACK 192d (combo).

Usage:
  python scripts/tools/eval_nupack_combo.py \
    --outer-seeds 30 --models XGB,GBR \
    --screen-k 100 --n-jobs 4 \
    --log-dir logs/v11 \
    --output artifacts/v11/nupack_combo_30seed.csv
"""

from __future__ import annotations

import argparse, json, re, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from scipy.stats import pearsonr, rankdata
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASES = "ACGU"
MUT_POS = [30, 31, 32, 33, 89, 90, 91, 92, 127, 128, 129, 130, 131]
WT_141 = ("UAUCCUUCGGGGCAGGGUGGAAAUCCCGACCGGCGGUAGUAAAGCACAUUUGCUUUAGAGCCC"
          "GUGACCCGUGUGCAUAAGCACGCGGUGGAUUCAGUUUAAGCUGAAGCCGACAGUGAAAGUCUGGA"
          "UGGGAGAAGGAUG")


def _spearman(a, b):
    return pearsonr(rankdata(a), rankdata(b))[0]


# ═══════════════════════════════════════════════════════════════════════
# 7 USEFUL NUPACK SUB-BLOCKS (from v10b)
# ═══════════════════════════════════════════════════════════════════════

USEFUL_BLOCKS = {
    "contact_pca": [r"nupack_contact_"],
    "cluster_pair": [r"nupack_inter_cluster_\d+_\d+$", r"nupack_intra_cluster_\d+$",
                     r"nupack_mut_long_range$"],
    "energy_land": [r"nupack_n_very_stable_pairs$", r"nupack_n_moderate_pairs$",
                    r"nupack_stable_pair_ratio$", r"nupack_eig_ipr$",
                    r"nupack_eig_n_eff_modes$"],
    "pos_entropy": [r"nupack_pos\d+_entropy$"],
    "motif_global": [r"nupack_motif_global_", r"nupack_motif_apt_"],
    "pos_pair_prob": [r"nupack_pos\d+_pair_prob$", r"nupack_pos\d+_likely_paired$",
                      r"nupack_c\d+_mean_pair_prob$", r"nupack_mean_pair_prob$",
                      r"nupack_std_pair_prob$", r"nupack_frac_paired_gt_0_5$"],
    "motif_positional": [r"nupack_motif_pos\d+_is_"],
}


def select_useful_columns(nupack_cols):
    """Select columns matching the 7 useful blocks."""
    selected = []
    for block_name, patterns in USEFUL_BLOCKS.items():
        for i, col in enumerate(nupack_cols):
            for pat in patterns:
                if re.search(pat, col):
                    selected.append(i)
                    break
    # Deduplicate while preserving order
    return sorted(set(selected))


# ═══════════════════════════════════════════════════════════════════════
# SEQUENCE FEATURES (303d)
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════

def make_model(name):
    if name == "GBR":
        return GradientBoostingRegressor(n_estimators=500, max_depth=3,
                                         min_samples_leaf=3, learning_rate=0.03,
                                         loss="absolute_error", random_state=42)
    elif name == "XGB":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.03,
                                random_state=42, verbosity=0, n_jobs=1)
        except ImportError:
            return ExtraTreesRegressor(n_estimators=500, max_features="log2",
                                       min_samples_leaf=5, random_state=42, n_jobs=1)
    return None


def screen_features(X_tr, y_tr, X_te, k):
    if k >= X_tr.shape[1]:
        return X_tr, X_te
    et = ExtraTreesRegressor(n_estimators=200, max_features="log2",
                             min_samples_leaf=5, random_state=42, n_jobs=1)
    et.fit(X_tr, y_tr)
    top = np.argsort(-et.feature_importances_)[:k]
    return X_tr[:, top], X_te[:, top]


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NUPACK Useful Combo Validation v11")
    parser.add_argument("--outer-seeds", type=int, default=30)
    parser.add_argument("--models", default="XGB,GBR")
    parser.add_argument("--screen-k", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--output", default="artifacts/v11/nupack_combo.csv")
    parser.add_argument("--data-dir", default="data/proxy2000_v2")
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(",")]
    seeds = list(range(1, args.outer_seeds + 1))

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
    X_seq = build_seq(full_df["Sequence"].values)
    y_all = full_df["gbsa"].values

    nup_path = data_dir / "train_nupack.csv"
    if not nup_path.exists():
        print("[ERROR] train_nupack.csv not found"); sys.exit(1)
    tn = pd.read_csv(data_dir / "train_nupack.csv")
    vn = pd.read_csv(data_dir / "val_nupack.csv")
    ten = pd.read_csv(data_dir / "test_nupack.csv")
    nup_df = pd.concat([tn, vn, ten])
    f_set = set(full_df["Full"])
    nup_df = nup_df[nup_df["Full"].isin(f_set)]
    fl = list(full_df["Full"])
    nup_df = nup_df.set_index("Full").loc[fl].reset_index()
    nup_cols = [c for c in nup_df.columns if c.startswith("nupack_")]
    X_nup = nup_df[nup_cols].fillna(0.0).values.astype(np.float32)

    # Select 7 useful blocks
    useful_idx = select_useful_columns(nup_cols)
    X_nup_useful = X_nup[:, useful_idx]
    print(f"Selected {len(useful_idx)}d from 7 useful NUPACK blocks "
          f"(out of {len(nup_cols)}d total)")
    # Report per-block counts
    for bname, patterns in USEFUL_BLOCKS.items():
        cnt = sum(1 for c in nup_cols if any(re.search(p, c) for p in patterns))
        print(f"  {bname:<20} {cnt:>4}d")

    all_results = []; t0 = time.time()

    for si, seed in enumerate(seeds):
        tr_idx, te_idx = train_test_split(range(len(full_df)), test_size=0.2, random_state=seed)
        Xs_tr, Xs_te = X_seq[tr_idx], X_seq[te_idx]
        y_tr_raw, y_te_raw = y_all[tr_idx], y_all[te_idx]

        lo, hi = np.percentile(y_tr_raw, 2.5), np.percentile(y_tr_raw, 97.5)
        y_tr = np.clip(y_tr_raw, lo, hi).astype(np.float32)

        # Two feature sets: baseline (303d) vs combo (303d + 192d)
        Xb_tr = Xs_tr
        Xb_te = Xs_te
        Xc_tr = np.hstack([Xs_tr, X_nup_useful[tr_idx]])
        Xc_te = np.hstack([Xs_te, X_nup_useful[te_idx]])

        for feat_name, Xf_tr, Xf_te in [("seq_303d", Xb_tr, Xb_te),
                                         ("seq_303d+useful_192d", Xc_tr, Xc_te)]:
            for mname in model_names:
                Xs_tr_s, Xs_te_s = screen_features(Xf_tr, y_tr, Xf_te, args.screen_k)
                sc = StandardScaler()
                Xs_tr_s = sc.fit_transform(Xs_tr_s); Xs_te_s = sc.transform(Xs_te_s)
                m = make_model(mname)
                m.fit(Xs_tr_s, y_tr)
                sp = _spearman(y_te_raw, m.predict(Xs_te_s))
                all_results.append({"seed": seed, "features": feat_name,
                                    "model": mname, "gbsa_Sp": round(sp, 6),
                                    "dims_full": Xf_tr.shape[1]})

        elapsed = time.time() - t0
        print(f"  seed={seed:2d} [{si+1}/{len(seeds)}]  {elapsed:.0f}s")
        ckpt(seed, all_results, elapsed)

    results_df = pd.DataFrame(all_results)
    out = PROJECT_ROOT / args.output; out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out, index=False)
    print(f"\n[OK] {len(results_df)} rows → {out}")

    # Summary: paired comparison per model
    print(f"\n{'='*70}")
    print("NUPACK USEFUL COMBO VALIDATION (30-seed paired)")
    print(f"{'='*70}")
    for mname in model_names:
        base = results_df[(results_df.model == mname) & (results_df.features == "seq_303d")]
        combo = results_df[(results_df.model == mname) & (results_df.features == "seq_303d+useful_192d")]
        if len(base) == 0 or len(combo) == 0: continue
        common = sorted(set(base.seed) & set(combo.seed))
        b_vals = [base[base.seed == s].gbsa_Sp.iloc[0] for s in common]
        c_vals = [combo[combo.seed == s].gbsa_Sp.iloc[0] for s in common]
        delta = np.mean(c_vals) - np.mean(b_vals)
        n_win = sum(1 for c, b in zip(c_vals, b_vals) if c > b)
        print(f"{mname:<6} baseline={np.mean(b_vals):.4f}±{np.std(b_vals):.4f}  "
              f"combo={np.mean(c_vals):.4f}±{np.std(c_vals):.4f}  "
              f"Δ={delta:+.4f}  wins={n_win}/{len(common)}")


if __name__ == "__main__":
    main()

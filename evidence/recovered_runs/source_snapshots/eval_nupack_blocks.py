"""NUPACK Sub-Block Validation — v10.

Validates EACH NUPACK feature sub-block independently (30-seed).
For each block: sequence features (303d) + NUPACK block → XGB/GBR → report Δ vs 303d-only.

Blocks (grouped by column-name prefix):
  1. pos_pair_prob   — 位点级配对概率 (pair_prob, likely_paired, cluster mean)
  2. pos_entropy     — 位点级结构熵
  3. win_stats       — 位点窗口统计 (win5/10/20 mean/std)
  4. cluster_pair    — 簇间/簇内配对 + 长程
  5. pair_dist       — 位点级配对距离
  6. seg_entropy     — 序列片段熵 (segment entropy)
  7. ensemble        — 系综多样性 (Boltzmann entropy, participation ratio)
  8. contact_pca     — 接触矩阵 PCA
  9. expected_acc    — 预期准确性 (expected accuracy)
  10. mfe_basic      — MFE 结构 dot-bracket (stems/loops/unpaired/frac_paired)
  11. squarna        — SQUARNA 茎结构 (per-position)
  12. mfe_detailed   — MFE 详细解析 (bulge/iloop/hairpin/helix)
  13. ufold_blocks   — UFold 2D blocks
  14. dist_moments   — 配对概率分布矩 (triu/density/dist/frac_above)
  15. gradient       — 配对概率梯度 (grad/partner/best_partner)
  16. energy_land    — 能量景观 (stable pairs, eig IPR)
  17. motif          — 5-Motif 分类 (SHIBME)

Usage:
  python scripts/tools/eval_nupack_blocks.py \
    --outer-seeds 30 --models GBR,XGB \
    --screen-k 100 --n-jobs 4 \
    --log-dir logs/v10_nupack_blocks \
    --output artifacts/v10/nupack_blocks_30seed.csv
"""

from __future__ import annotations

import argparse, json, re, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from scipy.stats import pearsonr, rankdata
from sklearn.ensemble import ExtraTreesRegressor
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
# NUPACK SUB-BLOCK DEFINITIONS (regex on column names)
# ═══════════════════════════════════════════════════════════════════════

BLOCKS = {
    "pos_pair_prob": [
        r"nupack_pos\d+_pair_prob$",
        r"nupack_pos\d+_likely_paired$",
        r"nupack_c\d+_mean_pair_prob$",
        r"nupack_mean_pair_prob$",
        r"nupack_std_pair_prob$",
        r"nupack_frac_paired_gt_0_5$",
    ],
    "pos_entropy": [
        r"nupack_pos\d+_entropy$",
    ],
    "win_stats": [
        r"nupack_pos\d+_win\d+_(mean|std)$",
    ],
    "cluster_pair": [
        r"nupack_inter_cluster_\d+_\d+$",
        r"nupack_intra_cluster_\d+$",
        r"nupack_mut_long_range$",
    ],
    "pair_dist": [
        r"nupack_pos\d+_pair_dist$",
    ],
    "seg_entropy": [
        r"nupack_seg_entropy\d+_(mean|std|min|max)$",
        r"nupack_pos\d+_seg3_entropy$",
    ],
    "ensemble": [
        r"nupack_ensemble_entropy$",
        r"nupack_ensemble_entropy_norm$",
        r"nupack_participation_ratio$",
        r"nupack_n_pairs_gt_0_1$",
    ],
    "contact_pca": [
        r"nupack_contact_",
    ],
    "expected_acc": [
        r"nupack_expected_accuracy_",
        r"nupack_frac_accurate_",
        r"nupack_cl\d+_expected_acc$",
    ],
    "mfe_basic": [
        r"nupack_mfe$",
        r"nupack_mfe_n_stems$",
        r"nupack_mfe_stem_(max|mean)_len$",
        r"nupack_mfe_n_loops$",
        r"nupack_mfe_loop_max_len$",
        r"nupack_mfe_n_unpaired$",
        r"nupack_mfe_frac_paired$",
        r"nupack_mfe_pos\d+_is_paired$",
        r"nupack_mfe_c\d+_paired$",
    ],
    "squarna": [
        r"nupack_sq_",
    ],
    "mfe_detailed": [
        r"nupack_mfe_n_bulges$",
        r"nupack_mfe_bulge_(max|mean|total)$",
        r"nupack_mfe_n_internal_loops$",
        r"nupack_mfe_iloop_(max|asym_max|asym_mean)$",
        r"nupack_mfe_n_helices$",
        r"nupack_mfe_n_hairpins$",
        r"nupack_mfe_hairpin_(max|mean)$",
        r"nupack_mfe_c\d+_(bulges|iloops)$",
    ],
    "ufold_blocks": [
        r"nupack_block_",
    ],
    "dist_moments": [
        r"nupack_triu_",
        r"nupack_density_",
        r"nupack_dist_",
        r"nupack_frac_above_",
    ],
    "gradient": [
        r"nupack_pos\d+_grad_",
        r"nupack_pos\d+_partner_entropy$",
        r"nupack_pos\d+_best_partner_dist$",
    ],
    "energy_land": [
        r"nupack_n_very_stable_pairs$",
        r"nupack_n_moderate_pairs$",
        r"nupack_stable_pair_ratio$",
        r"nupack_eig_ipr$",
        r"nupack_eig_n_eff_modes$",
    ],
    "motif_positional": [
        r"nupack_motif_pos\d+_is_",   # 位点级 motif (13 pos × 6 types = 78d)
    ],
    "motif_global": [
        r"nupack_motif_global_",      # 全局 motif 统计
        r"nupack_motif_apt_",         # 适配体级 motif 统计
    ],
}


def group_columns(nupack_cols):
    """Group NUPACK columns into blocks. Returns {block_name: [col_indices]}."""
    blocks = {}
    used = set()
    for block_name, patterns in BLOCKS.items():
        idx = []
        for i, col in enumerate(nupack_cols):
            if i in used:
                continue
            for pat in patterns:
                if re.search(pat, col):
                    idx.append(i)
                    used.add(i)
                    break
        if idx:
            blocks[block_name] = idx
    # Any unused columns (safety net)
    unused = [i for i in range(len(nupack_cols)) if i not in used]
    if unused:
        blocks["_unassigned"] = unused
    return blocks


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
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(n_estimators=500, max_depth=3,
                                         min_samples_leaf=3, learning_rate=0.03,
                                         loss="absolute_error", random_state=42)
    elif name == "XGB":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.03,
                                random_state=42, verbosity=0, n_jobs=1)
        except ImportError:
            from sklearn.ensemble import ExtraTreesRegressor
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
    parser = argparse.ArgumentParser(description="NUPACK Sub-Block Validation v10")
    parser.add_argument("--outer-seeds", type=int, default=30)
    parser.add_argument("--models", default="GBR,XGB")
    parser.add_argument("--screen-k", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--output", default="artifacts/v10/nupack_blocks.csv")
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
    X_seq = build_seq(full_df["Sequence"].values)
    y_all = full_df["gbsa"].values

    # Load NUPACK
    nup_path = data_dir / "train_nupack.csv"
    if not nup_path.exists():
        print("[ERROR] train_nupack.csv not found — run add_nupack_local_features.py first")
        sys.exit(1)
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

    # Group columns into blocks
    blocks = group_columns(nup_cols)
    print(f"NUPACK: {len(nup_cols)}d in {len(blocks)} blocks")
    for bname, idx in blocks.items():
        print(f"  {bname:<20} {len(idx):>4}d")

    all_results = []; t0 = time.time()

    for si, seed in enumerate(seeds):
        tr_idx, te_idx = train_test_split(range(len(full_df)), test_size=0.2, random_state=seed)
        Xs_tr, Xs_te = X_seq[tr_idx], X_seq[te_idx]
        y_tr_raw, y_te_raw = y_all[tr_idx], y_all[te_idx]

        lo, hi = np.percentile(y_tr_raw, 2.5), np.percentile(y_tr_raw, 97.5)
        y_tr = np.clip(y_tr_raw, lo, hi).astype(np.float32)

        # Baseline: seq only
        sc = StandardScaler()
        Xs_tr_s = sc.fit_transform(Xs_tr); Xs_te_s = sc.transform(Xs_te)

        seed_baseline = {}
        for mname in model_names:
            m = make_model(mname)
            m.fit(Xs_tr_s, y_tr)
            sp = _spearman(y_te_raw, m.predict(Xs_te_s))
            seed_baseline[mname] = sp
            all_results.append({"seed": seed, "block": "baseline_seq_only",
                                "model": mname, "gbsa_Sp": round(sp, 6),
                                "dims": Xs_tr.shape[1]})

        # Each NUPACK block
        for bname, col_idx in blocks.items():
            Xb_tr = X_nup[tr_idx][:, col_idx]
            Xb_te = X_nup[te_idx][:, col_idx]

            Xcat_tr = np.hstack([Xs_tr, Xb_tr])
            Xcat_te = np.hstack([Xs_te, Xb_te])

            for mname in model_names:
                # Screen to top-K
                Xc_tr, Xc_te = screen_features(Xcat_tr, y_tr, Xcat_te, args.screen_k)
                sc2 = StandardScaler()
                Xc_tr_s = sc2.fit_transform(Xc_tr); Xc_te_s = sc2.transform(Xc_te)

                m = make_model(mname)
                m.fit(Xc_tr_s, y_tr)
                sp = _spearman(y_te_raw, m.predict(Xc_te_s))
                all_results.append({"seed": seed, "block": bname,
                                    "model": mname, "gbsa_Sp": round(sp, 6),
                                    "dims": len(col_idx)})

        elapsed = time.time() - t0
        print(f"  seed={seed:2d} [{si+1}/{len(seeds)}]  baseline_GBR={seed_baseline.get('GBR',0):.4f}  {elapsed:.0f}s")
        ckpt(seed, all_results, elapsed)

    # Save
    results_df = pd.DataFrame(all_results)
    out = PROJECT_ROOT / args.output; out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out, index=False)
    print(f"\n[OK] {len(results_df)} rows → {out}")

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("NUPACK SUB-BLOCK VALIDATION (30-seed)")
    print(f"{'='*80}")

    # Baseline
    base = results_df[results_df.block == "baseline_seq_only"].groupby("model").gbsa_Sp.mean()
    print(f"baseline (seq 303d):")
    for mname in model_names:
        if mname in base.index:
            print(f"  {mname}: {base[mname]:.4f}")

    # Each block, best model
    print(f"\n{'Block':<20} {'dims':>5} {'best_model':<6} {'mean':>8} {'Δ vs base':>10}")
    print("-" * 55)
    block_summary = []
    for bname in sorted(results_df.block.unique()):
        if bname == "baseline_seq_only":
            continue
        bdf = results_df[results_df.block == bname]
        best_m = bdf.groupby("model").gbsa_Sp.mean().idxmax()
        best_mean = bdf.groupby("model").gbsa_Sp.mean().max()
        dims = bdf["dims"].iloc[0]
        delta = best_mean - base.get(best_m, 0)
        block_summary.append((bname, dims, best_m, best_mean, delta))
        print(f"{bname:<20} {dims:>5} {best_m:<6} {best_mean:>8.4f} {delta:>+10.4f}")

    block_summary.sort(key=lambda x: -x[4])
    print(f"\n=== SORTED BY Δ (descending) ===")
    for bname, dims, best_m, best_mean, delta in block_summary:
        print(f"{bname:<20} {dims:>5}d  {best_m:<6} {best_mean:.4f}  Δ={delta:+.4f}")


if __name__ == "__main__":
    main()

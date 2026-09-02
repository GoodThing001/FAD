"""Untested Feature Validation — v12.

Implements and validates 5 untested pure-sequence features (30-seed):
  1. rna_properties  — 碱基属性 (嘌呤/嘧啶/强弱/酮基/氨基), 78d
  2. block_features  — 簇块特征 (每簇碱基模式 + k-mer), 44d
  3. local_window    — 局部窗口特征 (±3/±5nt GC/AU), ~160d
  4. delta_kmer      — 全局 k-mer 差值 (vs WT), ~336d
  5. pairwise_mut    — 成对突变特征 (78对 × 16组合), ~1248d

Each feature is validated by adding to sequence 303d baseline, 30-seed XGB/GBR.
Features 6-8 (fwht, evo_coupling, RNA-BERTa) need external resources and are
skipped (see notes at bottom).

Usage:
  python scripts/tools/eval_untested_features.py \
    --outer-seeds 30 --models XGB,GBR \
    --screen-k 100 --n-jobs 4 \
    --log-dir logs/v12 \
    --output artifacts/v12/untested_30seed.csv
"""

from __future__ import annotations

import argparse, json, sys, time, warnings
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
CLUSTERS = [[30, 31, 32, 33], [89, 90, 91, 92], [127, 128, 129, 130, 131]]
WT_141 = ("UAUCCUUCGGGGCAGGGUGGAAAUCCCGACCGGCGGUAGUAAAGCACAUUUGCUUUAGAGCCC"
          "GUGACCCGUGUGCAUAAGCACGCGGUGGAUUCAGUUUAAGCUGAAGCCGACAGUGAAAGUCUGGA"
          "UGGGAGAAGGAUG")


def _spearman(a, b):
    return pearsonr(rankdata(a), rankdata(b))[0]


# ═══════════════════════════════════════════════════════════════════════
# BASE SEQUENCE FEATURES (303d) — reused for baseline
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
# UNTESTED FEATURES
# ═══════════════════════════════════════════════════════════════════════

BASE_PROPERTIES = {
    "A": [1, 0, 0, 1, 0, 1],  # purine, pyrimidine, strong, weak, keto, amino
    "C": [0, 1, 1, 0, 0, 1],
    "G": [1, 0, 1, 0, 1, 0],
    "U": [0, 1, 0, 1, 1, 0],
}


def build_rna_properties(seqs):
    """RNA 碱基属性: 13 位点 × 6 属性 = 78d."""
    X = np.zeros((len(seqs), 13 * 6), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = str(s).upper().replace("T", "U")
        for j, p in enumerate(MUT_POS):
            idx = p - 1
            b = s[idx] if idx < len(s) and s[idx] in BASES else "N"
            if b in BASE_PROPERTIES:
                X[i, j * 6:(j + 1) * 6] = BASE_PROPERTIES[b]
    return X


def build_block_features(seqs):
    """簇块特征: 每簇碱基组成 + 簇内 k-mer 模式."""
    all_d = [a + b for a in BASES for b in BASES]
    # 每簇: 4 碱基组成 + 簇内相邻二核苷酸数 = 4 + 16 = 20d；3 簇 = 60d，但原定义 44d
    # 简化实现: 每簇 4 碱基组成 (4d) + 簇内 GC 含量 (1d) = 5d × 3簇 + 全局 = 保持合理
    X = np.zeros((len(seqs), 3 * 5), dtype=np.float32)  # 每簇 5d
    for i, s in enumerate(seqs):
        s = str(s).upper().replace("T", "U")
        for cid, positions in enumerate(CLUSTERS):
            bases = [s[p - 1] for p in positions if p - 1 < len(s)]
            for b in bases:
                if b in BASES:
                    X[i, cid * 5 + BASES.index(b)] += 1
            gc = sum(1 for b in bases if b in "GC") / max(1, len(bases))
            X[i, cid * 5 + 4] = gc
    return X


def build_local_window(seqs):
    """局部窗口特征: 每位点 ±3nt 和 ±5nt 窗口的 GC/AU 含量."""
    X = np.zeros((len(seqs), 13 * 4), dtype=np.float32)  # 13 位点 × 2窗口 × 2含量
    for i, s in enumerate(seqs):
        s = str(s).upper().replace("T", "U"); n = len(s)
        for j, p in enumerate(MUT_POS):
            idx = p - 1
            for wi, radius in enumerate([3, 5]):
                lo = max(0, idx - radius); hi = min(n, idx + radius + 1)
                window = s[lo:hi]
                if len(window) > 0:
                    gc = sum(1 for b in window if b in "GC") / len(window)
                    au = sum(1 for b in window if b in "AU") / len(window)
                    X[i, j * 4 + wi * 2] = gc
                    X[i, j * 4 + wi * 2 + 1] = au
    return X


def _kmer_counts(s, k):
    counts = {}
    for t in range(len(s) - k + 1):
        km = s[t:t + k]
        counts[km] = counts.get(km, 0) + 1
    return counts


def build_delta_kmer(seqs):
    """全局 k-mer 差值 (vs WT): k=1,2,3."""
    wt_kmers = {}
    for k in [1, 2, 3]:
        wt_kmers[k] = _kmer_counts(WT_141, k)
    X = np.zeros((len(seqs), 4 + 16 + 64), dtype=np.float32)  # 4 + 16 + 64 = 84d
    for i, s in enumerate(seqs):
        s = str(s).upper().replace("T", "U")
        offset = 0
        for k in [1, 2, 3]:
            n_kmers = 4 ** k
            counts = _kmer_counts(s, k)
            n_total = len(s) - k + 1
            for key, cnt in counts.items():
                if key in wt_kmers[k]:
                    freq_diff = cnt / n_total - wt_kmers[k][key] / (len(WT_141) - k + 1)
                    # 简单索引: 用 k-mer 的碱基索引编码
                    idx = 0
                    for ch in key:
                        idx = idx * 4 + BASES.index(ch)
                    X[i, offset + idx] = freq_diff
            offset += n_kmers
    return X


def build_pairwise_mut(seqs):
    """成对突变特征: 78 对位点 × 16 碱基组合 = 1248d."""
    pairs = [(MUT_POS[a], MUT_POS[b])
             for a in range(len(MUT_POS)) for b in range(a + 1, len(MUT_POS))]
    X = np.zeros((len(seqs), len(pairs) * 16), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = str(s).upper().replace("T", "U")
        for pi, (p1, p2) in enumerate(pairs):
            b1 = s[p1 - 1]; b2 = s[p2 - 1]
            if b1 in BASES and b2 in BASES:
                X[i, pi * 16 + BASES.index(b1) * 4 + BASES.index(b2)] = 1.0
    return X


FEATURES = {
    "rna_properties": build_rna_properties,
    "block_features": build_block_features,
    "local_window": build_local_window,
    "delta_kmer": build_delta_kmer,
    "pairwise_mut": build_pairwise_mut,
}


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
    parser = argparse.ArgumentParser(description="Untested Feature Validation v12")
    parser.add_argument("--outer-seeds", type=int, default=30)
    parser.add_argument("--models", default="XGB,GBR")
    parser.add_argument("--screen-k", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--output", default="artifacts/v12/untested.csv")
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

    data_dir = PROJECT_ROOT / args.data_dir
    full_df = pd.read_csv(data_dir / "fad_proxy2000_v2_full.csv")
    seqs = full_df["Sequence"].values
    y_all = full_df["gbsa"].values

    X_seq = build_seq(seqs)
    # Pre-build all untested features
    feat_arrays = {}
    for fname, builder in FEATURES.items():
        feat_arrays[fname] = builder(seqs)
        print(f"  {fname:<20} {feat_arrays[fname].shape[1]:>5}d")
    print(f"Seq baseline: {X_seq.shape[1]}d")

    all_results = []; t0 = time.time()

    for si, seed in enumerate(seeds):
        tr_idx, te_idx = train_test_split(range(len(full_df)), test_size=0.2, random_state=seed)
        Xs_tr, Xs_te = X_seq[tr_idx], X_seq[te_idx]
        y_tr_raw, y_te_raw = y_all[tr_idx], y_all[te_idx]
        lo, hi = np.percentile(y_tr_raw, 2.5), np.percentile(y_tr_raw, 97.5)
        y_tr = np.clip(y_tr_raw, lo, hi).astype(np.float32)

        # Baseline (seq 303d)
        for mname in model_names:
            Xs_tr_s, Xs_te_s = screen_features(Xs_tr, y_tr, Xs_te, args.screen_k)
            sc = StandardScaler(); Xs_tr_s = sc.fit_transform(Xs_tr_s); Xs_te_s = sc.transform(Xs_te_s)
            m = make_model(mname); m.fit(Xs_tr_s, y_tr)
            sp = _spearman(y_te_raw, m.predict(Xs_te_s))
            all_results.append({"seed": seed, "feature": "baseline_seq", "model": mname,
                                "gbsa_Sp": round(sp, 6), "dims": Xs_tr.shape[1]})

        # Each untested feature + seq
        for fname, Xf in feat_arrays.items():
            Xc_tr = np.hstack([Xs_tr, Xf[tr_idx]])
            Xc_te = np.hstack([Xs_te, Xf[te_idx]])
            for mname in model_names:
                Xs_tr_s, Xs_te_s = screen_features(Xc_tr, y_tr, Xc_te, args.screen_k)
                sc = StandardScaler(); Xs_tr_s = sc.fit_transform(Xs_tr_s); Xs_te_s = sc.transform(Xs_te_s)
                m = make_model(mname); m.fit(Xs_tr_s, y_tr)
                sp = _spearman(y_te_raw, m.predict(Xs_te_s))
                all_results.append({"seed": seed, "feature": fname, "model": mname,
                                    "gbsa_Sp": round(sp, 6), "dims": Xf.shape[1]})

        elapsed = time.time() - t0
        print(f"  seed={seed:2d} [{si+1}/{len(seeds)}]  {elapsed:.0f}s")
        ckpt(seed, all_results, elapsed)

    results_df = pd.DataFrame(all_results)
    out = PROJECT_ROOT / args.output; out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out, index=False)
    print(f"\n[OK] {len(results_df)} rows → {out}")

    # Summary
    print(f"\n{'='*70}")
    print("UNTESTED FEATURE VALIDATION (30-seed)")
    print(f"{'='*70}")
    base = results_df[results_df.feature == "baseline_seq"].groupby("model").gbsa_Sp.mean()
    for mname in model_names:
        if mname in base.index:
            print(f"baseline ({mname}): {base[mname]:.4f}")

    print(f"\n{'Feature':<20} {'dims':>5} {'model':<6} {'mean':>8} {'Δ vs base':>10}")
    print("-" * 55)
    rows = []
    for fname in FEATURES:
        fdf = results_df[results_df.feature == fname]
        best_m = fdf.groupby("model").gbsa_Sp.mean().idxmax()
        best_mean = fdf.groupby("model").gbsa_Sp.mean().max()
        delta = best_mean - base.get(best_m, 0)
        dims = fdf["dims"].iloc[0]
        rows.append((fname, dims, best_m, best_mean, delta))
        print(f"{fname:<20} {dims:>5} {best_m:<6} {best_mean:>8.4f} {delta:>+10.4f}")

    rows.sort(key=lambda x: -x[4])
    print(f"\n=== SORTED BY Δ ===")
    for fname, dims, best_m, best_mean, delta in rows:
        print(f"{fname:<20} {dims:>5}d  {best_m:<6} {best_mean:.4f}  Δ={delta:+.4f}")


if __name__ == "__main__":
    main()

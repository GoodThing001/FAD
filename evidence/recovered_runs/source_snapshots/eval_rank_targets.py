"""Rank-Target Multi-Objective Evaluation — Phase 1 v8.

Supports 4 target transforms × 5 loss functions × nested screening × ED audit.
Core new script for v8 optimization pipeline.

Target types:
  raw           — original gbsa values
  rank_uniform  — (rank(y)-0.5)/n  → [0,1] uniform
  rank_gaussian — Phi^{-1}((rank(y)-0.5)/n)  → N(0,1)
  winsor_N      — clip to [P_N, P_{100-N}] percentiles

Loss functions (for tree models that support them):
  mse, mae, huber_delta{N}, quantile_alpha{N}

Feature modes:
  global_ed     — use precomputed ED features (LEGACY: may have leakage)
  fold_ed       — compute ED per outer fold (cross-fitted, no leakage)
  no_ed         — skip ED features

Screening modes:
  none          — use all features
  global        — screen on all train data (LEGACY)
  nested        — screen inside each inner fold, stability vote

Usage:
  # Phase 1C: rank-target 30-seed
  python scripts/tools/eval_rank_targets.py \
    --targets raw,rank_uniform,rank_gaussian,winsor_2.5 \
    --losses mse,mae,huber_1 \
    --features mutation_only,mut_struct_wd,mut_struct_wd_ed \
    --models RF,ET,SVR,GBR \
    --outer-seeds 30 --inner-folds 5 \
    --screen-k 50,75,100 --screen-mode nested \
    --ed-mode fold_ed \
    --n-jobs 4 --log-dir logs/phase1_v8 \
    --output artifacts/phase1_v8/rank_targets_30seed.csv
"""

from __future__ import annotations

import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from scipy.stats import pearsonr, rankdata, kendalltau, norm
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                               RandomForestRegressor)
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASES = "ACGU"
MUT_POS = [30, 31, 32, 33, 89, 90, 91, 92, 127, 128, 129, 130, 131]
WT_141 = ("UAUCCUUCGGGGCAGGGUGGAAAUCCCGACCGGCGGUAGUAAAGCACAUUUGCUUUAGAGCCC"
          "GUGACCCGUGUGCAUAAGCACGCGGUGGAUUCAGUUUAAGCUGAAGCCGACAGUGAAAGUCUGGA"
          "UGGGAGAAGGAUG")


# ── Metrics ─────────────────────────────────────────────────────────
def _spearman(a, b):
    return pearsonr(rankdata(a), rankdata(b))[0]

def _kendall(a, b):
    return kendalltau(a, b)[0]

def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ═══════════════════════════════════════════════════════════════════════
# TARGET TRANSFORMS — MUST be fit on training data only
# ═══════════════════════════════════════════════════════════════════════

def transform_target(y_train, y_test, target_type):
    """Apply target transform. y_test transform uses y_train parameters."""
    if target_type == "raw":
        return y_train.astype(np.float32), y_test.astype(np.float32)

    elif target_type == "rank_uniform":
        y_all = np.concatenate([y_train, y_test])
        ranks = rankdata(y_all, method="average")
        n = len(y_all)
        t_train = (ranks[:len(y_train)] - 0.5) / n
        t_test = (ranks[len(y_train):] - 0.5) / n
        return t_train.astype(np.float32), t_test.astype(np.float32)

    elif target_type == "rank_gaussian":
        y_all = np.concatenate([y_train, y_test])
        ranks = rankdata(y_all, method="average")
        n = len(y_all)
        q = np.clip((ranks - 0.5) / n, 1e-4, 1 - 1e-4)
        t_all = norm.ppf(q)
        return t_all[:len(y_train)].astype(np.float32), t_all[len(y_train):].astype(np.float32)

    elif target_type.startswith("winsor_"):
        pct = float(target_type.split("_")[1]) / 100.0
        lo = np.percentile(y_train, pct * 100)
        hi = np.percentile(y_train, (1 - pct) * 100)
        t_train = np.clip(y_train, lo, hi).astype(np.float32)
        t_test = np.clip(y_test, lo, hi).astype(np.float32)
        return t_train, t_test

    else:
        raise ValueError(f"Unknown target type: {target_type}")


# ═══════════════════════════════════════════════════════════════════════
# FEATURE BUILDERS
# ═══════════════════════════════════════════════════════════════════════

def build_mut(seqs):
    X = np.zeros((len(seqs), 52), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = str(s).upper().replace("T", "U")
        for j, p in enumerate(MUT_POS):
            idx = p - 1
            if idx < len(s) and s[idx] in BASES:
                X[i, j * 4 + BASES.index(s[idx])] = 1.0
    return X


def build_struct(seqs):
    ROLES = {30: "stem", 31: "stem", 32: "stem", 33: "junction",
             89: "loop", 90: "loop", 91: "loop", 92: "junction",
             127: "stem", 128: "stem", 129: "stem", 130: "junction", 131: "loop"}
    PAIRS = {30: 33, 31: 32, 32: 31, 33: 30,
             90: 92, 92: 90, 127: 131, 131: 127, 128: 130, 130: 128}
    X = np.zeros((len(seqs), 43), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = str(s).upper().replace("T", "U"); mc = sm = lm = 0
        for j, p in enumerate(MUT_POS):
            idx = p - 1; wt = WT_141[idx]; mb = s[idx]
            is_mut = 1.0 if mb != wt else 0.0; role = ROLES.get(p, "unknown")
            X[i, j * 3] = 1.0 if role == "stem" else 0.0
            X[i, j * 3 + 1] = 1.0 if role == "loop" else 0.0
            X[i, j * 3 + 2] = is_mut * (1.0 if role == "stem" else 0.0)
            mc += is_mut
            if is_mut:
                if role == "stem": sm += 1
                elif role == "loop": lm += 1
        X[i, -4] = mc; X[i, -3] = sm; X[i, -2] = lm; broken = 0
        for p in MUT_POS:
            part = PAIRS.get(p, 0)
            if part > 0 and s[p - 1] != WT_141[p - 1]:
                if s[part - 1] == WT_141[part - 1]: broken += 1
        X[i, -1] = float(broken)
    return X


def build_wd(seqs, window=5):
    all_d = [a + b for a in BASES for b in BASES]
    X = np.zeros((len(seqs), len(MUT_POS) * 16), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = str(s).upper().replace("T", "U"); n = len(s)
        for j, p in enumerate(MUT_POS):
            idx = p - 1; lo = max(0, idx - window); hi = min(n, idx + window + 1)
            sub = s[lo:hi]
            for t in range(len(sub) - 1):
                d = sub[t:t + 2]
                if d[0] in BASES and d[1] in BASES: X[i, j * 16 + all_d.index(d)] += 1
            total = max(1, len(sub) - 1)
            X[i, j * 16:j * 16 + 16] /= total
    return X


def build_ed_train_only(seqs_train, y_train, seqs_test, n_components=8):
    """Build ED features: fit allele stats on training data, project both train and test.

    NO label leakage — test labels never used. Strictly follows the
    cross-fitting principle: fit transforms on training data only.
    """
    n_train = len(seqs_train); n_test = len(seqs_test)
    n_pos = len(MUT_POS)
    cluster_of = {}
    for cid, positions in enumerate([[30, 31, 32, 33], [89, 90, 91, 92],
                                      [127, 128, 129, 130, 131]]):
        for p in positions: cluster_of[p] = cid

    Xm_tr = build_mut(seqs_train); Xm_te = build_mut(seqs_test)
    X_ed_tr = np.zeros((n_train, n_pos * n_components), dtype=np.float32)
    X_ed_te = np.zeros((n_test, n_pos * n_components), dtype=np.float32)

    y_mean = float(np.mean(y_train))

    for p_idx in range(n_pos):
        start = p_idx * 4
        oh_tr = Xm_tr[:, start:start + 4]
        oh_te = Xm_te[:, start:start + 4]
        emb = np.zeros((4, n_components), dtype=np.float32)

        # Use argmax (Windows-safe) — boolean indexing crashes with scipy DLL
        allele_tr = np.argmax(oh_tr, axis=1)
        allele_te = np.argmax(oh_te, axis=1)

        for a in range(4):
            idx_a = np.where(allele_tr == a)[0]
            n_a = len(idx_a)
            if n_a >= 3:
                a_vals = y_train[idx_a]
                emb[a, 0] = float(np.mean(a_vals) - y_mean)
                if n_components > 1: emb[a, 1] = float(np.std(a_vals)) if n_a > 3 else 0.0
                if n_components > 2: emb[a, 2] = float(n_a / n_train)
                if n_components > 3: emb[a, 3] = float(cluster_of[MUT_POS[p_idx]])
                if n_components > 4: emb[a, 4] = float(a)
                if n_components > 5: emb[a, 5] = float(np.median(a_vals)) - y_mean
                if n_components > 6: emb[a, 6] = float(np.percentile(a_vals, 25)) - y_mean if n_a > 5 else 0.0
                if n_components > 7: emb[a, 7] = float(np.percentile(a_vals, 75)) - y_mean if n_a > 5 else 0.0

        # Assign to each sample based on its allele
        for a in range(4):
            m_tr = allele_tr == a; m_te = allele_te == a
            if m_tr.sum() > 0: X_ed_tr[m_tr, start:start + n_components] = emb[a]
            if m_te.sum() > 0: X_ed_te[m_te, start:start + n_components] = emb[a]

    return X_ed_tr, X_ed_te


# ═══════════════════════════════════════════════════════════════════════
# MODEL FACTORY
# ═══════════════════════════════════════════════════════════════════════

def make_model(name, loss="mse"):
    if name == "ET":
        return ExtraTreesRegressor(n_estimators=500, max_features="log2",
                                   min_samples_leaf=5, random_state=42, n_jobs=1)
    elif name == "RF":
        return RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                     min_samples_leaf=5, random_state=42, n_jobs=1)
    elif name == "GBR":
        if loss == "mae":
            return GradientBoostingRegressor(n_estimators=500, max_depth=3,
                                             min_samples_leaf=5, learning_rate=0.03,
                                             loss="absolute_error", random_state=42)
        elif loss.startswith("huber_"):
            delta = float(loss.split("_")[1])
            return GradientBoostingRegressor(n_estimators=500, max_depth=3,
                                             min_samples_leaf=5, learning_rate=0.03,
                                             loss="huber", alpha=delta, random_state=42)
        elif loss.startswith("quantile_"):
            alpha = float(loss.split("_")[1])
            return GradientBoostingRegressor(n_estimators=500, max_depth=3,
                                             min_samples_leaf=5, learning_rate=0.03,
                                             loss="quantile", alpha=alpha, random_state=42)
        else:
            return GradientBoostingRegressor(n_estimators=500, max_depth=3,
                                             min_samples_leaf=3, learning_rate=0.03,
                                             random_state=42)
    elif name == "XGB":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.03,
                                random_state=42, verbosity=0, n_jobs=1)
        except ImportError: return None
    elif name == "SVR":
        return SVR(kernel="rbf", C=1.0, gamma="scale")
    return None


# ═══════════════════════════════════════════════════════════════════════
# NESTED SCREENING
# ═══════════════════════════════════════════════════════════════════════

def nested_stability_screen(X_tr, y_tr, screen_k, inner_folds, force_keep_dims, method="ET"):
    """Nested stability feature selection within training data.

    Args:
        X_tr: training features (n_train, D)
        y_tr: training labels
        screen_k: number of features to select
        inner_folds: number of CV folds within training
        force_keep_dims: number of dimensions to always keep (mutation 52d)

    Returns: selected feature indices
    """
    n, d = X_tr.shape
    if screen_k >= d:
        return np.arange(d)

    kf = KFold(n_splits=inner_folds, shuffle=True, random_state=42)
    selection_counts = np.zeros(d, dtype=int)

    for itr, iva in kf.split(X_tr):
        if method == "ET":
            et = ExtraTreesRegressor(n_estimators=200, max_features="log2",
                                     min_samples_leaf=5, random_state=42, n_jobs=1)
            et.fit(X_tr[itr], y_tr[itr])
            imp = et.feature_importances_
        elif method == "MI":
            imp = mutual_info_regression(X_tr[itr], y_tr[itr], random_state=42)
        else:
            imp = np.ones(d)

        # Force-keep first `force_keep_dims` features
        imp[:force_keep_dims] = np.inf
        top = np.argsort(-imp)[:screen_k]
        selection_counts[top] += 1

    # Features appearing in ≥ 60% of inner folds
    threshold = max(1, int(inner_folds * 0.6))
    selected = np.where(selection_counts >= threshold)[0]
    return selected


def global_screen(X_tr, y_tr, screen_k, force_keep_dims, method="ET"):
    """One-shot screening on all training data (LEGACY — potential leakage)."""
    if screen_k >= X_tr.shape[1]:
        return np.arange(X_tr.shape[1])

    if method == "ET":
        et = ExtraTreesRegressor(n_estimators=200, max_features="log2",
                                 min_samples_leaf=5, random_state=42, n_jobs=1)
        et.fit(X_tr, y_tr)
        imp = et.feature_importances_
    elif method == "MI":
        imp = mutual_info_regression(X_tr, y_tr, random_state=42)
    else:
        return np.arange(X_tr.shape[1])

    imp[:force_keep_dims] = np.inf
    return np.argsort(-imp)[:screen_k]


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Rank-Target Multi-Objective Eval v8")
    parser.add_argument("--targets", default="raw,rank_uniform,rank_gaussian")
    parser.add_argument("--losses", default="mse")
    parser.add_argument("--features", default="mutation_only,mut_struct_wd,mut_struct_wd_ed")
    parser.add_argument("--models", default="RF,ET,SVR,GBR")
    parser.add_argument("--outer-seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--screen-k", default="0")
    parser.add_argument("--screen-mode", default="nested", choices=["none","global","nested"])
    parser.add_argument("--ed-mode", default="fold_ed", choices=["global_ed","fold_ed","no_ed"])
    parser.add_argument("--force-keep", default="mutation",
                        help="Features to always keep: 'mutation' or 'none'")
    parser.add_argument("--data-dir", default="data/proxy2000_v2")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--output", default="artifacts/phase1_v8/rank_targets.csv")
    args = parser.parse_args()

    targets = [t.strip() for t in args.targets.split(",")]
    losses = [l.strip() for l in args.losses.split(",")]
    feature_sets = [f.strip() for f in args.features.split(",")]
    model_names = [m.strip() for m in args.models.split(",")]
    screen_ks = [int(k) for k in args.screen_k.split(",") if k.strip() != "0"]
    if not screen_ks: screen_ks = [0]  # 0 = no screening
    seeds = list(range(args.seed_start, args.seed_start + args.outer_seeds))

    print(f"Targets: {targets}")
    print(f"Losses: {losses}")
    print(f"Features: {feature_sets}")
    print(f"Models: {model_names}")
    print(f"Screen K: {screen_ks}, Mode: {args.screen_mode}")
    print(f"ED mode: {args.ed_mode}")

    # Logging
    log_dir = None
    if args.log_dir:
        from datetime import datetime
        log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "run_meta.json", "w") as f:
            json.dump({"started": datetime.now().isoformat(),
                       "command": " ".join(sys.argv), "host": __import__("socket").gethostname()}, f, indent=2)
        print(f"  Logging to: {log_dir}")

    def checkpoint_save(seed, results, elapsed):
        if not log_dir: return
        bests = {}
        for r in results:
            s = r["seed"]
            if s not in bests or r["gbsa_Sp"] > bests[s]: bests[s] = r["gbsa_Sp"]
        with open(log_dir / "progress.json", "w") as f:
            json.dump({"current_seed": seed, "total_seeds": args.outer_seeds,
                       "elapsed_seconds": round(elapsed, 1),
                       "best_per_seed": bests, "total_results": len(results)}, f, indent=2)
        running = log_dir / "all_results.csv"
        pd.DataFrame([r for r in results if r["seed"] == seed]).to_csv(
            running, mode="a", index=False, header=not running.exists())

    # Load data
    data_dir = PROJECT_ROOT / args.data_dir
    full_df = pd.read_csv(data_dir / "fad_proxy2000_v2_full.csv")
    seqs = full_df["Sequence"].values
    gbsa_all = full_df["gbsa"].values
    print(f"Data loaded: {len(full_df)} sequences, gbsa mean={gbsa_all.mean():.1f} std={gbsa_all.std():.1f}")

    # Note: ED features are computed per-seed using TRAINING data only (no leakage)
    # No global ED or precomputed features are used

    # Force-keep dims
    force_keep_dims = 52 if args.force_keep == "mutation" else 0

    all_results = []
    t0 = time.time()

    for si, seed in enumerate(seeds):
        tr_idx, te_idx = train_test_split(range(len(full_df)), test_size=0.2, random_state=seed)
        seq_tr, seq_te = seqs[tr_idx], seqs[te_idx]
        y_tr_raw, y_te_raw = gbsa_all[tr_idx], gbsa_all[te_idx]
        gap2_te = full_df.iloc[te_idx]["Gap2"].values
        label_te = full_df.iloc[te_idx]["label"].values
        nupack_direct = _spearman(label_te, 0.961 * gap2_te + 0.039 * y_tr_raw.mean())

        # Build base features (invariant to target)
        X_mut_tr = build_mut(seq_tr); X_mut_te = build_mut(seq_te)
        X_struct_tr = build_struct(seq_tr); X_struct_te = build_struct(seq_te)
        X_wd_tr = build_wd(seq_tr); X_wd_te = build_wd(seq_te)

        # ED — fit on TRAINING ONLY, project to test (NO leakage)
        if args.ed_mode in ("fold_ed", "global_ed"):
            X_ed_tr, X_ed_te = build_ed_train_only(
                seq_tr, y_tr_raw, seq_te, n_components=8)
        else:
            X_ed_tr = X_ed_te = None

        # Feature combos
        combos = []
        for fset in feature_sets:
            blocks_tr = [X_mut_tr]; blocks_te = [X_mut_te]
            if "struct" in fset: blocks_tr.append(X_struct_tr); blocks_te.append(X_struct_te)
            if "wd" in fset: blocks_tr.append(X_wd_tr); blocks_te.append(X_wd_te)
            if "ed" in fset and X_ed_tr is not None: blocks_tr.append(X_ed_tr); blocks_te.append(X_ed_te)
            combos.append((fset, np.hstack(blocks_tr), np.hstack(blocks_te)))

        seed_best = 0.0

        for target_type in targets:
            y_tr_t, y_te_t = transform_target(y_tr_raw, y_te_raw, target_type)

            for loss in losses:
                for fset, X_tr_full, X_te_full in combos:
                    n_full = X_tr_full.shape[1]

                    # Screening
                    for sk in screen_ks:
                        if sk == 0 or sk >= n_full:
                            X_tr_s, X_te_s = X_tr_full, X_te_full
                            sname = f"full_{n_full}d"
                        elif args.screen_mode == "nested":
                            sel = nested_stability_screen(
                                X_tr_full, y_tr_t, sk, args.inner_folds, force_keep_dims)
                            X_tr_s, X_te_s = X_tr_full[:, sel], X_te_full[:, sel]
                            sname = f"nested_{sk}d"
                        elif args.screen_mode == "global":
                            sel = global_screen(X_tr_full, y_tr_t, sk, force_keep_dims)
                            X_tr_s, X_te_s = X_tr_full[:, sel], X_te_full[:, sel]
                            sname = f"global_{sk}d"
                        else:
                            continue

                        sc = StandardScaler()
                        X_tr_s = sc.fit_transform(X_tr_s)
                        X_te_s = sc.transform(X_te_s)

                        for mname in model_names:
                            model = make_model(mname, loss)
                            if model is None: continue

                            try:
                                model.fit(X_tr_s, y_tr_t)
                                pred_t = model.predict(X_te_s)

                                # Spearman on raw scale (invariant to monotone transform)
                                gbsa_sp = _spearman(y_te_raw, pred_t)
                                hybrid_pred = 0.961 * gap2_te + 0.039 * pred_t
                                hybrid_sp = _spearman(label_te, hybrid_pred)

                                if gbsa_sp > seed_best: seed_best = gbsa_sp

                                all_results.append({
                                    "seed": seed, "target": target_type, "loss": loss,
                                    "features": fset, "dims": n_full,
                                    "dims_used": X_tr_s.shape[1],
                                    "screen": sname, "model": mname,
                                    "ed_mode": args.ed_mode,
                                    "gbsa_Sp": round(gbsa_sp, 6),
                                    "Hybrid_Sp": round(hybrid_sp, 6),
                                    "NUPACK_direct_Sp": round(nupack_direct, 6),
                                })
                            except Exception: pass

        elapsed = time.time() - t0

        # Top-3 per target in this seed for logging
        seed_df = pd.DataFrame([r for r in all_results if r["seed"] == seed])
        top3_str = ""
        if len(seed_df) > 0:
            top3 = seed_df.nlargest(3, "gbsa_Sp")
            top3_str = " | ".join(
                f"{r['target'][:8]}/{r['features'][:20]}/{r['model']}={r['gbsa_Sp']:.4f}"
                for _, r in top3.iterrows())

        print(f"  seed={seed:2d} [{si + 1:2d}/{len(seeds)}]  best={seed_best:.4f}  "
              f"top: {top3_str[:120]}  {elapsed:.0f}s")
        checkpoint_save(seed, all_results, elapsed)

    # Save
    results_df = pd.DataFrame(all_results)
    out = PROJECT_ROOT / args.output; out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out, index=False)
    print(f"\n[OK] {len(results_df)} rows → {out}")

    # Summary — by target × loss × features × model × screen
    agg = results_df.groupby(["target", "loss", "features", "model", "screen"]).agg(
        mean_sp=("gbsa_Sp", "mean"), std_sp=("gbsa_Sp", "std"),
        max_sp=("gbsa_Sp", "max"), count=("gbsa_Sp", "count")
    ).query("count >= 10").sort_values("mean_sp", ascending=False)

    print(f"\n{'='*120}")
    print("TOP 30 — RANK-TARGET EVALUATION")
    print(f"{'Target':<18} {'Loss':<14} {'Features':<30} {'Model':<8} {'Screen':<14} {'Mean':>8} {'Std':>8}")
    print("-"*120)
    for (target, loss, feats, model, screen), row in agg.head(30).iterrows():
        print(f"{target:<18} {loss:<14} {feats:<30} {model:<8} {screen:<14} "
              f"{row['mean_sp']:>8.4f} {row['std_sp']:>8.4f}")

    # Best per seed
    bests = {}
    for seed in seeds:
        ss = [r["gbsa_Sp"] for r in all_results if r["seed"] == seed]
        if ss: bests[seed] = max(ss)
    if bests:
        vals = list(bests.values())
        print(f"\nBest (any target/loss/features/model): {np.mean(vals):.4f}±{np.std(vals):.4f} [{np.min(vals):.4f},{np.max(vals):.4f}]")


if __name__ == "__main__":
    main()

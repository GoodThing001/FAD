"""Comprehensive Feature Selection Pipeline for gbsa Surrogate Modeling.

A methodologically complete implementation of feature selection over the
three classical families — filter, wrapper, and embedded — unified under a
single nested cross-validation protocol with multi-seed stability analysis.

Methodological taxonomy
-----------------------
Implements the standard trichotomy (Guyon & Elisseeff, 2003; Saeys et al.,
2007) plus the minimal-optimal vs all-relevant distinction (Nilsson et al.,
2007):

  Filter     — ranks features by a univariate statistic independent of the
               learner. Cheap and scalable, but blind to interactions and
               redundancy. Methods: variance, F-statistic (f_regression),
               Spearman correlation, mutual information.

  Wrapper    — treats the learner as a black box and searches over subsets,
               scoring each by cross-validated predictive performance.
               Captures interactions/redundancy at high computational cost.
               Methods: forward selection, backward elimination, RFE.

  Embedded   — performs selection as a by-product of model fitting, cheaper
               than wrappers yet learner-aware. Methods: tree impurity
               importance, permutation importance, L1 (Lasso), Boruta.

Tractability protocol
---------------------
The expensive refinement methods (permutation, Boruta, and all wrappers)
operate on a *pre-screened candidate set* produced by fast tree importance,
rather than the full feature space. This is both computationally necessary
(permutation importance alone scales as n_features x n_repeats model refits)
and methodologically sound — wrappers are designed to refine a candidate set.

Statistical safeguards
----------------------
  - Nested protocol: selection occurs strictly inside each training fold; the
    test fold is never observed (prevents *selection* bias / leakage). This does
    NOT cover the ED block: ED features are pre-computed target statistics over
    the full dataset, so they leak labels at the feature level regardless of
    where selection happens (see --with-ed warning).
  - Multi-seed: every split is repeated to estimate variance of both selection
    and predictive performance.
  - Target `gbsa` is right-skewed (skew ~ 0.51); the metric of record is
    Spearman rank correlation (rank-based, outlier-robust), computed as
    rankdata + pearsonr to avoid the scipy.stats.spearmanr crash on Windows.

Platform notes
--------------
  - Lasso.fit() hard-crashes on Windows (exit 127, BLAS/OpenBLAS) and cannot
    be caught by try/except; the method degrades to ExtraTrees top-k on
    Windows and runs only on Linux.
  - Validated NUPACK-derived features are loaded from the canonical
    `nupack_features_full.csv` table and aligned by the unique `Sequence` key.

Usage
-----
  # Fast baseline (filter + embedded importance)
  python scripts/tools/feature_selection_pipeline.py \
    --seeds 10 --methods spearman,mutual_info,tree_importance

  # Exhaustive (all methods + wrappers + stability + multi-model)
  python scripts/tools/feature_selection_pipeline.py \
    --seeds 30 --methods spearman,mutual_info,tree_importance,permutation,boruta,forward,rfe \
    --with-constrained --target winsor --loss mae \
    --eval-models ET,RF,GBR,XGB,LGBM \
    --stability-inner-folds 5 --stability-threshold 0.6 \
    --output artifacts/feat_sel/full.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

IS_WINDOWS = sys.platform == "win32"

# ── Windows-safe imports -------------------------------------------------
# scipy.stats.spearmanr crashes on Windows (exit 127); the project convention
# is rankdata + pearsonr.
from scipy.stats import pearsonr, rankdata, kendalltau, binom
from sklearn.ensemble import (ExtraTreesRegressor, RandomForestRegressor,
                              GradientBoostingRegressor)
from sklearn.feature_selection import (VarianceThreshold, f_regression,
                                       mutual_info_regression, RFE)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASES = "ACGU"
ALL_DIMERS = [a + b for a in BASES for b in BASES]
MUT_POS = [30, 31, 32, 33, 89, 90, 91, 92, 127, 128, 129, 130, 131]
WT_141 = ("UAUCCUUCGGGGCAGGGUGGAAAUCCCGACCGGCGGUAGUAAAGCACAUUUGCUUUAGAGCCC"
          "GUGACCCGUGUGCAUAAGCACGCGGUGGAUUCAGUUUAAGCUGAAGCCGACAGUGAAAGUCUGGA"
          "UGGGAGAAGGAUG")


# ════════════════════════════════════════════════════════════════════════
# Regression metrics
# ════════════════════════════════════════════════════════════════════════

def _spearman(a, b):
    """Spearman rank correlation (invariant to monotonic transforms)."""
    if len(a) < 2 or len(set(a)) == 1 or len(set(b)) == 1:
        return 0.0
    return pearsonr(rankdata(a), rankdata(b))[0]


def _pearson(a, b):
    if len(a) < 2:
        return 0.0
    return pearsonr(a, b)[0]


def _kendall(a, b):
    try:
        return kendalltau(a, b)[0]
    except Exception:
        return float("nan")


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def _r2(a, b):
    a = np.asarray(a); b = np.asarray(b)
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


METRICS = {
    "RMSE": _rmse,
    "MAE": _mae,
    "R2": _r2,
    "Pearson": _pearson,
    "Spearman": _spearman,
    "Kendall": _kendall,
}


# ════════════════════════════════════════════════════════════════════════
# Target transforms (fit on the training fold only — no leakage)
# ════════════════════════════════════════════════════════════════════════

def fit_target_transform(y_train, name):
    """Fit a target transform on the training labels only.

    Returns ``(y_train_transformed, test_transform_fn)``, where
    ``test_transform_fn`` maps held-out labels into the same target space, or
    is ``None`` when no scale-consistent mapping exists.

    Monotone transforms preserve rank order, so Spearman and Kendall are
    invariant; the scale-dependent metrics (RMSE/MAE/R2) and Pearson are only
    meaningful when ``test_transform_fn`` is available. rank-gaussianization
    has no clean scalar inverse, hence returns ``None``.
    """
    if name == "raw":
        return y_train, (lambda y: y)
    if name == "winsor":
        lo = np.percentile(y_train, 2.5)
        hi = np.percentile(y_train, 97.5)
        return np.clip(y_train, lo, hi), (lambda y: np.clip(y, lo, hi))
    if name == "rank_gaussian":
        from scipy.stats import norm
        r = rankdata(y_train)
        return norm.ppf((r - 0.5) / len(y_train)), None
    raise ValueError(f"unknown target transform: {name}")


# ════════════════════════════════════════════════════════════════════════
# Feature construction (sequence-derived) and cached loaders
# ════════════════════════════════════════════════════════════════════════

def build_mut(seqs):
    X = np.zeros((len(seqs), len(MUT_POS) * 4), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = str(s).upper().replace("T", "U")
        for j, p in enumerate(MUT_POS):
            idx = p - 1
            if idx < len(s) and s[idx] in BASES:
                X[i, j * 4 + BASES.index(s[idx])] = 1.0
    names = [f"mut_p{p}_{b}" for p in MUT_POS for b in BASES]
    return X, names


def build_wd(seqs, window=5):
    X = np.zeros((len(seqs), len(MUT_POS) * 16), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = str(s).upper().replace("T", "U")
        n = len(s)
        for j, p in enumerate(MUT_POS):
            idx = p - 1
            lo = max(0, idx - window)
            hi = min(n, idx + window + 1)
            sub = s[lo:hi]
            for t in range(len(sub) - 1):
                d = sub[t:t + 2]
                if d[0] in BASES and d[1] in BASES:
                    X[i, j * 16 + ALL_DIMERS.index(d)] += 1.0
            total = max(1, len(sub) - 1)
            X[i, j * 16:j * 16 + 16] /= total
    names = [f"wd_p{p}_{d}" for p in MUT_POS for d in ALL_DIMERS]
    return X, names


def build_struct(seqs):
    ROLES = {30: "stem", 31: "stem", 32: "stem", 33: "junction",
             89: "loop", 90: "loop", 91: "loop", 92: "junction",
             127: "stem", 128: "stem", 129: "stem", 130: "junction", 131: "loop"}
    PAIRS = {30: 33, 31: 32, 32: 31, 33: 30,
             90: 92, 92: 90, 127: 131, 131: 127, 128: 130, 130: 128}
    X = np.zeros((len(seqs), 43), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = str(s).upper().replace("T", "U")
        mc = sm = lm = 0
        for j, p in enumerate(MUT_POS):
            idx = p - 1
            wt = WT_141[idx]
            mb = s[idx]
            is_mut = 1.0 if mb != wt else 0.0
            role = ROLES.get(p, "unknown")
            X[i, j * 3] = 1.0 if role == "stem" else 0.0
            X[i, j * 3 + 1] = 1.0 if role == "loop" else 0.0
            X[i, j * 3 + 2] = is_mut * (1.0 if role == "stem" else 0.0)
            mc += is_mut
            if is_mut:
                if role == "stem":
                    sm += 1
                elif role == "loop":
                    lm += 1
        X[i, -4] = mc
        X[i, -3] = sm
        X[i, -2] = lm
        broken = sum(1 for p in MUT_POS
                     if PAIRS.get(p, 0) > 0 and s[p - 1] != WT_141[p - 1]
                     and s[PAIRS[p] - 1] == WT_141[PAIRS[p] - 1])
        X[i, -1] = float(broken)
    names = []
    for p in MUT_POS:
        names += [f"struct_p{p}_is_stem", f"struct_p{p}_is_loop", f"struct_p{p}_mut_in_stem"]
    names += ["struct_total_mut", "struct_stem_mut", "struct_loop_mut", "struct_broken_pairs"]
    return X, names


SEQUENCE_BUILDERS = {"mut": build_mut, "wd": build_wd, "struct": build_struct}


def load_cached_csv(path):
    df = pd.read_csv(path)
    return df.to_numpy(dtype=np.float32), list(df.columns)


# ════════════════════════════════════════════════════════════════════════
# NUPACK features (canonical safe table, aligned by Sequence)
# ════════════════════════════════════════════════════════════════════════

# The 7 NUPACK sub-blocks with confirmed positive independent Δ (192d total),
# as regexes over column names (from eval_nupack_blocks.py):
#   contact_pca 15d, cluster_pair 7d, energy_land 5d, pos_entropy 13d,
#   motif_global 42d, pos_pair_prob 32d, motif_positional 78d.
NUPACK_USEFUL_PATTERNS = [
    r"nupack_contact_",                 # contact_pca
    r"nupack_inter_cluster_\d+_\d+$",   # cluster_pair
    r"nupack_intra_cluster_\d+$",
    r"nupack_mut_long_range$",
    r"nupack_n_very_stable_pairs$",     # energy_land
    r"nupack_n_moderate_pairs$",
    r"nupack_stable_pair_ratio$",
    r"nupack_eig_ipr$",
    r"nupack_eig_n_eff_modes$",
    r"nupack_pos\d+_entropy$",          # pos_entropy
    r"nupack_motif_global_",            # motif_global
    r"nupack_motif_apt_",
    r"nupack_pos\d+_pair_prob$",        # pos_pair_prob
    r"nupack_pos\d+_likely_paired$",
    r"nupack_c\d+_mean_pair_prob$",
    r"nupack_mean_pair_prob$",
    r"nupack_std_pair_prob$",
    r"nupack_frac_paired_gt_0_5$",
    r"nupack_motif_pos\d+_is_",         # motif_positional
]


def load_nupack_features(data_dir, full_df, blocks="useful"):
    """Load the validated safe NUPACK table and align it to canonical rows.

    The clean mainline has one full table rather than historical split files.
    A requested NUPACK block is mandatory: missing files, unsafe columns,
    incomplete coverage, duplicates, or non-finite values abort the run.
    """
    nup_path = data_dir / "nupack_features_full.csv"
    if not nup_path.is_file():
        raise FileNotFoundError(f"validated NUPACK feature table is missing: {nup_path}")
    nup_df = pd.read_csv(nup_path)
    if "Sequence" not in nup_df.columns:
        raise ValueError("NUPACK feature table is missing the Sequence key")
    if nup_df["Sequence"].isna().any() or nup_df["Sequence"].duplicated().any():
        raise ValueError("NUPACK Sequence keys must be non-missing and unique")
    nup_cols = [c for c in nup_df.columns if c != "Sequence"]
    unsafe_columns = [
        c for c in nup_cols
        if not c.startswith(("nupack_", "delta_nupack_"))
    ]
    if unsafe_columns:
        raise ValueError(
            f"NUPACK table contains non-feature columns: {unsafe_columns[:10]}"
        )
    canonical_sequences = full_df["Sequence"].astype(str).tolist()
    nupack_sequences = set(nup_df["Sequence"].astype(str))
    missing = set(canonical_sequences) - nupack_sequences
    extra = nupack_sequences - set(canonical_sequences)
    if missing or extra:
        raise ValueError(
            "NUPACK/canonical Sequence coverage mismatch: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    nup_df["Sequence"] = nup_df["Sequence"].astype(str)
    nup_df = nup_df.set_index("Sequence").loc[canonical_sequences]
    if blocks == "useful":
        keep = [i for i, c in enumerate(nup_cols)
                if any(re.search(p, c) for p in NUPACK_USEFUL_PATTERNS)]
        nup_cols = [nup_cols[i] for i in keep]
    if not nup_cols:
        raise ValueError(f"no NUPACK columns matched block selection: {blocks}")
    X = nup_df[nup_cols].to_numpy(dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError("NUPACK features contain missing or non-finite values")
    return X, list(nup_cols)


# ════════════════════════════════════════════════════════════════════════
# Inner-CV evaluator (used by wrappers)
# ════════════════════════════════════════════════════════════════════════

def _cv_spearman(X, y, model_factory, n_folds=3, random_state=42):
    """Mean inner-fold Spearman for a candidate feature subset."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    scores = []
    for tr, va in kf.split(X):
        m = model_factory()
        m.fit(X[tr], y[tr])
        scores.append(_spearman(y[va], m.predict(X[va])))
    return float(np.mean(scores))


# ════════════════════════════════════════════════════════════════════════
# Model factory (deterministic, honors loss criterion)
# ════════════════════════════════════════════════════════════════════════

def _make_tree(model, loss, n_jobs, random_state=42):
    if model == "ET":
        return ExtraTreesRegressor(
            n_estimators=200, max_features="log2", min_samples_leaf=5,
            criterion="absolute_error" if loss == "mae" else "squared_error",
            n_jobs=n_jobs, random_state=random_state)
    if model == "RF":
        return RandomForestRegressor(
            n_estimators=200, max_features="log2", min_samples_leaf=5,
            criterion="absolute_error" if loss == "mae" else "squared_error",
            n_jobs=n_jobs, random_state=random_state)
    if model == "GBR":
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=3,
            loss="absolute_error" if loss == "mae" else "squared_error",
            random_state=random_state)
    raise ValueError(f"unknown tree model: {model}")


# ════════════════════════════════════════════════════════════════════════
# FILTER methods — univariate ranking (return a boolean mask over columns)
# ════════════════════════════════════════════════════════════════════════

def _top_k_mask(scores, k):
    d = len(scores)
    if k is None or k >= d:
        return np.ones(d, dtype=bool)
    order = np.argsort(-scores)[:k]
    mask = np.zeros(d, dtype=bool)
    mask[order] = True
    return mask


def filter_variance(X, y, k, **kw):
    """Drop near-constant features (variance below threshold)."""
    vt = VarianceThreshold(threshold=kw.get("var_threshold", 0.0))
    vt.fit(X)
    return vt.get_support()


def filter_f_regression(X, y, k, **kw):
    """Univariate linear F-statistic; rank by -log10(p), keep top-k."""
    F, pval = f_regression(X, y)
    scores = -np.log10(np.clip(pval, 1e-300, 1.0))
    return _top_k_mask(scores, k)


def filter_spearman(X, y, k, **kw):
    """Absolute Spearman correlation with the target; keep top-k (robust)."""
    scores = np.array([abs(_spearman(X[:, j], y)) for j in range(X.shape[1])])
    return _top_k_mask(scores, k)


def filter_mutual_info(X, y, k, **kw):
    """Mutual information (k-NN estimator); captures non-linear dependence."""
    scores = mutual_info_regression(X, y, random_state=kw.get("random_state", 42))
    return _top_k_mask(np.maximum(scores, 0.0), k)


# ════════════════════════════════════════════════════════════════════════
# EMBEDDED methods
# ════════════════════════════════════════════════════════════════════════

def embedded_tree_importance(X, y, k, model="ET", loss="mse", n_jobs=1,
                             random_state=42, **kw):
    """Impurity-based importance; keep top-k.

    Caveat (project-documented): impurity importance over-ranks features that
    are merely correlated with a true signal (the 'squarna' artifact). Treat
    as a candidate ranking, not a final verdict.
    """
    m = _make_tree(model, loss, n_jobs, random_state)
    m.fit(X, y)
    return _top_k_mask(m.feature_importances_, k)


def embedded_permutation_importance(X, y, k, candidate_idx=None, n_repeats=3,
                                    model="ET", loss="mse", n_jobs=1, **kw):
    """Model-agnostic permutation importance over a pre-screened candidate set.

    Permutation importance scales as n_features x n_repeats model refits, so it
    is applied to `candidate_idx` (pre-screened via tree importance), not the
    full space. Returns top-k among the candidates.
    """
    if candidate_idx is None:
        candidate_idx = np.where(
            embedded_tree_importance(X, y, k, model, loss, n_jobs, **kw))[0]
    cands = np.asarray(candidate_idx, dtype=int)
    Xc = X[:, cands]
    m = _make_tree(model, loss, n_jobs)
    m.fit(Xc, y)
    r = permutation_importance(m, Xc, y, n_repeats=n_repeats,
                               random_state=42, n_jobs=n_jobs)
    kk = min(k, len(cands))
    order = np.argsort(-r.importances_mean)[:kk]
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[cands[order]] = True
    return mask


def embedded_lasso(X, y, k, **kw):
    """L1-regularized linear model; select non-zero coefficients.

    Lasso.fit() hard-crashes on Windows (exit 127, BLAS); degrade to ExtraTrees
    top-k there. On Linux, standardize and select |coef| > eps, falling back to
    |coef| top-k if all coefficients vanish.
    """
    if IS_WINDOWS:
        print("  [lasso] Lasso.fit() hard-crashes on Windows (BLAS); "
              "using ExtraTrees top-k fallback.")
        return embedded_tree_importance(X, y, k, **kw)
    try:
        from sklearn.linear_model import Lasso
        Xs = StandardScaler().fit_transform(X)
        m = Lasso(alpha=kw.get("lasso_alpha", 0.01), max_iter=10000)
        m.fit(Xs, y)
        coef = np.abs(m.coef_)
        nonzero = coef > 1e-8
        if nonzero.sum() == 0:
            return _top_k_mask(coef, k)
        return nonzero
    except Exception as e:  # pragma: no cover
        print(f"  [lasso] failed ({type(e).__name__}); ExtraTrees fallback.")
        return embedded_tree_importance(X, y, k, **kw)


def embedded_boruta(X, y, k, candidate_idx=None, n_iter=20, alpha=0.05,
                    random_state=42, n_jobs=1, **kw):
    """Boruta (Kursa & Rudnicki, 2010) — all-relevant selection.

    Compares each real feature's importance against 'shadow' permutations
    (preserving column marginals). A feature is retained iff it beats the max
    shadow importance significantly often (one-sided binomial test). Identifies
    all signal-carrying features, including redundant ones that wrappers/L1 drop.
    Runs over `candidate_idx` for tractability.
    """
    if candidate_idx is None:
        cands = np.arange(X.shape[1])
    else:
        cands = np.asarray(candidate_idx, dtype=int)
    Xc = X[:, cands]
    d = Xc.shape[1]
    rng = np.random.RandomState(random_state)
    hits = np.zeros(d, dtype=int)
    for it in range(n_iter):
        X_shadow = np.array([rng.permutation(Xc[:, j]) for j in range(d)]).T
        X_aug = np.hstack([Xc, X_shadow])
        m = ExtraTreesRegressor(n_estimators=100, max_features="log2",
                                min_samples_leaf=5,
                                random_state=random_state + it, n_jobs=n_jobs)
        m.fit(X_aug, y)
        imp = m.feature_importances_
        shadow_max = imp[d:].max()
        hits += (imp[:d] > shadow_max).astype(int)
    selected_local = np.zeros(d, dtype=bool)
    for j in range(d):
        # P(X >= hits) under H0: indistinguishable from shadow (p = 0.5)
        p = binom.sf(hits[j] - 1, n_iter, 0.5)
        if p < alpha:
            selected_local[j] = True
    if selected_local.sum() == 0:
        selected_local = _top_k_mask(hits.astype(float), min(k, d))
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[cands[selected_local]] = True
    return mask


# ════════════════════════════════════════════════════════════════════════
# WRAPPER methods (operate on a pre-screened candidate set for tractability)
# ════════════════════════════════════════════════════════════════════════

def wrapper_forward(X, y, candidate_idx, model_factory, max_features=20,
                    tol=0.002, n_folds=3, random_state=42, **kw):
    """Greedy forward selection maximizing inner-CV Spearman."""
    candidates = list(candidate_idx)
    selected = []
    best = -np.inf
    for _ in range(max_features):
        if not candidates:
            break
        best_add, best_new = None, best
        for c in candidates:
            trial = selected + [c]
            s = _cv_spearman(X[:, trial], y, model_factory,
                             n_folds=n_folds, random_state=random_state)
            if s > best_new:
                best_new, best_add = s, c
        if best_add is None or (best_new - best) < tol:
            break
        selected.append(best_add)
        candidates.remove(best_add)
        best = best_new
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[selected] = True
    return mask


def wrapper_backward(X, y, candidate_idx, model_factory, min_features=10,
                     n_folds=3, random_state=42, **kw):
    """Greedy backward elimination down to `min_features`."""
    current = list(candidate_idx)
    while len(current) > min_features:
        worst, worst_score = None, -np.inf
        for c in current:
            trial = [x for x in current if x != c]
            s = _cv_spearman(X[:, trial], y, model_factory,
                             n_folds=n_folds, random_state=random_state)
            if s > worst_score:
                worst_score, worst = s, c
        if worst is None:
            break
        current.remove(worst)
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[current] = True
    return mask


def wrapper_rfe(X, y, candidate_idx, model_factory, n_features=50, step=1, **kw):
    """Recursive Feature Elimination (Guyon et al., 2002)."""
    m = model_factory()
    n_features = min(n_features, len(candidate_idx))
    rfe = RFE(estimator=m, n_features_to_select=n_features, step=step)
    rfe.fit(X[:, candidate_idx], y)
    mask = np.zeros(X.shape[1], dtype=bool)
    sel_local = np.asarray(candidate_idx)[rfe.support_]
    mask[sel_local] = True
    return mask


# ════════════════════════════════════════════════════════════════════════
# Method registry
# ════════════════════════════════════════════════════════════════════════

METHOD_REGISTRY = {
    "all":             (None, "control"),
    "variance":        (filter_variance, "filter"),
    "f_regression":    (filter_f_regression, "filter"),
    "spearman":        (filter_spearman, "filter"),
    "mutual_info":     (filter_mutual_info, "filter"),
    "tree_importance": (embedded_tree_importance, "embedded"),
    "permutation":     (embedded_permutation_importance, "embedded"),
    "lasso":           (embedded_lasso, "embedded"),
    "boruta":          (embedded_boruta, "embedded"),
    "forward":         (wrapper_forward, "wrapper"),
    "backward":        (wrapper_backward, "wrapper"),
    "rfe":             (wrapper_rfe, "wrapper"),
}

# Methods that operate on a pre-screened candidate set (tractability)
REFINE_METHODS = {"permutation", "boruta", "forward", "backward", "rfe"}


def _dispatch_select(method, family, fn, X, y, cand_idx, args, seed):
    """Invoke one selection method with its full parameter set.

    Centralizing dispatch here ensures the main per-seed path and the stability
    inner-fold path cannot drift apart (which previously dropped lasso_alpha,
    var_threshold, and loss/n_jobs in the generic else-branch).
    """
    if method == "all":
        return np.ones(X.shape[1], dtype=bool)
    wf = lambda: _make_tree("ET", args.loss, args.n_jobs, random_state=seed)
    if family == "wrapper":
        return fn(X, y, cand_idx, model_factory=wf,
                  max_features=args.max_forward, tol=args.forward_tol,
                  min_features=args.backward_min, n_features=args.rfe_features,
                  n_folds=3, random_state=seed)
    if method == "permutation":
        return fn(X, y, args.permutation_k, candidate_idx=cand_idx,
                  n_repeats=args.permutation_repeats, loss=args.loss,
                  n_jobs=args.n_jobs)
    if method == "boruta":
        return fn(X, y, args.top_k, candidate_idx=cand_idx,
                  n_iter=args.boruta_iters, n_jobs=args.n_jobs, random_state=seed)
    if method == "lasso":
        return fn(X, y, args.top_k, lasso_alpha=args.lasso_alpha)
    if method == "variance":
        return fn(X, y, args.top_k, var_threshold=args.var_threshold)
    if method == "tree_importance":
        return fn(X, y, args.top_k, loss=args.loss, n_jobs=args.n_jobs,
                  random_state=seed)
    # remaining filters: f_regression, spearman, mutual_info
    return fn(X, y, args.top_k, random_state=seed)


def make_force_keep_mask(names, groups):
    """Build a deterministic mask for feature families that must be retained."""
    prefixes = {
        "mut": "mut_",
        "wd": "wd_",
        "struct": "struct_",
        "nupack": "nupack_",
    }
    requested = [g.strip() for g in groups.split(",") if g.strip()]
    unknown = sorted(set(requested) - set(prefixes))
    if unknown:
        raise ValueError(f"unknown force-keep group(s): {unknown}")
    mask = np.zeros(len(names), dtype=bool)
    for group in requested:
        mask |= np.char.startswith(names.astype(str), prefixes[group])
    return mask


# ════════════════════════════════════════════════════════════════════════
# Model factory for final evaluation (multi-model, honors loss criterion)
# ════════════════════════════════════════════════════════════════════════

def make_eval_model(name, loss, n_jobs):
    name = name.upper()
    if name == "ET":
        return ExtraTreesRegressor(
            n_estimators=500, max_features="log2", min_samples_leaf=5,
            criterion="absolute_error" if loss == "mae" else "squared_error",
            n_jobs=n_jobs, random_state=42)
    if name == "RF":
        return RandomForestRegressor(
            n_estimators=500, max_features="log2", min_samples_leaf=5,
            criterion="absolute_error" if loss == "mae" else "squared_error",
            n_jobs=n_jobs, random_state=42)
    if name == "GBR":
        return GradientBoostingRegressor(
            n_estimators=500, max_depth=3,
            loss="absolute_error" if loss == "mae" else "squared_error",
            random_state=42)
    if name == "XGB":
        try:
            import xgboost as xgb
            major_version = int(xgb.__version__.split(".", 1)[0])
            if major_version < 2:
                raise RuntimeError(
                    f"XGB requires xgboost>=2.0 for reg:absoluteerror; "
                    f"found {xgb.__version__}"
                )
            return xgb.XGBRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.05,
                objective="reg:absoluteerror" if loss == "mae" else "reg:squarederror",
                n_jobs=n_jobs, random_state=42)
        except ImportError as exc:
            raise RuntimeError(
                "XGB was requested but xgboost is not installed. "
                "Install the project's boosting dependencies before running."
            ) from exc
    if name == "LGBM":
        try:
            import lightgbm as lgb
            return lgb.LGBMRegressor(
                n_estimators=500, num_leaves=31, learning_rate=0.05,
                objective="mae" if loss == "mae" else "regression",
                n_jobs=n_jobs, random_state=42, verbose=-1)
        except ImportError as exc:
            raise RuntimeError(
                "LGBM was requested but lightgbm is not installed. "
                "Install the project's boosting dependencies before running."
            ) from exc
    if name == "SVR":
        try:
            from sklearn.svm import SVR
            return SVR(kernel="rbf", C=1.0, epsilon=0.1)
        except ImportError as exc:  # pragma: no cover - sklearn is required
            raise RuntimeError("SVR was requested but scikit-learn is unavailable.") from exc
    raise ValueError(f"unknown eval model: {name}")


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Comprehensive feature selection pipeline")
    p.add_argument("--data-dir", default="data/proxy2000_v2")
    # features
    p.add_argument("--seq-groups", default="mut,wd,struct")
    p.add_argument("--with-constrained", action="store_true")
    p.add_argument("--with-ed", action="store_true",
                   help="load ED features (deprecated — label leakage)")
    p.add_argument("--with-nupack", action="store_true",
                   help="load validated NUPACK features from "
                        "nupack_features_full.csv")
    p.add_argument("--nupack-blocks", default="useful", choices=["useful", "all"],
                   help="useful = 7 confirmed sub-blocks (192d); "
                        "all = all validated NUPACK/delta columns (1192d)")
    p.add_argument("--expected-features", type=int, default=None,
                   help="abort if the assembled input dimension differs")
    # methods
    p.add_argument("--methods", default="spearman,mutual_info,tree_importance",
                   help="comma-separated: " + ",".join(METHOD_REGISTRY))
    p.add_argument("--top-k", type=int, default=100,
                   help="top-k for ranking methods")
    p.add_argument("--force-keep-groups", default="mut",
                   help="comma-separated feature families always retained; "
                        "choices: mut,wd,struct,nupack; empty disables")
    p.add_argument("--var-threshold", type=float, default=0.0)
    p.add_argument("--lasso-alpha", type=float, default=0.01)
    p.add_argument("--boruta-iters", type=int, default=20)
    p.add_argument("--permutation-repeats", type=int, default=3)
    p.add_argument("--permutation-k", type=int, default=20,
                   help="number of features permutation importance keeps "
                        "(must be < --candidate-k to actually prune the "
                        "candidate set)")
    p.add_argument("--candidate-k", type=int, default=50,
                   help="pre-screen size for refinement methods "
                        "(permutation/boruta/wrappers)")
    p.add_argument("--max-forward", type=int, default=20)
    p.add_argument("--forward-tol", type=float, default=0.002)
    p.add_argument("--rfe-features", type=int, default=30)
    p.add_argument("--backward-min", type=int, default=10)
    # target / loss
    p.add_argument("--target", default="raw", choices=["raw", "winsor", "rank_gaussian"])
    p.add_argument("--loss", default="mse", choices=["mse", "mae"])
    # evaluation
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--eval-models", default="ET",
                   help="comma-separated: ET,RF,GBR,XGB,LGBM,SVR")
    p.add_argument("--stability-inner-folds", type=int, default=0,
                   help="if >0, require cross-inner-fold agreement per method")
    p.add_argument("--stability-threshold", type=float, default=0.6)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--log-dir", default=None,
                   help="checkpoint directory for progress and per-seed rows")
    p.add_argument("--output", default="artifacts/feat_sel/feature_selection.csv")
    args = p.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = None
    if args.log_dir:
        log_dir = Path(args.log_dir)
        if not log_dir.is_absolute():
            log_dir = PROJECT_ROOT / log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "run_meta.json").write_text(
            json.dumps(
                {
                    "started": datetime.now().isoformat(),
                    "command": " ".join(sys.argv),
                    "host": socket.gethostname(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # ── Build features ──────────────────────────────────────────────────
    full_path = data_dir / "fad_proxy2000_v2_full.csv"
    if not full_path.is_file():
        raise FileNotFoundError(f"canonical data is missing: {full_path}")
    full_df = pd.read_csv(full_path)
    required_columns = {"Sequence", "gbsa"}
    missing_columns = required_columns - set(full_df.columns)
    if missing_columns:
        raise ValueError(f"canonical data is missing columns: {sorted(missing_columns)}")
    if len(full_df) != 2000:
        raise ValueError(f"canonical data must contain 2000 rows, found {len(full_df)}")
    if full_df["Sequence"].isna().any() or full_df["Sequence"].duplicated().any():
        raise ValueError("canonical Sequence values must be non-missing and unique")
    normalized_sequences = full_df["Sequence"].astype(str).str.upper().str.replace("T", "U")
    invalid_lengths = normalized_sequences.str.len() != len(WT_141)
    if invalid_lengths.any():
        raise ValueError(
            f"canonical sequences must be {len(WT_141)} nt; "
            f"found {int(invalid_lengths.sum())} invalid rows"
        )
    if not np.isfinite(full_df["gbsa"].to_numpy(dtype=np.float64)).all():
        raise ValueError("canonical gbsa target contains missing or non-finite values")
    seqs = full_df["Sequence"].values
    gbsa = full_df["gbsa"].values.astype(np.float64)

    blocks, names = [], []
    seq_groups = [x.strip() for x in args.seq_groups.split(",") if x.strip()]
    unknown_seq_groups = sorted(set(seq_groups) - set(SEQUENCE_BUILDERS))
    if unknown_seq_groups:
        raise ValueError(f"unknown sequence feature group(s): {unknown_seq_groups}")
    if len(seq_groups) != len(set(seq_groups)):
        raise ValueError("sequence feature groups must not be repeated")
    for g in seq_groups:
        Xg, ng = SEQUENCE_BUILDERS[g](seqs)
        blocks.append(Xg)
        names += ng
        print(f"[build] {g}: {Xg.shape[1]}d")
    if args.with_constrained:
        path = data_dir / "features" / "constrained" / "constrained_interactions_full.csv"
        if path.exists():
            Xg, ng = load_cached_csv(path)
            blocks.append(Xg)
            names += ng
            print(f"[load] constrained: {Xg.shape[1]}d")
    if args.with_ed:
        path = data_dir / "features" / "ed" / "ed_features_full.csv"
        if path.exists():
            Xg, ng = load_cached_csv(path)
            blocks.append(Xg)
            names += ng
            print("[load] ed: 104d — WARNING: ED features leak target statistics "
                  "at the feature level (pre-computed over the full dataset); "
                  "nested CV cannot remove this. Do NOT use for final results.")
    if args.with_nupack:
        Xn, nn = load_nupack_features(data_dir, full_df, blocks=args.nupack_blocks)
        blocks.append(Xn)
        names += nn
        print(f"[load] nupack ({args.nupack_blocks}): {Xn.shape[1]}d")

    if not blocks:
        print("[error] no feature blocks produced; provide --seq-groups or "
              "a cached block (--with-constrained/--with-ed).")
        sys.exit(1)
    X_full = np.hstack(blocks).astype(np.float32)
    names = np.array(names)
    n, d = X_full.shape
    if args.expected_features is not None and d != args.expected_features:
        raise ValueError(
            f"assembled feature dimension mismatch: expected "
            f"{args.expected_features}, found {d}"
        )
    force_keep = make_force_keep_mask(names, args.force_keep_groups)
    print(f"[data] {n} samples x {d} features; target=gbsa; transform={args.target}; loss={args.loss}")
    print(f"[policy] force-keep={args.force_keep_groups or 'none'} ({int(force_keep.sum())}d)")
    if not args.with_nupack:
        print("[info] NUPACK features not loaded (pass --with-nupack on Linux).")
    if IS_WINDOWS:
        print("[info] lasso degrades to ExtraTrees top-k on Windows (BLAS).")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if len(methods) != len(set(methods)):
        raise ValueError("selection methods must not be repeated")
    for m in methods:
        if m not in METHOD_REGISTRY:
            print(f"[error] unknown method: {m}; aborting.")
            sys.exit(1)
    eval_models = [m.strip() for m in args.eval_models.split(",") if m.strip()]
    eval_models = [m.upper() for m in eval_models]
    if not eval_models:
        raise ValueError("at least one evaluation model is required")
    if len(eval_models) != len(set(eval_models)):
        raise ValueError("evaluation models must not be repeated")
    # Fail before an expensive multi-seed run if an optional requested backend
    # is unavailable or a model name is invalid.
    for model_name in eval_models:
        make_eval_model(model_name, args.loss, args.n_jobs)

    rows = []
    selection_records = []
    selection_freq = {m: Counter() for m in methods}

    for seed in range(1, args.seeds + 1):
        tr_idx, te_idx = train_test_split(range(n), test_size=0.2, random_state=seed)
        X_tr, y_tr = X_full[tr_idx], gbsa[tr_idx]
        X_te, y_te = X_full[te_idx], gbsa[te_idx]
        y_tr_t, te_transform = fit_target_transform(y_tr, args.target)

        # Pre-screen candidate set for refinement methods (fast tree importance)
        cand_idx = None
        if any(m in REFINE_METHODS for m in methods):
            cm = embedded_tree_importance(X_tr, y_tr_t, args.candidate_k,
                                          loss=args.loss, n_jobs=args.n_jobs,
                                          random_state=seed)
            cand_idx = np.where(cm & ~force_keep)[0]

        for method in methods:
            fn, family = METHOD_REGISTRY[method]
            mask = _dispatch_select(method, family, fn, X_tr, y_tr_t, cand_idx,
                                    args, seed)
            mask |= force_keep

            # Optional stability: cross-inner-fold agreement
            if args.stability_inner_folds > 0:
                kf = KFold(n_splits=args.stability_inner_folds, shuffle=True,
                           random_state=seed)
                agree = Counter()
                for fold_index, (itr, _) in enumerate(kf.split(X_tr)):
                    inner_y_t, _ = fit_target_transform(y_tr[itr], args.target)
                    inner_cand_idx = None
                    if method in REFINE_METHODS:
                        inner_cm = embedded_tree_importance(
                            X_tr[itr], inner_y_t, args.candidate_k,
                            loss=args.loss, n_jobs=args.n_jobs,
                            random_state=seed * 100 + fold_index,
                        )
                        inner_cand_idx = np.where(inner_cm & ~force_keep)[0]
                    sub = _dispatch_select(method, family, fn, X_tr[itr],
                                           inner_y_t, inner_cand_idx, args,
                                           seed * 100 + fold_index)
                    sub |= force_keep
                    for j in np.where(sub)[0]:
                        agree[j] += 1
                min_folds = max(
                    1,
                    int(np.ceil(args.stability_inner_folds * args.stability_threshold)),
                )
                sel_idx = [j for j, c in agree.items() if c >= min_folds]
                mask = np.zeros(d, dtype=bool)
                mask[sel_idx] = True
                mask |= force_keep

            sel_idx = np.where(mask)[0]
            if len(sel_idx) == 0:
                print(f"  [warn] {method} selected 0 features at seed {seed}; skipping.")
                continue
            for j in sel_idx:
                selection_freq[method][int(j)] += 1
            selected_names = names[sel_idx].tolist()
            selected_hash = hashlib.sha256(
                "\n".join(selected_names).encode("utf-8")
            ).hexdigest()
            selection_records.append(
                {
                    "seed": seed,
                    "method": method,
                    "n_selected": len(sel_idx),
                    "selected_sha256": selected_hash,
                    "selected_features": "|".join(selected_names),
                }
            )

            for model_name in eval_models:
                fm = make_eval_model(model_name, args.loss, args.n_jobs)
                fm.fit(X_tr[:, sel_idx], y_tr_t)
                pred = fm.predict(X_te[:, sel_idx])
                row = {"seed": seed, "method": method, "model": model_name,
                       "n_selected": len(sel_idx),
                       "selected_sha256": selected_hash}
                # Rank metrics are invariant to the monotone target transform.
                row["Spearman"] = _spearman(y_te, pred)
                row["Kendall"] = _kendall(y_te, pred)
                # Scale metrics require a consistent target space.
                if te_transform is not None:
                    y_te_s = te_transform(y_te)
                    row["RMSE"] = _rmse(y_te_s, pred)
                    row["MAE"] = _mae(y_te_s, pred)
                    row["R2"] = _r2(y_te_s, pred)
                    row["Pearson"] = _pearson(y_te_s, pred)
                else:
                    row["RMSE"] = row["MAE"] = row["R2"] = row["Pearson"] = float("nan")
                rows.append(row)

        if log_dir:
            seed_rows = [row for row in rows if row["seed"] == seed]
            running = log_dir / "all_results.csv"
            pd.DataFrame(seed_rows).to_csv(
                running,
                mode="a",
                index=False,
                header=not running.exists(),
            )
            seed_selections = [row for row in selection_records if row["seed"] == seed]
            selection_checkpoint = log_dir / "selected_features.csv"
            pd.DataFrame(seed_selections).to_csv(
                selection_checkpoint,
                mode="a",
                index=False,
                header=not selection_checkpoint.exists(),
            )
            (log_dir / "progress.json").write_text(
                json.dumps(
                    {
                        "completed_seeds": seed,
                        "requested_seeds": args.seeds,
                        "result_rows": len(rows),
                        "updated": datetime.now().isoformat(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        print(f"  seed={seed:2d} done [{seed}/{args.seeds}]")

    # ── Save + summarize ────────────────────────────────────────────────
    if not rows:
        print("\n[error] no results produced; check method/feature configuration.")
        sys.exit(1)

    res = pd.DataFrame(rows)
    expected_keys = {
        (seed, method, model)
        for seed in range(1, args.seeds + 1)
        for method in methods
        for model in eval_models
    }
    observed_keys = set(
        res[["seed", "method", "model"]].itertuples(index=False, name=None)
    )
    duplicated_keys = res.duplicated(["seed", "method", "model"]).sum()
    if duplicated_keys or observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)[:10]
        unexpected = sorted(observed_keys - expected_keys)[:10]
        raise RuntimeError(
            "incomplete result matrix: "
            f"rows={len(res)}, expected={len(expected_keys)}, "
            f"duplicates={int(duplicated_keys)}, missing={missing}, "
            f"unexpected={unexpected}"
        )
    res.to_csv(out_path, index=False)
    print(f"\n[OK] per-seed results -> {out_path}")

    selection_path = out_path.with_name(out_path.stem + "_selected_features.csv")
    pd.DataFrame(selection_records).to_csv(selection_path, index=False)
    stability_rows = []
    for method, counts in selection_freq.items():
        for feature_index, count in counts.items():
            stability_rows.append(
                {
                    "method": method,
                    "feature": names[feature_index],
                    "selected_seeds": count,
                    "frequency": count / args.seeds,
                }
            )
    stability_path = out_path.with_name(out_path.stem + "_stability.csv")
    pd.DataFrame(stability_rows).sort_values(
        ["method", "frequency", "feature"], ascending=[True, False, True]
    ).to_csv(stability_path, index=False)

    control = res[res["method"] == "all"][["seed", "model", "Spearman"]].rename(
        columns={"Spearman": "control_spearman"}
    )
    paired_rows = []
    if not control.empty:
        paired_rng = np.random.default_rng(20260902)
        candidates = res[res["method"] != "all"].merge(
            control, on=["seed", "model"], how="inner"
        )
        candidates["delta_vs_all"] = (
            candidates["Spearman"] - candidates["control_spearman"]
        )
        for (method, model), group in candidates.groupby(["method", "model"]):
            deltas = group["delta_vs_all"].to_numpy(dtype=float)
            bootstrap_means = paired_rng.choice(
                deltas, size=(20000, len(deltas)), replace=True
            ).mean(axis=1)
            paired_rows.append(
                {
                    "method": method,
                    "model": model,
                    "paired_seeds": len(group),
                    "mean_delta_vs_all": group["delta_vs_all"].mean(),
                    "bootstrap_ci95_low": np.percentile(bootstrap_means, 2.5),
                    "bootstrap_ci95_high": np.percentile(bootstrap_means, 97.5),
                    "wins_vs_all": int((group["delta_vs_all"] > 0).sum()),
                    "mean_spearman": group["Spearman"].mean(),
                    "control_mean_spearman": group["control_spearman"].mean(),
                }
            )
    paired_path = out_path.with_name(out_path.stem + "_paired.csv")
    pd.DataFrame(paired_rows).to_csv(paired_path, index=False)
    print(f"[OK] selected features -> {selection_path}")
    print(f"[OK] stability table -> {stability_path}")
    print(f"[OK] paired comparison -> {paired_path}")

    print(f"\n{'='*90}")
    print("SUMMARY — mean test Spearman by method x model (across seeds)")
    print(f"{'='*90}")
    pivot = res.pivot_table(index="method", columns="model", values="Spearman",
                            aggfunc=["mean", "std"])
    print(pivot.round(4).to_string())

    # Feature stability (proportion of seeds each feature was selected)
    print(f"\n{'='*90}")
    print("FEATURE STABILITY — features selected in >= 50% of seeds, per method")
    print(f"{'='*90}")
    for method in methods:
        cnt = selection_freq[method]
        top = [(names[i], cnt[i] / args.seeds) for i in cnt
               if cnt[i] / args.seeds >= 0.5]
        top.sort(key=lambda x: -x[1])
        print(f"\n[{method}] {len(top)} features >= 50% stability (top 15):")
        for nm, f in top[:15]:
            print(f"    {nm:<28s} {f:.0%}")


if __name__ == "__main__":
    main()

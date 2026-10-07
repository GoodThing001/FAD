"""Active-learning acquisition and batch selection for the FAD closed loop.

Implements the 汇报's per-round batch of ~100 = 45 high epistemic uncertainty
+ 25 coverage/novelty + 20 predicted promising + 10 random control, with
physics-space distance as the diversity/coverage signal (Hamming distance over
the 13 mutation sites is a special case; physics distance is used here because
it captures functionally meaningful differences).

This replaces the paper's pure UCB + Kriging-Believer selection: the FAD target
is first to improve the surrogate (uncertainty/coverage), then to find low-GBSA
candidates (promising / LCB). `bo.py` remains available for the paper-faithful
UCB+KB path.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def standardize(X: np.ndarray) -> np.ndarray:
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - X.mean(axis=0)) / sd


def standardize_jointly(pool_phy: np.ndarray, labeled_phy: np.ndarray):
    """Standardize pool and labeled sets with ONE shared scaling.

    Fixes the space-inconsistency bug where the two sets were standardized
    independently before computing their pairwise distances.
    """
    if len(labeled_phy) == 0:
        return standardize(pool_phy), labeled_phy
    union = np.vstack([pool_phy, labeled_phy])
    sd = union.std(axis=0)
    sd[sd == 0] = 1.0
    mu = union.mean(axis=0)
    return (pool_phy - mu) / sd, (labeled_phy - mu) / sd


def novelty_to_labeled(pool_phy_std: np.ndarray, labeled_phy_std: np.ndarray) -> np.ndarray:
    """Min physics-space distance from each pool point to the labeled set."""
    if len(labeled_phy_std) == 0:
        return np.zeros(len(pool_phy_std))
    return cdist(pool_phy_std, labeled_phy_std).min(axis=1)


def select_batch(mu, sigma, pool_phy, labeled_phy, n_batch=100,
                 n_unc=45, n_cov=25, n_prom=20, n_rand=10,
                 minimize=True, seed=0):
    """Stratified batch selection. Returns sorted list of pool indices.

    mu, sigma: (M,) predictions / uncertainty. pool_phy: (M, d) physics.
    labeled_phy: (L, d) physics of the currently labeled set.
    """
    M = len(mu)
    rng = np.random.RandomState(seed)
    pool_std, labeled_std = standardize_jointly(pool_phy, labeled_phy)
    nov = novelty_to_labeled(pool_std, labeled_std)

    # Scale the 45:25:20:10 category ratio to the requested batch size.
    total_cat = n_unc + n_cov + n_prom + n_rand
    if total_cat != n_batch:
        n_unc = int(round(n_batch * n_unc / total_cat))
        n_cov = int(round(n_batch * n_cov / total_cat))
        n_prom = int(round(n_batch * n_prom / total_cat))
        n_rand = n_batch - n_unc - n_cov - n_prom

    unc_score = sigma
    cov_score = nov
    # ascending argsort must return the BEST candidates first:
    #   minimize -> smallest mu first (low gbsa = stronger binding)
    #   maximize -> largest mu first
    # NOTE (2026-09-27): this line used to read `-mu if minimize else mu`, which
    # inverted the "promising" bucket (it selected the worst-predicted
    # sequences). Historical AL runs used the inverted version; the AL≈random
    # conclusion should be re-verified with this fix.
    prom_score = mu if minimize else -mu

    selected: set[int] = set()

    def _take(order, k):
        got = []
        for i in order:
            if i not in selected:
                selected.add(int(i))
                got.append(int(i))
                if len(got) >= k:
                    break
        return got

    _take(np.argsort(-unc_score), n_unc)   # highest uncertainty first
    _take(np.argsort(-cov_score), n_cov)   # most novel first
    _take(np.argsort(prom_score), n_prom)  # lowest mu (minimize) first

    remaining = [i for i in range(M) if i not in selected]
    n_rand = min(n_rand, len(remaining))
    for i in rng.choice(remaining, size=n_rand, replace=False):
        selected.add(int(i))

    # If dedup left us short, backfill by uncertainty.
    if len(selected) < n_batch:
        for i in np.argsort(-unc_score):
            if len(selected) >= n_batch:
                break
            selected.add(int(i))

    return sorted(selected)

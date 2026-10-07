"""Shared regression metrics (rank-based Spearman to avoid Windows scipy crash)."""

from __future__ import annotations

import numpy as np


def spearman(a, b) -> float:
    from scipy.stats import rankdata, pearsonr
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(pearsonr(rankdata(a), rankdata(b))[0])


def mae(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b)))


def rmse(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))

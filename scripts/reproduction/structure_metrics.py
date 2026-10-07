"""Metrics for a NUPACK pairs matrix, whose diagonal is unpaired probability."""
from __future__ import annotations

import numpy as np


def target_structure_metrics(pairs, target_partner, mutation_positions):
    """Return dimensional ensemble defect and per-site correctness probability.

    Partners are zero-based; -1 means unpaired. Mutation positions are one-based.
    NUPACK includes unpaired probability on the diagonal, so row sums cannot
    be used as pairing probabilities. Tiny floating point error is clipped.
    """
    matrix = np.asarray(pairs, dtype=float)
    n = len(target_partner)
    if matrix.shape != (n, n) or not np.isfinite(matrix).all():
        raise ValueError("invalid pair probability matrix")
    if np.any(matrix < -1e-8) or np.any(matrix > 1 + 1e-8):
        raise ValueError("probability outside [0, 1]")
    correct = np.empty(n, dtype=float)
    for i, j in enumerate(target_partner):
        if j < -1 or j >= n:
            raise ValueError("invalid target partner")
        correct[i] = matrix[i, i if j == -1 else j]
    correct = np.clip(correct, 0, 1)
    positions = np.asarray(mutation_positions, dtype=int) - 1
    if np.any(positions < 0) or np.any(positions >= n):
        raise ValueError("mutation position outside sequence")
    return float(n - correct.sum()), correct[positions].tolist()

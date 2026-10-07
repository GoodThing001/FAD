"""Batch selection policies for the closed loop (interface v0.2.0).

Consumes ONLY the surrogate's `Prediction` (pred_gbsa, optional sigma) plus
sequences; never touches labels. Every policy receives the same availability set
(labelled, previously selected and awaiting-label sequences are removed by the
caller through `exclude`).

Strategies (see §8.2/§8.3 of the implementation plan):
  random            uniform sample (the control every strategy must beat)
  greedy            lowest pred_gbsa first (direction-aware)
  greedy_diverse    top-M by pred_gbsa, then batch-internal max-min Hamming
  maxmin_coverage   purely coverage-driven max-min Hamming
  mixed_control     configurable buckets (uncertainty/coverage/promising/random)
  uncertainty_mixed alias of mixed_control that REQUIRES a usable sigma

Strict rules enforced here:
  - negative budget / negative ratios / wrong ratio length -> ValueError;
  - zero quota yields zero picks (never silently steals a candidate);
  - budget larger than the available pool shrinks explicitly and is reported;
  - ties are broken deterministically (pred_gbsa, then pool index).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

try:
    from .interface import (MUT_POS, Prediction, candidate_id, normalize)
except ImportError:  # plain-script invocation
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from scripts.loop.interface import (MUT_POS, Prediction, candidate_id,
                                        normalize)

DEFAULT_RATIOS = (45, 25, 20, 10)   # uncertainty, coverage, promising, random
STRATEGIES = ("random", "greedy", "greedy_diverse", "maxmin_coverage",
              "mixed_control", "uncertainty_mixed")
NOVELTY_CHUNK = 512


def _mutation_matrix(seqs: Sequence[str]) -> np.ndarray:
    out = np.zeros((len(seqs), len(MUT_POS)), dtype=np.int16)
    for i, s in enumerate(seqs):
        s = normalize(s)
        for j, p in enumerate(MUT_POS):
            out[i, j] = ord(s[p - 1])
    return out


def _min_distance_to(P: np.ndarray, L: np.ndarray, chunk: int = NOVELTY_CHUNK) -> np.ndarray:
    """Chunked min Hamming distance from each row of P to any row of L."""
    n = P.shape[0]
    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk):
        block = P[start:start + chunk]
        d = np.zeros((block.shape[0], L.shape[0]), dtype=np.int16)
        for j in range(block.shape[1]):
            d += (block[:, j][:, None] != L[None, :, j]).astype(np.int16)
        out[start:start + chunk] = d.min(axis=1)
    return out


def loo_min_distance(T: np.ndarray, chunk: int = NOVELTY_CHUNK) -> np.ndarray:
    """Minimal Hamming distance from each row of `T` to the *other* rows of `T`.

    Leave-one-out neighbour distance. Taking a plain minimum first would always
    return the self-distance 0 and make any downstream "far from training"
    threshold degenerate, which is exactly the §7.1 OOD rule input.
    """
    n = T.shape[0]
    out = np.full(n, np.inf, dtype=np.float64)
    if n < 2:
        return out
    big = np.iinfo(np.int16).max
    for start in range(0, n, chunk):
        block = T[start:start + chunk]
        d = np.zeros((block.shape[0], n), dtype=np.int16)
        for j in range(T.shape[1]):
            d += (block[:, j][:, None] != T[None, :, j]).astype(np.int16)
        idx = np.arange(block.shape[0])
        d[idx, start + idx] = big
        out[start:start + chunk] = d.min(axis=1)
    return out


def hamming_novelty(pool_seqs: Sequence[str], labeled_seqs: Sequence[str],
                    chunk: int = NOVELTY_CHUNK) -> np.ndarray:
    P = _mutation_matrix(pool_seqs)
    if len(labeled_seqs) == 0:
        return np.zeros(len(pool_seqs))
    L = _mutation_matrix(labeled_seqs)
    return _min_distance_to(P, L, chunk)


def _validate_budget_and_ratios(budget: int, ratios: Sequence[int],
                                available: int) -> tuple[int, list[int], str | None]:
    if budget < 0:
        raise ValueError(f"budget must be non-negative, got {budget}")
    if len(ratios) != 4:
        raise ValueError(f"ratios must have exactly 4 entries, got {len(ratios)}")
    if any(r < 0 for r in ratios):
        raise ValueError(f"ratios must be non-negative, got {list(ratios)}")
    if sum(ratios) == 0:
        raise ValueError("ratios must not sum to zero")
    effective = min(budget, available)
    note = None
    if effective < budget:
        note = f"pool exhausted: requested {budget}, using {effective}"
    return effective, [int(r) for r in ratios], note


def _scale(ratios: Sequence[int], budget: int) -> list[int]:
    total = float(sum(ratios))
    counts = [int(np.floor(budget * r / total)) for r in ratios]
    short = budget - sum(counts)
    order = np.argsort([-r for r in ratios])       # remainder to the largest buckets
    for k in range(short):
        counts[int(order[k % 4])] += 1
    return counts


def select_batch(
    pool_seqs: Sequence[str],
    prediction: Prediction,
    *,
    budget: int,
    strategy: str = "greedy",
    labeled_seqs: Sequence[str] = (),
    exclude: Iterable[str] = (),
    ratios: Sequence[int] = DEFAULT_RATIOS,
    direction: str = "minimize",
    seed: int = 0,
    diverse_multiplier: int = 5,
) -> dict:
    """Choose `budget` sequences from the available pool. Returns a ledger dict."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy '{strategy}'; choices: {list(STRATEGIES)}")
    if direction not in ("minimize", "maximize"):
        raise ValueError("direction must be 'minimize' or 'maximize'")
    if len(pool_seqs) != len(prediction):
        raise ValueError("pool size and prediction length differ")

    blocked = {normalize(s) for s in labeled_seqs} | {normalize(s) for s in exclude}
    available = np.array([i for i, s in enumerate(pool_seqs)
                          if normalize(s) not in blocked], dtype=int)
    effective, ratios, note = _validate_budget_and_ratios(budget, ratios, len(available))
    if effective == 0:
        return {"indices": [], "roles": {}, "strategy": strategy,
                "effective_budget": 0, "pool_size": len(pool_seqs),
                "available": int(len(available)), "note": note or "nothing available",
                "candidate_ids": []}

    mu = prediction.mu
    sign = 1.0 if direction == "minimize" else -1.0
    order_mu = available[np.lexsort((available, sign * mu[available]))]
    rng = np.random.RandomState(seed)
    labeled_sets = list(labeled_seqs)

    def batch_distance(candidates: np.ndarray, chosen: list[int]) -> np.ndarray:
        ref = list(labeled_sets) + [pool_seqs[i] for i in chosen]
        return hamming_novelty([pool_seqs[i] for i in candidates], ref)

    chosen: list[int] = []
    roles: dict[int, str] = {}

    if strategy == "random":
        chosen = [int(i) for i in rng.choice(available, size=effective, replace=False)]
        roles = {i: "random" for i in chosen}

    elif strategy == "greedy":
        chosen = [int(i) for i in order_mu[:effective]]
        roles = {i: "promising" for i in chosen}

    elif strategy == "greedy_diverse":
        m = min(len(available), max(effective, effective * diverse_multiplier))
        head = [int(i) for i in order_mu[:m]]
        chosen = [head[0]]
        rest = head[1:]
        while rest and len(chosen) < effective:
            dist = batch_distance(np.array(rest, dtype=int), chosen)
            pick = min(range(len(rest)),
                       key=lambda k: (-dist[k], sign * mu[rest[k]], rest[k]))
            chosen.append(rest.pop(pick))
        roles = {i: "greedy_diverse" for i in chosen}

    elif strategy == "maxmin_coverage":
        rest = [int(i) for i in available]
        while rest and len(chosen) < effective:
            dist = batch_distance(np.array(rest, dtype=int), chosen)
            pick = min(range(len(rest)),
                       key=lambda k: (-dist[k], sign * mu[rest[k]], rest[k]))
            chosen.append(rest.pop(pick))
        roles = {i: "coverage" for i in chosen}

    else:  # mixed_control / uncertainty_mixed
        if strategy == "uncertainty_mixed" and not prediction.has_sigma:
            raise ValueError("uncertainty_mixed requires sigma; this prediction has none "
                             "(use greedy/random/greedy_diverse/maxmin_coverage)")
        n_unc, n_cov, n_prom, n_rand = _scale(ratios, effective)
        if n_unc > 0 and not prediction.has_sigma:
            raise ValueError("ratios request uncertainty picks but prediction has no sigma")
        taken: set[int] = set()

        def take(order, k: int, role: str) -> None:
            got = 0
            for i in order:
                i = int(i)
                if i in taken or got >= k:
                    continue
                taken.add(i)
                chosen.append(i)
                roles[i] = role
                got += 1

        if n_unc > 0:
            take(available[np.argsort(-prediction.sigma[available])], n_unc, "uncertainty")
        if n_cov > 0:
            rest = [int(i) for i in available if int(i) not in taken]
            picked: list[int] = []
            while rest and len(picked) < n_cov:
                dist = batch_distance(np.array(rest, dtype=int), chosen + picked)
                k = min(range(len(rest)), key=lambda z: (-dist[z], sign * mu[rest[z]], rest[z]))
                picked.append(rest.pop(k))
            for i in picked:
                taken.add(i)
                chosen.append(i)
                roles[i] = "coverage"
        if n_prom > 0:
            take([int(i) for i in order_mu], n_prom, "promising")
        if n_rand > 0:
            rest = [int(i) for i in available if int(i) not in taken]
            k = min(n_rand, len(rest))
            if k > 0:
                for i in rng.choice(rest, size=k, replace=False):
                    taken.add(int(i))
                    chosen.append(int(i))
                    roles[int(i)] = "random"
        # explicit deficit fill (recorded, never silent) in the declared priority order
        if len(chosen) < effective:
            for i in order_mu:
                if len(chosen) >= effective:
                    break
                i = int(i)
                if i not in taken:
                    taken.add(i)
                    chosen.append(i)
                    roles[i] = "deficit_fill"

    chosen = chosen[:effective]
    return {
        "indices": sorted(int(i) for i in chosen),
        "roles": {int(i): roles.get(int(i), "unassigned") for i in chosen},
        "strategy": strategy,
        "effective_budget": len(chosen),
        "requested_budget": budget,
        "pool_size": len(pool_seqs),
        "available": int(len(available)),
        "direction": direction,
        "candidate_ids": [candidate_id(pool_seqs[i]) for i in chosen],
        "note": note,
    }

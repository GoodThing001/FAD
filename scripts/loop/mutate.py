"""Sequence mutation / candidate generation v1 (interface v0.2.0).

Works purely on sequences: the 13 allowed mutation sites are sampled against the
WT scaffold, deduplicated against every known sequence, and recorded with full
provenance:

    candidate_id = sha256(normalized sequence)[:16]
    parent_sequence, changed_positions_from_parent, n_from_parent (vs parent),
    n_from_wt (vs WT reference), generator, generation_seed, constraints_version

Two branches (plan §7.1):
  * global stratified generation by `n_from_wt` quotas (`generate_candidates`);
  * local refinement around measured-good parents (`mutate_from_parent`).
No labels, dock or Gap2 are consulted — generation is label-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

try:
    from .interface import (BASES, MUT_POS, WT_141, candidate_id,
                            changed_positions, n_from_wt as _n_from_wt, normalize)
except ImportError:  # plain-script invocation
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from scripts.loop.interface import (BASES, MUT_POS, WT_141, candidate_id,
                                        changed_positions, normalize)
    from scripts.loop.interface import n_from_wt as _n_from_wt

CONSTRAINTS_VERSION = "v1"


@dataclass
class CandidatePool:
    """Generated candidates with per-candidate provenance."""

    sequences: list[str] = field(default_factory=list)
    provenance: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.sequences)

    def to_rows(self) -> list[dict]:
        return [{"sequence": s, **p} for s, p in zip(self.sequences, self.provenance)]

    def mutation_histogram(self) -> dict[int, int]:
        hist: dict[int, int] = {}
        for p in self.provenance:
            hist[int(p["n_from_wt"])] = hist.get(int(p["n_from_wt"]), 0) + 1
        return dict(sorted(hist.items()))


def mutation_count(seq: str, reference: str = WT_141) -> int:
    return _n_from_wt(seq, reference)


def _gc(seq: str) -> float:
    s = normalize(seq)
    return (s.count("G") + s.count("C")) / len(s)


def _make_provenance(seq: str, parent: str, source: str, seed: int,
                     round_idx: int = 0) -> dict:
    ch = changed_positions(seq, parent)
    return {
        "candidate_id": candidate_id(seq),
        "parent_sequence": normalize(parent),
        "changed_positions_from_parent": list(ch),
        "n_from_parent": len(ch),
        "n_from_wt": _n_from_wt(seq),
        "source": source,
        "generator": source,
        "generation_seed": seed,
        "round": round_idx,
        "constraints_version": CONSTRAINTS_VERSION,
    }


def stratum_capacity(k: int) -> int:
    """Number of distinct sequences with exactly k substitutions (3 alternatives
    per site): C(13, k) * 3**k."""
    from math import comb
    if not 0 <= k <= len(MUT_POS):
        return 0
    return comb(len(MUT_POS), k) * (len(BASES) - 1) ** k


def _waterfill_plan(ks: Sequence[int], total: int) -> dict[int, int]:
    """Distribute `total` across strata `ks`, respecting each stratum's capacity."""
    caps = {k: stratum_capacity(k) for k in ks}
    if total > sum(caps.values()):
        raise ValueError(f"requested {total} candidates exceeds capacity "
                         f"{sum(caps.values())} of strata {list(ks)}")
    plan = {k: 0 for k in ks}
    remaining = total
    active = list(ks)
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        progressed = False
        for k in list(active):
            room = caps[k] - plan[k]
            give = min(room, share, remaining)
            if give > 0:
                plan[k] += give
                remaining -= give
                progressed = True
            if plan[k] >= caps[k]:
                active.remove(k)
        if not progressed:
            break
    return plan


def generate_candidates(
    n: int | None = None,
    *,
    quotas: dict[int, int] | None = None,
    reference: str = WT_141,
    min_mutations: int = 1,
    max_mutations: int = 5,
    exclude: Iterable[str] = (),
    seed: int = 0,
    max_gc_shift: float | None = None,
    max_attempts_factor: int = 200,
) -> CandidatePool:
    """Stratified global generation.

    Either pass `n` (uniform over [min_mutations, max_mutations]) or `quotas`
    (e.g. ``{8: 200, 10: 200, 12: 100}`` for explicit per-distance counts).
    """
    ref = normalize(reference)
    if quotas is not None:
        if not quotas or any(k < 1 or k > len(MUT_POS) for k in quotas):
            raise ValueError("quota keys must be within 1..13")
        if any(v < 0 for v in quotas.values()):
            raise ValueError("quota values must be non-negative")
        plan = {int(k): int(v) for k, v in quotas.items()}
        total = sum(plan.values())
    else:
        if n is None or n < 0:
            raise ValueError("provide a non-negative n or a quotas dict")
        if not 1 <= min_mutations <= max_mutations <= len(MUT_POS):
            raise ValueError("require 1 <= min_mutations <= max_mutations <= 13")
        total = n
        plan = _waterfill_plan(list(range(min_mutations, max_mutations + 1)), total)

    rng = np.random.RandomState(seed)
    excluded = {normalize(s) for s in exclude}
    ref_gc = _gc(ref)
    seen: set[str] = set()
    sequences: list[str] = []
    provenance: list[dict] = []
    rejected = {"excluded": 0, "duplicate": 0, "gc": 0}

    for k, want in plan.items():
        got = 0
        attempts = 0
        limit = max(1000, want * max_attempts_factor)
        while got < want and attempts < limit:
            attempts += 1
            positions = rng.choice(len(MUT_POS), size=k, replace=False)
            s = list(ref)
            for pi in positions:
                p = MUT_POS[int(pi)] - 1
                choices = [b for b in BASES if b != s[p]]
                s[p] = choices[int(rng.randint(len(choices)))]
            cand = "".join(s)
            if cand in excluded:
                rejected["excluded"] += 1
                continue
            if cand in seen:
                rejected["duplicate"] += 1
                continue
            if max_gc_shift is not None and abs(_gc(cand) - ref_gc) > max_gc_shift:
                rejected["gc"] += 1
                continue
            seen.add(cand)
            sequences.append(cand)
            provenance.append(_make_provenance(cand, ref, "global_mutate", seed))
            got += 1
        if got < want:
            raise RuntimeError(
                f"generated only {got}/{want} unique candidates at n_from_wt={k} "
                f"after {attempts} attempts; relax filters or shrink the quota")

    stats = {"requested": total, "generated": len(sequences),
             "unique": len(set(sequences)), "rejected": rejected,
             "n_from_wt_histogram": dict(sorted(
                 {k: sum(1 for p in provenance if p["n_from_wt"] == k)
                  for k in set(p["n_from_wt"] for p in provenance)}.items()))}
    return CandidatePool(sequences=sequences, provenance=provenance, stats=stats)


def mutate_from_parent(
    parent: str,
    n: int,
    *,
    n_mutations: int = 1,
    exclude: Iterable[str] = (),
    seed: int = 0,
    round_idx: int = 0,
) -> CandidatePool:
    """Local refinement: `n` unique neighbours at exactly `n_mutations` sites.

    `n_from_parent` counts changes relative to `parent`; `n_from_wt` always
    counts against the WT reference (they differ when the parent is itself
    mutated).
    """
    parent = normalize(parent)
    rng = np.random.RandomState(seed)
    excluded = {normalize(s) for s in exclude} | {parent}
    seen: set[str] = set()
    sequences: list[str] = []
    provenance: list[dict] = []
    attempts = 0
    limit = max(1000, n * 500)
    while len(sequences) < n and attempts < limit:
        attempts += 1
        positions = rng.choice(len(MUT_POS), size=n_mutations, replace=False)
        s = list(parent)
        for pi in positions:
            p = MUT_POS[int(pi)] - 1
            choices = [b for b in BASES if b != s[p]]
            s[p] = choices[int(rng.randint(len(choices)))]
        cand = "".join(s)
        if cand in excluded or cand in seen:
            continue
        seen.add(cand)
        sequences.append(cand)
        provenance.append(_make_provenance(cand, parent, "local_mutate", seed, round_idx))
    if len(sequences) < n:
        raise RuntimeError(f"local generation produced only {len(sequences)}/{n}")
    return CandidatePool(sequences=sequences, provenance=provenance,
                         stats={"requested": n, "generated": len(sequences),
                                "unique": len(seen), "branch": "local"})


def dedupe_against(pool: CandidatePool, known: Iterable[str]) -> CandidatePool:
    known_set = {normalize(s) for s in known}
    keep = [i for i, s in enumerate(pool.sequences) if s not in known_set]
    stats = dict(pool.stats)
    stats["deduped_out"] = len(pool.sequences) - len(keep)
    return CandidatePool(sequences=[pool.sequences[i] for i in keep],
                         provenance=[pool.provenance[i] for i in keep],
                         stats=stats)

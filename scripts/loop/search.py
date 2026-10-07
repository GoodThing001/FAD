"""Iterative mutation search layer (SEARCH_INTERFACE_VERSION 0.1.1).

Design doc: docs/项目记录/突变序列迭代搜索算法与代理接口实施方案_20261002.md

The search layer sits between the frozen surrogate interface (v0.2.0) and the
batch-selection loop.  Its inner loop spends only *compute* budget:

    pick parents -> mutate -> surrogate.predict -> keep parents  (multi-generation)

It never calls `Oracle.query` and never writes predictions as real labels; the
outer loop (`scripts/loop/loop.py`) keeps consuming the real GBSA budget
unchanged (export -> PARKED -> ingest -> retrain).

The only model capability the search layer needs is ordered batch prediction
(`SequenceScorer.score`).  Everything else comes from `SearchContext`; policies
must not read private surrogate attributes.  Any change to the dataclasses or
protocol signatures must bump `SEARCH_INTERFACE_VERSION` and
`tests/test_search_contract.py`.
"""

from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

try:
    from .interface import (MUT_POS, WT_141, Prediction, Surrogate,
                            candidate_id, normalize, validate_sequences)
    from .selection import _min_distance_to, _mutation_matrix, loo_min_distance
except ImportError:  # plain-script invocation
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from scripts.loop.interface import (MUT_POS, WT_141, Prediction, Surrogate,
                                        candidate_id, normalize, validate_sequences)
    from scripts.loop.selection import (_min_distance_to, _mutation_matrix,
                                        loo_min_distance)

SEARCH_INTERFACE_VERSION = "0.1.1"

#: terminal stop reasons a policy may report
STOP_REASONS = (
    "budget_exhausted",      # max_unique_predictions reached
    "space_exhausted",       # no new feasible candidate can be generated
    "neighbors_exhausted",   # no parent has an un-evaluated feasible neighbour
    "stagnation",            # max_stagnant_generations without a new best
    "max_generations",       # generation cap reached
    "local_optima",          # every hill-climb walk terminated by no improvement
    "interrupted",           # test-only hard stop (state stays resumable)
    "no_seeds",              # no eligible initial seed could be formed
)

PARENT_EVIDENCE = ("measured", "predicted_only", "none")


class BudgetExceededError(ValueError):
    """Raised when a score() call would consume new unique predictions over budget."""


@dataclass(frozen=True)
class SearchContext:
    """Frozen description of one search (one outer-loop round).

    `measured_sequences` / `measured_gbsa` are the real labelled training set at
    search start (used for seed eligibility and the training-only OOD rule).
    `forbidden_sequences` must contain every sequence that may not be proposed
    (the whole canonical table, historical batches, QC failures, ...).  Measured
    sequences may be *parents* even though they are forbidden from re-export.

    `data_sha256` is recorded into the search snapshot for provenance; it is the
    caller's canonical-table hash (empty for pure mock searches).
    """

    search_id: str
    outer_round: int
    measured_sequences: tuple[str, ...]
    measured_gbsa: tuple[float, ...]
    forbidden_sequences: frozenset[str]
    wt_sequence: str = WT_141
    mutable_positions_1based: tuple[int, ...] = MUT_POS
    allowed_n_from_wt: tuple[int, int] = (4, 13)
    seed: int = 0
    max_unique_predictions: int = 256
    max_generations: int = 8
    constraints_version: str = "v1"
    data_sha256: str = ""
    archive_dir: str = ""

    def __post_init__(self) -> None:
        if not self.search_id:
            raise ValueError("search_id must be non-empty")
        if len(self.measured_sequences) != len(self.measured_gbsa):
            raise ValueError("measured_sequences and measured_gbsa must have equal length")
        if not np.isfinite(np.asarray(self.measured_gbsa, dtype=float)).all():
            raise ValueError("measured_gbsa contains non-finite values")
        lo, hi = self.allowed_n_from_wt
        if not (0 <= lo <= hi <= len(self.mutable_positions_1based)):
            raise ValueError(
                f"allowed_n_from_wt must satisfy 0 <= lo <= hi <= "
                f"{len(self.mutable_positions_1based)}, got ({lo}, {hi})")
        if self.max_unique_predictions < 1:
            raise ValueError("max_unique_predictions must be >= 1")
        if self.max_generations < 1:
            raise ValueError("max_generations must be >= 1")
        if not self.mutable_positions_1based:
            raise ValueError("mutable_positions_1based must be non-empty")
        if any(p < 1 or p > len(self.wt_sequence)
               for p in self.mutable_positions_1based):
            raise ValueError("mutable positions outside the sequence length")
        # normalize silently-lowered inputs so fingerprints compare cleanly
        object.__setattr__(self, "measured_sequences",
                           tuple(normalize(s) for s in self.measured_sequences))
        object.__setattr__(self, "forbidden_sequences",
                           frozenset(normalize(s) for s in self.forbidden_sequences))
        object.__setattr__(self, "wt_sequence", normalize(self.wt_sequence))

    def fingerprint(self) -> dict:
        """Identity fields that a resume must match exactly."""
        return {
            "search_id": self.search_id,
            "outer_round": self.outer_round,
            "seed": self.seed,
            "measured_sha256": _seqs_sha256(self.measured_sequences),
            "measured_values_sha256": hashlib.sha256(
                np.asarray(self.measured_gbsa, dtype='<f8').tobytes()).hexdigest(),
            "wt_sequence": self.wt_sequence,
            "mutable_positions_1based": list(self.mutable_positions_1based),
            "forbidden_sha256": _seqs_sha256(sorted(self.forbidden_sequences)),
            "allowed_n_from_wt": list(self.allowed_n_from_wt),
            "max_unique_predictions": self.max_unique_predictions,
            "max_generations": self.max_generations,
            "constraints_version": self.constraints_version,
            "data_sha256": self.data_sha256,
        }

    def to_dict(self) -> dict:
        return {
            "search_id": self.search_id,
            "outer_round": self.outer_round,
            "measured_sequences": list(self.measured_sequences),
            "measured_gbsa": [float(v) for v in self.measured_gbsa],
            "forbidden_sequences": sorted(self.forbidden_sequences),
            "wt_sequence": self.wt_sequence,
            "mutable_positions_1based": list(self.mutable_positions_1based),
            "allowed_n_from_wt": list(self.allowed_n_from_wt),
            "seed": self.seed,
            "max_unique_predictions": self.max_unique_predictions,
            "max_generations": self.max_generations,
            "constraints_version": self.constraints_version,
            "data_sha256": self.data_sha256,
            "archive_dir": self.archive_dir,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SearchContext":
        return cls(
            search_id=d["search_id"],
            outer_round=int(d["outer_round"]),
            measured_sequences=tuple(d["measured_sequences"]),
            measured_gbsa=tuple(float(v) for v in d["measured_gbsa"]),
            forbidden_sequences=frozenset(d["forbidden_sequences"]),
            wt_sequence=d["wt_sequence"],
            mutable_positions_1based=tuple(int(p) for p in d["mutable_positions_1based"]),
            allowed_n_from_wt=(int(d["allowed_n_from_wt"][0]),
                               int(d["allowed_n_from_wt"][1])),
            seed=int(d["seed"]),
            max_unique_predictions=int(d["max_unique_predictions"]),
            max_generations=int(d["max_generations"]),
            constraints_version=d["constraints_version"],
            data_sha256=d.get("data_sha256", ""),
            archive_dir=d.get("archive_dir", ""),
        )


@dataclass(frozen=True)
class ScoredCandidate:
    """One scored sequence with full search provenance.

    Every intermediate score and every eliminated child is recorded (the
    evaluated log), not just the winners, so prediction drift across the search
    stays auditable.
    """

    candidate_id: str
    sequence: str
    search_id: str
    outer_round: int
    generation: int
    parent_candidate_id: str | None
    parent_evidence: str            # measured | predicted_only | none
    operator: str                   # seed | one_hop | two_hop | global_restart
    n_from_parent: int
    n_from_wt: int
    changed_positions_from_parent: tuple[int, ...]
    pred_gbsa: float
    sigma_gbsa: float | None
    uncertainty_kind: str | None
    model_id: str
    feature_version: str
    ood_flag: bool
    nearest_train_hamming: float
    constraints_version: str
    rng_seed: int
    score_time: float

    def __post_init__(self) -> None:
        if self.parent_evidence not in PARENT_EVIDENCE:
            raise ValueError(f"parent_evidence must be one of {PARENT_EVIDENCE}")
        if not np.isfinite(self.pred_gbsa):
            raise ValueError("pred_gbsa must be finite")
        if self.sigma_gbsa is not None and (not np.isfinite(self.sigma_gbsa)
                                            or self.sigma_gbsa < 0):
            raise ValueError("sigma_gbsa must be finite and non-negative when given")

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "sequence": self.sequence,
            "search_id": self.search_id,
            "outer_round": self.outer_round,
            "generation": self.generation,
            "parent_candidate_id": self.parent_candidate_id,
            "parent_evidence": self.parent_evidence,
            "operator": self.operator,
            "n_from_parent": self.n_from_parent,
            "n_from_wt": self.n_from_wt,
            "changed_positions_from_parent": list(self.changed_positions_from_parent),
            "pred_gbsa": float(self.pred_gbsa),
            "sigma_gbsa": (None if self.sigma_gbsa is None else float(self.sigma_gbsa)),
            "uncertainty_kind": self.uncertainty_kind,
            "model_id": self.model_id,
            "feature_version": self.feature_version,
            "ood_flag": bool(self.ood_flag),
            "nearest_train_hamming": (
                None if not np.isfinite(self.nearest_train_hamming)
                else float(self.nearest_train_hamming)),
            "constraints_version": self.constraints_version,
            "rng_seed": int(self.rng_seed),
            "score_time": float(self.score_time),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScoredCandidate":
        h = d.get("nearest_train_hamming")
        return cls(
            candidate_id=d["candidate_id"],
            sequence=d["sequence"],
            search_id=d["search_id"],
            outer_round=int(d["outer_round"]),
            generation=int(d["generation"]),
            parent_candidate_id=d.get("parent_candidate_id"),
            parent_evidence=d["parent_evidence"],
            operator=d["operator"],
            n_from_parent=int(d["n_from_parent"]),
            n_from_wt=int(d["n_from_wt"]),
            changed_positions_from_parent=tuple(d["changed_positions_from_parent"]),
            pred_gbsa=float(d["pred_gbsa"]),
            sigma_gbsa=(None if d.get("sigma_gbsa") is None else float(d["sigma_gbsa"])),
            uncertainty_kind=d.get("uncertainty_kind"),
            model_id=d["model_id"],
            feature_version=d["feature_version"],
            ood_flag=bool(d["ood_flag"]),
            nearest_train_hamming=(float("inf") if h is None else float(h)),
            constraints_version=d["constraints_version"],
            rng_seed=int(d["rng_seed"]),
            score_time=float(d["score_time"]),
        )


@dataclass(frozen=True)
class CandidateGroup:
    """One proposed candidate group (1 or 2 groups; never padded to two)."""

    group_id: str
    group_role: str                 # predicted_best | diverse_backup
    prototype_id: str
    member_ids: tuple[str, ...]
    within_group_distance_summary: str
    group_selection_rule: str

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "group_role": self.group_role,
            "prototype_id": self.prototype_id,
            "member_ids": list(self.member_ids),
            "within_group_distance_summary": self.within_group_distance_summary,
            "group_selection_rule": self.group_selection_rule,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CandidateGroup":
        return cls(
            group_id=d["group_id"],
            group_role=d["group_role"],
            prototype_id=d["prototype_id"],
            member_ids=tuple(d["member_ids"]),
            within_group_distance_summary=d["within_group_distance_summary"],
            group_selection_rule=d["group_selection_rule"],
        )


@dataclass(frozen=True)
class SearchResult:
    """Search output: a full scored archive plus 1-2 candidate groups.

    `ranked_candidates` excludes measured seeds (they must never be re-exported);
    all *new* predicted sequences are ranked by pred_gbsa ascending.
    """

    ranked_candidates: tuple[ScoredCandidate, ...]
    groups: tuple[CandidateGroup, ...]
    n_unique_predicted: int
    stop_reason: str
    search_state_path: str

    def __post_init__(self) -> None:
        if self.stop_reason not in STOP_REASONS:
            raise ValueError(f"unknown stop_reason {self.stop_reason!r}")


class SearchPolicy(Protocol):
    """One search algorithm. Implementations must be deterministic given the
    context seed and must checkpoint into `context.archive_dir` after every
    generation (crash-resume must reproduce the uninterrupted run)."""

    name: str

    def search(self, context: SearchContext,
               scorer: "SequenceScorer") -> SearchResult: ...


@runtime_checkable
class SequenceScorer(Protocol):
    """Ordered batch prediction with a hard unique-prediction budget.

    Implementations count only *first-time* sequences against the budget,
    validate batch alignment and finiteness, and isolate caches per model
    snapshot.
    """

    def score(self, sequences: Sequence[str]) -> Prediction: ...

    @property
    def n_used(self) -> int: ...

    @property
    def remaining(self) -> int: ...


def _seqs_sha256(seqs) -> str:
    import hashlib
    return hashlib.sha256("\n".join(sorted(normalize(s) for s in seqs))
                          .encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# training-only OOD helpers (mirror of loop.py §7.1, duplicated on purpose so
# the search layer never imports loop.py; the equivalence is locked by a test)
# --------------------------------------------------------------------------

def training_ood_info(train_seqs) -> dict:
    """n_from_wt range + 95th percentile of training LOO nearest-neighbour
    Hamming distances. Training sequences only; no labels involved."""
    if not list(train_seqs):
        return {"n_from_wt_range": None, "hamming_threshold": None}
    train = [normalize(s) for s in train_seqs]
    train_n = [sum(1 for p in MUT_POS if s[p - 1] != WT_141[p - 1]) for s in train]
    d_train = loo_min_distance(_mutation_matrix(train))
    finite = d_train[np.isfinite(d_train)]
    thresh = float(np.percentile(finite, 95)) if finite.size else float("inf")
    return {"n_from_wt_range": [int(min(train_n)), int(max(train_n))],
            "hamming_threshold": thresh}


def ood_flags(pool_seqs, train_seqs) -> np.ndarray:
    """Binary OOD flags: n_from_wt outside the training range, or nearest-train
    Hamming distance above the training LOO 95th percentile."""
    pool = [normalize(s) for s in pool_seqs]
    if not list(train_seqs) or not pool:
        return np.zeros(len(pool), dtype=bool)
    info = training_ood_info(train_seqs)
    lo, hi = info["n_from_wt_range"]
    thresh = info["hamming_threshold"]
    pool_n = [sum(1 for p in MUT_POS if s[p - 1] != WT_141[p - 1]) for s in pool]
    out_of_range = [(n < lo) or (n > hi) for n in pool_n]
    d_pool = _min_distance_to(_mutation_matrix(pool), _mutation_matrix(list(train_seqs)))
    return np.array(out_of_range, dtype=bool) | (d_pool > thresh)


def nearest_train_hamming(pool_seqs, train_seqs) -> np.ndarray:
    """Min Hamming distance (13 sites) from each pool sequence to the training
    set; +inf when the training set is empty."""
    pool = [normalize(s) for s in pool_seqs]
    train = [normalize(s) for s in train_seqs]
    if not train:
        return np.full(len(pool), np.inf)
    return _min_distance_to(_mutation_matrix(pool), _mutation_matrix(train))


# --------------------------------------------------------------------------
# scorer adapter over the frozen surrogate
# --------------------------------------------------------------------------

class SurrogateScorer:
    """Thin adapter: fitted Surrogate -> SequenceScorer with cache + OOD tags.

    - validates every batch (order alignment, finite mu, provenance records);
    - caches by normalized sequence; the cache carries a single model snapshot
      (model_id + feature_version) and a mismatching model is an error, so a
      replaced model can never silently reuse old predictions;
    - counts only first-time sequences against `max_unique_predictions` and
      raises `BudgetExceededError` past the cap;
    - attaches `ood_flag` / `nearest_train_hamming` computed from the fixed
      pre-search training set (never updated with search-generated sequences).
    """

    def __init__(self, surrogate, context: SearchContext, *,
                 cache=None, ood_train_sequences=None,
                 model_id: str | None = None, feature_version: str | None = None):
        from .search_archive import PredictionCache
        self.surrogate = surrogate
        self.context = context
        self.cache = cache if cache is not None else PredictionCache()
        self._ood_train = list(ood_train_sequences or context.measured_sequences)
        self._ood_matrix = _mutation_matrix(self._ood_train)
        self._ood_info = training_ood_info(self._ood_train) if self._ood_train else None
        self._ood_tags: dict[str, tuple[bool, float]] = {}
        self.prediction_seconds = 0.0
        self.prediction_calls = 0
        self._declared_model_id = model_id
        self._declared_feature_version = feature_version
        # provenance observed at the first scored batch (recorded into cache)
        self._seen_model_id: str | None = None
        self._seen_feature_version: str | None = None
        if self.cache.provenance is not None:
            self._seen_model_id, self._seen_feature_version = self.cache.provenance
        if self._declared_model_id and self._seen_model_id \
                and self._declared_model_id != self._seen_model_id:
            raise ValueError(
                "surrogate model snapshot does not match the archive cache "
                f"({self._declared_model_id!r} vs cached {self._seen_model_id!r}); "
                "a replaced model must start a new archive (cache key must change)")

    @property
    def n_used(self) -> int:
        return len(self.cache)

    @property
    def remaining(self) -> int:
        return max(0, self.context.max_unique_predictions - self.n_used)

    @property
    def model_id(self) -> str:
        return self._seen_model_id or self._declared_model_id or \
            str(getattr(self.surrogate, "model_id", "unset"))

    @property
    def feature_version(self) -> str:
        return self._seen_feature_version or self._declared_feature_version or \
            str(getattr(self.surrogate, "feature_version", "unset"))

    def score(self, sequences: Sequence[str]) -> Prediction:
        seqs = validate_sequences(sequences, "search score sequences",
                                  check_scaffold=True)
        # keep input order, drop within-batch duplicates (repeats never cost budget)
        batch = list(dict.fromkeys(seqs))
        new = [s for s in batch if s not in self.cache]
        if len(new) > self.remaining:
            raise BudgetExceededError(
                f"prediction budget exceeded: {len(new)} new sequence(s) but "
                f"{self.remaining} remaining (max {self.context.max_unique_predictions})")
        if new:
            t0 = time.perf_counter()
            pred = self.surrogate.predict(new)
            elapsed = time.perf_counter() - t0
            self.prediction_seconds += elapsed
            self.prediction_calls += 1
            if len(pred.mu) != len(new):
                raise ValueError(
                    f"surrogate returned {len(pred.mu)} predictions for "
                    f"{len(new)} sequences; the scorer requires strict alignment")
            model_id = str(getattr(pred, "model_id", None)
                           or self._declared_model_id
                           or getattr(self.surrogate, "model_id", "unset"))
            feature_version = str(getattr(pred, "feature_version", None)
                                  or self._declared_feature_version
                                  or getattr(self.surrogate, "feature_version", "unset"))
            sigmas = (list(pred.sigma) if pred.has_sigma
                      else [None] * len(pred.mu))
            for s, mu, sig in zip(new, pred.mu, sigmas):
                self.cache.put(s, float(mu), sig, pred.uncertainty_kind,
                               model_id, feature_version, elapsed)
            self._seen_model_id = self._seen_model_id or model_id
            self._seen_feature_version = self._seen_feature_version or feature_version

        self._tag_sequences(seqs)
        mu = np.array([float(self.cache.mu(s)) for s in seqs])
        sigs = [self.cache.sigma(s) for s in seqs]
        if any(s is not None for s in sigs):
            sigma = np.array([s if s is not None else 0.0 for s in sigs])
            kind = next((self.cache.kind(s) for s in seqs if self.cache.kind(s)), None)
            return Prediction(mu=mu, sigma=sigma, uncertainty_kind=kind,
                              model_id=self.model_id,
                              feature_version=self.feature_version)
        return Prediction(mu=mu, model_id=self.model_id,
                          feature_version=self.feature_version)

    def _tag_sequences(self, seqs) -> None:
        missing = list(dict.fromkeys(s for s in seqs if s not in self._ood_tags))
        if not missing:
            return
        if self._ood_info is None:
            for s in missing:
                self._ood_tags[s] = (False, float("inf"))
            return
        distances = _min_distance_to(_mutation_matrix(missing), self._ood_matrix)
        lo, hi = self._ood_info["n_from_wt_range"]
        threshold = self._ood_info["hamming_threshold"]
        for s, distance in zip(missing, distances):
            n = sum(s[p - 1] != WT_141[p - 1] for p in MUT_POS)
            self._ood_tags[s] = (bool(n < lo or n > hi or distance > threshold),
                                 float(distance))

    def candidate_for(self, s: str, *, generation: int, parent_candidate_id: str | None,
                      parent_evidence: str, operator: str, n_from_parent: int,
                      changed_positions_from_parent: Sequence[int] = (),
                      rng_seed: int = 0) -> ScoredCandidate:
        """Build a ScoredCandidate from cache values + OOD tags (post-score)."""
        s = normalize(s)
        if s not in self.cache:
            raise ValueError(f"sequence not scored yet: {s[:20]}...")
        entry = self.cache.get(s)
        self._tag_sequences([s])
        ood, nearest = self._ood_tags[s]
        mut_pos = self.context.mutable_positions_1based
        wt = self.context.wt_sequence
        return ScoredCandidate(
            candidate_id=candidate_id(s),
            sequence=s,
            search_id=self.context.search_id,
            outer_round=self.context.outer_round,
            generation=generation,
            parent_candidate_id=parent_candidate_id,
            parent_evidence=parent_evidence,
            operator=operator,
            n_from_parent=n_from_parent,
            n_from_wt=sum(1 for p in mut_pos if s[p - 1] != wt[p - 1]),
            changed_positions_from_parent=tuple(changed_positions_from_parent),
            pred_gbsa=float(entry["mu"]),
            sigma_gbsa=entry["sigma"],
            uncertainty_kind=entry.get("uncertainty_kind"),
            model_id=entry["model_id"],
            feature_version=entry["feature_version"],
            ood_flag=ood,
            nearest_train_hamming=nearest,
            constraints_version=self.context.constraints_version,
            rng_seed=rng_seed,
            score_time=float(entry["score_time"]),
        )

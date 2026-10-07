"""Frozen interface contract between the surrogate model and the downstream loop.

INTERFACE_VERSION 0.2.0 (2026-09-30). Breaking change vs 0.1.0:
  - `Prediction.sigma` is now OPTIONAL (`None` = uncertainty not provided). It is
    never filled with 0 to fake certainty; when provided it must be finite,
    non-negative, and labelled with `uncertainty_kind`.
  - `Prediction.mu` is renamed in the export layer to `pred_gbsa`; it stays on the
    GBSA scale (kcal/mol).
  - `validate_sequences()` now enforces the fixed scaffold: every position
    outside the 13 allowed mutation sites must equal the WT reference.
  - `Oracle` exposes `eligible_to_query` / `already_labeled` and reveals truth
    only through `query(batch_id, sequences)`.

Any change to the dataclasses, protocol signatures, or the sequence contract must
bump INTERFACE_VERSION and `tests/test_loop_contract.py`.

Target policy (decided by the project)
--------------------------------------
The surrogate predicts **standalone GBSA** (`gbsa`, MM-GBSA binding score, lower
= stronger binding by project convention). The historical weighted label
`label = 0.961*Gap2 + 0.039*gbsa` is NOT a training target, acquisition score,
final candidate score, or success metric: its weights have not been validated.
`TargetSpec.label_policy = "gbsa_only"` records that decision.

Downstream modules (mutation, selection, oracle, loop) talk to the surrogate ONLY
through `Surrogate.fit` / `Surrogate.predict`, and to labels ONLY through
`Oracle.query`. Labels of unqueried sequences must never be visible.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

INTERFACE_VERSION = "0.2.0"

MUT_POS: tuple[int, ...] = (30, 31, 32, 33, 89, 90, 91, 92, 127, 128, 129, 130, 131)
BASES = "ACGU"
WT_141 = ("UAUCCUUCGGGGCAGGGUGGAAAUCCCGACCGGCGGUAGUAAAGCACAUUUGCUUUAGAGCCC"
          "GUGACCCGUGUGCAUAAGCACGCGGUGGAUUCAGUUUAAGCUGAAGCCGACAGUGAAAGUCUGGA"
          "UGGGAGAAGGAUG")
SEQ_LEN = len(WT_141)
TARGET_COLUMN = "gbsa"
PREDICTION_COLUMN = "pred_gbsa"

#: batch / candidate bookkeeping vocabulary shared by loop, oracle and runner
BATCH_STATUS = ("selected", "awaiting_labels", "partially_labeled", "labeled",
                "failed", "retrained")
MEASUREMENT_STATUS = ("pass", "fail", "pending")


@dataclass(frozen=True)
class TargetSpec:
    """What the surrogate predicts. Frozen: standalone gbsa only."""

    name: str = TARGET_COLUMN
    unit: str = "kcal/mol"
    direction: str = "minimize"          # lower gbsa = stronger binding
    label_policy: str = "gbsa_only"      # weighted label intentionally NOT used
    weighted_label_used: bool = False
    report_column: str = PREDICTION_COLUMN
    note: str = ("0.961*Gap2 + 0.039*gbsa weights are unvalidated; the loop "
                 "consumes standalone gbsa until the label policy is re-decided")

    def describe(self) -> dict:
        return {
            "name": self.name,
            "unit": self.unit,
            "direction": self.direction,
            "label_policy": self.label_policy,
            "weighted_label_used": self.weighted_label_used,
            "report_column": self.report_column,
            "note": self.note,
        }


@dataclass(frozen=True)
class Prediction:
    """Surrogate output on the GBSA scale.

    mu             predicted gbsa (kcal/mol), shape (n,), finite, order-aligned
    sigma          optional non-negative finite uncertainty, shape (n,)
    uncertainty_kind  e.g. "ensemble_spread" (never a calibrated CI unless said)
    model_id / feature_version  provenance for the batch ledger
    """

    mu: np.ndarray
    sigma: np.ndarray | None = None
    uncertainty_kind: str | None = None
    feature_version: str = "seq303"
    model_id: str = "unset"
    members: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        mu = np.asarray(self.mu, dtype=np.float64).reshape(-1)
        if mu.size == 0:
            raise ValueError("mu is empty")
        if not np.isfinite(mu).all():
            raise ValueError("mu contains non-finite values")
        object.__setattr__(self, "mu", mu)
        if self.sigma is None:
            if self.uncertainty_kind is not None:
                raise ValueError("uncertainty_kind given without sigma")
            return
        sigma = np.asarray(self.sigma, dtype=np.float64).reshape(-1)
        if sigma.shape != mu.shape:
            raise ValueError(f"mu/sigma shape mismatch: {mu.shape} vs {sigma.shape}")
        if not np.isfinite(sigma).all():
            raise ValueError("sigma contains non-finite values")
        if (sigma < 0).any():
            raise ValueError("sigma must be non-negative")
        if not self.uncertainty_kind:
            raise ValueError("sigma provided without uncertainty_kind")
        object.__setattr__(self, "sigma", sigma)

    def __len__(self) -> int:
        return int(self.mu.shape[0])

    @property
    def has_sigma(self) -> bool:
        return self.sigma is not None


@runtime_checkable
class Surrogate(Protocol):
    """Model contract. Implementations must be deterministic given their seed."""

    feature_version: str
    model_id: str

    def fit(self, sequences: Sequence[str], y: np.ndarray) -> "Surrogate":
        """Fit on labeled (sequence, gbsa) pairs. Returns self."""

    def predict(self, sequences: Sequence[str]) -> Prediction:
        """Predict pred_gbsa for arbitrary 141-nt sequences (no labels needed)."""

    def save(self, path) -> None:
        """Persist the fitted model."""


@runtime_checkable
class Oracle(Protocol):
    """Label source contract (table / parked batch / external MD pipeline)."""

    def eligible_to_query(self, sequences: Sequence[str]) -> np.ndarray:
        """Mask: sequences the oracle is allowed to label at all."""

    def already_labeled(self, sequences: Sequence[str]) -> np.ndarray:
        """Mask: sequences whose label has already been revealed."""

    def query(self, batch_id: str, sequences: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Reveal labels for a batch: (values, status_mask).

        values: gbsa or NaN;  status_mask: True where a label is available now.
        This is the ONLY path that reveals truth.
        """


def candidate_id(seq: str, length: int = 16) -> str:
    """Stable candidate identifier: sha256 of the normalized sequence."""
    return hashlib.sha256(normalize(seq).encode()).hexdigest()[:length]


def normalize(seq: str) -> str:
    return str(seq).upper().replace("T", "U").strip()


def changed_positions(seq: str, reference: str = WT_141) -> tuple[int, ...]:
    s = normalize(seq)
    return tuple(p for p in MUT_POS if s[p - 1] != reference[p - 1])


def n_from_wt(seq: str, reference: str = WT_141) -> int:
    return len(changed_positions(seq, reference))


def validate_sequences(sequences: Sequence[str], context: str = "sequences",
                       reference: str = WT_141, check_scaffold: bool = True) -> list[str]:
    """Enforce the sequence contract.

    - 141 nt, ACGU alphabet, non-empty input;
    - every position outside the 13 allowed mutation sites must equal `reference`.
    """
    out: list[str] = []
    for i, s in enumerate(sequences):
        n = normalize(s)
        if len(n) != SEQ_LEN:
            raise ValueError(f"{context}[{i}] has length {len(n)}, expected {SEQ_LEN}")
        bad = sorted({c for c in n if c not in BASES})
        if bad:
            raise ValueError(f"{context}[{i}] has non-ACGU characters: {bad}")
        if check_scaffold:
            ref = normalize(reference)
            off = [j + 1 for j in range(SEQ_LEN)
                   if (j + 1) not in MUT_POS and n[j] != ref[j]]
            if off:
                raise ValueError(
                    f"{context}[{i}] changes positions outside the 13 allowed "
                    f"sites: {off[:5]}{'...' if len(off) > 5 else ''}"
                )
        out.append(n)
    if not out:
        raise ValueError(f"{context} is empty")
    return out

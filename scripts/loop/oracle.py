"""Label sources (oracles) for the closed loop (interface v0.2.0).

All oracles expose the same three entry points:

  eligible_to_query(seqs)  mask: may this oracle ever label the sequence?
  already_labeled(seqs)    mask: has the label already been revealed?
  query(batch_id, seqs)    -> (gbsa values, available mask)  # the ONLY reveal path

Implementations
---------------
TableOracle    retrospective bring-up over the canonical table. Truth is revealed
               only for sequences inside the declared pool; rows outside the
               pool (including the evaluation holdout) always return NaN, and a
               sequence is never labelled twice through different batches.
ParkOracle     prospective hand-off: writes a persistent batch CSV plus a manifest
               for the external MD/MM-GBSA pipeline and returns NaN. Labels come
               back through `ingest()`, which validates the batch, the recomputed
               `candidate_id`, the protocol allow-list and the QC status, persists
               pass **and** fail rows atomically, rejects non-finite/conflicting
               values, and is idempotent.
CommandOracle  prospective hook that runs an external command (not used in this
               phase; requires an agreed protocol and budget).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from .interface import (MEASUREMENT_STATUS, TARGET_COLUMN, candidate_id,
                            normalize)
except ImportError:  # plain-script invocation
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.loop.interface import (MEASUREMENT_STATUS, TARGET_COLUMN,
                                        candidate_id, normalize)

INGEST_COLUMNS = ("batch_id", "candidate_id", "sequence", TARGET_COLUMN,
                  "measurement_status", "protocol_id")


class TableOracle:
    """Retrospective oracle restricted to an explicit, hidden pool."""

    def __init__(self, df: pd.DataFrame, pool: Sequence[str],
                 sequence_col: str = "Sequence", target_col: str = TARGET_COLUMN,
                 holdout: Sequence[str] = ()):
        frame = df[[sequence_col, target_col]].copy()
        frame[sequence_col] = frame[sequence_col].map(normalize)
        if frame[sequence_col].duplicated().any():
            raise ValueError("label table contains duplicate sequences")
        values = frame[target_col].astype(float).to_numpy()
        if not np.isfinite(values).all():
            raise ValueError("label table contains non-finite gbsa values")
        self.table = dict(zip(frame[sequence_col], values))
        self.pool = {normalize(s) for s in pool}
        self.holdout = {normalize(s) for s in holdout}
        self.revealed: set[str] = set()
        self.queries: list[dict] = []

    # --- contract ---
    def eligible_to_query(self, sequences: Sequence[str]) -> np.ndarray:
        return np.array([(normalize(s) in self.pool)
                         and (normalize(s) in self.table)
                         and (normalize(s) not in self.holdout)
                         for s in sequences], dtype=bool)

    def already_labeled(self, sequences: Sequence[str]) -> np.ndarray:
        return np.array([normalize(s) in self.revealed for s in sequences], dtype=bool)

    def query(self, batch_id: str, sequences: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        seqs = [normalize(s) for s in sequences]
        eligible = self.eligible_to_query(seqs)
        values = np.full(len(seqs), np.nan, dtype=np.float64)
        for i, (s, ok) in enumerate(zip(seqs, eligible)):
            if ok:
                values[i] = self.table[s]
                self.revealed.add(s)
        self.queries.append({"batch_id": batch_id, "n": len(seqs),
                             "n_revealed": int(eligible.sum())})
        return values, eligible

    def peek(self, sequences: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Report labels already revealed (never reveals anything new)."""
        seqs = [normalize(s) for s in sequences]
        known = np.array([s in self.revealed for s in seqs], dtype=bool)
        values = np.array([self.table.get(s, np.nan) if k else np.nan
                           for s, k in zip(seqs, known)], dtype=np.float64)
        return values, known


class ParkOracle:
    """Prospective oracle with a persistent, idempotent batch hand-off.

    Ingest rules (plan §7 P0-2):
      * the file is validated **in full before anything is written**; a rejected row
        never leaves a partial side effect;
      * `candidate_id` must equal `sha256(normalized sequence)[:16]`;
      * `protocol_id` must be non-empty and, when an allow-list is configured, approved;
      * `measurement_status` must be `pass` (needs a finite `gbsa`) or `fail` (needs no
        value); both are persisted, so a QC failure survives a restart;
      * conflicting values for an already-accepted candidate are rejected and the
        confirmed value is left untouched; an identical re-submission is a duplicate.
    """

    PASS_FILE = "labels.csv"
    FAIL_FILE = "failures.csv"

    def __init__(self, park_dir: str | Path, approved_protocols: Sequence[str] = ()):
        self.root = Path(park_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.batches_dir = self.root / "batches"
        self.batches_dir.mkdir(parents=True, exist_ok=True)
        self.approved_protocols = [str(p).strip() for p in approved_protocols if str(p).strip()]
        self.pending: dict[str, float] = {}
        self.batch_of: dict[str, str] = {}          # candidate -> batch_id
        self.failed: set[str] = set()
        self.fail_batch_of: dict[str, str] = {}
        self._load_state()

    # --- persistence ---
    def _load_state(self) -> None:
        for bdir in sorted(self.batches_dir.glob("batch_*")):
            pass_path = bdir / self.PASS_FILE
            if pass_path.is_file():
                for _, row in pd.read_csv(pass_path).iterrows():
                    if str(row.get("measurement_status", "")).lower() == "pass" \
                            and pd.notna(row.get(TARGET_COLUMN)):
                        s = normalize(row["sequence"])
                        self.pending[s] = float(row[TARGET_COLUMN])
                        self.batch_of[s] = bdir.name.replace("batch_", "", 1)
            fail_path = bdir / self.FAIL_FILE
            if fail_path.is_file():
                for _, row in pd.read_csv(fail_path).iterrows():
                    s = normalize(row["sequence"])
                    self.failed.add(s)
                    self.fail_batch_of[s] = bdir.name.replace("batch_", "", 1)

    def _atomic_append(self, path: Path, rows: list[dict], header: list[str]) -> None:
        """Append rows by rewriting the file through a temp path + os.replace."""
        existing = pd.read_csv(path) if path.is_file() else pd.DataFrame(columns=header)
        combined = pd.concat([existing, pd.DataFrame(rows, columns=header)],
                             ignore_index=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        combined.to_csv(tmp, index=False)
        os.replace(tmp, path)

    # --- contract ---
    def eligible_to_query(self, sequences: Sequence[str]) -> np.ndarray:
        return np.ones(len(sequences), dtype=bool)

    def already_labeled(self, sequences: Sequence[str]) -> np.ndarray:
        return np.array([normalize(s) in self.pending for s in sequences], dtype=bool)

    def failed_sequences(self) -> set[str]:
        return set(self.failed)

    def all_batch_sequences(self) -> set[str]:
        """Every sequence ever written into a batch (pending, labelled or failed)."""
        out: set[str] = set()
        for bdir in sorted(self.batches_dir.glob("batch_*")):
            batch_csv = bdir / "batch.csv"
            if batch_csv.is_file():
                out.update(pd.read_csv(batch_csv)["sequence"].map(normalize).tolist())
        return out

    def batch_path(self, batch_id: str) -> Path:
        return self.batches_dir / f"batch_{batch_id}"

    def query(self, batch_id: str, sequences: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        seqs = [normalize(s) for s in sequences]
        bdir = self.batch_path(batch_id)
        bdir.mkdir(parents=True, exist_ok=True)
        batch_csv = bdir / "batch.csv"
        if not batch_csv.is_file():                 # never overwrite a parked batch
            pd.DataFrame({
                "batch_id": [batch_id] * len(seqs),
                "candidate_id": [candidate_id(s) for s in seqs],
                "sequence": seqs,
                "status": ["awaiting_labels"] * len(seqs),
            }).to_csv(batch_csv, index=False)
        values = np.array([self.pending.get(s, np.nan) for s in seqs], dtype=np.float64)
        available = np.array([s in self.pending for s in seqs], dtype=bool)
        return values, available

    def peek(self, sequences: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Report labels already returned (never writes a new batch)."""
        seqs = [normalize(s) for s in sequences]
        known = np.array([s in self.pending for s in seqs], dtype=bool)
        values = np.array([self.pending.get(s, np.nan) if k else np.nan
                           for s, k in zip(seqs, known)], dtype=np.float64)
        return values, known

    def _validate_ingest_row(self, row: pd.Series, seen: dict[str, dict]) -> tuple[dict | None, str]:
        """Return (payload, reason); payload is None when the row must be rejected."""
        raw_seq = row["sequence"]
        if pd.isna(raw_seq):
            return None, "missing sequence"
        s = normalize(str(raw_seq))
        batch_id = str(row["batch_id"]).strip()
        bdir = self.batch_path(batch_id)
        batch_csv = bdir / "batch.csv"
        if not batch_csv.is_file():
            return None, "unknown batch"
        expected_id = candidate_id(s)
        given_id = str(row["candidate_id"]).strip()
        if given_id != expected_id:
            return None, f"candidate_id does not match sequence (expected {expected_id})"
        batch = pd.read_csv(batch_csv)
        known = set(batch["sequence"].map(normalize))
        if s not in known:
            return None, "not in batch"
        protocol = str(row.get("protocol_id", "")).strip()
        if not protocol or protocol.lower() in ("nan", "none"):
            return None, "missing protocol_id"
        if self.approved_protocols and protocol not in self.approved_protocols:
            return None, f"unapproved protocol_id {protocol!r}"
        status = str(row["measurement_status"]).strip().lower()
        if status not in MEASUREMENT_STATUS:
            return None, f"status={status or 'missing'}"
        if status == "fail":
            if s in self.pending:
                return None, "already accepted as pass"
            if s in self.failed:
                return None, "duplicate failure"
            return {"kind": "fail", "batch_id": batch_id, "sequence": s,
                    "protocol_id": protocol}, ""
        value = row[TARGET_COLUMN]
        if pd.isna(value) or not np.isfinite(float(value)):
            return None, "non-finite gbsa"
        value = float(value)
        if s in self.pending:
            if abs(self.pending[s] - value) > 1e-9:
                return None, "conflicting gbsa"
            return {"kind": "duplicate", "batch_id": batch_id, "sequence": s,
                    "function": value}, ""
        if s in self.failed:
            return None, "already failed QC"
        previous = seen.get(s)
        if previous is not None:
            if previous["kind"] == "pass" and abs(previous["value"] - value) > 1e-9:
                return None, "conflicting gbsa within the same file"
            return {"kind": "duplicate", "batch_id": batch_id, "sequence": s,
                    "function": value}, ""
        return {"kind": "pass", "batch_id": batch_id, "sequence": s,
                "value": value, "protocol_id": protocol}, ""

    def ingest(self, labeled_csv: str | Path) -> dict:
        """Validate the whole file, then persist accepted rows atomically."""
        df = pd.read_csv(labeled_csv)
        missing = [c for c in INGEST_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"ingest file missing columns: {missing}")
        report = {"accepted": 0, "duplicate": 0, "failed": 0, "rejected": [],
                  "batches": set(), "closed_batches": []}
        accepted: list[dict] = []
        failures: list[dict] = []
        seen: dict[str, dict] = {}
        for _, row in df.iterrows():
            payload, reason = self._validate_ingest_row(row, seen)
            if payload is None:
                report["rejected"].append({"sequence": str(row.get("sequence"))[:32],
                                           "reason": reason})
                continue
            kind = payload["kind"]
            if kind == "duplicate":
                report["duplicate"] += 1
                continue
            if kind == "fail":
                failures.append({"batch_id": payload["batch_id"],
                                 "candidate_id": candidate_id(payload["sequence"]),
                                 "sequence": payload["sequence"],
                                 TARGET_COLUMN: "", "measurement_status": "fail",
                                 "protocol_id": payload["protocol_id"]})
                self.failed.add(payload["sequence"])
                self.fail_batch_of[payload["sequence"]] = payload["batch_id"]
                report["failed"] += 1
                report["batches"].add(payload["batch_id"])
                seen[payload["sequence"]] = {"kind": "fail"}
                continue
            accepted.append({"batch_id": payload["batch_id"],
                             "candidate_id": candidate_id(payload["sequence"]),
                             "sequence": payload["sequence"],
                             TARGET_COLUMN: payload["value"],
                             "measurement_status": "pass",
                             "protocol_id": payload["protocol_id"]})
            seen[payload["sequence"]] = {"kind": "pass", "value": payload["value"]}
            self.pending[payload["sequence"]] = payload["value"]
            self.batch_of[payload["sequence"]] = payload["batch_id"]
            report["accepted"] += 1
            report["batches"].add(payload["batch_id"])

        by_batch: dict[str, dict[str, list[dict]]] = {}
        for entry in accepted:
            by_batch.setdefault(entry["batch_id"], {"pass": [], "fail": []})["pass"].append(entry)
        for entry in failures:
            by_batch.setdefault(entry["batch_id"], {"pass": [], "fail": []})["fail"].append(entry)
        for batch_id, groups in by_batch.items():
            bdir = self.batch_path(batch_id)
            if groups["pass"]:
                self._atomic_append(bdir / self.PASS_FILE, groups["pass"],
                                    list(INGEST_COLUMNS))
            if groups["fail"]:
                self._atomic_append(bdir / self.FAIL_FILE, groups["fail"],
                                    list(INGEST_COLUMNS))
            status = self.batch_status(batch_id)
            if status.get("closed"):
                report["closed_batches"].append(batch_id)

        report["batches"] = sorted(report["batches"])
        (self.root / "ingest_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        return report

    def batch_status(self, batch_id: str) -> dict:
        bdir = self.batch_path(batch_id)
        if not bdir.is_dir():
            return {"batch_id": batch_id, "exists": False}
        batch = pd.read_csv(bdir / "batch.csv")
        seqs = batch["sequence"].map(normalize).tolist()
        labeled = [s for s in seqs if s in self.pending]
        failed = [s for s in seqs if s in self.failed]
        closed = len(labeled) + len(failed) == len(seqs)
        status = ("closed" if closed else
                  "partially_labeled" if (labeled or failed) else "awaiting_labels")
        return {"batch_id": batch_id, "exists": True, "n": len(seqs),
                "n_labeled": len(labeled), "n_failed": len(failed),
                "closed": closed, "status": status}


class CommandOracle:
    """Prospective hook driving an external command (not used in this phase)."""

    def __init__(self, command: str, workdir: str | Path = ".",
                 sequence_col: str = "sequence", target_col: str = TARGET_COLUMN):
        self.command = command
        self.workdir = Path(workdir)
        self.sequence_col = sequence_col
        self.target_col = target_col
        self.cache: dict[str, float] = {}

    def eligible_to_query(self, sequences: Sequence[str]) -> np.ndarray:
        return np.ones(len(sequences), dtype=bool)

    def already_labeled(self, sequences: Sequence[str]) -> np.ndarray:
        return np.array([normalize(s) in self.cache for s in sequences], dtype=bool)

    def query(self, batch_id: str, sequences: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        seqs = [normalize(s) for s in sequences]
        todo = [s for s in seqs if s not in self.cache]
        if todo:
            with tempfile.TemporaryDirectory() as tmp:
                inp, out = Path(tmp) / "candidates.csv", Path(tmp) / "labels.csv"
                pd.DataFrame({self.sequence_col: todo}).to_csv(inp, index=False)
                subprocess.run(self.command.format(input=str(inp), output=str(out)),
                               shell=True, cwd=self.workdir, check=True)
                df = pd.read_csv(out)
                for s, y in zip(df[self.sequence_col], df[self.target_col]):
                    if pd.notna(y):
                        self.cache[normalize(s)] = float(y)
        values = np.array([self.cache.get(s, np.nan) for s in seqs], dtype=np.float64)
        return values, np.isfinite(values)

    def peek(self, sequences: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        seqs = [normalize(s) for s in sequences]
        known = np.array([s in self.cache for s in seqs], dtype=bool)
        values = np.array([self.cache.get(s, np.nan) if k else np.nan
                           for s, k in zip(seqs, known)], dtype=np.float64)
        return values, known


ORACLES = {"table": TableOracle, "park": ParkOracle, "command": CommandOracle}

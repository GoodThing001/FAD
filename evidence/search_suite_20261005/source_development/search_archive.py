"""Per-search prediction cache and on-disk archive (snapshot / state / artefacts).

Lifecycle of one search in `context.archive_dir`:

    search_snapshot.json   written once at search start; a resume refuses to run
                           when the identity fields no longer match
    predictions.jsonl      append/rewrite cache of every scored sequence with
                           model provenance (budget accounting + replay)
    search_evaluated.jsonl append-only ScoredCandidate records (all scores,
                           eliminated children included)
    search_state.json      incremental checkpoint after every generation
                           (parents, RNG state, best, stop info)
    search_candidates.csv  final artefacts written by finalize()
    search_generations.csv
    search_groups.json
    search_summary.json
    files.sha256

All writes go through a temp file + os.replace (atomic).  The archive layer is
pure persistence: it never imports policy or surrogate code.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class PredictionCache:
    """Sequence -> prediction mapping bound to ONE model snapshot.

    `provenance` is fixed on the first put and enforced on every later put, so a
    cache can never silently mix two models (a replaced model must start a new
    archive — "cache key must change").
    """

    def __init__(self, rows: list[dict] | None = None):
        self._entries: dict[str, dict] = {}
        self.provenance: tuple[str, str] | None = None
        if rows:
            for row in rows:
                self._load_row(row)

    def _load_row(self, row: dict) -> None:
        key = (str(row["model_id"]), str(row["feature_version"]))
        if self.provenance is None:
            self.provenance = key
        elif self.provenance != key:
            raise ValueError(
                "archive prediction cache mixes model snapshots "
                f"({self.provenance!r} vs {key!r}); refuse to reuse it")
        seq = str(row["sequence"])
        if seq in self._entries:
            raise ValueError(f"duplicate sequence in prediction cache: {seq[:20]}...")
        self._entries[seq] = {
            "sequence": seq,
            "model_id": str(row["model_id"]),
            "feature_version": str(row["feature_version"]),
            "mu": float(row["mu"]),
            "sigma": (None if row.get("sigma") is None else float(row["sigma"])),
            "uncertainty_kind": row.get("uncertainty_kind"),
            "score_time": float(row.get("score_time", 0.0)),
        }

    def __contains__(self, seq: str) -> bool:
        return seq in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def put(self, seq: str, mu: float, sigma: float | None,
            uncertainty_kind: str | None, model_id: str, feature_version: str,
            score_time: float) -> None:
        key = (str(model_id), str(feature_version))
        if self.provenance is None:
            self.provenance = key
        elif self.provenance != key:
            raise ValueError(
                f"prediction cache is bound to model snapshot {self.provenance!r}; "
                f"refusing to store a prediction from {key!r} (cache key must change)")
        if seq in self._entries:
            return
        if not np.isfinite(mu):
            raise ValueError("refusing to cache a non-finite prediction")
        if sigma is not None and (not np.isfinite(sigma) or sigma < 0):
            raise ValueError("refusing to cache an invalid sigma")
        self._entries[seq] = {
            "sequence": seq, "model_id": str(model_id),
            "feature_version": str(feature_version), "mu": float(mu),
            "sigma": (None if sigma is None else float(sigma)),
            "uncertainty_kind": uncertainty_kind, "score_time": float(score_time),
        }

    def get(self, seq: str) -> dict | None:
        return self._entries.get(seq)

    def mu(self, seq: str) -> float:
        return float(self._entries[seq]["mu"])

    def sigma(self, seq: str) -> float | None:
        return self._entries[seq]["sigma"]

    def kind(self, seq: str) -> str | None:
        return self._entries[seq].get("uncertainty_kind")

    def sequences(self) -> list[str]:
        return list(self._entries)

    def to_rows(self) -> list[dict]:
        return [dict(e) for e in self._entries.values()]


class SearchArchive:
    """On-disk lifecycle of one search (root = context.archive_dir)."""

    SNAPSHOT = "search_snapshot.json"
    STATE = "search_state.json"
    PREDICTIONS = "predictions.jsonl"
    EVALUATED = "search_evaluated.jsonl"
    CANDIDATES = "search_candidates.csv"
    GENERATIONS = "search_generations.csv"
    GROUPS = "search_groups.json"
    SUMMARY = "search_summary.json"
    HASHES = "files.sha256"

    #: identity fields a resume compares strictly (policy adds policy/config keys)
    STRICT_SNAPSHOT_KEYS = (
        "search_interface_version", "search_id", "outer_round", "seed",
        "data_sha256", "measured_sha256", "forbidden_sha256",
        "allowed_n_from_wt", "max_unique_predictions", "max_generations",
        "constraints_version", "policy", "policy_config", "model_snapshot",
    )

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.root / name

    def _atomic_write(self, name: str, payload: str) -> Path:
        path = self._path(name)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
        return path

    def _atomic_lines(self, name: str, rows: list[dict]) -> Path:
        text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                       for r in rows)
        return self._atomic_write(name, text)

    # ---- snapshot ---------------------------------------------------------
    def write_snapshot(self, payload: dict) -> Path:
        payload = dict(payload)
        payload["created_at"] = pd.Timestamp.now().isoformat()
        return self._atomic_write(self.SNAPSHOT,
                                  json.dumps(payload, ensure_ascii=False, indent=2))

    def read_snapshot(self) -> dict | None:
        path = self._path(self.SNAPSHOT)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def check_snapshot(self, payload: dict) -> None:
        """Resume guard: identity fields must match exactly, otherwise refuse."""
        stored = self.read_snapshot()
        if stored is None:
            raise FileNotFoundError(
                f"{self._path(self.SNAPSHOT)} is missing; cannot resume")
        diffs = {}
        for key in self.STRICT_SNAPSHOT_KEYS:
            if stored.get(key) != payload.get(key):
                diffs[key] = {"stored": stored.get(key), "current": payload.get(key)}
        if diffs:
            detail = "; ".join(sorted(diffs))
            raise ValueError(
                f"resume refuses: search snapshot mismatch ({detail}). "
                "Re-run with the original context/config or start a new search "
                "in a new archive dir.")

    # ---- state --------------------------------------------------------------
    def write_state(self, payload: dict) -> Path:
        return self._atomic_write(self.STATE,
                                  json.dumps(payload, ensure_ascii=False, indent=2))

    def read_state(self) -> dict | None:
        path = self._path(self.STATE)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    # ---- prediction cache ---------------------------------------------------
    def load_cache(self) -> PredictionCache:
        path = self._path(self.PREDICTIONS)
        if not path.is_file():
            return PredictionCache()
        rows = [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return PredictionCache(rows)

    def reconcile_to_state(self, state: dict) -> PredictionCache:
        """Crash guard for resume: keep only what the last checkpoint committed.

        The checkpoint write order is cache -> evaluated log -> state, so a crash
        window can leave the evaluated log longer than `evaluated_rows_committed`
        or the cache holding rows that never reached the log.  Both are trimmed
        back to the committed prefix here, making a boundary resume deterministic.
        """
        committed = int(state.get("evaluated_rows_committed", 0))
        rows = self.read_evaluated()
        if len(rows) > committed:
            rows = rows[:committed]
            self._atomic_lines(self.EVALUATED, rows)
        keep = {r["sequence"] for r in rows}
        cache = self.load_cache()
        pruned_rows = [r for r in cache.to_rows() if r["sequence"] in keep]
        if len(pruned_rows) != len(cache):
            cache = PredictionCache(pruned_rows)
            self.save_cache(cache)
        return cache

    def save_cache(self, cache: PredictionCache) -> Path:
        return self._atomic_lines(self.PREDICTIONS, cache.to_rows())

    # ---- evaluated log ------------------------------------------------------
    def append_evaluated(self, rows: list[dict]) -> None:
        path = self._path(self.EVALUATED)
        with open(path, "a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def read_evaluated(self) -> list[dict]:
        path = self._path(self.EVALUATED)
        if not path.is_file():
            return []
        return [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # ---- final artefacts ----------------------------------------------------
    def finalize(self, *, candidates: list[dict], generations: list[dict],
                 groups: list[dict], summary: dict) -> dict:
        """Write the four artefacts plus files.sha256; returns the hash table."""
        cdf = pd.DataFrame(candidates)
        if cdf.empty:
            cdf = pd.DataFrame(columns=["candidate_id", "sequence", "generation"])
        cdf.to_csv(self._path(self.CANDIDATES), index=False)
        gdf = pd.DataFrame(generations)
        if gdf.empty:
            gdf = pd.DataFrame(columns=["generation"])
        gdf.to_csv(self._path(self.GENERATIONS), index=False)
        self._atomic_write(self.GROUPS, json.dumps(groups, ensure_ascii=False, indent=2))
        summary = dict(summary)
        summary["files_sha256"] = {
            name: sha256_file(self._path(name))
            for name in (self.SNAPSHOT, self.STATE, self.PREDICTIONS,
                         self.EVALUATED, self.CANDIDATES, self.GENERATIONS,
                         self.GROUPS)
            if self._path(name).is_file()
        }
        self._atomic_write(self.SUMMARY,
                           json.dumps(summary, ensure_ascii=False, indent=2))
        hashes = summary["files_sha256"]
        hashes[self.SUMMARY] = sha256_file(self._path(self.SUMMARY))
        self._atomic_write(self.HASHES, json.dumps(hashes, ensure_ascii=False, indent=2))
        return hashes

    def load_final_summary(self) -> dict | None:
        path = self._path(self.SUMMARY)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

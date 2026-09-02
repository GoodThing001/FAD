"""Safely move reviewed exact duplicates into a recoverable quarantine.

The tool validates both the retained copy and candidate against the manifest
SHA256 before moving. It refuses paths outside the explicitly supplied legacy
and quarantine roots and writes an append-only movement journal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def io_path(path: Path) -> Path:
    """Return a Windows extended-length path for filesystem operations."""
    absolute = path.absolute()
    if os.name != "nt":
        return absolute
    raw = str(absolute)
    if raw.startswith("\\\\?\\"):
        return absolute
    if raw.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + raw.lstrip("\\"))
    return Path("\\\\?\\" + raw)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contained(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    root_resolved = root.resolve()
    return resolved == root_resolved or root_resolved in resolved.parents


def append_journal(path: Path, row: dict) -> None:
    fields = [
        "timestamp", "status", "sha256", "size_bytes", "retained_path",
        "source_path", "quarantine_path", "message",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a reviewed exact-duplicate quarantine plan")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--quarantine-root", type=Path, required=True)
    parser.add_argument("--min-size-mib", type=float, default=0.0)
    parser.add_argument("--max-files", type=int, default=0, help="0 means no limit")
    parser.add_argument(
        "--retry-failed-from",
        type=Path,
        help="only select candidates whose latest status in this journal is failed",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    plan = args.plan.resolve()
    legacy_root = args.legacy_root.resolve()
    quarantine_root = args.quarantine_root.absolute()
    if not legacy_root.is_dir():
        raise FileNotFoundError(f"legacy root does not exist: {legacy_root}")
    if contained(quarantine_root, legacy_root) or contained(legacy_root, quarantine_root):
        raise ValueError("legacy root and quarantine root must be separate trees")
    if not plan.is_file():
        raise FileNotFoundError(f"plan does not exist: {plan}")

    with plan.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    threshold = int(args.min_size_mib * 1024 * 1024)
    selected = [row for row in rows if int(row["size_bytes"]) >= threshold]
    if args.retry_failed_from:
        retry_journal = args.retry_failed_from.resolve()
        if not retry_journal.is_file():
            raise FileNotFoundError(f"retry journal does not exist: {retry_journal}")
        latest_status: dict[str, str] = {}
        with retry_journal.open("r", encoding="utf-8-sig", newline="") as handle:
            for journal_row in csv.DictReader(handle):
                key = os.path.normcase(os.path.abspath(journal_row["source_path"]))
                latest_status[key] = journal_row["status"]
        selected = [
            row
            for row in selected
            if latest_status.get(
                os.path.normcase(
                    os.path.abspath(str(legacy_root / row["candidate_to_quarantine"]))
                )
            )
            == "failed"
        ]
    selected.sort(key=lambda row: int(row["size_bytes"]), reverse=True)
    if args.max_files > 0:
        selected = selected[: args.max_files]

    preview = {
        "mode": "execute" if args.execute else "dry_run",
        "legacy_root": str(legacy_root),
        "quarantine_root": str(quarantine_root),
        "selected_files": len(selected),
        "selected_bytes": sum(int(row["size_bytes"]) for row in selected),
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    if not args.execute:
        for row in selected[:20]:
            print(f"[dry-run] {int(row['size_bytes']) / 1024**2:.2f} MiB  {row['candidate_to_quarantine']}")
        return

    quarantine_root.mkdir(parents=True, exist_ok=True)
    journal = quarantine_root / "MOVE_JOURNAL.csv"
    moved = skipped = failed = 0
    moved_bytes = 0

    for row in selected:
        expected_hash = row["sha256"].lower()
        expected_size = int(row["size_bytes"])
        keep = legacy_root / row["proposed_keep"]
        source = legacy_root / row["candidate_to_quarantine"]
        target = quarantine_root / row["candidate_to_quarantine"]
        keep_io = io_path(keep)
        source_io = io_path(source)
        target_io = io_path(target)
        record = {
            "timestamp": datetime.now().isoformat(),
            "status": "",
            "sha256": expected_hash,
            "size_bytes": expected_size,
            "retained_path": str(keep),
            "source_path": str(source),
            "quarantine_path": str(target),
            "message": "",
        }
        try:
            if not contained(source, legacy_root) or not contained(keep, legacy_root):
                raise ValueError("source or retained path escapes legacy root")
            target_parent = target.parent.absolute()
            if quarantine_root.absolute() not in target_parent.parents and target_parent != quarantine_root.absolute():
                raise ValueError("target escapes quarantine root")
            if not keep_io.is_file():
                raise FileNotFoundError(f"retained copy missing: {keep}")
            if keep_io.stat().st_size != expected_size or sha256(keep_io) != expected_hash:
                raise ValueError(f"retained copy hash/size mismatch: {keep}")
            if not source_io.exists() and target_io.is_file():
                if target_io.stat().st_size == expected_size and sha256(target_io) == expected_hash:
                    record["status"] = "already_quarantined"
                    skipped += 1
                    append_journal(journal, record)
                    continue
            if not source_io.is_file():
                raise FileNotFoundError(f"source missing: {source}")
            if source_io.stat().st_size != expected_size or sha256(source_io) != expected_hash:
                raise ValueError(f"source hash/size mismatch: {source}")
            if target_io.exists():
                raise FileExistsError(f"quarantine target already exists: {target}")
            target_io.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_io), str(target_io))
            if not target_io.is_file() or target_io.stat().st_size != expected_size:
                raise OSError(f"post-move verification failed: {target}")
            record["status"] = "moved"
            moved += 1
            moved_bytes += expected_size
        except Exception as exc:  # journal every failure before continuing
            record["status"] = "failed"
            record["message"] = f"{type(exc).__name__}: {exc}"
            failed += 1
        append_journal(journal, record)
        print(f"[{record['status']}] {row['candidate_to_quarantine']}")

    result = {
        "completed_at": datetime.now().isoformat(),
        "moved": moved,
        "moved_bytes": moved_bytes,
        "already_quarantined": skipped,
        "failed": failed,
        "journal": str(journal),
    }
    (quarantine_root / "BATCH_SUMMARY.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

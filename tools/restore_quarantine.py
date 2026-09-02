"""Restore quarantined files without overwriting anything in the legacy tree."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from apply_quarantine_plan import contained, io_path, sha256


FIELDS = [
    "timestamp",
    "status",
    "sha256",
    "size_bytes",
    "source_path",
    "quarantine_path",
    "message",
]


def append_row(path: Path, row: dict[str, object]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore files from a quarantine journal")
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--quarantine-root", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="legacy-root-relative path to restore; repeat for multiple files",
    )
    parser.add_argument("--all", action="store_true", help="explicitly select every journaled file")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.all and args.candidate:
        parser.error("use either --all or --candidate, not both")
    if args.execute and not args.all and not args.candidate:
        parser.error("execution requires --all or at least one --candidate")

    journal = args.journal.resolve()
    legacy_root = args.legacy_root.resolve()
    quarantine_root = args.quarantine_root.absolute()
    if contained(quarantine_root, legacy_root) or contained(legacy_root, quarantine_root):
        raise ValueError("legacy root and quarantine root must be separate trees")

    latest: dict[str, dict[str, str]] = {}
    with journal.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = os.path.normcase(os.path.abspath(row["source_path"]))
            latest[key] = row
    journal_eligible = [
        row for row in latest.values() if row["status"] in {"moved", "already_quarantined"}
    ]
    eligible: list[dict[str, str]] = []
    already_restored = 0
    state_conflicts = 0
    for row in journal_eligible:
        source_exists = io_path(Path(row["source_path"])).exists()
        target_exists = io_path(Path(row["quarantine_path"])).is_file()
        if target_exists and not source_exists:
            eligible.append(row)
        elif source_exists and not target_exists:
            already_restored += 1
        else:
            state_conflicts += 1

    requested = {
        os.path.normcase(os.path.abspath(str(legacy_root / candidate)))
        for candidate in args.candidate
    }
    if requested:
        selected = [
            row
            for row in eligible
            if os.path.normcase(os.path.abspath(row["source_path"])) in requested
        ]
        missing = requested - {
            os.path.normcase(os.path.abspath(row["source_path"])) for row in selected
        }
        if missing:
            raise ValueError(f"candidate not found in latest movement journal: {sorted(missing)}")
    else:
        selected = eligible
    selected.sort(key=lambda row: row["source_path"])

    preview = {
        "mode": "execute" if args.execute else "dry_run",
        "selected_files": len(selected),
        "selected_bytes": sum(int(row["size_bytes"]) for row in selected),
        "already_restored": already_restored,
        "state_conflicts": state_conflicts,
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    if not args.execute:
        for row in selected[:20]:
            print(f"[dry-run] {row['source_path']}")
        return

    restore_journal = quarantine_root / "RESTORE_JOURNAL.csv"
    restored = failed = restored_bytes = 0
    for movement in selected:
        source = Path(movement["source_path"])
        target = Path(movement["quarantine_path"])
        expected_hash = movement["sha256"].lower()
        expected_size = int(movement["size_bytes"])
        record: dict[str, object] = {
            "timestamp": datetime.now().isoformat(),
            "status": "",
            "sha256": expected_hash,
            "size_bytes": expected_size,
            "source_path": str(source),
            "quarantine_path": str(target),
            "message": "",
        }
        try:
            if not contained(source, legacy_root) or not contained(target, quarantine_root):
                raise ValueError("source or quarantine path escapes its declared root")
            source_io = io_path(source)
            target_io = io_path(target)
            if source_io.exists():
                raise FileExistsError(f"refusing to overwrite existing source: {source}")
            if not target_io.is_file():
                raise FileNotFoundError(f"quarantine copy missing: {target}")
            if target_io.stat().st_size != expected_size or sha256(target_io) != expected_hash:
                raise ValueError(f"quarantine copy hash/size mismatch: {target}")
            source_io.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(target_io), str(source_io))
            if not source_io.is_file() or source_io.stat().st_size != expected_size:
                raise OSError(f"post-restore verification failed: {source}")
            record["status"] = "restored"
            restored += 1
            restored_bytes += expected_size
        except Exception as exc:
            record["status"] = "failed"
            record["message"] = f"{type(exc).__name__}: {exc}"
            failed += 1
        append_row(restore_journal, record)

    result = {
        "restored": restored,
        "restored_bytes": restored_bytes,
        "failed": failed,
        "restore_journal": str(restore_journal),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

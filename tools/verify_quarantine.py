"""Verify the final state of an exact-duplicate quarantine batch."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from apply_quarantine_plan import contained, io_path, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify quarantined exact duplicates")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--quarantine-root", type=Path, required=True)
    parser.add_argument("--min-size-mib", type=float, default=0.0)
    parser.add_argument(
        "--verify-retained-hash",
        action="store_true",
        help="rehash retained copies as well as quarantine copies",
    )
    args = parser.parse_args()

    plan = args.plan.resolve()
    legacy_root = args.legacy_root.resolve()
    quarantine_root = args.quarantine_root.absolute()
    if contained(quarantine_root, legacy_root) or contained(legacy_root, quarantine_root):
        raise ValueError("legacy root and quarantine root must be separate trees")

    with plan.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    threshold = int(args.min_size_mib * 1024 * 1024)
    selected = [row for row in rows if int(row["size_bytes"]) >= threshold]

    failures: list[dict[str, str]] = []
    verified_bytes = 0
    for row in selected:
        expected_hash = row["sha256"].lower()
        expected_size = int(row["size_bytes"])
        keep = legacy_root / row["proposed_keep"]
        source = legacy_root / row["candidate_to_quarantine"]
        target = quarantine_root / row["candidate_to_quarantine"]
        try:
            if not contained(source, legacy_root) or not contained(keep, legacy_root):
                raise ValueError("source or retained path escapes legacy root")
            if io_path(source).exists():
                raise FileExistsError("source still exists in the legacy tree")
            keep_io = io_path(keep)
            target_io = io_path(target)
            if not keep_io.is_file() or keep_io.stat().st_size != expected_size:
                raise FileNotFoundError("retained copy missing or has unexpected size")
            if args.verify_retained_hash and sha256(keep_io) != expected_hash:
                raise ValueError("retained copy hash mismatch")
            if not target_io.is_file() or target_io.stat().st_size != expected_size:
                raise FileNotFoundError("quarantine copy missing or has unexpected size")
            if sha256(target_io) != expected_hash:
                raise ValueError("quarantine copy hash mismatch")
            verified_bytes += expected_size
        except Exception as exc:
            failures.append(
                {
                    "candidate": row["candidate_to_quarantine"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    result = {
        "selected_files": len(selected),
        "verified_files": len(selected) - len(failures),
        "verified_bytes": verified_bytes,
        "failed": len(failures),
        "retained_hashes_rechecked": args.verify_retained_hash,
        "failure_sample": failures[:20],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

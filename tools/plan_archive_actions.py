"""Turn a completed SHA256 manifest into conservative archive plans.

The generated CSVs are proposals only. This tool never moves or deletes files.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def preference(path: str) -> tuple[int, int, str]:
    normalized = path.replace("\\", "/").lower()
    if normalized.startswith("data/proxy2000_v2/"):
        rank = 0
    elif "/summaries/" in normalized or normalized.endswith("summary.json"):
        rank = 10
    elif normalized.startswith("artifacts/"):
        rank = 20
    elif normalized.startswith("archived/fad_rnaernie/"):
        rank = 25
    elif normalized.startswith("archived/"):
        rank = 30
    elif normalized.startswith("docs/"):
        rank = 40
    else:
        rank = 25
    return rank, len(normalized), normalized


def write_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan safe archive actions from SHA256 manifest")
    parser.add_argument(
        "--manifest-db", type=Path,
        default=ROOT / "reports/archive_batch_20260902/archive_manifest.sqlite",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/archive_batch_20260902/plans")
    parser.add_argument("--large-mib", type=int, default=50)
    args = parser.parse_args()
    db_path = args.manifest_db if args.manifest_db.is_absolute() else ROOT / args.manifest_db
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(db_path)
    rows = connection.execute(
        "SELECT path,size_bytes,sha256,category,recommended_action,error FROM files"
    ).fetchall()
    if not rows:
        raise RuntimeError("manifest database is empty")

    by_hash: dict[tuple[str, int], list[str]] = {}
    cache_rows: list[tuple] = []
    large_rows: list[tuple] = []
    for path, size, digest, category, action, error in rows:
        if error:
            continue
        if digest and size > 0:
            by_hash.setdefault((digest, size), []).append(path)
        if category == "regenerable_cache":
            cache_rows.append((path, size, digest, "quarantine_then_verify"))
        if size >= args.large_mib * 1024 * 1024:
            large_rows.append((path, size, round(size / 1024**2, 2), digest, category, action))

    duplicate_rows: list[tuple] = []
    duplicate_groups = 0
    duplicate_bytes = 0
    for (digest, size), paths in by_hash.items():
        if len(paths) < 2:
            continue
        duplicate_groups += 1
        ordered = sorted(paths, key=preference)
        keep = ordered[0]
        for candidate in ordered[1:]:
            duplicate_rows.append(
                (digest, size, len(paths), keep, candidate, "quarantine_exact_duplicate")
            )
            duplicate_bytes += size

    large_rows.sort(key=lambda row: row[1], reverse=True)
    duplicate_rows.sort(key=lambda row: row[1], reverse=True)
    cache_rows.sort(key=lambda row: row[1], reverse=True)

    write_csv(
        output_dir / "exact_duplicate_move_plan.csv",
        ["sha256", "size_bytes", "copies", "proposed_keep", "candidate_to_quarantine", "action"],
        duplicate_rows,
    )
    write_csv(
        output_dir / "regenerable_cache_move_plan.csv",
        ["path", "size_bytes", "sha256", "action"],
        cache_rows,
    )
    write_csv(
        output_dir / "large_file_review.csv",
        ["path", "size_bytes", "size_mib", "sha256", "category", "recommended_action"],
        large_rows,
    )

    summary = {
        "generated_at": datetime.now().isoformat(),
        "files_considered": len(rows),
        "exact_duplicate_groups": duplicate_groups,
        "exact_duplicate_files_proposed_for_quarantine": len(duplicate_rows),
        "exact_duplicate_reclaimable_bytes": duplicate_bytes,
        "regenerable_cache_files": len(cache_rows),
        "regenerable_cache_bytes": sum(row[1] for row in cache_rows),
        "large_files_for_manual_review": len(large_rows),
        "large_file_bytes": sum(row[1] for row in large_rows),
        "action": "review only; no file moved or deleted",
    }
    (output_dir / "archive_action_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

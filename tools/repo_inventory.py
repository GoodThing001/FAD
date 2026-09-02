"""Create a non-destructive inventory of the legacy FAD workspace."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


CLEAN_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = CLEAN_ROOT.parent
SCAN_ROOTS = ("docs", "scripts", "src", "data", "artifacts", "logs", "resources", "archived")

MAINLINE_SCRIPTS = {
    "scripts/tools/baseline_30seed.py",
    "scripts/tools/data_registry.py",
    "scripts/tools/phase3_fusion_v8.py",
    "scripts/tools/generate_cluster_splits.py",
    "scripts/tools/baseline_cluster_split.py",
}
ACTIVE_SCRIPTS = {"scripts/tools/feature_selection_pipeline.py"}
AUDIT_REQUIRED_SCRIPTS = {"scripts/tools/eval_rank_targets.py"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify(relative: str) -> tuple[str, str]:
    normalized = relative.replace("\\", "/")
    if normalized in MAINLINE_SCRIPTS:
        return "mainline", "copy_and_validate"
    if normalized in ACTIVE_SCRIPTS:
        return "active_experiment", "copy_as_exploratory"
    if normalized in AUDIT_REQUIRED_SCRIPTS:
        return "audit_required", "do_not_promote"
    if normalized.startswith("data/proxy2000_v2/"):
        if "/features/ed/" in normalized:
            return "invalid_leakage", "registry_only"
        return "mainline_data", "selective_copy"
    if normalized.startswith("artifacts/") or normalized.startswith("logs/"):
        return "generated_result", "select_by_registry"
    if normalized.startswith("archived/"):
        return "historical", "leave_in_legacy"
    if normalized.startswith("docs/"):
        return "documentation", "review_then_promote_or_archive"
    if normalized.startswith("scripts/legacy"):
        return "historical", "leave_in_legacy"
    return "unclassified", "manual_review"


def all_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and CLEAN_ROOT not in path.parents:
            yield path


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def git_summary() -> dict:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
            cwd=LEGACY_ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"available": False}
    codes = Counter(line[:2] for line in output.splitlines() if len(line) >= 3)
    return {"available": True, "total_entries": sum(codes.values()), "codes": dict(codes)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory the legacy FAD workspace")
    parser.add_argument("--output-dir", type=Path, default=CLEAN_ROOT / "reports")
    parser.add_argument("--max-hash-mib", type=int, default=32)
    args = parser.parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = CLEAN_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    top_rows: list[dict] = []
    by_size: dict[int, list[Path]] = defaultdict(list)
    max_hash_bytes = args.max_hash_mib * 1024 * 1024

    for root_name in SCAN_ROOTS:
        root = LEGACY_ROOT / root_name
        if not root.exists():
            continue
        root_files = list(all_files(root))
        top_rows.append(
            {
                "directory": root_name,
                "file_count": len(root_files),
                "size_bytes": sum(path.stat().st_size for path in root_files),
                "latest_mtime": max((path.stat().st_mtime for path in root_files), default=0),
            }
        )
        for path in root_files:
            stat = path.stat()
            relative = path.relative_to(LEGACY_ROOT).as_posix()
            category, action = classify(relative)
            records.append(
                {
                    "path": relative,
                    "size_bytes": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "suffix": path.suffix.lower(),
                    "category": category,
                    "suggested_action": action,
                }
            )
            if root_name in {"data", "scripts", "docs"} and stat.st_size <= max_hash_bytes:
                by_size[stat.st_size].append(path)

    duplicates: list[dict] = []
    for size, paths in by_size.items():
        if len(paths) < 2:
            continue
        by_hash: dict[str, list[Path]] = defaultdict(list)
        for path in paths:
            by_hash[sha256(path)].append(path)
        for digest, matches in by_hash.items():
            if len(matches) < 2:
                continue
            group_id = digest[:16]
            for path in matches:
                duplicates.append(
                    {
                        "group_id": group_id,
                        "sha256": digest,
                        "size_bytes": size,
                        "path": path.relative_to(LEGACY_ROOT).as_posix(),
                    }
                )

    write_csv(
        output_dir / "file_inventory.csv", records,
        ["path", "size_bytes", "modified", "suffix", "category", "suggested_action"],
    )
    write_csv(
        output_dir / "top_level_summary.csv", top_rows,
        ["directory", "file_count", "size_bytes", "latest_mtime"],
    )
    write_csv(
        output_dir / "duplicate_candidates.csv", duplicates,
        ["group_id", "sha256", "size_bytes", "path"],
    )

    category_counts = Counter(record["category"] for record in records)
    summary = {
        "generated_at": datetime.now().isoformat(),
        "legacy_root": str(LEGACY_ROOT),
        "clean_root": str(CLEAN_ROOT),
        "files_scanned": len(records),
        "bytes_scanned": sum(record["size_bytes"] for record in records),
        "category_counts": dict(category_counts),
        "duplicate_file_entries": len(duplicates),
        "git": git_summary(),
        "safety": "inventory only; no legacy file was modified, moved, or deleted",
    }
    (output_dir / "inventory_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


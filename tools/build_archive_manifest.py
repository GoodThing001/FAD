"""Build a resumable SHA256 manifest for legacy archive candidates.

This tool is deliberately non-destructive. It records files in SQLite as it
goes, exports a CSV manifest, and reports exact duplicate groups. Interrupted
runs resume without rehashing unchanged files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import time
from collections import Counter
from datetime import datetime
from pathlib import Path


CLEAN_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = CLEAN_ROOT.parent
DEFAULT_ROOTS = (
    "artifacts",
    "archived",
    "FAD_KmerTreeModels",
    "docs",
    "data",
    "FAD_SVR",
    "FAD_RegularizedLinear",
    "resources",
    "logs",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classification(relative: str) -> tuple[str, str]:
    p = relative.replace("\\", "/").lower()
    suffix = Path(p).suffix
    if "/__pycache__/" in f"/{p}" or suffix in {".pyc", ".pyo"}:
        return "regenerable_cache", "delete_after_review"
    if "/nupack-4.0.0.27/" in f"/{p}" or "/source/build/" in f"/{p}":
        return "third_party_source_or_build", "move_to_external_vendor"
    if suffix in {".pth", ".pt", ".ckpt", ".safetensors", ".joblib", ".pkl"}:
        return "model_or_checkpoint", "keep_unique_final_only"
    if suffix in {".zip", ".7z", ".tar", ".gz"}:
        return "archive_bundle", "compare_with_unpacked_copy"
    if p.startswith("data/") or "/data/" in f"/{p}":
        return "dataset_or_copy", "keep_canonical_or_unique_raw"
    if suffix in {".pdf", ".docx", ".pptx", ".xlsx", ".xls"}:
        return "document_or_raw_office", "move_to_reference_or_raw_store"
    if p.startswith("logs/") or suffix in {".log"}:
        return "log", "keep_registered_runs_only"
    if p.startswith("artifacts/"):
        return "generated_artifact", "keep_registered_runs_only"
    if p.startswith("archived/"):
        return "historical", "move_to_cold_archive"
    return "source_or_metadata", "manual_review"


def connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY,
            root_name TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            sha256 TEXT,
            category TEXT NOT NULL,
            recommended_action TEXT NOT NULL,
            error TEXT
        )
        """
    )
    connection.commit()
    return connection


def discover(roots: list[str]) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    legacy_resolved = LEGACY_ROOT.resolve()
    for root_name in roots:
        root = (LEGACY_ROOT / root_name).resolve()
        if root != legacy_resolved and legacy_resolved not in root.parents:
            raise ValueError(f"root escapes legacy workspace: {root_name}")
        if not root.exists():
            print(f"[skip] missing root: {root_name}")
            continue
        for path in root.rglob("*"):
            if path.is_file() and CLEAN_ROOT.resolve() not in path.resolve().parents:
                files.append((root_name, path))
    return files


def export(connection: sqlite3.Connection, output_dir: Path) -> dict:
    columns = [
        "path", "root_name", "size_bytes", "mtime_ns", "sha256",
        "category", "recommended_action", "error",
    ]
    rows = connection.execute(
        "SELECT path,root_name,size_bytes,mtime_ns,sha256,category,recommended_action,error "
        "FROM files ORDER BY root_name,path"
    ).fetchall()
    with (output_dir / "archive_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)

    duplicate_rows = connection.execute(
        """
        SELECT sha256, size_bytes, COUNT(*) AS copies,
               SUM(size_bytes) - MAX(size_bytes) AS reclaimable_bytes,
               GROUP_CONCAT(path, ' | ') AS paths
        FROM files
        WHERE error IS NULL AND sha256 IS NOT NULL AND size_bytes > 0
        GROUP BY sha256, size_bytes
        HAVING COUNT(*) > 1
        ORDER BY reclaimable_bytes DESC, copies DESC
        """
    ).fetchall()
    with (output_dir / "exact_duplicates.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sha256", "size_bytes", "copies", "reclaimable_bytes", "paths"])
        writer.writerows(duplicate_rows)

    category_counts = dict(
        connection.execute(
            "SELECT category,COUNT(*) FROM files GROUP BY category ORDER BY COUNT(*) DESC"
        ).fetchall()
    )
    total_files, total_bytes, errors = connection.execute(
        "SELECT COUNT(*),COALESCE(SUM(size_bytes),0),SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) FROM files"
    ).fetchone()
    summary = {
        "generated_at": datetime.now().isoformat(),
        "legacy_root": str(LEGACY_ROOT),
        "total_files": total_files,
        "total_bytes": total_bytes,
        "errors": errors or 0,
        "category_counts": category_counts,
        "exact_duplicate_groups": len(duplicate_rows),
        "exact_duplicate_entries": sum(row[2] for row in duplicate_rows),
        "potential_reclaimable_bytes": sum(row[3] for row in duplicate_rows),
        "safety": "manifest only; no file was moved, modified, or deleted",
    }
    (output_dir / "archive_manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build resumable legacy SHA256 manifest")
    parser.add_argument("--roots", default=",".join(DEFAULT_ROOTS))
    parser.add_argument("--output-dir", type=Path, default=CLEAN_ROOT / "reports/archive_batch_20260902")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = CLEAN_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    connection = connect(output_dir / "archive_manifest.sqlite")
    roots = [item.strip() for item in args.roots.split(",") if item.strip()]

    files = discover(roots)
    total = len(files)
    total_bytes = sum(path.stat().st_size for _, path in files)
    print(f"[discover] {total} files, {total_bytes / 1024**3:.2f} GiB")
    started = time.time()
    hashed_bytes = 0
    skipped = 0
    errors = 0

    for index, (root_name, path) in enumerate(files, start=1):
        stat = path.stat()
        relative = path.relative_to(LEGACY_ROOT).as_posix()
        existing = connection.execute(
            "SELECT size_bytes,mtime_ns,sha256,error FROM files WHERE path=?", (relative,)
        ).fetchone()
        if existing and existing[0] == stat.st_size and existing[1] == stat.st_mtime_ns and existing[2] and not existing[3]:
            skipped += 1
            hashed_bytes += stat.st_size
            continue

        category, action = classification(relative)
        digest = None
        error = None
        try:
            digest = sha256(path)
        except (OSError, PermissionError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            errors += 1
        connection.execute(
            """
            INSERT INTO files(path,root_name,size_bytes,mtime_ns,sha256,category,recommended_action,error)
            VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(path) DO UPDATE SET
              root_name=excluded.root_name,size_bytes=excluded.size_bytes,mtime_ns=excluded.mtime_ns,
              sha256=excluded.sha256,category=excluded.category,
              recommended_action=excluded.recommended_action,error=excluded.error
            """,
            (relative, root_name, stat.st_size, stat.st_mtime_ns, digest, category, action, error),
        )
        hashed_bytes += stat.st_size
        if index % args.progress_every == 0:
            connection.commit()
            elapsed = max(time.time() - started, 0.001)
            progress = {
                "updated_at": datetime.now().isoformat(),
                "current_file": index,
                "total_files": total,
                "processed_gib": round(hashed_bytes / 1024**3, 3),
                "total_gib": round(total_bytes / 1024**3, 3),
                "throughput_mib_s": round(hashed_bytes / 1024**2 / elapsed, 2),
                "resumed_files": skipped,
                "errors": errors,
            }
            (output_dir / "progress.json").write_text(
                json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"[progress] {index}/{total} files, "
                f"{hashed_bytes / 1024**3:.2f}/{total_bytes / 1024**3:.2f} GiB, errors={errors}"
            )

    connection.commit()
    summary = export(connection, output_dir)
    connection.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


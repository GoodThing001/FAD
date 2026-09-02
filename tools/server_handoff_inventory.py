"""Create a small, hash-aware inventory for recovering FAD server evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


RUN_TOKENS = {
    "phase1_v8", "phase2_v8", "phase3_v8", "phase4_v8",
    "v8", "v10", "v10b", "v11", "v12", "v13", "feature_selection",
}
RESULT_SUFFIXES = {".csv", ".json", ".log", ".txt", ".yaml", ".yml"}
SCRIPT_NAMES = {
    "eval_rank_targets.py",
    "hyperparam_search.py",
    "phase3_fusion_v8.py",
    "final_validation_v8.py",
    "eval_nupack_blocks.py",
    "eval_nupack_combo.py",
    "eval_untested_features.py",
    "eval_final_combo.py",
    "feature_selection_pipeline.py",
}
EXPECTED_PATHS = (
    "data/proxy2000_v2/fad_proxy2000_v2_full.csv",
    "data/proxy2000_v2/train_nupack.csv",
    "data/proxy2000_v2/val_nupack.csv",
    "data/proxy2000_v2/test_nupack.csv",
    "artifacts/phase1_v8/rank_targets_30seed.csv",
    "artifacts/phase3_v8/fusion_results.csv",
    "artifacts/v10/nupack_blocks_30seed.csv",
    "artifacts/v10b/nupack_blocks_motif_split.csv",
    "artifacts/v11/nupack_combo_30seed.csv",
    "artifacts/v12/untested_30seed.csv",
    "artifacts/v13/final_combo_30seed.csv",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def classify(relative: str, name: str, suffix: str) -> str | None:
    normalized = "/" + relative.lower().replace("\\", "/") + "/"
    parts = set(part for part in normalized.split("/") if part)
    lower_name = name.lower()
    if lower_name in {"train_nupack.csv", "val_nupack.csv", "test_nupack.csv"}:
        return "nupack_augmented_data"
    if lower_name == "fad_proxy2000_v2_full.csv":
        return "canonical_data"
    if suffix == ".py" and (lower_name in SCRIPT_NAMES or "nupack" in lower_name):
        return "relevant_source"
    if suffix in RESULT_SUFFIXES and parts.intersection(RUN_TOKENS):
        return "experiment_evidence"
    if suffix in RESULT_SUFFIXES and (
        "per_frame" in lower_name or "mmpbsa" in lower_name or "gbsa" in lower_name
    ):
        return "gbsa_physics_evidence"
    if suffix in RESULT_SUFFIXES and "candidate" in normalized and "nupack" in normalized:
        return "candidate_nupack_progress"
    if lower_name in {"environment.yml", "environment.yaml", "requirements.txt", "pyproject.toml"}:
        return "environment"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory FAD server files for handoff")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("server_handoff_manifest.csv"))
    parser.add_argument("--max-hash-mib", type=float, default=256.0)
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output.resolve()
    hash_limit = int(args.max_hash_mib * 1024 * 1024)
    rows: list[dict[str, object]] = []
    scan_errors: list[str] = []

    def on_error(error: OSError) -> None:
        scan_errors.append(str(error))

    for directory, _, filenames in os.walk(root, onerror=on_error, followlinks=False):
        directory_path = Path(directory)
        for filename in filenames:
            path = directory_path / filename
            try:
                relative = path.relative_to(root).as_posix()
                category = classify(relative, path.name, path.suffix.lower())
                if category is None:
                    continue
                stat = path.stat()
                rows.append(
                    {
                        "category": category,
                        "relative_path": relative,
                        "size_bytes": stat.st_size,
                        "modified_utc": datetime.fromtimestamp(
                            stat.st_mtime, tz=timezone.utc
                        ).isoformat(),
                        "sha256": digest(path) if stat.st_size <= hash_limit else "SKIPPED_LARGE",
                    }
                )
            except (OSError, ValueError) as exc:
                scan_errors.append(f"{path}: {exc}")

    rows.sort(key=lambda row: (str(row["category"]), str(row["relative_path"])))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["category", "relative_path", "size_bytes", "modified_utc", "sha256"],
        )
        writer.writeheader()
        writer.writerows(rows)

    found = {str(row["relative_path"]).lower() for row in rows}
    counts = Counter(str(row["category"]) for row in rows)
    summary = {
        "root": str(root),
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "files": len(rows),
        "bytes": sum(int(row["size_bytes"]) for row in rows),
        "category_counts": dict(sorted(counts.items())),
        "expected_present": [path for path in EXPECTED_PATHS if path.lower() in found],
        "expected_missing": [path for path in EXPECTED_PATHS if path.lower() not in found],
        "scan_error_count": len(scan_errors),
        "scan_error_sample": scan_errors[:20],
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Validate recovered multi-seed runs and compute paired comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


RUNS = ("phase1_v8", "v10", "v10b", "v11", "v12", "v13")
GENERATED_OUTPUTS = {
    "manifest.csv", "run_completeness.csv", "paired_summary.csv", "summary.json"
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_record(
    source_run: str,
    frame: pd.DataFrame,
    key: str,
    baseline: str,
    candidate: str,
    model: str,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, object] | None:
    subset = frame[frame["model"] == model]
    pivot = subset.pivot_table(index="seed", columns=key, values="gbsa_Sp", aggfunc="first")
    if baseline not in pivot or candidate not in pivot:
        return None
    paired = pivot[[baseline, candidate]].dropna()
    delta = (paired[candidate] - paired[baseline]).to_numpy(dtype=float)
    if len(delta) == 0:
        return None
    samples = rng.choice(delta, size=(bootstrap_samples, len(delta)), replace=True).mean(axis=1)
    return {
        "source_run": source_run,
        "model": model,
        "baseline": baseline,
        "candidate": candidate,
        "paired_seeds": len(delta),
        "baseline_mean": paired[baseline].mean(),
        "candidate_mean": paired[candidate].mean(),
        "mean_delta": delta.mean(),
        "delta_std": delta.std(ddof=1) if len(delta) > 1 else 0.0,
        "bootstrap_ci95_low": np.percentile(samples, 2.5),
        "bootstrap_ci95_high": np.percentile(samples, 97.5),
        "wins": int((delta > 0).sum()),
        "ties": int((delta == 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize recovered FAD run evidence")
    parser.add_argument("--evidence-dir", type=Path, default="evidence/recovered_runs")
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    args = parser.parse_args()

    root = args.evidence_dir.resolve()
    rng = np.random.default_rng(20260902)
    completeness: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    frames: dict[str, pd.DataFrame] = {}

    source_files = [
        path
        for path in root.rglob("*")
        if path.is_file() and not (path.parent == root and path.name in GENERATED_OUTPUTS)
    ]
    for path in sorted(source_files):
        manifest.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    for run in RUNS:
        run_dir = root / run
        result_path = run_dir / "all_results.csv"
        progress_path = run_dir / "progress.json"
        frame = pd.read_csv(result_path)
        frames[run] = frame
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        seeds = sorted(int(seed) for seed in frame["seed"].unique())
        expected_rows = progress.get("total_results", progress.get("total"))
        completeness.append(
            {
                "run": run,
                "rows": len(frame),
                "expected_rows": expected_rows,
                "row_count_matches": expected_rows is None or len(frame) == expected_rows,
                "unique_seeds": len(seeds),
                "seed_min": min(seeds),
                "seed_max": max(seeds),
                "complete_30_seeds": seeds == list(range(1, 31)),
                "results_sha256": sha256(result_path),
            }
        )

    comparisons: list[dict[str, object]] = []

    v10b = frames["v10b"]
    for candidate in sorted(set(v10b["block"]) - {"baseline_seq_only"}):
        record = paired_record(
            "v10b", v10b, "block", "baseline_seq_only", candidate, "XGB",
            rng, args.bootstrap_samples,
        )
        if record:
            comparisons.append(record)

    v11 = frames["v11"]
    for model in sorted(v11["model"].unique()):
        record = paired_record(
            "v11", v11, "features", "seq_303d", "seq_303d+useful_192d",
            model, rng, args.bootstrap_samples,
        )
        if record:
            comparisons.append(record)

    v12 = frames["v12"]
    for model in sorted(v12["model"].unique()):
        for candidate in sorted(set(v12["feature"]) - {"baseline_seq"}):
            record = paired_record(
                "v12", v12, "feature", "baseline_seq", candidate, model,
                rng, args.bootstrap_samples,
            )
            if record:
                comparisons.append(record)

    v13 = frames["v13"]
    for model in sorted(v13["model"].unique()):
        for candidate in sorted(set(v13["combo"]) - {"seq_303d"}):
            record = paired_record(
                "v13", v13, "combo", "seq_303d", candidate, model,
                rng, args.bootstrap_samples,
            )
            if record:
                comparisons.append(record)

    manifest_path = root / "manifest.csv"
    completeness_path = root / "run_completeness.csv"
    paired_path = root / "paired_summary.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    pd.DataFrame(completeness).to_csv(completeness_path, index=False)
    pd.DataFrame(comparisons).sort_values(
        ["source_run", "mean_delta"], ascending=[True, False]
    ).to_csv(paired_path, index=False)

    failed_runs = [row["run"] for row in completeness if not row["complete_30_seeds"]]
    summary = {
        "runs_checked": len(completeness),
        "all_complete_30_seeds": not failed_runs,
        "incomplete_runs": failed_runs,
        "manifest": manifest_path.name,
        "run_completeness": completeness_path.name,
        "paired_summary": paired_path.name,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failed_runs:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

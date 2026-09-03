"""Run one configured experiment into an immutable run directory.

The runner records the exact config, data hash, environment, command, output,
and completion state. It deliberately does not overwrite an existing run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DATA = PROJECT_ROOT / "data/proxy2000_v2/fad_proxy2000_v2_full.csv"

# Windows may inherit a GBK console even though experiment logs are UTF-8.
# Keep console output and captured child output on one explicit encoding.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    for key in ("name", "entrypoint", "arguments"):
        if key not in config:
            raise ValueError(f"missing required config field: {key}")
    return config


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not cleaned:
        raise ValueError("experiment name becomes empty after normalization")
    return cleaned


def command_args(arguments: dict) -> list[str]:
    result: list[str] = []
    for key, value in arguments.items():
        option = f"--{key}"
        if isinstance(value, bool):
            if value:
                result.append(option)
        elif value is not None:
            result.extend([option, str(value)])
    return result


def environment_manifest() -> dict:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "lightgbm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "packages": packages,
    }


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={PROJECT_ROOT.as_posix()}",
                "rev-parse",
                "HEAD",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a registered FAD experiment")
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()
    config = load_config(config_path)

    entrypoint = (PROJECT_ROOT / config["entrypoint"]).resolve()
    if PROJECT_ROOT.resolve() not in entrypoint.parents or not entrypoint.is_file():
        raise FileNotFoundError(f"entrypoint is missing or outside project: {entrypoint}")
    if not CANONICAL_DATA.is_file():
        raise FileNotFoundError(f"canonical data is missing: {CANONICAL_DATA}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = safe_name(args.run_id or f"{stamp}_{config['name']}")
    run_dir = PROJECT_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    resolved_config = dict(config)
    resolved_arguments = dict(config["arguments"])
    resolved_arguments["output"] = str((run_dir / "metrics.csv").relative_to(PROJECT_ROOT))
    if config.get("supports_log_dir"):
        resolved_arguments["log-dir"] = str((run_dir / "checkpoints").relative_to(PROJECT_ROOT))
    resolved_config["arguments"] = resolved_arguments

    command = [sys.executable, str(entrypoint), *command_args(resolved_arguments)]
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "source_config": str(config_path),
        "command": command,
        "git_commit": git_commit(),
        "canonical_data": {
            "path": str(CANONICAL_DATA.relative_to(PROJECT_ROOT)),
            "size_bytes": CANONICAL_DATA.stat().st_size,
            "sha256": sha256(CANONICAL_DATA),
        },
        "environment": environment_manifest(),
    }
    (run_dir / "config.json").write_text(
        json.dumps(resolved_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log_path = run_dir / "stdout.log"
    print(f"[run] {run_id}")
    print(f"[data] {metadata['canonical_data']['sha256']}")
    print(f"[output] {run_dir}")

    child_env = os.environ.copy()
    child_env["PYTHONUTF8"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"
    # The runner is commonly launched under nohup on Linux.  Force the child
    # experiment to stream progress instead of block-buffering many seeds.
    child_env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return_code = process.wait()

    metrics_path = run_dir / "metrics.csv"
    metrics_rows = None
    if metrics_path.is_file():
        with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
            metrics_rows = sum(1 for _ in csv.DictReader(handle))
    expected_rows = config.get("acceptance", {}).get("expected_rows")
    acceptance_ok = expected_rows is None or metrics_rows == expected_rows

    completion = {
        "completed_at": datetime.now().isoformat(),
        "return_code": return_code,
        "metrics_exists": metrics_path.is_file(),
        "metrics_rows": metrics_rows,
        "expected_rows": expected_rows,
        "acceptance_ok": acceptance_ok,
    }
    marker = "DONE" if return_code == 0 and completion["metrics_exists"] and acceptance_ok else "FAILED"
    (run_dir / marker).write_text(
        json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[{marker}] return_code={return_code}")
    return 0 if marker == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())

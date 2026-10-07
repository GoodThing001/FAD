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

import pandas as pd


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


def git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}",
             "status", "--porcelain"],
            cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def registry_hash(dataset_id: str = "proxy2000_v2_full") -> str | None:
    """Expected canonical SHA256 from data/registry.csv (None when unavailable)."""
    path = PROJECT_ROOT / "data/registry.csv"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("dataset_id") == dataset_id:
                return (row.get("sha256") or "").strip().upper() or None
    return None


def source_manifest(entrypoint: Path) -> dict:
    """Hash the entrypoint plus every mainline python file it may import."""
    files: dict[str, str] = {}
    targets = [entrypoint]
    for pattern in ("scripts/**/*.py", "src/**/*.py"):
        targets.extend(sorted(PROJECT_ROOT.glob(pattern)))
    for path in targets:
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        if rel not in files:
            try:
                files[rel] = sha256(path)
            except OSError:
                continue
    return {"files": files, "n_files": len(files)}


def require_registry_hash(config: dict) -> str | None:
    """Formal runs must be able to name the dataset they used.

    A missing registry entry used to be reported as `registry_match: true`, which
    made an unregistered dataset look verified.
    """
    expected = registry_hash()
    if expected is None and not config.get("allow_missing_registry", False):
        raise ValueError(
            "canonical data is not registered in data/registry.csv (no entry for "
            f"{CANONICAL_DATA.name}); add the registry entry or set "
            "allow_missing_registry=true for a non-formal smoke run")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a registered FAD experiment")
    parser.add_argument("config", type=Path, nargs="?")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume-run", default=None,
                        help="reuse an existing runs/<run_id> with --resume (loop checkpoints)")
    parser.add_argument("--ingest", default=None,
                        help="with --resume-run: measurement CSV to absorb before continuing")
    parser.add_argument("--pin-protocol", default=None,
                        help="with --resume-run: audited protocol change (rewrites the "
                             "placeholder protocol in config/fingerprint/ledger/batches)")
    args = parser.parse_args()

    if args.resume_run:
        return resume_run(args.resume_run, args.ingest, args.pin_protocol)
    if args.config is None:
        parser.error("a config path is required unless --resume-run is used")

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
    canonical_hash = sha256(CANONICAL_DATA)
    expected_hash = require_registry_hash(config)
    if expected_hash and canonical_hash.upper() != expected_hash:
        raise ValueError(
            "canonical data SHA256 does not match data/registry.csv: "
            f"computed {canonical_hash.upper()}, registry {expected_hash}")
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "source_config": str(config_path),
        "command": command,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "source_manifest": source_manifest(entrypoint),
        "canonical_data": {
            "path": str(CANONICAL_DATA.relative_to(PROJECT_ROOT)),
            "size_bytes": CANONICAL_DATA.stat().st_size,
            "sha256": canonical_hash,
            "registry_sha256": expected_hash,
            "registry_match": (expected_hash is not None
                               and canonical_hash.upper() == expected_hash),
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

    return_code = run_child(command, log_path)
    return finish_run(run_dir, config, return_code)


def run_child(command: list[str], log_path: Path) -> int:
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
        return process.wait()


def finish_run(run_dir: Path, config: dict, return_code: int) -> int:
    """Apply the acceptance rules and write exactly one marker file."""
    metrics_path = run_dir / "metrics.csv"
    metrics_rows = None
    if metrics_path.is_file():
        with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
            metrics_rows = sum(1 for _ in csv.DictReader(handle))
    acceptance = config.get("acceptance", {}) or {}
    expected_rows = acceptance.get("expected_rows")
    expected_marker = acceptance.get("expect_marker")
    expected_exit = acceptance.get("expect_exit_code")
    acceptance_ok = (expected_rows is None or metrics_rows == expected_rows)
    if expected_exit is not None and return_code != expected_exit:
        acceptance_ok = False

    # A prospective loop parks its batch and exits 3 while labels are pending:
    # that is neither DONE (work finished) nor FAILED (error).  Only the *loop's*
    # marker counts here — `run_dir/PARKED` is this runner's own marker file.
    log_dir = run_dir / "checkpoints"
    configured_log_dir = (config.get("arguments") or {}).get("log-dir")
    if configured_log_dir:
        log_dir = PROJECT_ROOT / configured_log_dir
    parked_payload = None
    if (log_dir / "PARKED").is_file():
        try:
            parked_payload = json.loads((log_dir / "PARKED").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            parked_payload = {}
    is_parked = return_code == 3 or (parked_payload is not None and return_code == 0)

    # Terminal-state semantics are per-marker, not global:
    #  * a run that will be marked DONE must have `retrained` in the checkpoint —
    #    otherwise `--no-post-retrain-eval` (or a future config) could yield DONE
    #    while the ledger rows are still pending;
    #  * a PARKED run is *supposed* to sit in `awaiting_labels`/`labels_ingested`,
    #    so that must not count as an acceptance failure.
    state_path = log_dir / "state.json"
    checkpoint_status = None
    if state_path.is_file():
        try:
            checkpoint_status = json.loads(
                state_path.read_text(encoding="utf-8")).get("status")
        except (OSError, json.JSONDecodeError):
            checkpoint_status = None
    terminal_state_ok = True
    if not is_parked and return_code == 0 and state_path.is_file():
        if checkpoint_status != "retrained":
            terminal_state_ok = False
            acceptance_ok = False

    completion = {
        "completed_at": datetime.now().isoformat(),
        "return_code": return_code,
        "metrics_exists": metrics_path.is_file(),
        "metrics_rows": metrics_rows,
        "expected_rows": expected_rows,
        "acceptance_ok": acceptance_ok,
        "parked": is_parked,
        "parked_detail": parked_payload,
        "terminal_state_ok": terminal_state_ok,
        "checkpoint_status": checkpoint_status,
    }
    if is_parked:
        marker = "PARKED"
    elif return_code == 0 and completion["metrics_exists"] and acceptance_ok:
        marker = "DONE"
    else:
        marker = "FAILED"
    if expected_marker is not None and marker != expected_marker:
        # e.g. a prospective batch that returned 0/DONE instead of parking.
        acceptance_ok = False
        marker = "FAILED"
    completion["expected_marker"] = expected_marker
    completion["expected_exit_code"] = expected_exit
    completion["acceptance_ok"] = acceptance_ok
    for stale in ("DONE", "PARKED", "FAILED"):
        (run_dir / stale).unlink(missing_ok=True)
    (run_dir / marker).write_text(
        json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[{marker}] return_code={return_code}")
    if marker == "DONE":
        return 0
    if marker == "PARKED":
        return 3
    return 1


def pin_protocol(run_dir: Path, new_protocol: str) -> dict:
    """Audited protocol change for a parked run (P0 follow-up).

    The lab protocol was unknown when the 96-candidate batch was exported, so the
    batch carries a placeholder. Pinning the real protocol string must never be a
    silent edit of the run directory: it (a) appends an immutable audit line,
    (b) rewrites the protocol metadata in config.json / fingerprint.json /
    ledger.csv / batch CSVs / **both** batch-manifest copies (the checkpoint one
    AND the oracle's `pending_batches/batches/batch_<id>/batch_manifest.json`,
    including the fingerprint embedded in each manifest), and (c) re-matches the
    checkpoint fingerprint so the loop's resume check still holds for everything
    else.

    Guard rails: the protocol must be non-blank, and pinning is only allowed while
    the run is still PARKED and no measurement has been returned yet — a protocol
    must never be retroactively rewritten over an already-labelled batch.
    """
    new_protocol = str(new_protocol).strip()
    if not new_protocol:
        raise ValueError("--pin-protocol requires a non-blank protocol string")

    ckpt = run_dir / "checkpoints"
    if not (run_dir / "PARKED").is_file() and not (ckpt / "PARKED").is_file():
        raise ValueError(
            "protocol pinning is only allowed on a run that is still PARKED "
            f"({run_dir.name} has no PARKED marker)")
    returned_files = (sorted(ckpt.glob("pending_batches/batches/batch_*/labels.csv"))
                      + sorted(ckpt.glob("pending_batches/batches/batch_*/failures.csv")))
    for returned in returned_files:
        try:
            if pd.read_csv(returned).shape[0] > 0:
                raise ValueError(
                    "protocol pinning is not allowed once measurements have been "
                    f"returned ({returned.name} is non-empty); re-issue a new batch "
                    "with the agreed protocol instead of rewriting a labelled one")
        except pd.errors.EmptyDataError:
            continue

    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    arguments = dict(config.get("arguments", {}) or {})
    old_protocol = arguments.get("protocol-id", "unassigned")
    old_approved = arguments.get("approved-protocols", "")
    if old_protocol == new_protocol and old_approved == new_protocol:
        return {"changed": False, "protocol": new_protocol}

    fp_path = ckpt / "fingerprint.json"
    if not fp_path.is_file():
        raise FileNotFoundError(f"{fp_path} is missing; refusing to pin a protocol "
                                "on an unfingerprinted checkpoint")

    audit_line = {
        "at": datetime.now().isoformat(),
        "from_protocol_id": old_protocol,
        "to_protocol_id": new_protocol,
        "from_approved_protocols": old_approved,
        "to_approved_protocols": new_protocol,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "reason": "lab protocol confirmed after batch export",
    }
    audit_path = ckpt / "protocol_changes.jsonl"
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(audit_line, ensure_ascii=False) + "\n")

    # config.json + fingerprint.json keep the new pinned values
    arguments["protocol-id"] = new_protocol
    arguments["approved-protocols"] = new_protocol
    config["arguments"] = arguments
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    fingerprint = json.loads(fp_path.read_text(encoding="utf-8"))
    fingerprint.setdefault("config", {})["protocol_id"] = new_protocol
    fingerprint.setdefault("config", {})["approved_protocols"] = new_protocol
    fingerprint["protocol_pin_audit"] = {
        "at": audit_line["at"], "from": old_protocol, "to": new_protocol,
        "audit_file": "checkpoints/protocol_changes.jsonl"}
    fp_path.write_text(json.dumps(fingerprint, ensure_ascii=False, indent=2),
                       encoding="utf-8")

    def _rewrite_manifest(path: Path) -> bool:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["protocol_id"] = new_protocol
        manifest["approved_protocols"] = [new_protocol]
        manifest["protocol_pinned_to"] = new_protocol
        manifest["protocol_pinned_at"] = audit_line["at"]
        # the fingerprint embedded in the manifest must agree with the manifest
        # itself — and with checkpoints/fingerprint.json
        embedded = manifest.get("fingerprint")
        if isinstance(embedded, dict):
            embedded.setdefault("config", {})["protocol_id"] = new_protocol
            embedded.setdefault("config", {})["approved_protocols"] = new_protocol
            embedded["protocol_pin_audit"] = fingerprint["protocol_pin_audit"]
            manifest["fingerprint"] = embedded
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return True

    # ledger / batch CSVs carry the protocol string too
    rewritten = 0
    batch_handoffs = sorted(ckpt.glob(
        "pending_batches/batches/batch_*/batch_handoff.csv"))
    for csv_path in (sorted(ckpt.glob("round_*_batch.csv")) + batch_handoffs
                     + ([ckpt / "ledger.csv"] if (ckpt / "ledger.csv").is_file()
                        else [])):
        frame = pd.read_csv(csv_path)
        if "protocol_id" in frame.columns:
            frame["protocol_id"] = new_protocol
            frame.to_csv(csv_path, index=False)
            rewritten += len(frame)
    # BOTH manifest copies: the checkpoint round manifest and the oracle's copy
    n_manifests = 0
    for manifest_path in sorted(ckpt.glob("round_*_batch_manifest.json")):
        _rewrite_manifest(manifest_path)
        n_manifests += 1
    for manifest_path in sorted(ckpt.glob(
            "pending_batches/batches/batch_*/batch_manifest.json")):
        _rewrite_manifest(manifest_path)
        n_manifests += 1
    print(f"[pin-protocol] {old_protocol!r} -> {new_protocol!r} "
          f"(audited: {audit_path.name}, {rewritten} ledger/batch rows rewritten)")
    return {"changed": True, "protocol": new_protocol, "rows_rewritten": rewritten}


def resume_run(run_id: str, ingest: str | None, pin: str | None = None) -> int:
    """Continue an existing run in place (safe resume entry point).

    The run directory, its config and its checkpoint fingerprint are reused as-is;
    the loop refuses to continue if the configuration, data hash, pool or holdout no
    longer match the fingerprint written when the checkpoint was created.
    `pin` goes through `pin_protocol` (audited metadata rewrite) instead of a manual
    config edit, so the fingerprint keeps matching.
    """
    run_dir = PROJECT_ROOT / "runs" / safe_name(run_id)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"no run directory at {run_dir}")
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{run_dir} has no config.json; cannot resume safely")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    entrypoint = (PROJECT_ROOT / config["entrypoint"]).resolve()
    if not entrypoint.is_file():
        raise FileNotFoundError(f"entrypoint is missing: {entrypoint}")
    if not (run_dir / "checkpoints" / "state.json").is_file():
        raise FileNotFoundError(
            f"{run_dir}/checkpoints/state.json is missing; nothing to resume "
            "(expected a checkpointed run)")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if pin:
        pin_report = pin_protocol(run_dir, pin)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        print(f"[pin-protocol] audited change recorded: {pin_report}")
    resolved_arguments = dict(config["arguments"])
    resolved_arguments["output"] = str((run_dir / "metrics.csv").relative_to(PROJECT_ROOT))
    if config.get("supports_log_dir"):
        resolved_arguments["log-dir"] = str((run_dir / "checkpoints").relative_to(PROJECT_ROOT))
    resolved_arguments["resume"] = True
    if ingest:
        resolved_arguments["ingest"] = str(Path(ingest).resolve())
    command = [sys.executable, str(entrypoint), *command_args(resolved_arguments)]

    history_path = run_dir / "resume_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else []
    history.append({"at": datetime.now().isoformat(), "command": command,
                    "ingest": ingest, "pin_protocol": pin,
                    "git_commit": git_commit(), "git_dirty": git_dirty(),
                    "source_manifest": source_manifest(entrypoint)})
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[resume] {run_dir.name}")
    print(f"[command] {' '.join(command)}")
    # Drop the previous verdict before running: if the child dies half-way the run
    # must not keep advertising its old marker.
    for stale in ("DONE", "PARKED", "FAILED"):
        (run_dir / stale).unlink(missing_ok=True)
    return_code = run_child(command, run_dir / f"stdout.resume_{stamp}.log")

    # Data hash and source hashes are re-checked on every resume.
    canonical_hash = sha256(CANONICAL_DATA)
    expected_hash = require_registry_hash(config)
    if expected_hash and canonical_hash.upper() != expected_hash:
        raise ValueError("canonical data SHA256 no longer matches data/registry.csv")
    resume_acceptance = dict(config.get("acceptance", {}) or {})
    if ingest:
        # after a successful ingest the marker may legitimately flip to DONE,
        # so a pinned PARKED expectation would be wrong
        resume_acceptance.pop("expect_marker", None)
        resume_acceptance.pop("expect_exit_code", None)
    # a resume legitimately appends history rows; only enforce a row count when the
    # config explicitly provides one for the resume path
    resume_acceptance["expected_rows"] = resume_acceptance.get("resume_expected_rows")
    config = {**config, "acceptance": resume_acceptance}
    return finish_run(run_dir, config, return_code)


if __name__ == "__main__":
    raise SystemExit(main())

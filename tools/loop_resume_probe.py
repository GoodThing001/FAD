"""End-to-end probe of the prospective park/resume path on the FAD server.

Sequence:
  1. read the parked batch from runs/srv_loop_park/checkpoints/batches/batch_*/batch.csv
  2. fabricate a 5-row ingest file (synthetic values; this is a *plumbing* probe,
     the numbers are NOT measurements)
  3. resume the same checkpoint twice with --ingest
  4. assert: awaiting 50 -> 45 -> 45, same batch_id, no new batch directory,
     second resume reports duplicate=5 and accepts nothing

Usage: python tools/loop_resume_probe.py
"""
from __future__ import annotations

import csv
import io
import json
import posixpath
import socket
import sys
import warnings
from pathlib import Path

import paramiko

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
CFG = json.load(open(ROOT / ".vscode" / "sftp.json", encoding="utf-8"))
REMOTE = "/home/hzeng/project/FAD_CLEAN"
PARK = sys.argv[1] if len(sys.argv) > 1 else "runs/srv_loop_park"
CKPT = f"{PARK}/checkpoints"
# ParkOracle root (see oracle.py: ParkOracle(root=<log-dir>/pending_batches))
ORACLE = f"{CKPT}/pending_batches"
PY = "/home/hzeng/envs/FAD_env/bin/python"

RESUME_ARGS = (
    "scripts/loop/loop.py --mode prospective --resume "
    f"--log-dir {CKPT} --ingest /tmp/probe_ingest_5.csv "
    f"--output {PARK}/resume_probe_metrics.csv "
    "--candidate-source mutate --rounds 1 --budget 50 "
    "--n-init 400 --n-holdout 300 --pool-size 500 --surrogate xgb "
    "--n-members 10 --oracle park --seed 1 --data-dir data/proxy2000_v2 "
    "--strategy greedy"
)


def connect() -> paramiko.SSHClient:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(30)
    sock.connect((CFG["host"], 22))
    tr = paramiko.Transport(sock)
    tr.start_client(timeout=30)
    tr.auth_password(CFG["username"], CFG["password"])
    client = paramiko.SSHClient()
    client._transport = tr
    return client


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 1800) -> str:
    _, out, err = client.exec_command(cmd, timeout=timeout)
    return out.read().decode("utf-8", "replace") + err.read().decode("utf-8", "replace")


def main() -> int:
    client = connect()
    failures: list[str] = []
    try:
        batches = run(client, f"ls -d {REMOTE}/{ORACLE}/batches/batch_* 2>/dev/null").split()
        if len(batches) != 1:
            print(f"[fail] expected exactly 1 parked batch, found {len(batches)}")
            return 1
        batch_dir = batches[0].strip()
        batch_id = posixpath.basename(batch_dir).replace("batch_", "")
        raw = run(client, f"cat {batch_dir}/batch.csv")
        rows = list(csv.DictReader(io.StringIO(raw)))
        print(f"[park] batch_id={batch_id} rows={len(rows)}")
        if len(rows) != 50:
            failures.append(f"expected 50 parked rows, found {len(rows)}")

        probe = rows[:5]
        base_state = json.loads(run(client, f"cat {REMOTE}/{CKPT}/state.json"))
        base_awaiting = set(base_state.get("awaiting", {}))
        base_labeled = len(base_state.get("labeled_sequences", []))
        n_new_expected = sum(1 for row in probe if row["sequence"] in base_awaiting)
        print(f"[baseline] awaiting={len(base_awaiting)} labeled={base_labeled} "
              f"probe_rows_still_pending={n_new_expected}")

        buf = io.StringIO()
        buf.write("batch_id,candidate_id,sequence,gbsa,measurement_status,protocol_id\n")
        for i, row in enumerate(probe):
            # synthetic placeholder values: plumbing probe only
            buf.write(f"{batch_id},{row['candidate_id']},{row['sequence']},"
                      f"{-20.0 - i},pass,probe_synthetic\n")
        payload = buf.getvalue()
        quoted = "EOF_PROBE"
        write = (f"cat > /tmp/probe_ingest_5.csv <<'{quoted}'\n{payload}{quoted}\n")
        print(run(client, write, timeout=120).strip())

        expected_awaiting = len(base_awaiting) - n_new_expected
        first_labeled = None
        first_rows = None
        for attempt in (1, 2):
            label = "first" if attempt == 1 else "second (idempotency)"
            print(f"\n===== resume attempt {attempt}: {label} =====")
            shell = (f"cd {REMOTE} && {{ {PY} {RESUME_ARGS} ; echo rc=$?; }} 2>&1 "
                     f"| tail -n 14")
            chunk = run(client, shell)
            print(chunk)
            if "rc=3" not in chunk:
                failures.append(f"attempt {attempt}: expected rc=3 (PARKED) in output")
            progress = json.loads(run(client, f"cat {REMOTE}/{CKPT}/progress.json"))
            state = json.loads(run(client, f"cat {REMOTE}/{CKPT}/state.json"))
            awaiting = len(state.get("awaiting", {}))
            labeled = len(state.get("labeled_sequences", []))
            n_rows = int(run(client, f"wc -l < {REMOTE}/{PARK}/resume_probe_metrics.csv").strip())
            print(f"[state] status={progress['status']} round={progress['completed_rounds']} "
                  f"labeled={labeled} awaiting={awaiting} metrics_rows={n_rows} "
                  f"open_batch={state.get('open_batch')}")
            if awaiting != expected_awaiting:
                failures.append(f"attempt {attempt}: awaiting={awaiting} != {expected_awaiting}")
            if attempt == 1:
                first_labeled, first_rows = labeled, n_rows
                if labeled != base_labeled + n_new_expected:
                    failures.append(f"attempt 1: labeled={labeled} != "
                                    f"{base_labeled + n_new_expected}")
            else:
                if labeled != first_labeled:
                    failures.append(f"attempt 2 not idempotent: labeled "
                                    f"{first_labeled} -> {labeled}")
                if n_rows != first_rows:
                    failures.append(f"attempt 2 appended log rows: {first_rows} -> {n_rows}")
            if state.get("open_batch") != batch_id:
                failures.append(f"attempt {attempt}: open_batch changed: {state.get('open_batch')}")
            n_batches = int(run(client, f"ls -d {REMOTE}/{ORACLE}/batches/batch_* | wc -l").strip())
            if n_batches != 1:
                failures.append(f"attempt {attempt}: new batch directories appeared ({n_batches})")

        report = run(client, f"cat {REMOTE}/{ORACLE}/ingest_report.json")
        print("\n[ingest_report] " + report.strip().replace("\n", " "))
        metrics = run(client, f"cat {REMOTE}/{PARK}/resume_probe_metrics.csv")
        print("[resume metrics]\n" + metrics)
    finally:
        client.close()

    if failures:
        print("[FAIL] " + "; ".join(failures))
        return 1
    print("[ok] park/resume probe passed: same batch, awaiting 50->45->45, no new batch")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""One-off audit helper: upload configs and launch polite background jobs on the FAD server."""
import json
import socket
import sys
import warnings
from pathlib import Path

import paramiko

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
CFG = json.load(open(ROOT / ".vscode" / "sftp.json"))
REMOTE = "/home/hzeng/project/FAD_CLEAN"


def connect():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(30)
    sock.connect((CFG["host"], 22))
    tr = paramiko.Transport(sock)
    tr.start_client(timeout=30)
    tr.auth_password(CFG["username"], CFG["password"])
    client = paramiko.SSHClient()
    client._transport = tr
    return client


def run(client, cmd, timeout=600):
    _, out, err = client.exec_command(cmd, timeout=timeout)
    return out.read().decode("utf-8", "replace") + err.read().decode("utf-8", "replace")


def main():
    c = connect()
    try:
        sftp = c.open_sftp()
        for name in ("audit_baseline_rf_100seed.json", "audit_v8_10seed.json"):
            local = ROOT / "configs" / "experiments" / name
            remote = f"{REMOTE}/configs/experiments/{name}"
            sftp.put(str(local), remote)
            print("uploaded", name)
        sftp.close()

        py = "/home/hzeng/envs/FAD_env/bin/python"
        cmd = (
            f"cd {REMOTE} && "
            f"nohup nice -n 19 {py} scripts/run_experiment.py "
            f"configs/experiments/audit_baseline_rf_100seed.json --run-id audit_baseline_rf_100seed "
            f"> /tmp/audit_baseline_rf_100seed.launcher.log 2>&1 & "
            f"nohup nice -n 19 {py} scripts/run_experiment.py "
            f"configs/experiments/audit_v8_10seed.json --run-id audit_v8_10seed "
            f"> /tmp/audit_v8_10seed.launcher.log 2>&1 & "
            f"echo launched; sleep 5; ps -eo pid,args | grep -E 'audit_(baseline_rf|v8_10seed)' | grep -v grep"
        )
        print(run(c, cmd))
    finally:
        c.close()


if __name__ == "__main__":
    main()

"""Download audit run artifacts from the server into local evidence."""
import json
import socket
import warnings
from pathlib import Path

import paramiko

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
CFG = json.load(open(ROOT / ".vscode" / "sftp.json"))
REMOTE = "/home/hzeng/project/FAD_CLEAN"


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(30)
    sock.connect((CFG["host"], 22))
    tr = paramiko.Transport(sock)
    tr.start_client(timeout=30)
    tr.auth_password(CFG["username"], CFG["password"])
    client = paramiko.SSHClient()
    client._transport = tr
    sftp = client.open_sftp()
    local_dir = ROOT / "evidence" / "audit_20260924"
    for sub in ("audit_baseline_rf_100seed", "audit_v8_10seed"):
        (local_dir / sub).mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "run_manifest.json", "metrics.csv",
                     "metrics_summary.json", "stdout.log", "DONE", "FAILED"):
            remote = f"{REMOTE}/runs/{sub}/{name}"
            try:
                sftp.get(remote, str(local_dir / sub / name))
                print("downloaded", sub, name)
            except FileNotFoundError:
                pass
    sftp.close()
    client.close()


if __name__ == "__main__":
    main()

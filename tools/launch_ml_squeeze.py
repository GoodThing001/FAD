"""Launch the ML-squeeze experiment batch on the FAD server (polite, nice 19)."""
import json
import socket
import time
import warnings
from pathlib import Path

import paramiko

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
CFG = json.load(open(ROOT / ".vscode" / "sftp.json"))
REMOTE = "/home/hzeng/project/FAD_CLEAN"
PY = "/home/hzeng/envs/FAD_env/bin/python"

JOBS = [
    ("eval_negative_controls", "scripts/reproduction/eval_negative_controls.py",
     "--seeds 10 --output runs/ml_squeeze/negative_controls.json"),
    ("eval_smoothing_variants", "scripts/reproduction/eval_smoothing_variants.py",
     "--outer-seeds 30 --output runs/ml_squeeze/smoothing_variants.json"),
    ("eval_rank_objectives2", "scripts/reproduction/eval_rank_objectives2.py",
     "--outer-seeds 30 --output runs/ml_squeeze/rank_objectives2.json"),
    ("eval_extrapolation", "scripts/reproduction/eval_extrapolation.py",
     "--outer-seeds 30 --output runs/ml_squeeze/extrapolation.json"),
    ("eval_sparse_epistasis", "scripts/reproduction/eval_sparse_epistasis.py",
     "--outer-seeds 30 --output runs/ml_squeeze/sparse_epistasis.json"),
    ("eval_fusion_smoothing", "scripts/reproduction/eval_fusion_smoothing.py",
     "--outer-seeds 30 --output runs/ml_squeeze/fusion_smoothing.json"),
]


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


def main():
    c = connect()
    for name, entry, extra in JOBS:
        mod = entry[:-3].replace("/", ".")
        cmd = (f"cd {REMOTE} && mkdir -p runs/ml_squeeze && "
               f"nohup nice -n 19 {PY} -m {mod} {extra} "
               f"> /tmp/ml_{name}.log 2>&1 < /dev/null & echo LAUNCHED_{name}")
        c.exec_command(cmd, timeout=30)  # fire and forget; do not read
        time.sleep(2)
    c.close()
    print("all launch commands sent")


if __name__ == "__main__":
    main()

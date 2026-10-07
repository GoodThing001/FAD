"""Launch the three loop configs through run_experiment.py on the FAD server."""
import json
import socket
import time
import warnings

import paramiko

warnings.filterwarnings("ignore")

CFG = json.load(open(".vscode/sftp.json"))
REMOTE = "/home/hzeng/project/FAD_CLEAN"
PY = "/home/hzeng/envs/FAD_env/bin/python"

CONFIGS = [
    ("loop_smoke", "configs/experiments/loop_smoke.json"),
    ("loop_prospective_park", "configs/experiments/loop_prospective_park.json"),
    ("loop_retrospective_5round", "configs/experiments/loop_retrospective_5round.json"),
]


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(30)
    sock.connect((CFG["host"], 22))
    tr = paramiko.Transport(sock)
    tr.start_client(timeout=30)
    tr.auth_password(CFG["username"], CFG["password"])
    c = paramiko.SSHClient()
    c._transport = tr
    for run_id, cfg in CONFIGS:
        cmd = (f"cd {REMOTE} && nohup nice -n 19 env PYTHONUNBUFFERED=1 {PY} "
               f"scripts/run_experiment.py {cfg} --run-id {run_id} "
               f"> /tmp/{run_id}.launcher.log 2>&1 < /dev/null & echo L_{run_id}")
        c.exec_command(cmd, timeout=30)
        time.sleep(2)
    c.close()
    print("launched:", ", ".join(r for r, _ in CONFIGS))


if __name__ == "__main__":
    main()

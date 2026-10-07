"""Sync selected local paths to the FAD server and run remote commands.

Usage:
  python tools/sync_remote.py push scripts/loop configs/experiments/loop_smoke.json
  python tools/sync_remote.py pull runs/srv_loop_park_clean/PARKED evidence/loop_20260930/server
  python tools/sync_remote.py run "cd /home/hzeng/project/FAD_CLEAN && git status --short"

Only the paths you name are uploaded (plus any new files inside named directories);
credentials come from .vscode/sftp.json and are never printed.
"""
from __future__ import annotations

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


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 900) -> str:
    _, out, err = client.exec_command(cmd, timeout=timeout)
    return out.read().decode("utf-8", "replace") + err.read().decode("utf-8", "replace")


def _mkdirs(sftp, remote_dir: str) -> None:
    parts = remote_dir.strip("/").split("/")
    cur = ""
    for part in parts:
        cur = f"{cur}/{part}"
        try:
            sftp.stat(cur)
        except OSError:
            sftp.mkdir(cur)


def push(client: paramiko.SSHClient, paths: list[str]) -> int:
    sftp = client.open_sftp()
    n = 0
    try:
        for raw in paths:
            local = ROOT / raw
            if not local.exists():
                raise FileNotFoundError(local)
            files = [local] if local.is_file() else [
                p for p in sorted(local.rglob("*"))
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
            for path in files:
                rel = path.relative_to(ROOT).as_posix()
                remote = posixpath.join(REMOTE, rel)
                _mkdirs(sftp, posixpath.dirname(remote))
                sftp.put(str(path), remote)
                n += 1
                print("pushed", rel)
    finally:
        sftp.close()
    return n


def pull(client: paramiko.SSHClient, remote_paths: list[str], local_dir: str) -> int:
    sftp = client.open_sftp()
    out = ROOT / local_dir
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    try:
        for remote in remote_paths:
            remote = remote if remote.startswith("/") else posixpath.join(REMOTE, remote)
            name = posixpath.basename(remote.rstrip("/"))
            local = out / name
            try:
                sftp.get(remote, str(local))
            except IOError as exc:
                print(f"[skip] {remote}: {exc}")
                continue
            print(f"pulled {posixpath.relpath(remote, REMOTE)} -> "
                  f"{(out / name).relative_to(ROOT).as_posix()} ({local.stat().st_size} B)")
            n += 1
    finally:
        sftp.close()
    return n


def main() -> int:
    action = sys.argv[1] if len(sys.argv) > 1 else "push"
    args = sys.argv[2:]
    client = connect()
    try:
        if action == "push":
            n = push(client, args)
            print(f"[ok] {n} files -> {REMOTE}")
        elif action == "pull":
            if len(args) < 2:
                raise SystemExit("pull needs <remote...> <local-dir>")
            n = pull(client, args[:-1], args[-1])
            print(f"[ok] {n} files -> {args[-1]}")
        elif action == "run":
            print(run(client, " ".join(args)))
        else:
            raise SystemExit(f"unknown action: {action}")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

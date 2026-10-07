"""Minimal SSH helper for the FAD server (Tailscale, password auth).

Reads credentials from .vscode/sftp.json (never prints the password) and runs
a command via paramiko over an explicit IPv4 socket. The raw AF_INET socket is
required because getaddrinfo/create_connection intermittently time out on this
Windows host even though the TCP connect itself is instant.

Usage:
  python tools/ssh_helper.py "hostname; nvidia-smi -L"
"""
import json
import socket
import sys
import warnings
from pathlib import Path

import paramiko

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
CFG = json.load(open(ROOT / ".vscode" / "sftp.json"))


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


def run(client, cmd, timeout=300):
    _, out, err = client.exec_command(cmd, timeout=timeout)
    return out.read().decode("utf-8", "replace") + err.read().decode("utf-8", "replace")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "echo hi"
    c = connect()
    try:
        sys.stdout.write(run(c, cmd))
    finally:
        c.close()

"""Local markdown link checker (offline; skips http(s) and anchors).

Usage: python tools/check_doc_links.py [root]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

# Accept one nested pair (gkag145(2).pdf), angle delimiters and encoded paths.
PAT = re.compile(r"\[[^\]]+\]\((<[^>]+>|(?:[^()]|\([^()]*\))+)\)")


def main() -> int:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    files = [p for p in root.glob("docs/**/*.md")]
    files += [root / n for n in ("README.md", "CHANGELOG.md", "AGENTS.md",
                                 "PROJECT_STATUS.md", "scripts/README.md",
                                 "evidence/loop_20260930/README.md")]
    broken: list[tuple[str, str]] = []
    checked = 0
    for f in files:
        if not f.is_file():
            continue
        checked += 1
        text = f.read_text(encoding="utf-8", errors="replace")
        for m in PAT.finditer(text):
            target = unquote(m.group(1).strip().strip("<>").split("#", 1)[0])
            if target.startswith(("http://", "https://", "mailto:")) or not target:
                continue
            if not (f.parent / target).resolve().exists():
                broken.append((f.relative_to(root).as_posix(), target))
    print(f"checked {checked} files, {len(broken)} broken local links")
    for f, t in broken:
        print(f" - {f} -> {t}")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())

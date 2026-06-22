#!/usr/bin/env python3
"""Normalize Markdown LaTeX delimiters to GitHub-compatible dollar syntax."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = (ROOT / "README.md", ROOT / "docs")


def markdown_files() -> list[Path]:
    files: list[Path] = []
    for target in TARGETS:
        if target.is_file():
            files.append(target)
        elif target.exists():
            files.extend(sorted(target.rglob("*.md")))
    return files


def normalize(text: str) -> str:
    return text.replace(r"\[", "$$").replace(r"\]", "$$").replace(r"\(", "$").replace(r"\)", "$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report files that still need normalization")
    args = parser.parse_args()
    changed: list[Path] = []
    for path in markdown_files():
        original = path.read_text(encoding="utf-8")
        updated = normalize(original)
        if updated != original:
            changed.append(path)
            if not args.check:
                path.write_text(updated, encoding="utf-8")
    if changed:
        action = "needs normalization" if args.check else "normalized"
        for path in changed:
            print(f"{action}: {path.relative_to(ROOT)}")
    return int(args.check and bool(changed))


if __name__ == "__main__":
    raise SystemExit(main())

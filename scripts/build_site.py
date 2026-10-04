#!/usr/bin/env python3
"""Build site/ for GitHub Pages from the committed report pages (standard library only).

The report pages are written as page bodies (the form Claude artifacts take), so each one is
wrapped in a full HTML document here, with a small nav between them.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
PAGES = [  # (published file, source, nav label)
    ("index.html", ROOT / "reports" / "dashboard.html", "Breakout monitor"),
    ("scan.html", ROOT / "reports" / "scan_report.html", "Full scan"),
]
SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<style>
body {{ margin: 0; }}
.site-nav {{ max-width: 1180px; margin: 0 auto; padding: 14px 20px 0; display: flex; flex-wrap: wrap; gap: 6px 18px;
  font: 500 14px/1.4 "IBM Plex Sans", system-ui, sans-serif; }}
.site-nav [aria-current="page"] {{ color: inherit; font-weight: 700; text-decoration: none; }}
</style>
</head>
<body>
<nav class="site-nav" aria-label="Reports">{nav}</nav>
{content}
</body>
</html>
"""


def main() -> int:
    pages = [(name, src, label) for name, src, label in PAGES if src.exists()]
    if not pages:
        print("no report pages found under reports/; run `python -m monitor daily` first")
        return 1
    SITE.mkdir(exist_ok=True)
    for name, src, _ in pages:
        current = ' aria-current="page"'
        nav = " ".join(f'<a href="{n}"{current if n == name else ""}>{label}</a>' for n, _, label in pages)
        (SITE / name).write_text(SHELL.format(nav=nav, content=src.read_text()))
        print(f"site/{name} <- {src.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Breakout monitor for the contrarian cash scanner's names.

  python -m monitor daily               # score the watchlist; rescans and relearns when due
  python -m monitor daily --rescan      # force a full fundamental re-scan first
  python -m monitor daily --learn       # force the engine to re-fit
  python -m monitor add TICKER --note "why"
  python -m monitor remove TICKER
  python -m monitor list
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

from . import config as C
from . import report, run
from . import watchlist as W


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m monitor", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("daily", help="score the watchlist and write reports/")
    d.add_argument("--rescan", action="store_true", help="force a full re-scan")
    d.add_argument("--learn", action="store_true", help="force the engine to re-fit")
    a = sub.add_parser("add", help="monitor a ticker by hand")
    a.add_argument("ticker")
    a.add_argument("--note", default="")
    r = sub.add_parser("remove", help="stop monitoring a ticker")
    r.add_argument("ticker")
    sub.add_parser("list", help="print the active watchlist")
    args = ap.parse_args(argv)
    today = dt.date.today()

    if args.cmd == "daily":
        res = run.daily(force_rescan=args.rescan, force_learn=args.learn, today=today)
        print(report.write(res))
        pages = [p for p in (C.REPORTS / "dashboard.html", C.REPORTS / "scan_report.html") if p.exists()]
        print("Pages: " + ", ".join(str(p.relative_to(C.ROOT)) for p in pages))
        return 0
    wl = W.load()
    if args.cmd == "add":
        W.add_manual(wl, args.ticker, args.note, today)
    elif args.cmd == "remove":
        if args.ticker.upper() not in wl:
            print(f"{args.ticker.upper()} is not on the watchlist", file=sys.stderr)
            return 1
        W.remove(wl, args.ticker, today)
    else:
        for t in W.active(wl):
            e = wl[t]
            print(f"{t:6} {e['source']:7} added {e['added']}  {e.get('name', '')}")
        return 0
    W.save(wl)
    print(f"{args.ticker.upper()}: {wl[args.ticker.upper()]['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

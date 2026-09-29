"""The monitored names: seeded and refreshed from scans, plus the author's calls and manual adds."""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd

from . import config as C

SCAN_COLS = ["name", "description", "sector", "industry", "exchange", "close", "score", "tier", "n_checks",
             "timing", "fcf_yield", "sh_yield", "fwd_pe", "dd_52w", "target_upside", "nd_ebitda"]


def load() -> dict:
    return json.loads(C.WATCHLIST.read_text()) if C.WATCHLIST.exists() else {}


def save(wl: dict) -> None:
    C.WATCHLIST.parent.mkdir(parents=True, exist_ok=True)
    C.WATCHLIST.write_text(json.dumps(dict(sorted(wl.items())), indent=1, ensure_ascii=False) + "\n")


def active(wl: dict) -> list[str]:
    return [t for t, e in wl.items() if e["status"] == "active"]


def save_scan(final: pd.DataFrame) -> pd.DataFrame:
    """Keep a compact copy of the scan (top TRAIN_UNIVERSE names) in state/ for the next runs."""
    top = final.head(C.TRAIN_UNIVERSE).copy()
    top["why"] = top["thesis"].map(lambda x: " | ".join(x))
    top["risk"] = top["risks"].map(lambda x: " | ".join(x))
    top = top[SCAN_COLS + ["why", "risk"]]
    top.to_csv(C.SCAN, index=False, float_format="%.2f")
    return top


def load_scan() -> pd.DataFrame | None:
    return pd.read_csv(C.SCAN, keep_default_na=False, na_values=[""]) if C.SCAN.exists() else None


def qualifies(scan: pd.DataFrame) -> pd.Series:
    rank = pd.Series(range(len(scan)), index=scan.index)
    return (scan["tier"].isin(C.ADMIT_TIERS)
            | ((scan["tier"] == "B") & (scan["score"] >= C.ADMIT_MIN_SCORE_B))
            | (rank < C.ADMIT_TOP_N))


def _num(v, n=1):
    return None if pd.isna(v) else round(float(v), n)


def thesis(r) -> dict:
    split = lambda s: [x for x in str(s).split(" | ") if x and x != "nan"]
    return {"score": _num(r["score"]), "tier": r["tier"], "checks": int(r["n_checks"]), "timing": r["timing"],
            "fcf": _num(r["fcf_yield"]), "shy": _num(r["sh_yield"]), "pe": _num(r["fwd_pe"]),
            "dd": _num(r["dd_52w"]), "up": _num(r["target_upside"]), "nd": _num(r["nd_ebitda"]),
            "why": split(r["why"])[:5], "risk": split(r["risk"])}


def merge_scan(wl: dict, scan: pd.DataFrame, today: dt.date) -> tuple[list[str], list[str]]:
    """Add newly qualifying names, refresh the thesis of known ones, drop names that keep missing."""
    day = today.isoformat()
    ok = qualifies(scan)
    rank = {n: i + 1 for i, n in enumerate(scan["name"])}
    added, dropped = [], []
    for (_, r), good in zip(scan.iterrows(), ok):
        t = r["name"]
        e = wl.get(t)
        if e is None and not good:
            continue
        if e is None:
            e = wl[t] = {"ticker": t, "added": day, "source": "scan", "status": "active", "misses": 0}
            added.append(t)
        elif good and e["status"] == "dropped":
            e.update(status="active", misses=0, readded=day)
            added.append(t)
        e.update(name=r["description"], sector=r["sector"], exchange=r["exchange"], thesis=thesis(r),
                 scan_rank=rank[t], scanned=day)
        if good:
            e["misses"] = 0
    for t, e in wl.items():
        if e.get("source") != "scan" or e["status"] != "active":
            continue
        if t not in rank or not ok[scan["name"] == t].any():
            e["misses"] = e.get("misses", 0) + 1
            if t not in rank:
                e["scan_rank"] = None
            if e["misses"] >= C.DROP_AFTER_MISSES and not e.get("hold"):
                e.update(status="dropped", dropped=day)
                dropped.append(t)
    return added, dropped


def ensure_author_calls(wl: dict, today: dt.date) -> list[str]:
    """The author's own long calls stay on the list whatever the scan says."""
    from make_report import AUTHOR_CALLS
    added = []
    for t, ex, date, entry, note in AUTHOR_CALLS:
        if t not in wl:
            wl[t] = {"ticker": t, "added": today.isoformat(), "source": "author", "status": "active",
                     "exchange": ex, "misses": 0}
            added.append(t)
        wl[t]["author"] = {"date": date, "entry": entry, "note": note}
    return added


def add_manual(wl: dict, ticker: str, note: str, today: dt.date) -> None:
    t = ticker.upper()
    e = wl.setdefault(t, {"ticker": t, "added": today.isoformat(), "misses": 0})
    e.update(source="manual", status="active", note=note or e.get("note", ""))


def remove(wl: dict, ticker: str, today: dt.date) -> None:
    e = wl[ticker.upper()]
    e.update(status="dropped", dropped=today.isoformat(), source="manual")

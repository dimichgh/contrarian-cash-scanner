"""The trade ledger: every trigger followed as a trade from its close, kept after it closes.

A trade opens at the trigger-day close with a target and a stop sized to the stock's volatility on
that day. It closes the first time a close reaches either one, or after `horizon` sessions. Open
trades are re-followed each run; a closed trade keeps the numbers it closed with for good, so the
history no longer depends on what the watchlist or the trigger rule looks like today.
"""
from __future__ import annotations

import json

import pandas as pd

from . import config as C
from .model import predict, readiness
from .signals import barrier_labels, move_size

LOOKBACK = 90  # sessions of past triggers a name is backfilled with when the ledger first sees it


def load() -> list[dict]:
    if not C.TRADES.exists():
        return []
    return [json.loads(x) for x in C.TRADES.read_text().splitlines() if x.strip()]


def save(trades: list[dict]) -> None:
    trades = sorted(trades, key=lambda r: (r["date"], r["t"]))
    C.TRADES.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in trades))


def follow(t: str, f: pd.DataFrame, st: pd.DataFrame, spy: pd.Series, d: pd.Timestamp, eng: dict,
           frozen: dict | None = None) -> dict:
    """The trade from the close of trigger day `d`, up to its exit or, while open, the last close."""
    c = f["close"]
    i = c.index.get_loc(d)
    entry = float(c.iloc[i])
    horizon = eng["label"]["horizon"]
    if frozen:  # keep the target and stop the trade opened with
        up, down = frozen["up"], frozen["down"]
    else:
        size = float(move_size(c, horizon).iloc[i])
        up, down = size * eng["label"]["up"], size * eng["label"]["down"]
    o = barrier_labels(c.iloc[i:], horizon, up, down).iloc[0]
    if pd.isna(o["y"]):
        status, k = "open", len(c) - 1 - i
    else:
        k = int(o["bars"])
        status = "hit target" if o["y"] == 1 else "stopped out" if o["ret"] <= -down + 1e-9 else "timed out"
    path = c.iloc[i:i + k + 1]
    exit_px = float(path.iloc[-1])
    sp = spy.reindex(c.index).ffill()
    spy_chg = float(sp.iloc[i + k] / sp.iloc[i] - 1)
    r2 = lambda x: round(x * 100, 1)
    level = frozen["level"] if frozen else round(float(st.at[d, "pivot"]), 2)
    rec = {
        "t": t, "date": str(d.date()), "level": level, "entry": round(entry, 2),
        "up": round(up, 5), "down": round(down, 5), "target": r2(up), "stop": r2(down),
        "target_px": round(entry * (1 + up), 2), "stop_px": round(entry * (1 - down), 2),
        "status": status, "sessions": k, "resolved_in": None if status == "open" else k,
        "exit_date": None if status == "open" else str(path.index[-1].date()),
        "exit_px": None if status == "open" else round(exit_px, 2),
        "change": r2(exit_px / entry - 1), "best": r2(float(path.max()) / entry - 1),
        "worst": r2(float(path.min()) / entry - 1), "spy": r2(spy_chg), "excess": r2(exit_px / entry - 1 - spy_chg),
    }
    if frozen:
        rec.update({k: frozen[k] for k in ("ready_at", "strong", "live", "engine") if k in frozen})
    else:
        m = eng["model"]
        ready = int(readiness(m, predict(m, f.loc[[d], m["features"]].to_numpy())[0]))
        rec.update(ready_at=ready, strong=ready >= eng["trigger"].get("min_ready", 0), live=False,
                   engine=eng["version"])
    return rec


def update(ledger: list[dict], scored: dict, spy: pd.Series, eng: dict, seen: dict) -> tuple[list[dict], dict]:
    """Add new triggers and re-follow open trades.

    scored: {ticker: (features, stages)} for every name priced this run.
    seen: {ticker: last session the ledger looked at}; a name it has not seen is backfilled with
    its triggers from the last LOOKBACK sessions, a known one only gets triggers after that session.
    """
    by_key = {(r["t"], r["date"]): r for r in ledger}
    seen = dict(seen)
    for t, (f, st) in scored.items():
        days = st.index[st["trigger"].to_numpy(bool)]
        floor = st.index[max(0, len(st) - 1 - LOOKBACK)]
        since = pd.Timestamp(seen[t]) if t in seen else None
        days = days[(days >= floor) & ((days > since) if since is not None else True)]
        for d in days:
            day = str(d.date())
            # one trade per name at a time: a re-trigger while a trade is still open is part of that trade
            busy = any(r["t"] == t and r["date"] < day and (r["exit_date"] is None or r["exit_date"] > day)
                       for r in by_key.values())
            if (t, day) not in by_key and not busy:
                by_key[(t, day)] = follow(t, f, st, spy, d, eng)
        seen[t] = str(st.index[-1].date())
    for key, r in by_key.items():
        if r["status"] == "open" and r["t"] in scored:
            f, st = scored[r["t"]]
            d = pd.Timestamp(r["date"])
            if d in f.index:
                by_key[key] = follow(r["t"], f, st, spy, d, eng, frozen=r)
    return list(by_key.values()), seen


def mark_live(ledger: list[dict], alerts: list[dict]) -> None:
    """Trades whose trigger the monitor announced on the day take the readiness it announced."""
    live = {(a["t"], a.get("trigger_date") or a["date"]): a for a in alerts if a["type"] == "TRIGGERED"}
    for r in ledger:
        a = live.get((r["t"], r["date"]))
        if a and not r.get("live"):
            r.update(live=True, ready_at=a["ready"], strong=a.get("strong", True), engine=a.get("engine", r["engine"]))


def latest(ledger: list[dict], t: str, since: str) -> dict | None:
    """A name's most recent trade triggered on or after `since`."""
    mine = [r for r in ledger if r["t"] == t and r["date"] >= since]
    return max(mine, key=lambda r: r["date"]) if mine else None


def window(r: dict, eng: dict) -> dict | None:
    """When open trades that reach their target usually do, from the learner's trigger timing."""
    tm = eng.get("metrics", {}).get("trigger", {}).get("timing", {})
    if "median" not in tm or r["status"] != "open":
        return None
    d, n = pd.Timestamp(r["date"]), r["sessions"]
    day = lambda k: str((d + pd.offsets.BDay(k)).date())  # weekdays; holidays shift it a day or so
    phase = "early" if n < tm["p25"] else "in window" if n <= tm["p75"] else "late"
    return {"from": day(tm["p25"]), "median": day(tm["median"]), "to": day(tm["p75"]),
            "p25": tm["p25"], "mid": tm["median"], "p75": tm["p75"], "phase": phase}


def stats(ledger: list[dict]) -> dict:
    closed = [r for r in ledger if r["status"] != "open"]
    mean = lambda xs: round(sum(xs) / len(xs), 1) if xs else None
    return {"open": sum(r["status"] == "open" for r in ledger), "closed": len(closed),
            "hit": sum(r["status"] == "hit target" for r in closed),
            "stopped": sum(r["status"] == "stopped out" for r in closed),
            "timed_out": sum(r["status"] == "timed out" for r in closed),
            "avg_change": mean([r["change"] for r in closed]), "avg_excess": mean([r["excess"] for r in closed]),
            "live_closed": sum(bool(r.get("live")) for r in closed)}

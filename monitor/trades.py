"""The trade ledger: every trigger followed as a trade from its close, kept after it closes.

A trade opens at the trigger-day close with a target and a stop sized to the stock's volatility on
that day. It closes the first time a close reaches either one, or after `horizon` sessions. Open
trades are re-followed each run; a closed trade keeps the numbers it closed with for good, so the
history no longer depends on what the watchlist or the trigger rule looks like today. The trade
stages a name shows (triggered, running, retest, target, failed...) are read off its trade.
"""
from __future__ import annotations

import json

import pandas as pd

from . import config as C
from .model import predict, readiness
from .signals import FAIL_WINDOW, RUN_WINDOW, barrier_labels, move_size

LOOKBACK = 90   # sessions of past triggers a name is backfilled with when the ledger first sees it
RECHECK = 5     # sessions re-checked every run: a revised bar can reveal a trigger after the fact
NO_DATA_AFTER = 15  # business days past its horizon an open trade nobody can follow is closed


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
    vr = f.at[d, "vol_ratio"]
    rec = {
        "t": t, "date": str(d.date()), "level": level, "entry": round(entry, 2),
        "vol_ratio": None if pd.isna(vr) else round(float(vr), 2),
        "up": round(up, 5), "down": round(down, 5), "target": r2(up), "stop": r2(down),
        "target_px": round(entry * (1 + up), 2), "stop_px": round(entry * (1 - down), 2),
        "status": status, "sessions": k, "resolved_in": None if status == "open" else k,
        "exit_date": None if status == "open" else str(path.index[-1].date()),
        "exit_px": None if status == "open" else round(exit_px, 2),
        "as_of": str(path.index[-1].date()),
        "change": r2(exit_px / entry - 1), "best": r2(float(path.max()) / entry - 1),
        "worst": r2(float(path.min()) / entry - 1), "spy": r2(spy_chg), "excess": r2(exit_px / entry - 1 - spy_chg),
    }
    if frozen:
        rec.update({k: frozen[k] for k in ("ready_at", "quality", "strong", "live", "engine") if k in frozen})
    else:  # as the engine saw the trigger day itself
        rec.update(judge(f.loc[[d]], eng), live=False, engine=eng["version"])
    return rec


def judge(x: pd.DataFrame, eng: dict) -> dict:
    """How the engine rates a trigger from its day's features (one row): readiness, quality and
    strong or weak. Quality decides strong while the learner has it active, readiness otherwise."""
    m, q = eng["model"], eng.get("quality") or {}
    out = {"ready_at": int(readiness(m, predict(m, x[m["features"]].to_numpy())[0]))}
    if q.get("active"):
        out["quality"] = int(readiness(q, predict(q, x[q["features"]].to_numpy())[0]))
        out["strong"] = out["quality"] >= q["min_quality"]
    else:
        out["strong"] = out["ready_at"] >= eng["trigger"].get("min_ready", 0)
    return out


def rate(ledger: list[dict], scored: dict, eng: dict) -> list[str]:
    """Give trades from before the quality score its rating (and strong or weak from it), once:
    after that a trade keeps the rating it has, like its target and stop."""
    if not (eng.get("quality") or {}).get("active"):
        return []
    done = []
    for r in ledger:
        f_st = scored.get(r["t"])
        d = pd.Timestamp(r["date"])
        if "quality" in r or f_st is None or d not in f_st[0].index:
            continue
        r.update({k: v for k, v in judge(f_st[0].loc[[d]], eng).items() if k != "ready_at"})
        done.append(f"{r['t']} {r['date']}")
    return done


def update(ledger: list[dict], scored: dict, spy: pd.Series, eng: dict, seen: dict,
           today: str | None = None) -> tuple[list[dict], dict]:
    """Re-follow open trades, then add new triggers.

    scored: {ticker: (features, stages)} for every name priced this run.
    seen: {ticker: last session the ledger looked at}; a name it has not seen is backfilled with
    its triggers from the last LOOKBACK sessions, a known one gets the triggers since then plus the
    last RECHECK sessions again (a revised bar can turn a day into a trigger after the fact).
    """
    by_key = {(r["t"], r["date"]): r for r in ledger}
    horizon = eng["label"]["horizon"]
    for key, r in list(by_key.items()):  # first, so a trade that closed in between frees the name
        if r["status"] != "open":
            continue
        f_st = scored.get(r["t"])
        d = pd.Timestamp(r["date"])
        if f_st is not None and d in f_st[0].index:
            by_key[key] = follow(r["t"], *f_st, spy, d, eng, frozen=r)
        elif today and len(pd.bdate_range(r["date"], today)) > horizon + NO_DATA_AFTER:
            # delisted, renamed or out of the data: close it where it was last seen
            by_key[key] = {**r, "status": "no data", "exit_date": r.get("as_of", r["date"]),
                           "exit_px": round(r["entry"] * (1 + r["change"] / 100), 2), "resolved_in": r["sessions"]}
    seen = dict(seen)
    for t, (f, st) in scored.items():
        idx = st.index
        start = max(0, len(idx) - 1 - LOOKBACK)
        if t in seen:
            start = max(start, int(idx.searchsorted(pd.Timestamp(seen[t]), side="right")) - RECHECK)
        days = idx[start:][st["trigger"].to_numpy(bool)[start:]]
        for d in days:  # chronological, so each new trade is resolved before the next trigger is checked
            day = str(d.date())
            busy = any(r["t"] == t and r["date"] < day and (r["exit_date"] is None or r["exit_date"] > day)
                       for r in by_key.values())  # one trade per name at a time
            if (t, day) not in by_key and not busy:
                by_key[(t, day)] = follow(t, f, st, spy, d, eng)
        seen[t] = str(idx[-1].date())
    return list(by_key.values()), seen


def mark_live(ledger: list[dict], alerts: list[dict]) -> None:
    """Trades the monitor announced as a trigger at the time (not found in the history later)."""
    trig = [a for a in alerts if a["type"] == "TRIGGERED"]
    for r in ledger:
        if r.get("live"):
            continue
        for a in trig:
            if a["t"] != r["t"]:
                continue
            # older alerts carry no trigger date: they were sent within a few days of it
            d = a.get("trigger_date")
            if d == r["date"] or (d is None and 0 <= (pd.Timestamp(a["date"]) - pd.Timestamp(r["date"])).days <= 7):
                r["live"] = True
                break


def latest(ledger: list[dict], t: str, since: str, until: str) -> dict | None:
    """A name's most recent trade triggered between `since` and `until`."""
    mine = [r for r in ledger if r["t"] == t and since <= r["date"] <= until]
    return max(mine, key=lambda r: r["date"]) if mine else None


def stage_on(f: pd.DataFrame, st: pd.DataFrame, i: int, trade: dict | None, p: dict) -> tuple[str, float]:
    """(stage, level) on bar i: the trade decides the trade stages, the price the others."""
    base, piv = st["stage"].iloc[i], float(st["pivot"].iloc[i])
    day = str(f.index[i].date())
    if trade is None or base == "n/a" or trade["date"] > day:
        return base, piv
    j = f.index.get_indexer([pd.Timestamp(trade["date"])])[0]
    if j < 0:
        return base, piv
    n, c, lvl = i - j, float(f["close"].iloc[i]), float(trade["level"])
    if trade["exit_date"] is None or trade["exit_date"] > day:  # open on that day
        if c < lvl:
            return "RETEST", lvl
        if n < p["confirm"]:
            return "TRIGGERED", lvl
        if c > float(f["sma50"].iloc[i]) * 1.2 or float(f["rsi"].iloc[i]) > 75:
            return "EXTENDED", lvl
        return "RUNNING", lvl
    if trade["status"] == "hit target" and c >= lvl and n <= RUN_WINDOW:
        return "TARGET", lvl
    if trade["status"] == "stopped out" and c < lvl and n <= FAIL_WINDOW:
        return "FAILED", lvl
    return base, piv


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
    closed = [r for r in ledger if r["status"] not in ("open", "no data")]
    mean = lambda xs: round(sum(xs) / len(xs), 1) if xs else None
    return {"open": sum(r["status"] == "open" for r in ledger), "closed": len(closed),
            "hit": sum(r["status"] == "hit target" for r in closed),
            "stopped": sum(r["status"] == "stopped out" for r in closed),
            "timed_out": sum(r["status"] == "timed out" for r in closed),
            "no_data": sum(r["status"] == "no data" for r in ledger),
            "avg_change": mean([r["change"] for r in closed]), "avg_excess": mean([r["excess"] for r in closed]),
            "live_closed": sum(bool(r.get("live")) for r in closed)}

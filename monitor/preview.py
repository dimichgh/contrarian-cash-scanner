"""Pre-close preview: today's live prices against the trigger rule, about 30 minutes before the close.

It takes the cached daily bars, adds today's still-forming bar from the live quote and projects its
volume to a full session, then asks which names would trigger if they closed where they trade now.
Nothing is saved: the trade ledger, alerts, history and price cache change only in the run after the
close, on final prices. The preview writes reports/preview.md (the notification) and preview.html.
"""
from __future__ import annotations

import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import config as C
from . import trades as T
from . import watchlist as W
from .model import predict, readiness
from .prices import fetch_ohlcv, load_cached
from .signals import PIVOT_TEXT, features, move_size, stages

ET = ZoneInfo("America/New_York")
TEMPLATE = Path(__file__).with_name("preview_template.html")
NEAR = 0.01  # within this distance under the level counts as "could cross by the close"
# Share of a US session's volume traded by each minute after the open: U-shaped trading, with the
# closing auction in the last minutes. Approximate; used only to project today's volume so far.
VOLUME_CURVE = [(0, 0.0), (30, 0.13), (60, 0.21), (90, 0.28), (120, 0.34), (180, 0.45), (240, 0.56),
                (300, 0.67), (330, 0.73), (360, 0.81), (375, 0.86), (390, 1.0)]


def session_share(minutes: float) -> float:
    xs, ys = zip(*VOLUME_CURVE)
    return float(np.interp(minutes, xs, ys))


def live_bars(t: str) -> pd.DataFrame | None:
    """Cached history plus the latest days from Yahoo, today's bar included while it is forming."""
    fresh = fetch_ohlcv(t, "5d", partial=True)
    cached = load_cached(t)
    if fresh is None:
        return None
    live = fresh.attrs.get("live")
    bars = fresh if cached is None else pd.concat([cached[cached.index < fresh.index[0]], fresh])
    bars.attrs["live"] = live
    return bars


def project(bars: pd.DataFrame) -> pd.DataFrame:
    """Scale today's volume so far to a full session (no change once the session is over)."""
    live = bars.attrs.get("live")
    if not live:
        return bars
    b = bars.copy()
    b.loc[b.index[-1], "volume"] = b["volume"].iloc[-1] / max(session_share(live["minutes"]), 0.05)
    return b


def classify(t: str, f: pd.DataFrame, st: pd.DataFrame, eng: dict, busy: bool) -> dict | None:
    """Where today's live bar leaves a name against the trigger rule; None when it is in none of the lists."""
    p, lab, m = eng["trigger"], eng["label"], eng["model"]
    c, lvl = float(f["close"].iloc[-1]), float(st["pivot"].iloc[-1])
    if not np.isfinite(lvl):
        return None
    above = f["close"] > st["pivot"]
    below_days = int((~above.iloc[-21:-1]).sum())
    low_age = st["low_age"].iloc[-1]
    vr = float(f["vol_ratio"].iloc[-1]) if pd.notna(f["vol_ratio"].iloc[-1]) else 0.0
    base_ok = below_days >= p["min_below"] and low_age >= p["min_low_age"] and not bool(above.iloc[-2])
    prob = predict(m, f[m["features"]].iloc[-1:].to_numpy())[0]
    ready = int(readiness(m, prob))
    size = float(move_size(f["close"], lab["horizon"]).iloc[-1])
    row = {"t": t, "close": round(c, 2), "level": round(lvl, 2), "gap": round((c / lvl - 1) * 100, 2),
           "vol_ratio": round(vr, 2), "vol_need": p["vol_mult"], "ready": ready,
           "strong": ready >= p.get("min_ready", 0), "below_days": below_days,
           "stop_px": round(c * (1 - size * lab["down"]), 2), "target_px": round(c * (1 + size * lab["up"]), 2)}
    if busy or not base_ok:
        return None  # one trade per name; and without a stretch below the level no close can trigger
    if bool(st["trigger"].iloc[-1]):
        return {**row, "kind": "trigger"}
    if c > lvl:
        return {**row, "kind": "volume_short"}
    if c / lvl - 1 >= -NEAR:
        return {**row, "kind": "near"}
    return None


def open_trades_now(ledger: list[dict], bars: dict) -> list[dict]:
    out = []
    for r in ledger:
        if r["status"] != "open" or r["t"] not in bars:
            continue
        c = float(bars[r["t"]]["close"].iloc[-1])
        flag = "at target" if c >= r["target_px"] else "at stop" if c <= r["stop_px"] else ""
        out.append({"t": r["t"], "date": r["date"], "entry": r["entry"], "close": round(c, 2),
                    "change": round((c / r["entry"] - 1) * 100, 1), "stop_px": r["stop_px"],
                    "target_px": r["target_px"], "sessions": r["sessions"] + 1, "flag": flag})
    return sorted(out, key=lambda x: (x["flag"] == "", x["date"]))


def run(today: dt.date | None = None) -> dict:
    eng = json.loads(C.ENGINE.read_text())
    wl = W.load()
    ledger = T.load()
    names = W.active(wl)
    followed = sorted({r["t"] for r in ledger if r["status"] == "open"} - set(names))
    with ThreadPoolExecutor(max_workers=8) as ex:
        got = dict(zip(names + followed + ["SPY"], ex.map(live_bars, names + followed + ["SPY"])))
    spy = got.pop("SPY", None)
    if spy is None:
        raise RuntimeError("no SPY prices from Yahoo; cannot preview")
    live = spy.attrs.get("live")
    now = dt.datetime.now(ET)
    session = live["session"] if live else str(spy.index[-1].date())
    res = {"time": now.strftime("%Y-%m-%d %H:%M"), "session": session, "live": live,
           "rule": {"pivot": PIVOT_TEXT[eng["trigger"]["pivot"]], "vol_mult": eng["trigger"]["vol_mult"],
                    "min_below": eng["trigger"]["min_below"]},
           "trigger": [], "volume_short": [], "near": [], "trades": [], "missing": []}
    if session != now.date().isoformat():
        res["closed"] = True  # no session today (weekend or holiday): nothing to preview
        return res
    busy = {r["t"] for r in ledger if r["status"] == "open"}
    spy_close = project(spy)["close"]
    today_bars = {}
    for t in names + followed:
        b = got.get(t)
        if b is None or len(b) < 220 or str(b.index[-1].date()) != session:
            res["missing"].append(t)
            continue
        today_bars[t] = b
        if t not in names:
            continue
        f = features(project(b), spy_close)
        st = stages(f, eng["trigger"])
        row = classify(t, f, st, eng, t in busy)
        if row:
            row["name"] = wl[t].get("name", t)
            res[row["kind"]].append(row)
    for k in ("trigger", "volume_short"):
        res[k].sort(key=lambda r: -r["ready"])
    res["near"].sort(key=lambda r: -r["gap"])
    res["trades"] = open_trades_now(ledger, today_bars)
    return res


def _clock(res: dict) -> str:
    live = res["live"]
    t = dt.datetime.strptime(res["time"], "%Y-%m-%d %H:%M").strftime("%b %-d, %-I:%M%p").replace("AM", "am").replace("PM", "pm")
    if not live:
        return f"{t} ET, after the close: today's final prices"
    left = max(0, round(390 - live["minutes"]))
    return f"{t} ET, {left} minutes before the close: prices so far, volume projected to a full session"


def summary(res: dict) -> str:
    lines = [f"Pre-close preview · {_clock(res)}", ""]
    if res.get("closed"):
        return "\n".join(lines + ["No trading session today.", "", "NOTIFY: no"]) + "\n"
    need = f"{res['rule']['vol_mult']:g}x"
    if res["trigger"]:
        lines.append(f"On track to trigger at the close ({len(res['trigger'])}):")
        for r in res["trigger"]:
            lines.append(f"▲ {r['t']} ${r['close']:.2f}: {r['gap']:+.1f}% through the {res['rule']['pivot']} (${r['level']:.2f}), "
                         f"volume on pace for {r['vol_ratio']:.1f}x · readiness {r['ready']}{'' if r['strong'] else ' (weak)'} · "
                         f"if it closes here: stop ${r['stop_px']:.2f}, target ${r['target_px']:.2f}")
    if res["volume_short"]:
        lines.append(f"Crossing, but volume short of {need} so far ({len(res['volume_short'])}):")
        for r in res["volume_short"]:
            lines.append(f"△ {r['t']} ${r['close']:.2f}: {r['gap']:+.1f}% over ${r['level']:.2f}, volume on pace for {r['vol_ratio']:.1f}x")
    if res["near"]:
        lines.append(f"Within {NEAR:.0%} under the level, could cross by the close ({len(res['near'])}):")
        for r in res["near"]:
            lines.append(f"◇ {r['t']} ${r['close']:.2f}: needs {-r['gap']:.1f}% to close above ${r['level']:.2f}, "
                         f"volume on pace for {r['vol_ratio']:.1f}x (needs {need})")
    if not (res["trigger"] or res["volume_short"] or res["near"]):
        lines.append("No name is on track to trigger at the close.")
    flagged = [r for r in res["trades"] if r["flag"]]
    if res["trades"]:
        lines += ["", "Open trades now: " + ", ".join(
            f"{r['t']} {r['change']:+.1f}%{' (' + r['flag'] + ')' if r['flag'] else ''}" for r in res["trades"])]
    lines += ["", "Nothing here is recorded: the run after the close (about 6pm ET) decides on final prices."]
    notify = bool(res["trigger"] or res["volume_short"] or res["near"] or flagged)
    return "\n".join(lines + ["", f"NOTIFY: {'yes' if notify else 'no'}"]) + "\n"


def page(res: dict) -> str:
    data = {**res, "clock": _clock(res), "near_pct": NEAR}
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("NaN", "null")
    return TEMPLATE.read_text().replace("/*__DATA__*/null", blob)


def write(res: dict) -> str:
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    text = summary(res)
    (C.REPORTS / "preview.md").write_text(text)
    (C.REPORTS / "preview.html").write_text(page(res))
    return text

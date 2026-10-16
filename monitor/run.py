"""The daily run: rescan and relearn when due, score every active name, raise alerts, keep score."""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from . import config as C
from . import learn as L
from . import trades as T
from . import watchlist as W
from .model import auc, predict, readiness
from .prices import fetch_many
from .signals import PIVOT_TEXT, features, move_size, outcomes, stages

ALERT_STAGES = {"TRIGGERED", "SETUP", "TARGET", "RUNNING", "FAILED"}
HOLD_STAGES = {"TRIGGERED", "RUNNING", "EXTENDED", "RETEST", "TARGET"}


def _load(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def _due(last: str | None, days: int, today: dt.date) -> bool:
    return last is None or (today - dt.date.fromisoformat(last)).days >= days


def rescan(wl: dict, today: dt.date) -> tuple[list[str], list[str]]:
    import arden_scanner as A
    out = C.ROOT / "output"
    cached = out / "arden_scan.json"
    if cached.exists() and dt.date.fromtimestamp(cached.stat().st_mtime) == today:
        final = pd.read_json(cached)  # a full scan already ran today (e.g. by hand)
    else:
        final = A.run(2e9, 120, None, out)
    scan = W.save_scan(final)
    try:  # the scan's own report page, refreshed with every rescan
        import make_report
        make_report.main(C.REPORTS / "scan_report.html")
    except Exception as e:  # a report failure must not stop the monitor
        print(f"  ! scan report not built: {e.__class__.__name__}: {e}")
    return W.merge_scan(wl, scan, today)


def make_row(t: str, f: pd.DataFrame, st: pd.DataFrame, eng: dict, ledger: list[dict]) -> dict:
    """Today's call for one name: its stage (read off its trade when it has one), level and readiness."""
    m, p = eng["model"], eng["trigger"]
    prob = predict(m, f[m["features"]].iloc[-1:].to_numpy())[0]
    last = f.iloc[-1]
    since = str(st.index[max(0, len(st) - 1 - T.LOOKBACK)].date())
    day, yday = str(st.index[-1].date()), str(st.index[-2].date())
    trade = T.latest(ledger, t, since, day)
    stage, level = T.stage_on(f, st, len(st) - 1, trade, p)
    prev_stage, _ = T.stage_on(f, st, len(st) - 2, T.latest(ledger, t, since, yday), p)
    size = float(move_size(f["close"], eng["label"]["horizon"]).iloc[-1])
    window = f["close"].iloc[-60:]
    lo, hi = float(window.min()), float(window.max())
    norm = lambda x: int(round((x - lo) / (hi - lo) * 99)) if hi > lo and np.isfinite(x) else 50
    low_age = st["low_age"].iloc[-1]
    return {
        "t": t, "date": day, "close": round(float(last["close"]), 2),
        "target": round(size * eng["label"]["up"] * 100, 1), "stop": round(size * eng["label"]["down"] * 100, 1),
        "stage": stage, "prob": round(float(prob), 4), "ready": int(readiness(m, prob)),
        "level": round(level, 2), "gap": round((float(last["close"]) / level - 1) * 100, 1),
        "vol_ratio": round(float(last["vol_ratio"]), 2) if not pd.isna(last["vol_ratio"]) else None,
        "sma50": round(float(last["sma50"]), 2), "sma200": None if pd.isna(last["sma200"]) else round(float(last["sma200"]), 2),
        "slope": round(float(last["f_sma50_slope"]) * 100, 2),
        "rs20": round(float(last["f_rs_slope"]) * 100, 1), "rsi": round(float(last["rsi"]), 0),
        "low_age": None if pd.isna(low_age) else int(low_age),
        "ret20": round(float(last["f_ret20"]) * 100, 1), "dd52": round(float(last["f_dd_52w"]) * 100, 1),
        "spark": "".join(f"{norm(x):02d}" for x in window), "spark_level": norm(level),
        "prev_stage": prev_stage, "contrib": _contributions(m, last),
        "trade": None if trade is None else {**trade, "window": T.window(trade, eng)},
    }


def _contributions(m: dict, last: pd.Series) -> list:
    """The three features pushing readiness up the most and the one pulling it down the most."""
    z = (last[m["features"]].astype(float).to_numpy() - np.array(m["mean"])) / np.array(m["std"])
    c = np.nan_to_num(z * np.array(m["coef"]))
    order = np.argsort(c)
    pos = [(m["features"][i], round(float(c[i]), 2)) for i in order[::-1][:3] if c[i] > 0.02]
    neg = [(m["features"][i], round(float(c[i]), 2)) for i in order[:1] if c[i] < -0.02]
    return pos + neg


def _side(gap: float) -> str:
    return f"{abs(gap):.1f}% {'above' if gap >= 0 else 'under'}"


def alert_text(r: dict, p: dict) -> str:
    lvl = PIVOT_TEXT[p["pivot"]]
    t = r.get("trade") or {}
    if r["stage"] == "TRIGGERED":
        gap = (t["entry"] / t["level"] - 1) * 100
        when = "" if t["date"] == r["date"] else f" on {t['date']}"
        vol = f", volume {t['vol_ratio']:.1f}x its average" if t.get("vol_ratio") else ""
        return (f"closed {gap:.1f}% above the {lvl} (${t['level']:.2f}){when}{vol}; "
                f"stop ${t['stop_px']:.2f}, target ${t['target_px']:.2f}")
    if r["stage"] == "SETUP":
        held = f", lows held {r['low_age']} sessions" if r["low_age"] is not None else ""
        return (f"{abs(r['gap']):.1f}% under the trigger level ${r['level']:.2f} ({lvl}){held}; "
                f"if it triggers there: stop ${r['level'] * (1 - r['stop'] / 100):.2f}, target ${r['level'] * (1 + r['target'] / 100):.2f}")
    if r["stage"] == "TARGET":
        return (f"reached its target: {t['change']:+.1f}% from the {t['date']} trigger at ${t['entry']:.2f} "
                f"in {t['resolved_in']} sessions (target ${t['target_px']:.2f})")
    if r["stage"] == "RUNNING":
        trend = f"50-day average {'rising' if r['slope'] > 0 else 'falling'} {abs(r['slope']):.1f}%/2wk"
        return (f"move confirmed: {t['sessions']} sessions after the trigger, {_side(r['gap'])} the ${r['level']:.2f} level "
                f"({t['change']:+.1f}% from the entry), {trend}")
    if r["stage"] == "FAILED":
        return (f"breakout failed: stopped out at ${t['exit_px']:.2f} ({t['change']:+.1f}%) on {t['exit_date']}; "
                f"now {_side(r['gap'])} the ${r['level']:.2f} level")
    return ""


def _run_start(stages: pd.Series, value: str) -> int:
    """Position of the first day of the current unbroken run of `value` at the end of `stages`."""
    i = len(stages) - 1
    while i > 0 and stages.iloc[i - 1] == value:
        i -= 1
    return i


def _alert_due(r: dict, st: pd.DataFrame, past: list[dict]) -> tuple[bool, str | None]:
    """(is this news, the date that identifies the event). Decided from what was actually sent,
    so a missed run delays a signal by a day instead of losing it."""
    s, t = r["stage"], r.get("trade")
    if s not in ALERT_STAGES:
        return False, None
    sent = {(a["t"], a["type"], a.get("trigger_date") or a["date"]) for a in past}
    recent = str(st.index[max(0, len(st) - 6)].date())  # older than a week is not news
    if s in ("TRIGGERED", "RUNNING", "TARGET", "FAILED"):
        if s in ("TARGET", "FAILED") and t["exit_date"] < recent:
            return False, None
        if s == "RUNNING" and t["sessions"] > 15:  # the confirmation, not every return to RUNNING
            return False, None
        # alerts from before trade dates were recorded: one of this type during the trade covers it
        start = pd.Timestamp(t["date"])
        legacy = any(a["t"] == r["t"] and a["type"] == s and not a.get("trigger_date")
                     and 0 <= (pd.Timestamp(a["date"]) - start).days <= 90 for a in past)
        return (r["t"], s, t["date"]) not in sent and not legacy, t["date"]
    # SETUP: when the current run of SETUP days began in the last week, and no SETUP signal went
    # out in the 10 sessions before it (a name hovering under its level does not repeat)
    i = _run_start(st["stage"], "SETUP")
    start = str(st.index[i].date())
    quiet_from = str(st.index[max(0, i - 10)].date())
    repeat = any(a["t"] == r["t"] and a["type"] == "SETUP" and a["date"] >= quiet_from for a in past)
    return start >= recent and not repeat and (r["t"], "SETUP", start) not in sent, start


def track_record(hist: pd.DataFrame, bars: dict, label: dict, alerts: list[dict]) -> dict:
    """Out-of-sample scorekeeping: how the engine's own past calls turned out.

    A call counts only once its whole 40-session window has passed. Counting calls as soon as they
    hit a barrier would over-weight quick stops and targets and drop the slow timeouts.
    """
    h = label["horizon"]

    def labelled(t):
        bl = outcomes(bars[t]["close"], label)
        return bl.assign(elapsed=len(bl) - 1 - np.arange(len(bl)))

    out: dict = {}
    if len(hist):
        lab = [g.assign(_d=pd.to_datetime(g["date"])).join(labelled(t), on="_d")
               for t, g in hist.groupby("t") if t in bars]
        if lab:
            allrows = pd.concat(lab)
            done = allrows[(allrows["elapsed"] >= h) & allrows["y"].notna()]
            hi = done[done["ready"] >= 80]
            out["snapshots"] = {"resolved": len(done), "open": int(len(hist) - len(done)),
                                "hit_rate": round(float(done["y"].mean()), 3) if len(done) else None,
                                "hit_rate_ready80": round(float(hi["y"].mean()), 3) if len(hi) else None,
                                "n_ready80": len(hi), "auc": round(auc(done["y"], done["prob"]), 3) if len(done) >= 30 else None}
    res = []
    for a in alerts:
        if a["type"] != "TRIGGERED" or a["t"] not in bars:
            continue
        bl = labelled(a["t"])
        d = pd.Timestamp(a.get("trigger_date") or a["date"])
        if d in bl.index and bl.at[d, "elapsed"] >= h and not pd.isna(bl.at[d, "y"]):
            res.append({**a, "y": float(bl.at[d, "y"]), "ret": round(float(bl.at[d, "ret"]) * 100, 1),
                        "bars": int(bl.at[d, "bars"])})

    def stats(xs):
        return {"resolved": len(xs), "hit_rate": round(float(np.mean([r["y"] for r in xs])), 3) if xs else None,
                "avg_ret": round(float(np.mean([r["ret"] for r in xs])), 2) if xs else None}
    out["triggers"] = {**stats(res), "strong": stats([r for r in res if r.get("strong", True)])}
    return out


def daily(force_rescan: bool = False, force_learn: bool = False, today: dt.date | None = None) -> dict:
    today = today or dt.date.today()
    C.STATE.mkdir(parents=True, exist_ok=True)
    wl = W.load()
    last = _load(C.LAST_RUN, {})
    eng = _load(C.ENGINE, None)
    notes: list[str] = []

    ledger = T.load()
    for t, e in wl.items():  # before a rescan decides what to drop: names mid-move or mid-trade stay
        e["hold"] = any(r["t"] == t and r["status"] == "open" for r in ledger) \
            or last.get("stages", {}).get(t) in HOLD_STAGES
    added, dropped = [], []
    did_rescan = force_rescan or _due(last.get("rescan"), C.RESCAN_DAYS, today) or not C.SCAN.exists()
    if did_rescan:
        try:
            added, dropped = rescan(wl, today)
            last["rescan"] = today.isoformat()
        except Exception as e:  # scan sources down: keep the last scan and retry next run
            did_rescan = False
            notes.append(f"Rescan failed ({e.__class__.__name__}: {e}); kept the scan of {last.get('rescan', 'n/a')}")
    added += W.ensure_author_calls(wl, today)

    did_learn = force_learn or eng is None or _due(eng.get("updated"), C.LEARN_DAYS, today)
    names = W.active(wl)
    # names off the watchlist with a trade still open keep being followed until it closes
    followed = sorted({r["t"] for r in ledger if r["status"] == "open"} - set(names))
    fetch = list(names) + followed
    if did_learn:
        scan = W.load_scan()
        fetch += list(scan["name"]) if scan is not None else []
    got: dict = {}
    bars = fetch_many(fetch + ["SPY"], "5y" if did_learn else "2y", today=today, stats=got)
    if "SPY" not in bars:
        raise RuntimeError("no SPY bars (Yahoo down and none cached); cannot score")
    spy = bars["SPY"]["close"]
    stale = set(got.get("stale", []))
    notes.append("Prices: " + ", ".join(f"{got[k]} {k}" for k in ("update", "download", "cache", "missing") if got.get(k))
                 + (" (offline: cache only)" if got.get("offline") else ""))
    if "SPY" in stale:
        raise RuntimeError("no fresh SPY bars from Yahoo; refusing to signal on cached prices")
    if did_learn:
        prev_version = eng.get("version") if eng else None
        eng = L.learn({t: b for t, b in bars.items() if t != "SPY"}, spy, eng, today)
        C.ENGINE.write_text(json.dumps(eng, indent=1) + "\n")
        mt = eng["metrics"]
        what = [k for k, ch in (("trigger rule", mt["trigger"]["changed"]), ("prior strength", mt["model"]["changed"]),
                                ("strong-trigger bar", mt["trigger"]["ready_filter"]["changed"])) if ch]
        notes.append(f"Engine re-fit: {f'v{prev_version} → ' if prev_version else ''}v{eng['version']} (new weights"
                     + (f"; changed: {', '.join(what)})" if what else ")"))

    as_of = str(spy.index[-1].date())
    fresh_session = as_of != last.get("as_of")
    scored, missing = {}, []
    for t in names + followed:
        b = bars.get(t)
        if b is None or len(b) < 220 or t in stale:
            if t in names:
                missing.append(t)
            continue
        f = features(b, spy)
        scored[t] = (f, stages(f, eng["trigger"]))

    ledger, last["ledger_seen"] = T.update(ledger, scored, spy, eng, last.get("ledger_seen", {}), as_of)
    rows = []
    for t in names:
        if t not in scored:
            continue
        f, st = scored[t]
        if st["stage"].iloc[-1] == "n/a":
            missing.append(t)
            continue
        row = make_row(t, f, st, eng, ledger)
        e = wl[t]
        row.update(name=e.get("name", t), source=e.get("source"), thesis=e.get("thesis"), author=e.get("author"),
                   added=e.get("added"))
        e["hold"] = bool(row["trade"]) and row["trade"]["status"] == "open" or row["stage"] in HOLD_STAGES
        rows.append(row)
    rows.sort(key=lambda r: (-r["ready"], r["t"]))

    # alerts: stage changes that are news (see _alert_due)
    prev = last.get("stages", {})
    new_alerts = []
    if fresh_session:
        past = [json.loads(x) for x in C.ALERTS.read_text().splitlines()] if C.ALERTS.exists() else []
        for r in rows:
            due, key = _alert_due(r, scored[r["t"]][1], past + new_alerts)
            if not due:
                continue
            t = r["trade"] or {}
            trig = r["stage"] == "TRIGGERED"
            new_alerts.append({"date": r["date"], "t": r["t"], "type": r["stage"], "from": prev.get(r["t"], r["prev_stage"]),
                               "strong": bool(t.get("strong", True)) if trig else True,
                               "close": t["entry"] if trig else r["close"], "level": t["level"] if trig else r["level"],
                               "ready": t["ready_at"] if trig else r["ready"], "prob": r["prob"], "trigger_date": key,
                               "text": alert_text(r, eng["trigger"]), "engine": eng["version"]})
        with C.ALERTS.open("a") as fh:
            for a in new_alerts:
                fh.write(json.dumps(a) + "\n")
        # daily snapshot of every call, scored later against what the price actually did
        C.HISTORY.mkdir(parents=True, exist_ok=True)
        snap = pd.DataFrame([{k: r[k] for k in ("date", "t", "close", "stage", "prob", "ready", "level")}
                             | {"engine": eng["version"]} for r in rows])
        month = C.HISTORY / f"{as_of[:7]}.csv"
        old = pd.read_csv(month) if month.exists() else None
        snap = pd.concat([old, snap]).drop_duplicates(["date", "t"], keep="last") if old is not None else snap
        snap.to_csv(month, index=False)
    else:
        notes.append(f"No new session since {as_of}; alerts already sent")

    alerts = [json.loads(x) for x in C.ALERTS.read_text().splitlines()] if C.ALERTS.exists() else []
    T.mark_live(ledger, alerts)
    T.save(ledger)
    hist = pd.concat([pd.read_csv(p) for p in sorted(C.HISTORY.glob("*.csv"))]) if C.HISTORY.exists() and \
        any(C.HISTORY.glob("*.csv")) else pd.DataFrame()
    record = track_record(hist, bars, eng["label"], alerts)

    last.update(as_of=as_of, run=dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
                stages={**prev, **{r["t"]: r["stage"] for r in rows}}, record=record)
    C.LAST_RUN.write_text(json.dumps(last, indent=1) + "\n")
    W.save(wl)
    return {"as_of": as_of, "rows": rows, "alerts": new_alerts, "recent_alerts": alerts[-40:], "added": added,
            "trades": ledger, "closes": {t: float(f["close"].iloc[-1]) for t, (f, _) in scored.items()},
            "dropped": dropped, "missing": missing, "engine": eng, "record": record, "notes": notes,
            "rescanned": did_rescan, "learned": did_learn, "fresh": fresh_session, "watchlist": wl}

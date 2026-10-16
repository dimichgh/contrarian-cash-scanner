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


def score_ticker(t: str, b: pd.DataFrame, spy: pd.Series, eng: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    f = features(b, spy)
    st = stages(f, eng["trigger"])
    m = eng["model"]
    prob = predict(m, f[m["features"]].iloc[-1:].to_numpy())[0]
    last, s = f.iloc[-1], st.iloc[-1]
    level = float(s["level"])
    size = float(move_size(f["close"], eng["label"]["horizon"]).iloc[-1])
    window = f["close"].iloc[-60:]
    lo, hi = float(window.min()), float(window.max())
    norm = lambda x: int(round((x - lo) / (hi - lo) * 99)) if hi > lo and np.isfinite(x) else 50
    row = {
        "t": t, "date": str(f.index[-1].date()), "close": round(float(last["close"]), 2),
        "target": round(size * eng["label"]["up"] * 100, 1), "stop": round(size * eng["label"]["down"] * 100, 1),
        "stage": s["stage"], "prob": round(float(prob), 4), "ready": int(readiness(m, prob)),
        "level": round(level, 2), "gap": round((float(last["close"]) / level - 1) * 100, 1),
        "vol_ratio": round(float(last["vol_ratio"]), 2) if not pd.isna(last["vol_ratio"]) else None,
        "sma50": round(float(last["sma50"]), 2), "sma200": None if pd.isna(last["sma200"]) else round(float(last["sma200"]), 2),
        "slope": round(float(last["f_sma50_slope"]) * 100, 2),
        "rs20": round(float(last["f_rs_slope"]) * 100, 1), "rsi": round(float(last["rsi"]), 0),
        "low_age": None if pd.isna(last["low_age_days"]) else int(last["low_age_days"]),
        "ret20": round(float(last["f_ret20"]) * 100, 1), "dd52": round(float(last["f_dd_52w"]) * 100, 1),
        "spark": "".join(f"{norm(x):02d}" for x in window), "spark_level": norm(level),
        "prev_stage": st["stage"].iloc[-2] if len(st) > 1 else None,
        "contrib": _contributions(m, f.iloc[-1]),
    }
    return row, f, st


def attach_trade(row: dict, trade: dict | None, eng: dict) -> None:
    """Show the name's latest trade, and call a move that already reached its target what it is."""
    row["trade"] = None if trade is None else {**trade, "window": T.window(trade, eng)}
    moving = ("TRIGGERED", "RUNNING", "EXTENDED", "ABOVE")
    if trade and trade["status"] == "hit target":
        if row["stage"] in moving:
            row["stage"] = "TARGET"
        if trade["exit_date"] < row["date"] and row["prev_stage"] in moving:
            row["prev_stage"] = "TARGET"  # already there yesterday: no fresh alert for a name seen first today


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


def trigger_day(f: pd.DataFrame, st: pd.DataFrame) -> dict | None:
    """The numbers of the latest trigger, from the trigger day itself."""
    days = st.index[st["trigger"].to_numpy(bool)]
    if not len(days):
        return None
    d = days[-1]
    vr = f.at[d, "vol_ratio"]
    return {"date": str(d.date()), "close": round(float(f.at[d, "close"]), 2), "level": round(float(st.at[d, "pivot"]), 2),
            "gap": round((float(f.at[d, "close"]) / float(st.at[d, "pivot"]) - 1) * 100, 1),
            "vol_ratio": None if pd.isna(vr) else round(float(vr), 2)}


def alert_text(r: dict, p: dict, trig: dict | None = None) -> str:
    lvl = PIVOT_TEXT[p["pivot"]]
    t = r.get("trade") or {}
    if r["stage"] == "TRIGGERED" and trig:
        vol = f" on {trig['vol_ratio']:.1f}x average volume" if trig["vol_ratio"] else ""
        when = "" if trig["date"] == r["date"] else f" on {trig['date']}"
        plan = f"; stop ${t['stop_px']:.2f}, target ${t['target_px']:.2f}" if t.get("date") == trig["date"] else ""
        return f"closed {trig['gap']:.1f}% above the {lvl} (${trig['level']:.2f}){when}{vol}{plan}"
    if r["stage"] == "SETUP":
        return (f"{abs(r['gap']):.1f}% under the trigger level ${r['level']:.2f} ({lvl}), lows held {r['low_age']} sessions; "
                f"if it triggers there: stop ${r['level'] * (1 - r['stop'] / 100):.2f}, target ${r['level'] * (1 + r['target'] / 100):.2f}")
    if r["stage"] == "TARGET":
        return (f"reached its target: {t['change']:+.1f}% from the {t['date']} trigger at ${t['entry']:.2f} "
                f"in {t['resolved_in']} sessions (target ${t['target_px']:.2f})")
    if r["stage"] == "RUNNING":
        trend = f"50-day average {'rising' if r['slope'] > 0 else 'falling'} {abs(r['slope']):.1f}%/2wk"
        return f"move confirmed: held the breakout, now {_side(r['gap'])} the ${r['level']:.2f} level, {trend}"
    if r["stage"] == "FAILED":
        return f"breakout failed: closed 3%+ back under ${r['level']:.2f}; now {_side(r['gap'])} it"
    return ""


def _alert_due(r: dict, before: str | None, st: pd.DataFrame, trig: dict | None, sent: set) -> bool:
    """Signal a stage change only when it is news: one alert per trigger, no flip-flopping."""
    s = r["stage"]
    if s not in ALERT_STAGES or s == before:
        return False
    if s == "TRIGGERED":
        return trig is not None and (r["t"], "TRIGGERED", trig["date"]) not in sent
    if s == "RUNNING":
        return before in ("TRIGGERED", "RETEST") and trig is not None and (r["t"], "RUNNING", trig["date"]) not in sent
    # what happened more than a week ago is not news (e.g. after a rule change or a missed week)
    recent = str(st.index[max(0, len(st) - 6)].date())
    if s == "FAILED":
        return (trig is not None and (r["t"], "FAILED", trig["date"]) not in sent
                and "FAILED" not in set(st["stage"].iloc[-6:-1]))
    if s == "TARGET":
        return r["trade"]["exit_date"] >= recent and (r["t"], "TARGET", r["trade"]["date"]) not in sent
    # SETUP: only when it was not already set up in the previous 10 sessions
    return "SETUP" not in set(st["stage"].iloc[-11:-1])


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
    ledger = T.load()
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
    rows, missing, scored = [], [], {}
    for t in names + followed:
        b = bars.get(t)
        if b is None or len(b) < 220 or t in stale:
            if t in names:
                missing.append(t)
            continue
        row, f, st = score_ticker(t, b, spy, eng)
        scored[t] = (f, st)
        if t not in names:
            continue
        if row["stage"] == "n/a":
            missing.append(t)
            continue
        e = wl[t]
        row.update(name=e.get("name", t), source=e.get("source"), thesis=e.get("thesis"), author=e.get("author"),
                   added=e.get("added"))
        rows.append(row)

    ledger, last["ledger_seen"] = T.update(ledger, scored, spy, eng, last.get("ledger_seen", {}))
    for r in rows:
        f, st = scored[r["t"]]
        since = str(st.index[max(0, len(st) - 1 - T.LOOKBACK)].date())
        attach_trade(r, T.latest(ledger, r["t"], since), eng)
        open_trade = bool(r["trade"]) and r["trade"]["status"] == "open"
        wl[r["t"]]["hold"] = open_trade or r["stage"] in ("TRIGGERED", "TARGET", "RUNNING", "EXTENDED", "RETEST")
    rows.sort(key=lambda r: (-r["ready"], r["t"]))

    # alerts: stage changes since the last recorded run (or since yesterday's close on a first run)
    prev = last.get("stages", {})
    new_alerts = []
    if fresh_session:
        past = [json.loads(x) for x in C.ALERTS.read_text().splitlines()] if C.ALERTS.exists() else []
        sent = {(a["t"], a["type"], a.get("trigger_date") or a["date"]) for a in past}
        for r in rows:
            before = prev.get(r["t"], r["prev_stage"])
            f, st = scored[r["t"]]
            trig = trigger_day(f, st)
            if not _alert_due(r, before, st, trig, sent):
                continue
            strong = r["stage"] != "TRIGGERED" or r["ready"] >= eng["trigger"].get("min_ready", 0)
            key_date = r["trade"]["date"] if r["stage"] == "TARGET" else trig["date"] if r["stage"] in (
                "TRIGGERED", "RUNNING", "FAILED") else None
            new_alerts.append({"date": r["date"], "t": r["t"], "type": r["stage"], "from": before, "strong": strong,
                               "close": trig["close"] if r["stage"] == "TRIGGERED" else r["close"],
                               "level": trig["level"] if r["stage"] == "TRIGGERED" else r["level"],
                               "ready": r["ready"], "prob": r["prob"], "trigger_date": key_date,
                               "text": alert_text(r, eng["trigger"], trig), "engine": eng["version"]})
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

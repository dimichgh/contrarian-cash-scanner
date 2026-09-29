"""The daily run: rescan and relearn when due, score every active name, raise alerts, keep score."""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from . import config as C
from . import learn as L
from . import watchlist as W
from .model import auc, predict, readiness
from .prices import fetch_many
from .signals import PIVOT_TEXT, features, move_size, outcomes, stages

ALERT_STAGES = {"TRIGGERED", "SETUP", "RUNNING", "FAILED"}


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
    return W.merge_scan(wl, scan, today)


def score_ticker(t: str, b: pd.DataFrame, spy: pd.Series, eng: dict) -> tuple[dict, pd.DataFrame]:
    f = features(b, spy)
    st = stages(f, eng["trigger"])
    m = eng["model"]
    prob = predict(m, f[m["features"]].iloc[-1:].to_numpy())[0]
    last, s = f.iloc[-1], st.iloc[-1]
    p = eng["trigger"]
    level = float(s["pivot"]) if s["stage"] not in ("TRIGGERED", "RUNNING", "EXTENDED", "FAILED") \
        else float(s["trig_level"]) if not pd.isna(s["trig_level"]) else float(s["pivot"])
    size = float(move_size(f["close"], eng["label"]["horizon"]).iloc[-1])
    window = f["close"].iloc[-60:]
    lo, hi = float(window.min()), float(window.max())
    norm = lambda x: int(round((x - lo) / (hi - lo) * 99)) if hi > lo else 50
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
    return row, st


def _contributions(m: dict, last: pd.Series) -> list:
    """The three features pushing readiness up the most and the one pulling it down the most."""
    z = (last[m["features"]].astype(float).to_numpy() - np.array(m["mean"])) / np.array(m["std"])
    c = np.nan_to_num(z * np.array(m["coef"]))
    order = np.argsort(c)
    pos = [(m["features"][i], round(float(c[i]), 2)) for i in order[::-1][:3] if c[i] > 0.02]
    neg = [(m["features"][i], round(float(c[i]), 2)) for i in order[:1] if c[i] < -0.02]
    return pos + neg


def alert_text(r: dict, p: dict) -> str:
    lvl = PIVOT_TEXT[p["pivot"]]
    if r["stage"] == "TRIGGERED":
        vol = f" on {r['vol_ratio']:.1f}x average volume" if r["vol_ratio"] else ""
        return f"closed {r['gap']:+.1f}% through the {lvl} (${r['level']:.2f}){vol}"
    if r["stage"] == "SETUP":
        return f"{abs(r['gap']):.1f}% under the trigger level ${r['level']:.2f} ({lvl}), lows held {r['low_age']} sessions"
    if r["stage"] == "RUNNING":
        return f"move confirmed: holding {r['gap']:+.1f}% above ${r['level']:.2f}, 50-day average rising {r['slope']:+.1f}%/2wk"
    if r["stage"] == "FAILED":
        return f"breakout failed: back {r['gap']:.1f}% under ${r['level']:.2f}"
    return ""


def track_record(hist: pd.DataFrame, bars: dict, label: dict, alerts: list[dict]) -> dict:
    """Out-of-sample scorekeeping: how the engine's own past calls turned out."""
    out: dict = {}
    if len(hist):
        lab = []
        for t, g in hist.groupby("t"):
            if t not in bars:
                continue
            bl = outcomes(bars[t]["close"], label)
            lab.append(g.assign(_d=pd.to_datetime(g["date"])).join(bl, on="_d"))
        if lab:
            h = pd.concat(lab).dropna(subset=["y"])
            hi = h[h["ready"] >= 80]
            out["snapshots"] = {"resolved": len(h), "open": int(len(hist) - len(h)),
                                "hit_rate": round(float(h["y"].mean()), 3) if len(h) else None,
                                "hit_rate_ready80": round(float(hi["y"].mean()), 3) if len(hi) else None,
                                "n_ready80": len(hi), "auc": round(auc(h["y"], h["prob"]), 3) if len(h) >= 30 else None}
    res = []
    for a in alerts:
        if a["type"] != "TRIGGERED" or a["t"] not in bars:
            continue
        bl = outcomes(bars[a["t"]]["close"], label)
        d = pd.Timestamp(a["date"])
        if d in bl.index and not pd.isna(bl.at[d, "y"]):
            res.append({**a, "y": float(bl.at[d, "y"]), "ret": round(float(bl.at[d, "ret"]) * 100, 1),
                        "bars": int(bl.at[d, "bars"])})
    def stats(xs):
        return {"resolved": len(xs), "hit_rate": round(float(np.mean([r["y"] for r in xs])), 3) if xs else None,
                "avg_ret": round(float(np.mean([r["ret"] for r in xs])), 2) if xs else None}
    out["triggers"] = {**stats(res), "strong": stats([r for r in res if r.get("strong", True)]), "recent": res[-15:]}
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
        added, dropped = rescan(wl, today)
        last["rescan"] = today.isoformat()
    added += W.ensure_author_calls(wl, today)

    did_learn = force_learn or eng is None or _due(eng.get("updated"), C.LEARN_DAYS, today)
    names = W.active(wl)
    fetch = list(names)
    if did_learn:
        scan = W.load_scan()
        fetch += list(scan["name"]) if scan is not None else []
    bars = fetch_many(fetch + ["SPY"], "5y" if did_learn else "2y")
    if "SPY" not in bars:
        raise RuntimeError("no SPY bars from Yahoo; cannot score")
    spy = bars["SPY"]["close"]
    if did_learn:
        prev_version = eng.get("version") if eng else None
        eng = L.learn({t: b for t, b in bars.items() if t != "SPY"}, spy, eng, today)
        C.ENGINE.write_text(json.dumps(eng, indent=1) + "\n")
        if prev_version is not None and eng["version"] != prev_version:
            notes.append(f"Engine updated to v{eng['version']}")

    as_of = str(spy.index[-1].date())
    fresh_session = as_of != last.get("as_of")
    rows, missing = [], []
    for t in names:
        b = bars.get(t)
        if b is None or len(b) < 220:
            missing.append(t)
            continue
        row, _ = score_ticker(t, b, spy, eng)
        if row["stage"] == "n/a":
            missing.append(t)
            continue
        e = wl[t]
        row.update(name=e.get("name", t), source=e.get("source"), thesis=e.get("thesis"), author=e.get("author"),
                   added=e.get("added"))
        e["hold"] = row["stage"] in ("TRIGGERED", "RUNNING", "EXTENDED")
        rows.append(row)
    rows.sort(key=lambda r: (-r["ready"], r["t"]))

    # alerts: stage changes since the last recorded run (or since yesterday's close on a first run)
    prev = last.get("stages", {})
    new_alerts = []
    if fresh_session:
        for r in rows:
            before = prev.get(r["t"], r["prev_stage"])
            if r["stage"] in ALERT_STAGES and r["stage"] != before:
                strong = r["stage"] != "TRIGGERED" or r["ready"] >= eng["trigger"].get("min_ready", 0)
                new_alerts.append({"date": r["date"], "t": r["t"], "type": r["stage"], "from": before, "strong": strong,
                                   "close": r["close"], "level": r["level"], "ready": r["ready"], "prob": r["prob"],
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
    hist = pd.concat([pd.read_csv(p) for p in sorted(C.HISTORY.glob("*.csv"))]) if C.HISTORY.exists() and \
        any(C.HISTORY.glob("*.csv")) else pd.DataFrame()
    record = track_record(hist, bars, eng["label"], alerts)

    last.update(as_of=as_of, run=dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
                stages={r["t"]: r["stage"] for r in rows}, record=record)
    C.LAST_RUN.write_text(json.dumps(last, indent=1) + "\n")
    W.save(wl)
    return {"as_of": as_of, "rows": rows, "alerts": new_alerts, "recent_alerts": alerts[-40:], "added": added,
            "dropped": dropped, "missing": missing, "engine": eng, "record": record, "notes": notes,
            "rescanned": did_rescan, "learned": did_learn, "fresh": fresh_session, "watchlist": wl}

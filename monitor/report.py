"""Outputs of a run: a short summary (the routine's notification) and the dashboard page."""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import pandas as pd

from . import config as C
from . import trades as T
from .signals import PIVOT_TEXT, STAGE_TEXT, STAGES

TEMPLATE = Path(__file__).with_name("dashboard_template.html")
MARK = {"TRIGGERED": "▲", "SETUP": "◆", "TARGET": "✓", "RUNNING": "●", "FAILED": "✕"}
FEATURE_TEXT = {
    "f_sma50_gap": "Distance above the 50-day average", "f_sma200_gap": "Distance above the 200-day average",
    "f_sma50_slope": "50-day average slope", "f_hh40_gap": "Distance to the 40-day high",
    "f_low_age": "Time since the last 120-day low", "f_higher_low": "Higher lows", "f_rs_slope": "1-month strength vs S&P 500",
    "f_rs_pos": "Strength vs S&P 500 in its 6-month range", "f_rsi": "RSI", "f_squeeze": "Volatility vs its norm",
    "f_updown_vol": "Up-day vs down-day volume", "f_vol_surge": "Volume surge (5d vs 50d)", "f_ret20": "1-month return",
    "f_dd_52w": "Drop from the 52-week high", "f_mkt": "S&P 500 vs its 200-day average", "f_vol20": "Volatility",
}


def rule_text(p: dict, q: dict | None = None) -> str:
    vol = f" on at least {p['vol_mult']:g}x average volume" if p["vol_mult"] else ""
    base = f", after {p['min_below']}+ of the prior 20 sessions below it" if p["min_below"] else ""
    if (q or {}).get("active"):
        bar = f"; strong when trigger quality is {q['min_quality']}+" if q["min_quality"] else ""
    else:
        bar = f"; strong when readiness is {p['min_ready']}+" if p.get("min_ready") else ""
    return f"Close above the {PIVOT_TEXT[p['pivot']]}{vol}{base}{bar}"


def bar_text(eng: dict) -> str:
    """What a weak trigger fell short of."""
    return "quality" if (eng.get("quality") or {}).get("active") else "readiness"


def rating(a: dict) -> str:
    """A trigger's scores as one phrase: quality first when it has one."""
    q = a.get("quality")
    ready = a["ready_at"] if "ready_at" in a else a["ready"]
    return f"quality {q}, readiness {ready}" if q is not None else f"readiness {ready}"


def _md(iso: str) -> str:
    return dt.date.fromisoformat(iso).strftime("%b %-d")


def _sessions_since(iso: str, as_of: str) -> int:
    """Weekdays from `iso` to `as_of` (holidays count, which is close enough for "recent")."""
    return max(0, len(pd.bdate_range(iso, as_of)) - 1)


def _pct(x):
    return "n/a" if x is None else f"{x * 100:.0f}%"


def summary(res: dict) -> str:
    eng, rows, alerts = res["engine"], res["rows"], res["alerts"]
    mm, tm = eng["metrics"]["model"], eng["metrics"]["trigger"]
    lines = [f"Breakout monitor · close of {res['as_of']} · engine v{eng['version']}", ""]
    if alerts:
        lines.append(f"{len(alerts)} new signal{'s' * (len(alerts) != 1)}:")
        order = {s: i for i, s in enumerate(STAGES)}
        for a in sorted(alerts, key=lambda a: (order.get(a["type"], 9), -(a.get("quality") or 0), -a["ready"])):
            strong = a.get("strong", True)
            kind = a["type"] if a["type"] != "TRIGGERED" else "TRIGGERED (strong)" if strong \
                else f"TRIGGERED (weak: {bar_text(eng)} under the learned bar)"
            mark = MARK.get(a["type"], "·") if strong else "△"
            lines.append(f"{mark} {kind} {a['t']} ${a['close']:.2f}: {a['text']} · {rating(a)}")
    else:
        lines.append("No new signals." if res["fresh"] else "No new session since the last run.")
    near = [r for r in rows if r["stage"] in ("TRIGGERED", "SETUP")][:8]
    if near:
        lines += ["", "Nearest the trigger: " + ", ".join(
            f"{r['t']} {r['gap']:+.1f}% vs ${r['level']:.2f} (readiness {r['ready']})" for r in near)]
    top = [r for r in rows if r["stage"] not in ("RUNNING", "EXTENDED", "TARGET")][:8]
    if top:
        lines.append("Highest readiness: " + ", ".join(f"{r['t']} {r['ready']} ({r['stage'].lower()})" for r in top))
    ledger = res.get("trades", [])
    open_ = sorted((t for t in ledger if t["status"] == "open"), key=lambda t: t["sessions"])
    if open_:
        def plan(t):
            w = T.window(t, eng)
            if not w:
                return f"; stop ${t['stop_px']:.2f}, target ${t['target_px']:.2f}"
            return (f"; stop ${t['stop_px']:.2f}; target ${t['target_px']:.2f} usually hit {_md(w['from'])}–{_md(w['to'])}"
                    f" (day {t['sessions']} of a typical {w['p25']}–{w['p75']}, {w['phase']})")
        def grade(t):
            if t.get("quality") is None:
                return ""
            return f"quality {t['quality']} {'strong' if t['strong'] else 'weak'}, "
        lines += ["", "Open trades: " + ", ".join(
            f"{t['t']} {t['change']:+.1f}% in {t['sessions']}d ({grade(t)}S&P {t['spy']:+.1f}%, best {t['best']:+.1f}%{plan(t)})"
            for t in open_)]
    closed = [t for t in ledger if t["status"] != "open" and _sessions_since(t["exit_date"], res["as_of"]) <= 5]
    if closed:
        lines.append("Closed this week: " + ", ".join(
            f"{t['t']} {t['status']} {t['change']:+.1f}% on {_md(t['exit_date'])}" for t in closed))
    counts = {s: sum(r["stage"] == s for r in rows) for s in STAGES}
    lines += ["", f"Watchlist: {len(rows)} names · " + " · ".join(f"{k.lower()} {v}" for k, v in counts.items() if v)]
    if res["added"]:
        lines.append(f"Added: {', '.join(res['added'])}")
    if res["dropped"]:
        lines.append(f"Dropped: {', '.join(res['dropped'])}")
    if res["missing"]:
        lines.append(f"No price data: {', '.join(res['missing'])}")
    t_all = tm["all"]
    lines.append(f"Engine: readiness AUC {mm['test_auc']:.3f} on the held-out year (expert prior alone {mm['prior_only_auc']:.3f}); "
                 f"trigger: {rule_text(eng['trigger'], eng.get('quality')).lower()} — {_pct(t_all['hit_rate'])} reached their target "
                 f"before their stop vs {_pct(t_all['base_hit_rate'])} for any day ({t_all['n']} past triggers)")
    qm = eng["metrics"].get("quality")
    if qm and qm["active"]:
        hi = qm["by_threshold"][str(qm["min_quality"])]
        lines.append(f"Trigger quality: AUC {qm['test_auc']:.3f} on the held-out year's {qm['n_test']} triggers; "
                     f"the {hi['n']} at quality {qm['min_quality']}+ hit target {_pct(hi['hit_rate'])} of the time "
                     f"(avg {hi['expectancy'] * 100:+.1f}%) vs {_pct(qm['by_threshold']['0']['hit_rate'])} for all of them.")
    timing = tm.get("timing", {})
    if "median" in timing:
        lines.append(f"Past triggers that hit target took a median {timing['median']} sessions (middle half "
                     f"{timing['p25']}–{timing['p75']}); stops came in a median {timing['stop_median']}. "
                     f"{_pct(timing['hit'])} hit target, {_pct(timing['stopped'])} stopped, "
                     f"{_pct(timing['timed_out'])} timed out.")
    live = [t for t in res.get("trades", []) if t.get("live")]
    if live:  # counted the day a trade reaches its target or stop, not after the full 40 sessions
        n = lambda s: sum(t["status"] == s for t in live)
        closed = [t for t in live if t["status"] not in ("open", "no data")]
        avg = f"; closed ones averaged {sum(t['change'] for t in closed) / len(closed):+.1f}%" if closed else ""
        lines.append(f"Live triggers so far: {len(live)}: {n('hit target')} hit target, {n('stopped out')} stopped out, "
                     f"{n('timed out')} timed out, {n('open')} open{avg}.")
    lines += res["notes"]
    publish = res["fresh"] or res["rescanned"] or res["learned"]
    lines += ["", f"DASHBOARD: {'publish' if publish else 'skip'}"]
    return "\n".join(lines) + "\n"


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _trades(res: dict) -> list[dict]:
    """Every trade in the ledger, with today's close beside it and how long ago a closed one ended."""
    names = {t: e.get("name", t) for t, e in res.get("watchlist", {}).items()}
    out = []
    for t in res.get("trades", []):
        now = res.get("closes", {}).get(t["t"])
        out.append({k: v for k, v in t.items() if k not in ("up", "down")}
                   | {"name": names.get(t["t"], t["t"]), "window": T.window(t, res["engine"]),
                      "now": None if now is None else round(now, 2),
                      "now_change": None if now is None else round((now / t["entry"] - 1) * 100, 1),
                      "since_exit": None if t["status"] == "open" else _sessions_since(t["exit_date"], res["as_of"])})
    return sorted(out, key=lambda t: (t["status"] != "open", t["exit_date"] or t["date"]), reverse=False)


def dashboard(res: dict) -> str:
    eng = res["engine"]
    m = eng["model"]
    learned = sorted(({"f": FEATURE_TEXT.get(k, k), "prior": C.PRIOR.get(k, 0), "w": w}
                      for k, w in zip(m["features"], m["coef"])), key=lambda x: -abs(x["w"]))
    q = eng.get("quality") or {}
    q_learned = sorted(({"f": FEATURE_TEXT.get(k, k), "prior": 0, "w": w}
                        for k, w in zip(q.get("features", []), q.get("coef", []))), key=lambda x: -abs(x["w"]))
    history = [json.loads(x) for x in C.ENGINE_LOG.read_text().splitlines()] if C.ENGINE_LOG.exists() else []
    rows = [{k: v for k, v in r.items() if k not in ("prev_stage", "watch")} for r in res["rows"]]
    for r in rows:
        r["contrib"] = [[FEATURE_TEXT.get(k, k), v] for k, v in r["contrib"]]
        if r.get("thesis"):
            r["thesis"] = {**r["thesis"], "why": r["thesis"]["why"][:4], "risk": r["thesis"]["risk"][:2]}
        r.pop("date", None)
    by_trade = {(t["t"], t["date"]): t for t in res.get("trades", [])}

    def graded(a):  # a trigger shows the rating its trade carries (trades from before quality get it later)
        t = by_trade.get((a["t"], a.get("trigger_date")))
        return {**a, "quality": t.get("quality"), "strong": t["strong"]} if a["type"] == "TRIGGERED" and t else a
    data = {
        "asof": res["as_of"], "version": eng["version"], "updated": eng["updated"],
        "stages": [[s, STAGE_TEXT[s]] for s in STAGES], "rule": rule_text(eng["trigger"], q),
        "label": eng["label"], "trigger": eng["trigger"], "metrics": eng["metrics"],
        "quality": {k: q[k] for k in ("active", "min_quality", "lam") if k in q},
        "learned": learned, "q_learned": q_learned, "history": history[-20:], "record": res["record"],
        "alerts": [graded(a) for a in res["recent_alerts"][::-1]], "rows": rows, "trades": _trades(res),
        "ledger": T.stats(res.get("trades", [])),
        "added": res["added"], "dropped": res["dropped"],
    }
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=_clean)
    blob = blob.replace("NaN", "null")
    return TEMPLATE.read_text().replace("/*__DATA__*/null", blob)


def write(res: dict) -> str:
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    text = summary(res)
    (C.REPORTS / "summary.md").write_text(text)
    (C.REPORTS / "dashboard.html").write_text(dashboard(res))
    return text

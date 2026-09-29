"""Outputs of a run: a short summary (the routine's notification) and the dashboard page."""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import config as C
from .signals import PIVOT_TEXT, STAGE_TEXT, STAGES

TEMPLATE = Path(__file__).with_name("dashboard_template.html")
MARK = {"TRIGGERED": "▲", "SETUP": "◆", "RUNNING": "●", "FAILED": "✕"}
FEATURE_TEXT = {
    "f_sma50_gap": "Distance above the 50-day average", "f_sma200_gap": "Distance above the 200-day average",
    "f_sma50_slope": "50-day average slope", "f_hh40_gap": "Distance to the 40-day high",
    "f_low_age": "Time since the last 120-day low", "f_higher_low": "Higher lows", "f_rs_slope": "1-month strength vs S&P 500",
    "f_rs_pos": "Strength vs S&P 500 in its 6-month range", "f_rsi": "RSI", "f_squeeze": "Volatility vs its norm",
    "f_updown_vol": "Up-day vs down-day volume", "f_vol_surge": "Volume surge (5d vs 50d)", "f_ret20": "1-month return",
    "f_dd_52w": "Drop from the 52-week high", "f_mkt": "S&P 500 vs its 200-day average", "f_vol20": "Volatility",
}


def rule_text(p: dict) -> str:
    vol = f" on at least {p['vol_mult']:g}x average volume" if p["vol_mult"] else ""
    base = f", after {p['min_below']}+ of the prior 20 sessions below it" if p["min_below"] else ""
    ready = f"; strong when readiness is {p['min_ready']}+" if p.get("min_ready") else ""
    return f"Close above the {PIVOT_TEXT[p['pivot']]}{vol}{base}{ready}"


def _pct(x):
    return "n/a" if x is None else f"{x * 100:.0f}%"


def summary(res: dict) -> str:
    eng, rows, alerts = res["engine"], res["rows"], res["alerts"]
    mm, tm = eng["metrics"]["model"], eng["metrics"]["trigger"]
    lines = [f"Breakout monitor · close of {res['as_of']} · engine v{eng['version']}", ""]
    if alerts:
        lines.append(f"{len(alerts)} new signal{'s' * (len(alerts) != 1)}:")
        order = {s: i for i, s in enumerate(STAGES)}
        for a in sorted(alerts, key=lambda a: (order.get(a["type"], 9), -a["ready"])):
            kind = a["type"] if a.get("strong", True) else "TRIGGERED (weak: readiness under the learned bar)"
            mark = MARK.get(a["type"], "·") if a.get("strong", True) else "△"
            lines.append(f"{mark} {kind} {a['t']} ${a['close']:.2f}: {a['text']} · readiness {a['ready']}")
    else:
        lines.append("No new signals." if res["fresh"] else "No new session since the last run.")
    near = [r for r in rows if r["stage"] in ("TRIGGERED", "SETUP")][:8]
    if near:
        lines += ["", "Nearest the trigger: " + ", ".join(
            f"{r['t']} {r['gap']:+.1f}% vs ${r['level']:.2f} (readiness {r['ready']})" for r in near)]
    top = [r for r in rows if r["stage"] not in ("RUNNING", "EXTENDED")][:8]
    if top:
        lines.append("Highest readiness: " + ", ".join(f"{r['t']} {r['ready']} ({r['stage'].lower()})" for r in top))
    trades = sorted((r for r in rows if r.get("trade")), key=lambda r: r["trade"]["sessions"])
    open_ = [r for r in trades if r["trade"]["status"] == "open"]
    if open_:
        lines += ["", "Since the trigger (open): " + ", ".join(
            f"{r['t']} {r['trade']['change']:+.1f}% in {r['trade']['sessions']}d "
            f"(S&P {r['trade']['spy']:+.1f}%, best {r['trade']['best']:+.1f}%)" for r in open_)]
    closed = [r for r in trades if r["trade"]["status"] != "open"
              and r["trade"]["sessions"] - r["trade"]["resolved_in"] <= 5]
    if closed:
        lines.append("Resolved this week: " + ", ".join(
            f"{r['t']} {r['trade']['status']} ({r['trade']['change']:+.1f}% now)" for r in closed))
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
                 f"trigger: {rule_text(eng['trigger']).lower()} — {_pct(t_all['hit_rate'])} reached their target before their stop "
                 f"vs {_pct(t_all['base_hit_rate'])} for any day ({t_all['n']} past triggers)")
    tr = res["record"].get("triggers", {})
    if tr.get("resolved"):
        st = tr["strong"]
        lines.append(f"Live record: {tr['resolved']} trigger alerts resolved, {_pct(tr['hit_rate'])} hit target first, "
                     f"average {tr['avg_ret']:+.1f}% (strong ones: {st['resolved']}, {_pct(st['hit_rate'])})")
    lines += res["notes"]
    publish = res["rescanned"] or res["learned"] or any(a["type"] == "TRIGGERED" for a in alerts)
    lines += ["", f"DASHBOARD: {'publish' if publish else 'skip'}"]
    return "\n".join(lines) + "\n"


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def dashboard(res: dict) -> str:
    eng = res["engine"]
    m = eng["model"]
    learned = sorted(({"f": FEATURE_TEXT.get(k, k), "prior": C.PRIOR.get(k, 0), "w": w}
                      for k, w in zip(m["features"], m["coef"])), key=lambda x: -abs(x["w"]))
    history = [json.loads(x) for x in C.ENGINE_LOG.read_text().splitlines()] if C.ENGINE_LOG.exists() else []
    rows = [{k: v for k, v in r.items() if k not in ("prev_stage", "watch")} for r in res["rows"]]
    for r in rows:
        r["contrib"] = [[FEATURE_TEXT.get(k, k), v] for k, v in r["contrib"]]
        if r.get("thesis"):
            r["thesis"] = {**r["thesis"], "why": r["thesis"]["why"][:4], "risk": r["thesis"]["risk"][:2]}
        r.pop("date", None)
    data = {
        "asof": res["as_of"], "version": eng["version"], "updated": eng["updated"],
        "stages": [[s, STAGE_TEXT[s]] for s in STAGES], "rule": rule_text(eng["trigger"]),
        "label": eng["label"], "trigger": eng["trigger"], "metrics": eng["metrics"],
        "learned": learned, "history": history[-20:], "record": res["record"],
        "alerts": res["recent_alerts"][::-1], "rows": rows,
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

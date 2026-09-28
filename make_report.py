#!/usr/bin/env python3
"""Build output/scan_report.html from output/arden_scan.json (run arden_scanner.py first)."""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import pandas as pd

import arden_scanner as a

HERE = Path(__file__).parent
OUT = HERE / "output"
TOP_N = 150

# The author's long calls on single stocks / sector ETFs, with the price they stated.
AUTHOR_CALLS = [
    ("INTC", "NASDAQ", "2025-05-15", 20.00, "Recovery option on US fabs; fair value $36–37"),
    ("XOP", "AMEX", "2025-09-19", 130.00, "Contrarian energy long; fair value $143–160"),
    ("UDMY", "NASDAQ", "2025-10-27", 6.92, "First positive FCF, CEO buying; fair value $11–12"),
    ("F", "NYSE", "2026-03-08", 12.15, "Chairman bought, quants long; base $17–18"),
    ("T", "NYSE", "2026-06-07", 24.00, "3-year target $28–32; preferred entry $19–20"),
    ("PYPL", "NASDAQ", "2026-06-16", 41.50, "17% FCF yield, 11% share shrink; base $60–80"),
    ("SONY", "NYSE", "2026-06-28", 19.70, "Sum of parts $26–28; yen upside"),
    ("CMCSA", "NASDAQ", "2026-07-23", 21.92, "Sold on a beat; spin-off 2027; accumulate $21–23"),
    ("PEP", "NASDAQ", "2026-09-19", None, "Buy zone $120–125 (not reached yet); target $180–200"),
]

FACTORS = [
    ("Cash machine", a.WEIGHTS["cash"], "FCF yield against market cap. Always set against the 10-year Treasury and the S&P 500 average.",
     "PYPL: “$6.4B FCF on a ~$38B cap is 15–17% a year, vs 4.3% on Treasuries.” CMCSA: FCF yield ~17% vs 4–5% for the S&P.",
     "FCF TTM ÷ market cap, FCF margin"),
    ("Cheap vs peers", a.WEIGHTS["valuation"], "Multiples against the closest competitor or the company’s own history, plus sum-of-parts gaps.",
     "PEP 16–17× vs KO 26×. F 8× vs Toyota 12.7×. SONY EV/EBITDA 6.9 vs UMG 14–15. PYPL fwd P/E 8.5 vs 39 median.",
     "Fwd P/E, EV/EBITDA, P/FCF, P/S, P/B ÷ sector median"),
    ("Paid to wait", a.WEIGHTS["shareholder"], "Dividend plus the real fall in share count, and whether FCF covers the dividend.",
     "CMCSA: 6% dividend + 4–5% buyback = 10–11%/yr. PYPL share count 999M → 892M. T dividend covered 2.25×.",
     "Dividend yield + net share reduction, FCF/dividend cover"),
    ("Capitulation", a.WEIGHTS["capitulation"], "Price damage: 52-week lows, multi-year declines, a crowd that has given up.",
     "SONY −35% from the high at a 52w low. PYPL −80% from peak. PEP “falling for the third year in a row.”",
     "Drop from 52w high, 5-year return, distance from 52w low"),
    ("Balance sheet", a.WEIGHTS["balance"], "Debt the business can carry. Net cash is a bonus.",
     "CMCSA: exit trigger if net debt/EBITDA passes 2.9×. PYPL cash $13.5B vs debt $11.6B. UDMY no debt.",
     "Net debt ÷ EBITDA, current ratio"),
    ("Business still works", a.WEIGHTS["business"], "Records and beats hidden under a bad narrative. Guards against value traps.",
     "SONY record revenue and operating profit. CMCSA beat on revenue, EPS and EBITDA. F record $187B revenue.",
     "Revenue growth, operating margin, last EPS surprise, ROE"),
    ("Contrarian consensus", a.WEIGHTS["sentiment"], "Analysts sit on Hold and cut targets, yet their own average target is far above the price.",
     "CMCSA: 27 analysts, mostly Hold, target +40%. PYPL consensus Hold, target $52 from $41.",
     "Avg target upside × share of Hold/Sell ratings"),
    ("Smart money", None, "Insiders buying with their own cash, funds adding, shorts covering, and a sell-off on good news.",
     "F chairman bought $1.93M of stock. UDMY CEO bought $330k. PEP shorts fell 33.3M → 21.3M. CMCSA −6.8% on a beat.",
     "OpenInsider buys (90d), FINRA short interest, 2-day earnings reaction"),
]


def clean(v):
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (list, tuple)):
        return list(v)
    return v


def r1(v, n=1):
    v = clean(v)
    return None if v is None else round(float(v), n)


def build_rows(j: pd.DataFrame) -> list[dict]:
    rows = []
    for i, r in j.head(TOP_N).iterrows():
        rows.append({
            "rank": int(i) + 1, "t": r["name"], "n": r["description"], "sec": r["sector"], "ind": r["industry"],
            "ex": r["exchange"], "px": r1(r["close"], 2), "cap": r1(r["market_cap_basic"] / 1e9, 1),
            "score": r1(r["score"]), "tier": r["tier"], "checks": int(r["n_checks"]), "hit": list(r["checks_hit"]),
            "timing": r["timing"], "enriched": bool(r["enriched"]),
            "fcf": r1(r["fcf_yield"]), "shy": r1(r["sh_yield"]), "div": r1(r["div_yield"]), "bb": r1(r["buyback_yld"]),
            "pe": r1(r["fwd_pe"]), "pem": r1(r["fwd_pe_sector_med"]), "eve": r1(r["ev_ebitda"]),
            "evem": r1(r["ev_ebitda_sector_med"]), "nd": r1(r["nd_ebitda"]), "dd": r1(r["dd_52w"]),
            "low": r1(r["from_52w_low"]), "p5": r1(r["perf_5y"]), "rev": r1(r["rev_growth"]),
            "sur": r1(r["eps_surprise"]), "er": r1(r.get("earn_reaction")), "si": r1(r.get("si_change")),
            "ins": r1(r.get("insider_value"), 0), "up": r1(r["target_upside"]), "nb": r1(r["non_buy_share"], 2),
            "na": int(r["n_analysts"] or 0), "beta": r1(r["beta"], 2), "enr": r1(r["enrich_pts"]),
            "p": {k: round(float(r[f"s_{k}"]) * 100) for k in a.WEIGHTS},
            "why": list(r["thesis"]), "risk": list(r["risks"]),
        })
    return rows


def build_calls(j: pd.DataFrame) -> list[dict]:
    live = a.tv_scan({"symbols": {"tickers": [f"{ex}:{t}" for t, ex, *_ in AUTHOR_CALLS]}, "columns": ["name", "close"]})
    px = {d["d"][0]: d["d"][1] for d in live["data"]}
    ranks = {n: i + 1 for i, n in enumerate(j["name"])}
    uni = a.gate(a.add_metrics(a.fetch_universe(2e9)))
    reasons = {n: (f"{g} ({p:+.0f}% in 1 year)" if g == "not beaten down" else g)
               for n, g, p in zip(uni["name"], uni["gate_fail"], uni["perf_y"])}
    out = []
    for t, ex, date, entry, note in AUTHOR_CALLS:
        now = px.get(t)
        if t in ranks:
            row = j[j["name"] == t].iloc[0]
            status = f"#{ranks[t]} of {len(j)} · Tier {row['tier']} · {int(row['n_checks'])}/10"
        elif t in reasons and reasons[t]:
            status = f"Filtered out: {reasons[t]}"
        elif t in ("XOP",):
            status = "ETF, not scanned"
        else:
            status = "No longer listed under this ticker"
        out.append({"t": t, "date": date, "entry": entry, "now": r1(now, 2), "note": note, "status": status,
                    "ret": r1((now / entry - 1) * 100) if (now and entry) else None})
    return out


def main():
    j = pd.read_json(OUT / "arden_scan.json")
    uni_n = len(a.fetch_universe(2e9))
    data = {
        "date": dt.date.today().isoformat(),
        "universe": uni_n, "passed": len(j), "enriched": int(j["enriched"].sum()),
        "tierA": int((j["tier"] == "A").sum()), "tierB": int((j["tier"] == "B").sum()),
        "tenY": a.TEN_YEAR_YIELD,
        "checks": [name for name, _ in a.CHECKS],
        "weights": a.WEIGHTS,
        "factors": [dict(zip(("name", "w", "what", "quote", "metric"), f)) for f in FACTORS],
        "rows": build_rows(j),
        "calls": build_calls(j),
    }
    html = (HERE / "report_template.html").read_text()
    html = html.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, default=clean))
    (OUT / "scan_report.html").write_text(html)
    print(f"wrote {OUT / 'scan_report.html'}  ({len(data['rows'])} rows)")


if __name__ == "__main__":
    main()

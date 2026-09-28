#!/usr/bin/env python3
"""
Arden-style contrarian value scanner.

Reproduces the stock-selection framework used in arden_trading's long-term
TradingView ideas (PEP, CMCSA, T, PYPL, SONY, F, INTC, UDMY):

  "A beaten-down, hated company that still throws off lots of free cash,
   returns it to shareholders, trades far below peers, has a manageable
   balance sheet, a business that still works, analysts sitting on Hold
   with big target upside, and smart money quietly buying."

Data sources (all free, no API keys):
  - TradingView screener API  -> universe, fundamentals, analyst data, technicals
  - Yahoo Finance chart API   -> daily bars (earnings reaction, volume accumulation)
  - OpenInsider               -> open-market insider purchases (last 90 days)
  - FINRA API                 -> short-interest history

Usage:
  python arden_scanner.py                 # full scan, writes ./output/
  python arden_scanner.py --min-mcap 5e9  # large caps only
  python arden_scanner.py --enrich 150    # enrich more candidates
  python arden_scanner.py --tickers PEP CMCSA T   # score specific names
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

UA = "Mozilla/5.0"  # Yahoo rejects full browser UAs whose TLS fingerprint does not match
TEN_YEAR_YIELD = 4.3  # % — the benchmark the author compares FCF yield against

# ---------------------------------------------------------------------------
# Factor weights (base score, 0-100). FCF yield is the author's "main number".
# ---------------------------------------------------------------------------
WEIGHTS = {
    "cash": 22,         # FCF yield vs market cap
    "valuation": 18,    # cheap vs sector peers (fwd P/E, EV/EBITDA, P/FCF, P/S, P/B)
    "shareholder": 15,  # dividend + buyback yield, dividend coverage
    "capitulation": 15, # drawdown from highs, multi-year decline, near 52w low
    "balance": 10,      # net debt / EBITDA, liquidity
    "business": 10,     # revenue growth, margins, beat last quarter
    "sentiment": 10,    # analyst target upside while consensus is Hold-heavy
}
ENRICH_MAX = 15  # insider buys, short covering, sold-on-beat, accumulation, reclaim

TV_COLUMNS = [
    "name", "description", "exchange", "type", "subtype", "sector", "industry", "country",
    "close", "market_cap_basic",
    "non_gaap_price_to_earnings_per_share_forecast_next_fy", "price_earnings_ttm",
    "enterprise_value_ebitda_ttm", "price_free_cash_flow_ttm", "price_sales_current", "price_book_fq",
    "free_cash_flow_ttm", "free_cash_flow_margin_ttm",
    "dividends_yield_current", "dividend_payout_ratio_ttm", "share_buyback_ratio_fy",
    "net_debt_fq", "total_debt_fq", "ebitda_ttm", "cash_n_short_term_invest_fq", "current_ratio",
    "total_revenue_ttm", "total_revenue_yoy_growth_ttm", "operating_margin_ttm", "return_on_equity",
    "earnings_per_share_diluted_yoy_growth_ttm", "eps_surprise_percent_fq",
    "earnings_release_date", "earnings_release_next_date",
    "price_target_average", "recommendation_buy", "recommendation_hold", "recommendation_sell",
    "recommendation_mark",
    "Perf.1M", "Perf.3M", "Perf.6M", "Perf.Y", "Perf.YTD", "Perf.5Y",
    "price_52_week_high", "price_52_week_low", "High.All",
    "RSI", "SMA50", "SMA200", "beta_1_year",
    "average_volume_10d_calc", "average_volume_90d_calc", "float_shares_outstanding",
]
EXCLUDED_SECTORS = {"Finance", "Miscellaneous"}  # FCF / EBITDA meaningless for banks, funds
CAPTIVE_FINANCE_INDUSTRIES = {"Motor Vehicles", "Trucks/Construction/Farm Machinery"}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def http_get(url: str, headers: dict | None = None, timeout: int = 20, retries: int = 2) -> bytes:
    h = {"User-Agent": UA}
    h.update(headers or {})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
                return r.read()
        except Exception:
            if attempt == retries:
                raise
            time.sleep(1.5 * (attempt + 1))
    return b""


def tv_scan(body: dict) -> dict:
    req = urllib.request.Request(
        "https://scanner.tradingview.com/america/scan",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "User-Agent": UA},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


# ---------------------------------------------------------------------------
# 1. Universe
# ---------------------------------------------------------------------------
def fetch_universe(min_mcap: float, tickers: list[str] | None = None) -> pd.DataFrame:
    body: dict = {"columns": TV_COLUMNS, "range": [0, 6000],
                  "sort": {"sortBy": "market_cap_basic", "sortOrder": "desc"}}
    body["filter"] = [
        {"left": "market_cap_basic", "operation": "greater", "right": min_mcap},
        {"left": "type", "operation": "in_range", "right": ["stock", "dr"]},
        {"left": "exchange", "operation": "in_range", "right": ["NYSE", "NASDAQ", "AMEX"]},
    ]
    data = tv_scan(body)["data"]
    df = pd.DataFrame([dict(zip(TV_COLUMNS, row["d"]), symbol=row["s"]) for row in data])
    # common shares + ADRs only; one line per company (drop duplicate share classes)
    df = df[(df["type"] == "dr") | (df["subtype"].isin(["common", ""]))]
    df = df.sort_values("market_cap_basic", ascending=False)
    df = df.drop_duplicates(subset=["description"]).copy()
    df["description"] = df["description"].str.replace(r"\s+Class [A-Z]$", "", regex=True)
    df = df.drop_duplicates(subset=["market_cap_basic", "sector"])
    if tickers:
        df = df[df["name"].isin([t.upper() for t in tickers])]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Metrics
# ---------------------------------------------------------------------------
def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    num = lambda c: pd.to_numeric(d[c], errors="coerce")
    d["fcf_yield"] = num("free_cash_flow_ttm") / num("market_cap_basic") * 100
    d["div_yield"] = num("dividends_yield_current").fillna(0)
    # net share-count reduction over the fiscal year (what the author quotes: "share count down 4.9%");
    # gross buyback dollars overstate it for stock-comp-heavy companies
    d["buyback_yld"] = num("share_buyback_ratio_fy").fillna(0).clip(lower=0, upper=15)
    d["sh_yield"] = d["div_yield"] + d["buyback_yld"]
    d["div_coverage"] = np.where(d["div_yield"] > 0, d["fcf_yield"] / d["div_yield"], np.nan)
    d["fwd_pe"] = num("non_gaap_price_to_earnings_per_share_forecast_next_fy")
    d["ev_ebitda"] = num("enterprise_value_ebitda_ttm")
    d["p_fcf"] = num("price_free_cash_flow_ttm")
    d["p_s"] = num("price_sales_current")
    d["p_b"] = num("price_book_fq")
    ebitda = num("ebitda_ttm")
    d["nd_ebitda"] = np.where(ebitda > 0, num("net_debt_fq") / ebitda, np.nan)
    # auto/equipment makers consolidate a captive lender (Ford Credit, Cat Financial): its debt is
    # funded by loan receivables, so net debt/EBITDA overstates operating leverage
    d["captive_finance"] = d["industry"].isin(CAPTIVE_FINANCE_INDUSTRIES) & (d["nd_ebitda"] > 3)
    d["dd_52w"] = (num("close") / num("price_52_week_high") - 1) * 100
    d["from_52w_low"] = (num("close") / num("price_52_week_low") - 1) * 100
    d["dd_ath"] = (num("close") / num("High.All") - 1) * 100
    d["perf_y"] = num("Perf.Y")
    d["perf_5y"] = num("Perf.5Y")
    d["rev_growth"] = num("total_revenue_yoy_growth_ttm")
    d["op_margin"] = num("operating_margin_ttm")
    d["roe"] = num("return_on_equity")
    d["eps_surprise"] = num("eps_surprise_percent_fq")
    buy, hold, sell = (num(c).fillna(0) for c in ("recommendation_buy", "recommendation_hold", "recommendation_sell"))
    d["n_analysts"] = buy + hold + sell
    d["non_buy_share"] = np.where(d["n_analysts"] > 0, (hold + sell) / d["n_analysts"].replace(0, np.nan), np.nan)
    d["target_upside"] = (num("price_target_average") / num("close") - 1) * 100
    d["beta"] = num("beta_1_year")
    d["vol_ratio"] = num("average_volume_10d_calc") / num("average_volume_90d_calc")

    # sector-relative valuation: value / sector median (computed on the full cleaned universe)
    for col in ("fwd_pe", "ev_ebitda", "p_fcf", "p_s", "p_b"):
        positive = d[col].where(d[col] > 0)
        sect_med = positive.groupby(d["sector"]).transform("median")
        sect_cnt = positive.groupby(d["sector"]).transform("count")
        med = np.where(sect_cnt >= 15, sect_med, positive.median())
        d[f"{col}_sector_med"] = med
        # negative multiples (losses) are treated as "expensive"
        d[f"{col}_rel"] = np.where(d[col] > 0, d[col] / med, 5.0)
    return d


def gate(d: pd.DataFrame) -> pd.DataFrame:
    """Hard filters shared by every one of the author's picks."""
    ok_sector = ~d["sector"].isin(EXCLUDED_SECTORS)
    positive_fcf = d["free_cash_flow_ttm"].fillna(-1) > 0
    # loose on purpose — how beaten down is scored below; this only drops stocks near their highs
    beaten_down = (d["dd_52w"] <= -20) | (d["dd_ath"] <= -30) | (d["perf_y"] <= -10) | (d["perf_5y"] <= -30)
    d = d.copy()
    d["gate_fail"] = np.select(
        [~ok_sector, ~positive_fcf, ~beaten_down],
        ["financial/fund sector", "negative FCF", "not beaten down"], default="")
    return d


# ---------------------------------------------------------------------------
# 3. Base score
# ---------------------------------------------------------------------------
def pct(s: pd.Series, higher_better: bool = True, fill: float = 0.3) -> pd.Series:
    r = s.rank(pct=True, ascending=higher_better)
    return r.fillna(fill)


def score_base(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["s_cash"] = 0.8 * pct(g["fcf_yield"]) + 0.2 * pct(g["free_cash_flow_margin_ttm"])

    sh = pct(g["sh_yield"])
    uncovered = (g["div_yield"] > 0) & (g["div_coverage"] < 1.2)
    g["s_shareholder"] = np.where(uncovered, sh * 0.6, sh)

    rel_cols = ["fwd_pe_rel", "ev_ebitda_rel", "p_fcf_rel", "p_s_rel", "p_b_rel"]
    rel_w = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    rel_scores = np.column_stack([pct(g[c], higher_better=False, fill=0.4) for c in rel_cols])
    g["s_valuation"] = rel_scores @ rel_w

    near_low = (1 - g["from_52w_low"] / 30).clip(0, 1).fillna(0)
    g["s_capitulation"] = (0.35 * pct(-g["dd_52w"].clip(lower=-75))
                           + 0.35 * pct(-g["perf_5y"].clip(lower=-85), fill=0.4)
                           + 0.30 * near_low)

    nd = g["nd_ebitda"]
    nd_s = np.where(nd.isna(), 0.2, np.where(nd <= 0, 1.0, np.clip(1 - nd / 4.0, 0, 1) * 0.95))
    nd_s = np.where(g["captive_finance"], 0.5, nd_s)  # neutral: can't separate the finance arm
    cr = ((g["current_ratio"] - 0.6) / 0.9).clip(0, 1).fillna(0.4)
    g["s_balance"] = 0.75 * nd_s + 0.25 * cr

    rev = ((g["rev_growth"] + 10) / 15).clip(0, 1).fillna(0.4)
    surprise = np.where(g["eps_surprise"].isna(), 0.5, np.where(g["eps_surprise"] > 0, 1.0, 0.3))
    g["s_business"] = 0.35 * rev + 0.25 * pct(g["op_margin"]) + 0.25 * surprise + 0.15 * pct(g["roe"])

    up = (g["target_upside"] / 50).clip(0, 1)
    enough = g["n_analysts"] >= 3
    g["s_sentiment"] = np.where(enough, 0.6 * up.fillna(0) + 0.4 * g["non_buy_share"].fillna(0), 0.2)

    base = sum(WEIGHTS[k] * g[f"s_{k}"] for k in WEIGHTS)
    penalty = np.where(g["beta"] > 1.5, 4, 0) + np.where((g["nd_ebitda"] > 3.5) & ~g["captive_finance"], 4, 0)
    g["base_score"] = (base - penalty).clip(lower=0)
    return g


# ---------------------------------------------------------------------------
# 4. Enrichment (shortlist only)
# ---------------------------------------------------------------------------
def fetch_insider_buys(days: int = 90) -> pd.DataFrame:
    url = ("http://openinsider.com/screener?s=&o=&pl=&ph=&ll=&lh=&fd={d}&fdr=&td=0&tdr=&fdlyl=&fdlyh="
           "&daysago=&xp=1&vl=25&vh=&ocl=&och=&sic1=-1&sicl=100&sich=9999&grp=0&nfl=&nfh=&nil=&nih="
           "&nol=&noh=&v2l=&v2h=&oc2l=&oc2h=&sortcol=0&cnt=5000&page=1").format(d=days)
    try:
        html = http_get(url, timeout=60).decode("utf-8", "ignore")
        t = pd.read_html(io.StringIO(html), attrs={"class": "tinytable"})[0]
    except Exception as e:  # network or layout change: enrichment degrades gracefully
        print(f"  ! OpenInsider unavailable ({e.__class__.__name__}); insider factor skipped", file=sys.stderr)
        return pd.DataFrame(columns=["ticker", "n_insiders", "insider_value", "insider_titles"])
    t.columns = [str(c).replace("\xa0", " ").strip() for c in t.columns]
    t = t.rename(columns={"Ticker": "ticker", "Insider Name": "insider", "Title": "title", "Value": "value"})
    t["value"] = pd.to_numeric(t["value"].astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce")
    t["ticker"] = t["ticker"].astype(str).str.strip()
    # the author's signal is executives/directors buying with their own cash (Ford's chairman,
    # Udemy's CEO) — drop rows where the buyer is only a 10% holder (funds, foundations)
    t = t[t["title"].astype(str).str.contains(r"CEO|CFO|COO|CTO|Pres|Chair|COB|Dir|VP|Officer|GC|Treas|Sec",
                                              case=False, regex=True)]
    agg = t.groupby("ticker").agg(
        n_insiders=("insider", "nunique"),
        insider_value=("value", "sum"),
        insider_titles=("title", lambda s: ", ".join(sorted(set(map(str, s)))[:4])),
    ).reset_index()
    return agg


def fetch_short_interest(ticker: str) -> dict:
    """FINRA consolidated short interest (all US exchanges, bi-monthly settlements)."""
    start = (dt.date.today() - dt.timedelta(days=110)).isoformat()
    body = {"limit": 12,
            "compareFilters": [{"compareType": "EQUAL", "fieldName": "symbolCode", "fieldValue": ticker}],
            "dateRangeFilters": [{"fieldName": "settlementDate", "startDate": start,
                                  "endDate": dt.date.today().isoformat()}],
            "fields": ["settlementDate", "currentShortPositionQuantity"]}
    req = urllib.request.Request(
        "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json", "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            rows = sorted(json.load(r), key=lambda x: x["settlementDate"], reverse=True)
        vals = [float(x["currentShortPositionQuantity"]) for x in rows]
        if len(vals) >= 2:
            back = vals[min(3, len(vals) - 1)]  # ~6 weeks ago
            return {"si_latest": vals[0], "si_change": (vals[0] / back - 1) * 100 if back else np.nan,
                    "si_date": rows[0]["settlementDate"]}
    except Exception:
        pass
    return {}


def fetch_bars(ticker: str) -> pd.DataFrame | None:
    sym = ticker.replace(".", "-")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=2y&interval=1d"
    try:
        j = json.loads(http_get(url, timeout=15))
        res = j["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        bars = pd.DataFrame({"close": q["close"], "volume": q["volume"]},
                            index=pd.to_datetime(res["timestamp"], unit="s").normalize())
        return bars.dropna()
    except Exception:
        return None


def bar_signals(bars: pd.DataFrame | None, release_ts) -> dict:
    out: dict = {}
    if bars is None or len(bars) < 80:
        return out
    c, v = bars["close"], bars["volume"]
    # 2-day reaction to the last earnings release
    if release_ts and not pd.isna(release_ts):
        day = pd.Timestamp(dt.datetime.fromtimestamp(int(release_ts), dt.timezone.utc).date())
        before, after = c[c.index < day], c[c.index > day]
        if len(before) and len(after):
            out["earn_reaction"] = (after.iloc[0] / before.iloc[-1] - 1) * 100
    # volume accumulation over the last 30 sessions: volume on up days vs down days
    last = bars.tail(31)
    chg = last["close"].diff().iloc[1:]
    vol = last["volume"].iloc[1:]
    down_vol = vol[chg < 0].sum()
    out["updown_vol"] = vol[chg > 0].sum() / down_vol if down_vol else np.nan
    # reclaim of the 50-day average (the author's confirmation trigger for extra tranches)
    sma50 = c.rolling(50).mean()
    above = c > sma50
    out["reclaimed_sma50"] = bool(above.iloc[-1] and (~above.iloc[-21:-1]).any())
    # multi-year decline: count down calendar years among the last 3 (author: "falling 3rd year in a row")
    yearly = c.resample("YE").last().pct_change().dropna()
    out["down_years"] = int((yearly.tail(3) < 0).sum())
    return out


def enrich(g: pd.DataFrame, n: int) -> pd.DataFrame:
    g = g.copy()
    short = g.sort_values("base_score", ascending=False).head(n)
    print(f"  enriching top {len(short)} candidates (insiders, short interest, daily bars)...")
    ins = fetch_insider_buys()
    g = g.merge(ins, how="left", left_on="name", right_on="ticker").drop(columns=["ticker"], errors="ignore")

    def work(row):
        return row["name"], fetch_short_interest(row["name"]), bar_signals(
            fetch_bars(row["name"]), row["earnings_release_date"])

    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(work, [r for _, r in short.iterrows()]))
    extra = pd.DataFrame([{"name": t, **si, **bs} for t, si, bs in results])
    g = g.merge(extra, how="left", on="name")
    g["enriched"] = g["name"].isin(short["name"])
    return g


def score_enrichment(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    for col in ("n_insiders", "insider_value", "si_change", "earn_reaction", "updown_vol"):
        if col not in g:
            g[col] = np.nan
    if "reclaimed_sma50" not in g:
        g["reclaimed_sma50"] = False
    ins_any = g["insider_value"].fillna(0) >= 100_000
    ins_big = (g["insider_value"].fillna(0) >= 1_000_000) | (g["n_insiders"].fillna(0) >= 2)
    g["sig_insider"] = ins_any
    g["sig_short_cover"] = g["si_change"] <= -15
    g["sig_sold_on_beat"] = (g["eps_surprise"] > 0) & (g["earn_reaction"] <= -5)  # CMCSA −6.8%, PYPL −8%
    g["sig_accumulation"] = (g["updown_vol"] >= 1.3) & (g["from_52w_low"] <= 20)
    g["sig_reclaim"] = g["reclaimed_sma50"].fillna(False).astype(bool)
    pts = (3 * ins_any + 3 * (ins_any & ins_big)
           + np.where(g["sig_short_cover"], 3, np.where(g["si_change"] >= 25, -2, 0))
           + 3 * g["sig_sold_on_beat"] + 2 * g["sig_accumulation"] + 1 * g["sig_reclaim"])
    g["enrich_pts"] = np.clip(pts, -2, ENRICH_MAX)
    g["score"] = (g["base_score"] * 0.85 + g["enrich_pts"]).round(1)
    return g


# ---------------------------------------------------------------------------
# 5. Checklist, tiers, timing, thesis bullets
# ---------------------------------------------------------------------------
CHECKS = [
    ("FCF yield ≥ 8%", lambda r: r.fcf_yield >= 8),
    ("Shareholder yield ≥ 6%", lambda r: r.sh_yield >= 6),
    ("≥25% cheaper than sector", lambda r: min(r.fwd_pe_rel, r.ev_ebitda_rel) <= 0.75),
    ("Beaten down (≥30% off high or 5Y ≤ −40%)", lambda r: r.dd_52w <= -30 or r.perf_5y <= -40),
    ("Net debt/EBITDA ≤ 2.5", lambda r: not r.captive_finance and not (r.nd_ebitda > 2.5) and not pd.isna(r.nd_ebitda)),
    ("Business alive (rev ≥ 0 & EPS beat)", lambda r: r.rev_growth >= 0 and r.eps_surprise > 0),
    ("Hold-heavy consensus, target ≥ +25%", lambda r: r.target_upside >= 25 and r.non_buy_share >= 0.5),
    ("Insider open-market buying", lambda r: bool(r.sig_insider)),
    ("Shorts covering (−15%+)", lambda r: bool(r.sig_short_cover)),
    ("Sold off despite beat", lambda r: bool(r.sig_sold_on_beat)),
]


def safe(fn, r):
    try:
        return bool(fn(r))
    except Exception:
        return False


def timing(r) -> str:
    """Author's entry logic: first tranche near the lows, add after a reclaim of the 50-day average."""
    if r.close > r.SMA50 and r.SMA50 < r.SMA200 and r.from_52w_low <= 35:
        return "Turning up"          # confirmation tranche
    if r.RSI < 35 and r["Perf.1M"] < -8:
        return "Still falling"       # ladder in small, don't go all-in
    if r.from_52w_low <= 12:
        return "Near lows"           # first-tranche zone
    return "Mid-range"


def fmt_money(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            return f"${x / div:.1f}{suf}"
    return f"${x:.0f}"


def thesis(r) -> list[str]:
    b = []
    if r.fcf_yield >= 6:
        b.append(f"FCF yield {r.fcf_yield:.1f}% vs 10Y Treasury {TEN_YEAR_YIELD}% and S&P ~4–5%")
    if r.sh_yield >= 4:
        b.append(f"Returns {r.sh_yield:.1f}%/yr to holders (dividend {r.div_yield:.1f}% + buyback {r.buyback_yld:.1f}%)")
    if r.fwd_pe > 0 and r.fwd_pe_rel <= 0.8:
        b.append(f"Fwd P/E {r.fwd_pe:.1f} vs sector median {r.fwd_pe_sector_med:.1f}")
    elif r.ev_ebitda > 0 and r.ev_ebitda_rel <= 0.8:
        b.append(f"EV/EBITDA {r.ev_ebitda:.1f} vs sector median {r.ev_ebitda_sector_med:.1f}")
    if r.dd_52w <= -20:
        b.append(f"{abs(r.dd_52w):.0f}% below 52w high, {r.from_52w_low:.0f}% above 52w low")
    if not pd.isna(r.perf_5y) and r.perf_5y <= -30:
        b.append(f"5-year return {r.perf_5y:.0f}% — long capitulation")
    if r.target_upside >= 20 and r.n_analysts >= 3:
        b.append(f"Analysts: {int(r.recommendation_hold or 0)} Hold / {int(r.recommendation_buy or 0)} Buy / "
                 f"{int(r.recommendation_sell or 0)} Sell, avg target +{r.target_upside:.0f}%")
    if r.sig_sold_on_beat:
        b.append(f"Beat EPS by {r.eps_surprise:.0f}% yet stock moved {r.earn_reaction:.1f}% — capitulation signature")
    if r.sig_insider:
        b.append(f"Insiders bought {fmt_money(r.insider_value)} on the open market (last 90d, {int(r.n_insiders)} buyer(s))")
    if r.sig_short_cover:
        b.append(f"Short interest {r.si_change:.0f}% over ~6 weeks — shorts leaving")
    return b


def risks(r) -> list[str]:
    k = []
    if r.captive_finance:
        k.append(f"Net debt/EBITDA {r.nd_ebitda:.1f}x includes the captive finance arm — check industrial net cash")
    elif r.nd_ebitda > 2.5:
        k.append(f"Leverage: net debt/EBITDA {r.nd_ebitda:.1f}x (author's CMCSA trigger was 2.9x)")
    if r.rev_growth < 0:
        k.append(f"Revenue shrinking {r.rev_growth:.1f}% y/y — value-trap risk")
    if r.div_yield > 0 and r.div_coverage < 1.2:
        k.append(f"Dividend barely covered by FCF ({r.div_coverage:.1f}x)")
    if r.beta > 1.5:
        k.append(f"High beta {r.beta:.2f}")
    if r.fcf_yield > 25:
        k.append("FCF yield looks too high — check for one-offs (the author's 'P/E illusion' warning)")
    if r.eps_surprise < 0:
        k.append(f"Missed EPS last quarter ({r.eps_surprise:.0f}%)")
    if not pd.isna(r.get("si_change")) and r.si_change >= 25:
        k.append(f"Short interest rising {r.si_change:.0f}%")
    return k


def finalize(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    hits = []
    for _, r in g.iterrows():
        hits.append([name for name, fn in CHECKS if safe(fn, r)])
    g["checks_hit"] = hits
    g["n_checks"] = g["checks_hit"].str.len()
    g["tier"] = np.select([g["n_checks"] >= 7, g["n_checks"] >= 5], ["A", "B"], default="C")
    g["timing"] = [timing(r) for _, r in g.iterrows()]
    g["thesis"] = [thesis(r) for _, r in g.iterrows()]
    g["risks"] = [risks(r) for _, r in g.iterrows()]
    return g.sort_values("score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run(min_mcap: float, n_enrich: int, tickers: list[str] | None, out: Path) -> pd.DataFrame:
    print(f"[1/4] universe: US stocks & ADRs, mcap > {fmt_money(min_mcap)}")
    uni = add_metrics(fetch_universe(min_mcap))
    gated = gate(uni)
    pool = gated[gated["gate_fail"] == ""]
    print(f"      {len(uni)} companies, {len(pool)} pass the gates")
    wanted = [t.upper() for t in tickers or []]
    if wanted:  # score requested names even if they fail a gate, so you can see why
        pool = pd.concat([pool, gated[gated["name"].isin(wanted)]]).drop_duplicates("name")
    print("[2/4] base scoring")
    # percentiles are computed against everything that passed the gates
    scored = score_base(pool)
    if wanted:
        scored = scored[scored["name"].isin(wanted)]
        n_enrich = len(scored)
    print("[3/4] enrichment")
    scored = score_enrichment(enrich(scored, n_enrich))
    print("[4/4] checklist, tiers, thesis")
    final = finalize(scored)

    out.mkdir(parents=True, exist_ok=True)
    keep = ["name", "description", "sector", "industry", "close", "market_cap_basic", "score", "base_score",
            "enrich_pts", "tier", "n_checks", "timing", "fcf_yield", "sh_yield", "div_yield", "buyback_yld",
            "fwd_pe", "fwd_pe_sector_med", "ev_ebitda", "ev_ebitda_sector_med", "p_fcf", "nd_ebitda",
            "dd_52w", "from_52w_low", "perf_y", "perf_5y", "rev_growth", "op_margin", "eps_surprise",
            "target_upside", "non_buy_share", "n_analysts", "beta", "RSI", "earn_reaction", "si_change",
            "insider_value", "n_insiders", "updown_vol", "down_years"] + [f"s_{k}" for k in WEIGHTS]
    final[keep].to_csv(out / "arden_scan.csv", index=False, float_format="%.3f")
    final.to_json(out / "arden_scan.json", orient="records", default_handler=str, indent=1)
    return final


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-mcap", type=float, default=2e9, help="minimum market cap in USD (default 2e9)")
    ap.add_argument("--enrich", type=int, default=120, help="how many top candidates to enrich (default 120)")
    ap.add_argument("--tickers", nargs="*", help="score only these tickers (still ranked vs full universe)")
    ap.add_argument("--top", type=int, default=25, help="rows to print")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("output"))
    a = ap.parse_args()

    final = run(a.min_mcap, a.enrich, a.tickers, a.out)
    cols = ["name", "score", "tier", "n_checks", "timing", "fcf_yield", "sh_yield", "fwd_pe", "dd_52w",
            "target_upside", "nd_ebitda"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(final[cols].head(a.top).round(1).to_string(index=False))
    print(f"\nSaved: {a.out / 'arden_scan.csv'}  and  {a.out / 'arden_scan.json'}")


if __name__ == "__main__":
    main()

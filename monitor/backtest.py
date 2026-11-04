"""Portfolio backtest: the trigger rule traded from a $100,000 account, against the market.

Every trigger the rule finds in the last five years of daily bars is followed as a trade exactly as
the ledger follows one: bought at the trigger-day close, sold at the first close at its
volatility-sized target or stop, or after 40 sessions, one trade per name at a time. That is done
for the watchlist and for the learner's whole universe (the watchlist plus the scan's names).

The page simulates the account from these trades in the browser, so the period, position size,
number of positions, costs and which trades are taken can be changed there, and it sets the
account beside market, style, international and sector indexes.

Trigger quality is walk-forward here: every quarter the score is refitted on the triggers whose
outcome was known by then and rates only the triggers after it, so "strong only" never uses a
trade's own outcome. Readiness is left out: its weights were fitted on the whole period.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .model import fit_logit, predict, readiness, sigmoid
from .signals import barrier_labels, features, move_size, trigger_events

TEMPLATE = Path(__file__).with_name("backtest_template.html")
WARMUP = 252     # sessions of history before the first trade: the features need a year
REFIT = 63       # walk-forward quality is refitted every quarter
MIN_TRAIN = 200  # resolved triggers it needs before it rates any
STALE = 10       # sessions a name's bars may lag the calendar before an open trade counts as no data
BENCH = [  # (ticker, name, group): price indexes and ETFs, closes only (no dividends, like the trades)
    ("^GSPC", "S&P 500", "Market"), ("^DJI", "Dow Jones", "Market"), ("^IXIC", "Nasdaq Composite", "Market"),
    ("^RUT", "Russell 2000 (small caps)", "Market"), ("IWD", "Russell 1000 Value", "Market"),
    ("EFA", "Developed markets ex-US", "International"), ("EEM", "Emerging markets", "International"),
    ("XLK", "Technology", "Sector"), ("XLC", "Communication services", "Sector"), ("IYZ", "Telecom", "Sector"),
    ("XLY", "Consumer discretionary", "Sector"), ("XLP", "Consumer staples", "Sector"),
    ("XLV", "Health care", "Sector"), ("XLF", "Financials", "Sector"), ("XLI", "Industrials", "Sector"),
    ("XLB", "Materials", "Sector"), ("GDX", "Gold miners", "Sector"), ("XLE", "Energy", "Sector"),
    ("XLU", "Utilities", "Sector"), ("XLRE", "Real estate", "Sector"),
]
# the scan's sectors (TradingView's classification) to the index a stock is held against
SECTOR_ETF = {
    "Technology Services": "XLK", "Electronic Technology": "XLK", "Communications": "IYZ",
    "Consumer Services": "XLY", "Retail Trade": "XLY", "Consumer Durables": "XLY", "Consumer Non-Durables": "XLP",
    "Health Technology": "XLV", "Health Services": "XLV", "Finance": "XLF",
    "Producer Manufacturing": "XLI", "Industrial Services": "XLI", "Commercial Services": "XLI",
    "Transportation": "XLI", "Distribution Services": "XLI", "Process Industries": "XLB",
    "Non-Energy Minerals": "XLB", "Energy Minerals": "XLE", "Utilities": "XLU",
}
SECTOR_FIX = {"INTC": "Electronic Technology", "UDMY": "Consumer Services", "XOP": "Energy Minerals"}  # author calls


def sector_of(t: str, wl: dict, scan: pd.DataFrame | None) -> tuple[str, str]:
    """(sector, the index it is compared with); gold and silver miners go against the miners' index."""
    row = scan[scan["name"] == t] if scan is not None else None
    sector = (wl.get(t) or {}).get("sector") or (row["sector"].iloc[0] if row is not None and len(row) else None)
    sector = sector if isinstance(sector, str) else SECTOR_FIX.get(t, "Other")
    industry = row["industry"].iloc[0] if row is not None and len(row) else ""
    etf = "GDX" if industry == "Precious Metals" else SECTOR_ETF.get(sector, "^GSPC")
    return sector, etf


def follow_all(t: str, b: pd.DataFrame, spy: pd.Series, rule: dict, label: dict, dates: pd.DatetimeIndex):
    """Every trade the ledger would have taken in this name, plus every trigger's features."""
    f = features(b, spy)
    c = f["close"]
    h = label["horizon"]
    size = move_size(c, h)
    up, down = (size * label["up"]).to_numpy(), (size * label["down"]).to_numpy()
    o = barrier_labels(c, h, up, down)
    events = np.flatnonzero(trigger_events(f, rule).to_numpy())
    pos = np.minimum(dates.searchsorted(c.index), len(dates) - 1)  # each bar's place on the S&P 500 calendar
    trades, rows, free_from = [], [], 0
    for j in events:
        y = o["y"].iloc[j]
        k = len(c) - 1 - j if pd.isna(y) else int(o["bars"].iloc[j])
        i0, i1 = int(pos[j]), int(pos[j + k])
        # every trigger teaches the quality score, as in the learner; its outcome is known at its exit
        rows.append({"t": t, "i0": i0, "known": None if pd.isna(y) else i1, "y": None if pd.isna(y) else float(y),
                     **{name: float(v) for name, v in f.iloc[j][C.FEATURES].items()}})
        if j < free_from:  # one trade per name at a time
            continue
        free_from = len(c) if pd.isna(y) else j + k  # an open trade keeps the name busy
        status = "open" if pd.isna(y) else "hit" if y == 1 else "stop" if o["ret"].iloc[j] <= -down[j] + 1e-9 else "time"
        if status == "open":
            if len(dates) - 1 - i1 > STALE:
                status = "nodata"  # its bars stop for good: closed at the last price, as the ledger does
            else:
                i1 = len(dates) - 1  # a day or two behind the calendar: held at its last close
        entry = float(c.iloc[j])
        path = c.iloc[j:j + k + 1].reindex(dates[i0:i1 + 1]).ffill().fillna(entry) / entry - 1
        trades.append({"t": t, "i0": i0, "i1": i1, "px": round(entry, 4), "o": status,
                       "p": [int(round(x * 1e4)) for x in path.to_numpy()]})
    return trades, rows


def walk_forward_quality(rows: pd.DataFrame, start: int, end: int, lam: float) -> dict:
    """{(ticker, entry position): quality}, each scored by a fit on outcomes known before its quarter."""
    out = {}
    zero = np.zeros(len(C.FEATURES))
    for r in range(start, end + 1, REFIT):
        train = rows[rows["known"].notna() & (rows["known"] < r)]
        rate = rows[(rows["i0"] >= r) & (rows["i0"] < r + REFIT)]
        if len(train) < MIN_TRAIN or not len(rate):
            continue
        mean = train[C.FEATURES].mean()
        std = train[C.FEATURES].std().replace(0, 1).fillna(1)
        X = np.nan_to_num(((train[C.FEATURES] - mean) / std).to_numpy())
        w, b = fit_logit(X, train["y"].to_numpy(), zero, lam)
        m = {"features": C.FEATURES, "coef": w.tolist(), "intercept": float(b), "mean": mean.tolist(), "std": std.tolist(),
             "quantiles": np.quantile(sigmoid(X @ w + b), np.linspace(0, 1, 101)).tolist()}
        q = readiness(m, predict(m, rate[C.FEATURES].to_numpy()))
        out.update({(t, int(i)): int(v) for t, i, v in zip(rate["t"], rate["i0"], q)})
    return out


def build(bars: dict[str, pd.DataFrame], spy: pd.Series, eng: dict, wl: dict, scan: pd.DataFrame | None,
          watch: list[str], bench: dict[str, pd.DataFrame]) -> dict:
    """The page's data: the calendar, the index closes on it, and every trade with its daily path."""
    from .report import rule_text
    dates = spy.index
    start = WARMUP
    if len(dates) <= start + 20:
        raise ValueError(f"only {len(dates)} sessions of S&P 500 bars; the backtest needs {start + 20}")
    trades, rows = [], []
    for t, b in sorted(bars.items()):
        if len(b) < 260:
            continue
        tr, rw = follow_all(t, b, spy, eng["trigger"], eng["label"], dates)
        trades += tr
        rows += rw
    q = eng.get("quality") or {}
    rated = walk_forward_quality(pd.DataFrame(rows), start, len(dates) - 1, q.get("lam", 3.0)) if rows else {}
    names = {t: (wl.get(t) or {}).get("name") for t in bars}
    if scan is not None:
        names.update({t: d for t, d in zip(scan["name"], scan["description"]) if not names.get(t)})
    watch = set(watch)
    out = []
    for tr in trades:
        if tr["i0"] < start:
            continue
        sector, etf = sector_of(tr["t"], wl, scan)
        out.append({**tr, "i0": tr["i0"] - start, "i1": tr["i1"] - start, "n": names.get(tr["t"]) or tr["t"],
                    "s": sector, "e": etf, "w": int(tr["t"] in watch), "q": rated.get((tr["t"], tr["i0"]))})
    cal = dates[start:]
    idx = []
    for t, name, group in BENCH:
        b = bench.get(t)
        if b is None or not len(b):
            continue
        close = b["close"].reindex(cal).ffill().bfill()
        idx.append({"id": t, "name": name, "group": group, "c": [float(f"{x:.6g}") for x in close.to_numpy()]})
    if not any(x["id"] == "^GSPC" for x in idx):
        raise ValueError("no S&P 500 index bars; the page compares everything with it")
    rated_from = min((tr["i0"] for tr in out if tr["q"] is not None), default=None)
    return {"asof": str(cal[-1].date()), "built": dt.date.today().isoformat(), "version": eng.get("version"),
            "dates": [str(d.date()) for d in cal], "bench": idx, "trades": out,
            "rule": rule_text(eng["trigger"]), "label": eng["label"],
            "min_quality": q.get("min_quality") or 67, "rated_from": rated_from,
            "universe": {"wl": len(watch & set(bars)), "all": len(bars)}}


def page(data: dict) -> str:
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("NaN", "null")
    return TEMPLATE.read_text().replace("/*__DATA__*/null", blob)


def write(data: dict) -> Path:
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    out = C.REPORTS / "backtest.html"
    out.write_text(page(data))
    return out


def run(today: dt.date | None = None) -> dict:
    """Fetch what the backtest needs (from the cache where it can) and build its data."""
    from . import watchlist as W
    from .prices import fetch_many
    today = today or dt.date.today()
    wl, scan = W.load(), W.load_scan()
    eng = json.loads(C.ENGINE.read_text())
    watch = W.active(wl)
    universe = list(dict.fromkeys(watch + (list(scan["name"]) if scan is not None else [])))
    tickers = [t for t, _, _ in BENCH]
    bars = fetch_many(universe + ["SPY"] + tickers, "5y", today=today)
    spy = bars["SPY"]["close"]
    bench = {t: bars.pop(t) for t in tickers if t in bars}
    bars.pop("SPY")
    return build(bars, spy, eng, wl, scan, watch, bench)

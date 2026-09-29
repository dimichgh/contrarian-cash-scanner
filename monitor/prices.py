"""Daily OHLCV bars from the Yahoo Finance chart API."""
from __future__ import annotations

import datetime as dt
import json
import time
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo

import pandas as pd

from arden_scanner import http_get


def fetch_ohlcv(ticker: str, range_: str = "2y") -> pd.DataFrame | None:
    sym = ticker.replace(".", "-")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range={range_}&interval=1d"
    try:
        res = json.loads(http_get(url, timeout=20))["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        df = pd.DataFrame({k: q.get(k) for k in ("open", "high", "low", "close", "volume")},
                          index=pd.to_datetime(res["timestamp"], unit="s", utc=True), dtype=float)
    except Exception:
        return None
    meta = res["meta"]
    tz = meta.get("exchangeTimezoneName") or "America/New_York"
    df.index = df.index.tz_convert(tz).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")]
    end = meta.get("currentTradingPeriod", {}).get("regular", {}).get("end")
    if end and len(df):
        session_day = pd.Timestamp(dt.datetime.fromtimestamp(end, ZoneInfo(tz)).date())
        if df.index[-1] == session_day:
            if time.time() < end:
                # session still open: signals are on daily closes only
                df = df.iloc[:-1]
            elif pd.isna(df["close"].iloc[-1]) and meta.get("regularMarketTime", 0) >= end:
                # right after the close the daily bar can lag; the quote already has the final price
                df.loc[df.index[-1], "close"] = meta["regularMarketPrice"]
    df = df.dropna(subset=["close"])
    for col in ("high", "low", "open"):
        df[col] = df[col].fillna(df["close"])
    df["volume"] = df["volume"].fillna(0)
    return df if len(df) else None


def fetch_many(tickers: list[str], range_: str = "2y", workers: int = 8) -> dict[str, pd.DataFrame]:
    tickers = list(dict.fromkeys(tickers))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        got = dict(zip(tickers, ex.map(lambda t: fetch_ohlcv(t, range_), tickers)))
    return {t: b for t, b in got.items() if b is not None}

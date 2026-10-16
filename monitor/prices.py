"""Daily OHLCV bars from the Yahoo Finance chart API, kept in a local cache.

Every download is saved under state/bars/ (committed with the rest of the state), so a run only
asks Yahoo for the last few weeks of each name and merges them in. Past bars only change on a
split; the overlap between the cache and the fresh bars catches that and triggers a full download.
With MONITOR_OFFLINE=1 nothing is downloaded and the cache alone is used.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo

import pandas as pd

from arden_scanner import http_get

from . import config as C

RANGE_DAYS = {"1mo": 31, "3mo": 92, "6mo": 183, "1y": 366, "2y": 731, "5y": 1827}
FULL_EVERY = 60      # days between full re-downloads of a name, whatever the overlap says
FORGET_AFTER = 120   # days a cached name may go unrequested before its file is deleted
INDEX = C.BARS / "_index.json"


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


def _path(t: str):
    return C.BARS / f"{t}.csv"


def load_cached(t: str) -> pd.DataFrame | None:
    p = _path(t)
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col="date", parse_dates=["date"])
    return df if len(df) else None


def save_cached(t: str, df: pd.DataFrame) -> None:
    C.BARS.mkdir(parents=True, exist_ok=True)
    out = df[["open", "high", "low", "close", "volume"]].copy()
    out["volume"] = out["volume"].round().astype("int64")
    out.index.name = "date"
    out.to_csv(_path(t), float_format="%.6g", date_format="%Y-%m-%d")


def _consistent(cached: pd.DataFrame, fresh: pd.DataFrame) -> bool:
    """Do the fresh bars agree with the cache where they overlap? A split rescales all past bars."""
    # the cache's last bar may hold a provisional close taken from the quote; leave it out
    both = cached["close"].iloc[:-1].to_frame("old").join(fresh["close"].rename("new"), how="inner")
    if len(both) < 2:
        return False
    return bool(((both["new"] / both["old"] - 1).abs() <= 0.005).all())


def get_bars(t: str, range_: str, today: dt.date, entry: dict | None, offline: bool = False):
    """(bars trimmed to `range_`, new index entry or None, how they were obtained)."""
    cached = load_cached(t)
    start = pd.Timestamp(today - dt.timedelta(days=RANGE_DAYS[range_]))
    trim = lambda df: df[df.index >= start] if df is not None else None
    if offline:
        return trim(cached), None, "cache" if cached is not None else "missing"
    covers = (cached is not None and entry is not None
              and RANGE_DAYS[entry["range"]] >= RANGE_DAYS[range_]
              and (today - dt.date.fromisoformat(entry["full"])).days < FULL_EVERY)
    if covers:
        behind = (today - cached.index[-1].date()).days
        if behind <= 80:
            fresh = fetch_ohlcv(t, "1mo" if behind <= 25 else "6mo")
            if fresh is not None and _consistent(cached, fresh):
                merged = pd.concat([cached[cached.index < fresh.index[0]], fresh])
                save_cached(t, merged)
                return trim(merged), None, "update"
    # first sight, a split, a stale cache or a longer range: download it all
    want = range_ if entry is None or RANGE_DAYS[range_] >= RANGE_DAYS[entry["range"]] else entry["range"]
    full = fetch_ohlcv(t, want)
    if full is not None:
        save_cached(t, full)
        return trim(full), {"range": want, "full": today.isoformat()}, "download"
    return trim(cached), None, "cache" if cached is not None else "missing"  # Yahoo down: keep what we have


def fetch_many(tickers: list[str], range_: str = "2y", workers: int = 8, today: dt.date | None = None,
               stats: dict | None = None) -> dict[str, pd.DataFrame]:
    today = today or dt.date.today()
    offline = os.environ.get("MONITOR_OFFLINE") == "1"
    tickers = list(dict.fromkeys(tickers))
    index = json.loads(INDEX.read_text()) if INDEX.exists() else {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        got = list(ex.map(lambda t: get_bars(t, range_, today, index.get(t), offline), tickers))
    out, counts, stale = {}, {}, []
    for t, (b, entry, how) in zip(tickers, got):
        counts[how] = counts.get(how, 0) + 1
        if how == "cache" and not offline:
            stale.append(t)  # Yahoo failed for it: older bars only, fine to learn from, not to signal on
        if entry is not None:
            index[t] = entry
        if t in index:
            index[t]["used"] = today.isoformat()
        if b is not None and len(b):
            out[t] = b
    if not offline:
        for t, e in list(index.items()):  # names nobody has asked for in months
            if (today - dt.date.fromisoformat(e.get("used", e["full"]))).days > FORGET_AFTER:
                _path(t).unlink(missing_ok=True)
                del index[t]
        C.BARS.mkdir(parents=True, exist_ok=True)
        INDEX.write_text(json.dumps(dict(sorted(index.items())), indent=0) + "\n")
    if stats is not None:
        for k, v in counts.items():
            stats[k] = stats.get(k, 0) + v
        stats["stale"] = stats.get("stale", []) + stale
        stats["offline"] = offline
    return out

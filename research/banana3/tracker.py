"""Score a trader's posted alerts against Yahoo price data, losers included.

Built for the one-month audit of @Banana3Stocks (see README.md next to this file), but it scores
any alert list in the alerts.csv format.

  python research/banana3/tracker.py date 2045900452613742677       # X post id -> UTC time
  python research/banana3/tracker.py bars NVDA TSLA                  # bank daily + 5-minute bars
  python research/banana3/tracker.py bars --alerts research/banana3/data/alerts.csv
  python research/banana3/tracker.py score research/banana3/prior_calls.csv --baseline

An alert is filled at the first price after its post (or when price trades through its entry
level), then followed bar by bar until its stop, its targets, or its horizon. Five-minute bars are
used where they exist; Yahoo keeps them for 60 days only, so `bars` should run every session while
alerts are live. Older stretches fall back to daily bars, where a stop and a target touched on the
same bar can't be ordered: those count as stopped and are flagged `ambiguous`.

Levels are quoted in the prices of the day they were posted; later splits are undone before
comparing. Option alerts whose levels are premiums are scored on a Black-Scholes repricing of the
contract from the underlying path, with the volatility implied by the entry premium: an estimate
that ignores volatility crush and skew, flagged as such.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
BARS = DATA / "bars"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

X_EPOCH_MS = 1288834974657
DISCORD_EPOCH_MS = 1420070400000
DEFAULT_HORIZON = 90      # calendar days an alert without an expiry is followed
ENTRY_WINDOW = 5          # calendar days an alert with an entry level waits for its fill
BASELINE_DAYS = 500       # sessions before the post the random-entry baseline samples
RATE = 0.04               # risk-free rate for the option model
STALE_HOURS = 6           # cached bars younger than this aren't re-downloaded

COLUMNS = ["alert_id", "post_id", "posted_at", "source", "ticker", "side", "level_basis", "entry",
           "stop", "t1", "t2", "t3", "opt_type", "strike", "expiry", "horizon_days", "bucket",
           "advertised", "notes"]


# ---------------------------------------------------------------------------
# post ids

def snowflake_time(post_id: int | str, discord: bool = False) -> pd.Timestamp:
    ms = (int(post_id) >> 22) + (DISCORD_EPOCH_MS if discord else X_EPOCH_MS)
    return pd.Timestamp(ms, unit="ms", tz="UTC")


def posted_at(a: dict) -> pd.Timestamp:
    if a.get("posted_at"):
        ts = pd.Timestamp(a["posted_at"])
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    if str(a.get("post_id", "")).isdigit():
        return snowflake_time(a["post_id"], discord=a.get("source") == "discord")
    raise ValueError(f"{a['alert_id']}: needs posted_at or a numeric post_id")


# ---------------------------------------------------------------------------
# bars

def _get_json(url: str) -> dict:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
                return json.load(r)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    return {}


def _download(sym: str, interval: str) -> tuple[pd.DataFrame, dict]:
    range_ = "60d" if interval == "5m" else "10y"
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(sym)}"
           f"?range={range_}&interval={interval}&events=split")
    res = _get_json(url)["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({k: q.get(k) for k in ("open", "high", "low", "close")},
                      index=pd.to_datetime(res["timestamp"], unit="s", utc=True), dtype=float).dropna(subset=["close"])
    for col in ("open", "high", "low"):
        df[col] = df[col].fillna(df["close"])
    tz = res["meta"].get("exchangeTimezoneName") or "America/New_York"
    if interval == "1d":
        df.index = df.index.tz_convert(tz).tz_localize(None).normalize()
        df = df[~df.index.duplicated(keep="last")]
    splits = [{"date": dt.datetime.fromtimestamp(s["date"], dt.timezone.utc).date().isoformat(),
               "num": s["numerator"], "den": s["denominator"]}
              for s in res.get("events", {}).get("splits", {}).values()]
    return df, {"tz": tz, "splits": sorted(splits, key=lambda s: s["date"])}


def load_bars(sym: str, refresh: bool = True) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict]:
    """Daily bars, 5-minute bars (or None) and {tz, splits} for a Yahoo symbol, merged into the cache."""
    BARS.mkdir(parents=True, exist_ok=True)
    key = sym.replace("^", "_")
    meta_p = BARS / f"{key}.json"
    out, meta = {}, json.loads(meta_p.read_text()) if meta_p.exists() else {}
    for interval in ("1d", "5m"):
        p = BARS / f"{key}_{interval}.csv"
        old = pd.read_csv(p, index_col=0, parse_dates=[0]) if p.exists() else None
        if old is not None and interval == "5m":
            old.index = pd.to_datetime(old.index, utc=True)
        fresh = old is not None and time.time() - p.stat().st_mtime < STALE_HOURS * 3600
        if refresh and not fresh:
            try:
                new, new_meta = _download(sym, interval)
                # daily history is rewritten whole (splits revise it); intraday is appended, Yahoo drops it after 60 days
                old = new if old is None or interval == "1d" else pd.concat([old, new])
                old = old[~old.index.duplicated(keep="last")].sort_index()
                old.to_csv(p)
                if interval == "1d":            # the 5-minute request only sees 60 days of splits
                    meta = new_meta
                    meta_p.write_text(json.dumps(meta))
            except Exception as e:
                print(f"  {sym} {interval}: download failed ({e}); using cache", file=sys.stderr)
        out[interval] = old if old is not None and len(old) else None
    return out["1d"], out["5m"], meta


def _close_time(day: pd.Timestamp, tz: str) -> pd.Timestamp:
    if tz in ("UTC", "Etc/UTC", "GMT"):          # crypto: the daily bar is the UTC day
        return pd.Timestamp(day).tz_localize("UTC") + pd.Timedelta(days=1)
    return (pd.Timestamp(day) + pd.Timedelta(hours=16)).tz_localize(tz).tz_convert("UTC")


def price_path(daily: pd.DataFrame, intraday: pd.DataFrame | None, tz: str, t0: pd.Timestamp):
    """(price at t0, bars after t0 with a UTC end time `t`, resolution label)."""
    daily = daily.copy()
    daily["t"] = [_close_time(d, tz) for d in daily.index]
    span = pd.Timedelta(days=1) if tz in ("UTC", "Etc/UTC", "GMT") else pd.Timedelta(hours=6.5)
    if intraday is not None and len(intraday) and intraday.index[0] <= t0 <= intraday.index[-1] + pd.Timedelta(minutes=5):
        bars = intraday[intraday.index >= t0].copy()
        bars["t"] = bars.index + pd.Timedelta(minutes=5)
        before = intraday[intraday.index < t0]
        px0 = bars["open"].iloc[0] if len(bars) else before["close"].iloc[-1]
        last_day = bars["t"].iloc[-1].tz_convert(tz).tz_localize(None).normalize() if len(bars) else None
        tail = daily[daily.index > last_day] if last_day is not None else daily.iloc[0:0]
        cols = ["open", "high", "low", "close", "t"]
        path = pd.concat([bars[cols].reset_index(drop=True), tail[cols].reset_index(drop=True)], ignore_index=True)
        return px0, path, "5m" if not len(tail) else "5m+1d"
    after = daily[daily["t"] >= t0]
    if not len(after):
        return None, daily.iloc[0:0], "1d"
    first = after.iloc[0]
    if t0 <= first["t"] - span:                   # posted before that session opened: its open is tradable
        return first["open"], after, "1d"
    return first["close"], after.iloc[1:], "1d"   # posted mid-session: the close is the first known price


def split_factor(meta: dict, t0: pd.Timestamp) -> float:
    """Multiplier from prices quoted at t0 to today's split-adjusted prices."""
    f = 1.0
    for s in meta.get("splits", []):
        if pd.Timestamp(s["date"]).tz_localize("UTC") > t0:
            f *= s["den"] / s["num"]
    return f


# ---------------------------------------------------------------------------
# option model

def bs_price(s: float, k: float, years: float, vol: float, call: bool) -> float:
    if years <= 0 or vol <= 0:
        return max(0.0, s - k) if call else max(0.0, k - s)
    sq = vol * math.sqrt(years)
    d1 = (math.log(s / k) + (RATE + vol * vol / 2) * years) / sq
    d2 = d1 - sq
    n = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    if call:
        return s * n(d1) - k * math.exp(-RATE * years) * n(d2)
    return k * math.exp(-RATE * years) * n(-d2) - s * n(-d1)


def implied_vol(premium: float, s: float, k: float, years: float, call: bool) -> float | None:
    lo, hi = 0.01, 6.0
    if not bs_price(s, k, years, lo, call) <= premium <= bs_price(s, k, years, hi, call):
        return None
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if bs_price(s, k, years, mid, call) < premium else (lo, mid)
    return (lo + hi) / 2


def option_path(path: pd.DataFrame, k: float, expiry: pd.Timestamp, vol: float, call: bool) -> pd.DataFrame:
    yrs = ((expiry - path["t"]).dt.total_seconds() / (365.25 * 86400)).clip(lower=0)
    price = lambda col: [bs_price(s, k, y, vol, call) for s, y in zip(path[col], yrs)]
    hi, lo = (price("high"), price("low")) if call else (price("low"), price("high"))
    return pd.DataFrame({"open": price("open"), "high": hi, "low": lo, "close": price("close"), "t": path["t"]},
                        index=path.index)


# ---------------------------------------------------------------------------
# scoring

def _num(x) -> float | None:
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def follow(path: pd.DataFrame, fill: float, long: bool, stop: float | None, targets: list[float]) -> dict:
    """Walk the bars after a fill. Two exit plans are scored on the same path:
    all-out at the first target, and equal parts at each target with the stop moved to the fill
    price once the first target is hit. A bar that touches the stop and a target counts as stopped."""
    sign = 1 if long else -1
    ret = lambda px: sign * (px / fill - 1)
    reached = lambda bar, lvl: (bar.high >= lvl) if long else (bar.low <= lvl)
    breached = lambda bar, lvl: (bar.low <= lvl) if long else (bar.high >= lvl)
    stop_px = lambda bar, lvl: min(lvl, bar.open) if long else max(lvl, bar.open)   # a gap through the stop fills at the open

    out = {"hits": [], "t1_time": None, "stop_time": None, "ambiguous": False, "mae": 0.0}
    t1_ret = None                                   # plan A: all out at T1 or the stop
    left, held, realized, stop_b = list(targets), 1.0, 0.0, stop   # plan B: equal parts per target
    size = 1 / len(targets) if targets else 1.0
    for bar in path.itertuples():
        if t1_ret is None:
            out["mae"] = min(out["mae"], ret(bar.low if long else bar.high))
            if stop is not None and breached(bar, stop):
                t1_ret, out["stop_time"] = ret(stop_px(bar, stop)), bar.t
                out["ambiguous"] = bool(targets) and reached(bar, targets[0])
            elif targets and reached(bar, targets[0]):
                t1_ret, out["t1_time"] = ret(targets[0]), bar.t
        if held > 1e-9:
            if stop_b is not None and breached(bar, stop_b):
                realized, held = realized + held * ret(stop_px(bar, stop_b)), 0.0
            else:
                for lvl in [x for x in left if reached(bar, x)]:
                    realized, held = realized + size * ret(lvl), held - size
                    left.remove(lvl)
                    out["hits"].append((lvl, bar.t))
                    if lvl == targets[0]:
                        stop_b = fill
        if held <= 1e-9 and (t1_ret is not None or not targets):
            break
    last = path["close"].iloc[-1] if len(path) else fill
    out["ret_t1"] = t1_ret if t1_ret is not None else ret(last)
    out["ret_parts"] = realized + max(held, 0.0) * ret(last)
    out["finished"] = held <= 1e-9
    return out


def score_alert(a: dict, daily: pd.DataFrame, intraday: pd.DataFrame | None, meta: dict,
                now: pd.Timestamp | None = None, baseline: bool = False) -> dict:
    now = now or pd.Timestamp.now(tz="UTC")
    t0 = posted_at(a)
    tz = meta.get("tz", "America/New_York")
    f = split_factor(meta, t0)
    long = str(a.get("side") or "long").lower() != "short"
    lv = lambda k: None if _num(a.get(k)) is None else _num(a[k]) * f
    is_opt = str(a.get("level_basis") or "underlying").lower() == "option"
    expiry = (_close_time(pd.Timestamp(a["expiry"]), tz) if a.get("expiry") else None)
    horizon = expiry if expiry is not None else t0 + pd.Timedelta(days=_num(a.get("horizon_days")) or DEFAULT_HORIZON)

    px0, path, res = price_path(daily, intraday, tz, t0)
    row = {"alert_id": a["alert_id"], "ticker": a["ticker"], "posted_at": t0.strftime("%Y-%m-%d %H:%M"),
           "source": a.get("source", ""), "bucket": a.get("bucket", ""), "advertised": a.get("advertised", ""),
           "level_basis": "option" if is_opt else "underlying", "resolution": res,
           "px_at_post": None if px0 is None else round(px0 / f, 4)}
    if px0 is None:
        return row | {"status": "waiting" if t0 > _close_time(daily.index[-1], tz) - pd.Timedelta(days=1) else "no_data"}
    path = path[path["t"] <= horizon]
    over = now >= horizon

    if is_opt:
        k, call = _num(a.get("strike")), str(a.get("opt_type") or "C").upper().startswith("C")
        if k is None or expiry is None:
            return row | {"status": "bad_option", "note": "option alert needs strike and expiry"}
        k *= f
        if lv("entry") is None:
            # premium unknown: all that can be said is where the stock finished against the strike
            if not over:
                return row | {"status": "open", "note": "no entry premium; intrinsic known at expiry"}
            s_exp = path["close"].iloc[-1] if len(path) else px0
            intrinsic = max(0.0, (s_exp - k) if call else (k - s_exp)) / f
            return row | {"status": "itm_at_expiry" if intrinsic > 0 else "otm_at_expiry",
                          "underlying_at_expiry": round(s_exp / f, 2), "intrinsic_at_expiry": round(intrinsic, 2),
                          "win": intrinsic > 0, "note": "no entry premium; scored on expiry value only"}
        yrs0 = (expiry - t0).total_seconds() / (365.25 * 86400)
        vol = implied_vol(lv("entry"), px0, k, yrs0, call)
        if vol is None:
            return row | {"status": "bad_option", "note": "entry premium outside model range"}
        path = option_path(path, k, expiry, vol, call)
        fill, fill_t, long = lv("entry"), t0, True   # a bought call or put: long the premium
        row["note"] = f"modelled premium, iv {vol:.0%}"
    else:
        entry = lv("entry")
        if entry is None:
            fill, fill_t = px0, t0
        else:
            window = path[path["t"] <= t0 + pd.Timedelta(days=ENTRY_WINDOW)]
            need_up = entry >= px0
            touched = window[window["high"] >= entry] if need_up else window[window["low"] <= entry]
            if not len(touched):
                return row | {"status": "no_fill" if over or len(window) < len(path) else "waiting"}
            bar = touched.iloc[0]
            fill = max(entry, bar["open"]) if need_up else min(entry, bar["open"])
            fill_t = bar["t"]
            path = path[path["t"] > fill_t]

    targets = [x for x in (lv("t1"), lv("t2"), lv("t3")) if x is not None]
    stop = lv("stop")
    sign = 1 if long else -1
    if (targets and sign * (targets[0] - fill) <= 0) or (stop is not None and sign * (fill - stop) <= 0):
        # a target behind the fill (or a stop beyond it) is almost always a transcription slip
        return row | {"status": "bad_levels", "fill": round(fill / f, 4),
                      "note": "target or stop on the wrong side of the fill; check the alert"}
    fol = follow(path, fill, long, stop, targets)
    hits = fol["hits"]
    if fol["stop_time"] is not None:
        status = "stopped"
    elif hits:
        status = f"t{len(hits)}" + ("" if fol["finished"] or over else "+open")
    else:
        status = "expired" if over else "open"
    risk = abs(fill - stop) / fill if stop else None
    days = lambda t: None if t is None else round((t - fill_t).total_seconds() / 86400, 1)
    # a win is T1 before the stop; a target still unreached at the horizon is a loss; no targets, no verdict
    win = (True if fol["t1_time"] is not None else False if fol["stop_time"] is not None
           else False if targets and over else None)
    row |= {
        "status": status, "fill": round(fill / f, 4), "win": win,
        "ambiguous": fol["ambiguous"], "targets_hit": len(hits),
        "days_to_t1": days(fol["t1_time"]), "days_to_stop": days(fol["stop_time"]),
        "ret_t1": round(fol["ret_t1"], 4), "ret_parts": round(fol["ret_parts"], 4),
        "r_t1": round(fol["ret_t1"] / risk, 2) if risk else None, "mae": round(fol["mae"], 4),
    }
    if baseline and not is_opt and targets:
        row["baseline_win"] = random_entry_rate(daily, t0, tz, long, targets[0] / fill - 1,
                                                None if stop is None else stop / fill - 1, horizon - t0)
    return row


def random_entry_rate(daily: pd.DataFrame, t0: pd.Timestamp, tz: str, long: bool, tgt: float,
                      stp: float | None, span: pd.Timedelta) -> float | None:
    """Share of the BASELINE_DAYS sessions before t0 on which an entry at the close, with the same
    distances to target and stop, would have reached the target first within the same span."""
    closes = daily.index[[(_close_time(d, tz) < t0) for d in daily.index]]
    starts = closes[-BASELINE_DAYS:]
    wins = n = 0
    for d in starts:
        fwd = daily[(daily.index > d) & (daily.index <= d + span)]
        if fwd.empty or fwd.index[-1] < d + span - pd.Timedelta(days=5):
            continue          # not enough future left before the post to judge this start
        if fwd.index[-1] >= closes[-1]:
            continue          # would peek past the post
        c0 = daily.at[d, "close"]
        sign = 1 if long else -1
        fav = (fwd["high"] / c0 - 1) if long else (fwd["low"] / c0 - 1)
        adv = (fwd["low"] / c0 - 1) if long else (fwd["high"] / c0 - 1)
        hit_t = (sign * fav >= sign * tgt)
        hit_s = (sign * adv <= sign * stp) if stp is not None else pd.Series(False, index=fwd.index)
        first_t = hit_t.idxmax() if hit_t.any() else None
        first_s = hit_s.idxmax() if hit_s.any() else None
        n += 1
        wins += first_t is not None and (first_s is None or first_t < first_s)
    return round(wins / n, 3) if n else None


# ---------------------------------------------------------------------------
# summary

def summarize(out: pd.DataFrame) -> str:
    def col(g: pd.DataFrame, name: str) -> pd.Series:
        return pd.to_numeric(g[name], errors="coerce").dropna() if name in g else pd.Series(dtype=float)

    def block(g: pd.DataFrame) -> dict:
        judged = g[g["win"].isin([True, False])] if "win" in g else g.iloc[0:0]
        b, r1, rp, rr = col(g, "baseline_win"), col(g, "ret_t1"), col(g, "ret_parts"), col(g, "r_t1")
        return {"alerts": len(g), "judged": len(judged),
                "win rate": f"{judged['win'].astype(bool).mean():.0%}" if len(judged) else "-",
                "baseline": f"{b.mean():.0%}" if len(b) else "-",
                "avg ret (T1 plan)": f"{r1.mean():+.1%}" if len(r1) else "-",
                "avg ret (parts)": f"{rp.mean():+.1%}" if len(rp) else "-",
                "avg R": f"{rr.mean():+.2f}" if len(rr) else "-",
                "ambiguous": int(g["ambiguous"].eq(True).sum()) if "ambiguous" in g else 0}

    lines = []
    for by in (None, "bucket", "source", "level_basis", "advertised"):
        if by is not None and by not in out:
            continue
        groups = [("all", out)] if by is None else [(f"{by}={k}", g) for k, g in out.groupby(out[by].fillna("").astype(str)) if k != ""]
        if by is not None and len(groups) < 2:
            continue
        rows = [{"group": name} | block(g) for name, g in groups]
        cols = list(rows[0])
        lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows] + [""]
    return "\n".join(lines)


def read_alerts(path: Path) -> list[dict]:
    # lines starting with # are notes; a # inside a field (a hashtag) is data
    text = "\n".join(x for x in path.read_text().splitlines() if not x.startswith("#"))
    df = pd.read_csv(io.StringIO(text), dtype=str).fillna("")
    missing = [c for c in ("alert_id", "ticker") if c not in df]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}; expected {COLUMNS}")
    dupes = df["alert_id"][df["alert_id"].duplicated()].tolist()
    if dupes:
        raise SystemExit(f"{path}: duplicate alert_id {dupes}")
    return df.to_dict("records")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("date", help="post id(s) -> UTC time")
    d.add_argument("ids", nargs="+")
    d.add_argument("--discord", action="store_true", help="Discord message ids instead of X post ids")
    b = sub.add_parser("bars", help="download and cache bars (run every session while alerts are live)")
    b.add_argument("tickers", nargs="*")
    b.add_argument("--alerts", type=Path, help="cache every ticker in this alerts file")
    s = sub.add_parser("score", help="score an alerts file")
    s.add_argument("alerts", type=Path)
    s.add_argument("--out", type=Path, default=DATA, help="where outcomes.csv and summary.md go")
    s.add_argument("--baseline", action="store_true", help="add the random-entry baseline (slower)")
    s.add_argument("--offline", action="store_true", help="use cached bars only")
    args = ap.parse_args(argv)

    if args.cmd == "date":
        for i in args.ids:
            print(i, snowflake_time(i, args.discord).strftime("%Y-%m-%d %H:%M:%S UTC"))
        return 0
    if args.cmd == "bars":
        tickers = list(args.tickers) + ([a["ticker"] for a in read_alerts(args.alerts)] if args.alerts else [])
        for t in sorted(set(tickers)):
            daily, intra, _ = load_bars(t)
            span = "none" if intra is None else f"{intra.index[0]:%Y-%m-%d} .. {intra.index[-1]:%Y-%m-%d %H:%M}"
            print(f"{t}: {0 if daily is None else len(daily)} daily bars, 5-minute {span}")
        return 0

    alerts = read_alerts(args.alerts)
    cache = {t: load_bars(t, refresh=not args.offline) for t in sorted({a["ticker"] for a in alerts})}
    rows = []
    for a in alerts:
        daily, intra, meta = cache[a["ticker"]]
        if daily is None:
            rows.append({"alert_id": a["alert_id"], "ticker": a["ticker"], "status": "no_data"})
            continue
        try:
            rows.append(score_alert(a, daily, intra, meta, baseline=args.baseline))
        except Exception as e:
            rows.append({"alert_id": a["alert_id"], "ticker": a["ticker"], "status": "error", "note": str(e)})
    out = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out / "outcomes.csv", index=False)
    summary = summarize(out)
    (args.out / "summary.md").write_text(f"# Alert scores ({args.alerts.name}, {dt.date.today()})\n\n{summary}\n")
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.max_rows", 500):
        cols = [c for c in ("alert_id", "posted_at", "px_at_post", "status", "win", "days_to_t1", "ret_t1",
                            "mae", "baseline_win", "resolution", "note") if c in out]
        print(out[cols].to_string(index=False))
    print("\n" + summary)
    print(f"wrote {args.out / 'outcomes.csv'} and {args.out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

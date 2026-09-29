"""Per-ticker daily features, trigger events, stages and outcome labels.

Everything here is computed from daily bars alone, so any past day can be rebuilt: the learner
replays history with it and the daily run uses the same code for today.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURES

# display order, most actionable first
STAGES = ["TRIGGERED", "SETUP", "RUNNING", "EXTENDED", "BASING", "FAILED", "FALLING"]
STAGE_TEXT = {
    "TRIGGERED": "Move starting: fresh close through the trigger level",
    "SETUP": "Coiled just under the trigger level, lows holding",
    "RUNNING": "Move under way: holding above the level, 50-day average rising",
    "EXTENDED": "Running but stretched: over 20% above the 50-day average or RSI over 75",
    "BASING": "Lows holding, 50-day average flattening, not near the level yet",
    "FAILED": "Broke out, then closed back under the level",
    "FALLING": "Still making new lows",
}
PIVOT_TEXT = {
    "sma50": "50-day average",
    "hh20": "20-day closing high",
    "hh40": "40-day closing high",
    "sma50_hh20": "50-day average and 20-day closing high",
}


def features(b: pd.DataFrame, spy_close: pd.Series) -> pd.DataFrame:
    c, h, l, v = b["close"], b["high"], b["low"], b["volume"]
    f = pd.DataFrame(index=b.index)
    f["close"] = c
    f["volume"] = v
    f["sma50"] = sma50 = c.rolling(50).mean()
    f["sma200"] = sma200 = c.rolling(200, min_periods=150).mean()
    f["hh20"] = c.shift(1).rolling(20).max()
    f["hh40"] = c.shift(1).rolling(40).max()
    f["vol_avg50"] = v.shift(1).rolling(50).mean()
    f["vol_ratio"] = v / f["vol_avg50"].replace(0, np.nan)

    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    # sessions since the last new 120-day closing low: how long the lows have held
    new_low = c <= c.rolling(120, min_periods=60).min()
    f["low_age_days"] = c.groupby(new_low.cumsum()).cumcount().clip(upper=120).astype(float)
    f.loc[new_low.cumsum() == 0, "low_age_days"] = np.nan

    spy = spy_close.reindex(c.index).ffill()
    rs = c / spy
    rs_hi, rs_lo = rs.rolling(120).max(), rs.rolling(120).min()
    chg = c.diff()
    upv = v.where(chg > 0, 0).rolling(30).sum()
    dnv = v.where(chg < 0, 0).rolling(30).sum()

    f["f_sma50_gap"] = (c / sma50 - 1).clip(-0.4, 0.4)
    f["f_sma200_gap"] = (c / sma200 - 1).clip(-0.6, 0.6)
    f["f_sma50_slope"] = (sma50 / sma50.shift(10) - 1).clip(-0.2, 0.2)
    f["f_hh40_gap"] = (c / f["hh40"] - 1).clip(-0.5, 0.2)
    f["f_low_age"] = f["low_age_days"] / 120
    f["f_higher_low"] = (c.rolling(20).min() / c.shift(20).rolling(40).min() - 1).clip(-0.3, 0.3)
    f["f_rs_slope"] = (rs / rs.shift(20) - 1).clip(-0.4, 0.4)
    f["f_rs_pos"] = ((rs - rs_lo) / (rs_hi - rs_lo).replace(0, np.nan)).clip(0, 1)
    f["f_rsi"] = f["rsi"] / 100
    f["f_squeeze"] = (tr.rolling(14).mean() / tr.rolling(100).mean().replace(0, np.nan)).clip(0.3, 3)
    f["f_updown_vol"] = np.log((upv + 1) / (dnv + 1)).clip(-1.5, 1.5)
    f["f_vol_surge"] = np.log((v.rolling(5).mean() + 1) / (f["vol_avg50"] + 1)).clip(-1.5, 1.5)
    f["f_ret20"] = (c / c.shift(20) - 1).clip(-0.5, 0.5)
    f["f_dd_52w"] = (c / c.rolling(252, min_periods=120).max() - 1).clip(-0.9, 0)
    f["f_mkt"] = (spy / spy.rolling(200).mean() - 1).clip(-0.3, 0.3)
    f["f_vol20"] = (np.log(c).diff().rolling(20).std() * np.sqrt(252)).clip(0, 1.5)
    assert all(k in f for k in FEATURES)
    return f


def pivot(f: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "sma50_hh20":
        return pd.concat([f["sma50"], f["hh20"]], axis=1).max(axis=1, skipna=False)
    return f[mode]


def trigger_events(f: pd.DataFrame, p: dict) -> pd.Series:
    """Days the close crosses up through the trigger level after a real stretch below it."""
    piv = pivot(f, p["pivot"])
    above = f["close"] > piv
    cross = above & (above.shift(1) == False)  # noqa: E712 — NaN-safe: the first bar never crosses
    below_days = (~above).astype(float).shift(1).rolling(20).sum()
    ok = cross & piv.notna() & (below_days >= p["min_below"]) & (f["low_age_days"] >= p["min_low_age"])
    if p["vol_mult"] > 0:
        ok &= f["vol_ratio"] >= p["vol_mult"]
    return ok.fillna(False).astype(bool)


def stages(f: pd.DataFrame, p: dict) -> pd.DataFrame:
    piv = pivot(f, p["pivot"])
    trig = trigger_events(f, p)
    c, sma50, slope, low_age = f["close"], f["sma50"], f["f_sma50_slope"], f["low_age_days"]
    pos = pd.Series(np.arange(len(f), dtype=float), index=f.index)
    age = pos - pos.where(trig).ffill()          # sessions since the last trigger
    trig_level = piv.where(trig).ffill()           # the level it broke through
    floor = trig_level * (1 - p["fail_tol"])
    held = c.groupby(trig.cumsum()).cummin() >= floor  # never closed back under since the trigger
    gap = c / piv - 1

    running = ((age <= 60) & held & (c > sma50)) | ((c > sma50) & (slope > 0.005) & (low_age >= 30))
    conds = [
        (age < p["confirm"]) & (c >= floor),
        running & ((c > sma50 * 1.2) | (f["rsi"] > 75)),
        running,
        (age <= 20) & (c < floor),
        (gap >= -p["setup_gap"]) & (low_age >= 15),
        (low_age >= 15) & (slope > -0.02),
    ]
    st = np.select([x.fillna(False).to_numpy(bool) for x in conds],
                   ["TRIGGERED", "EXTENDED", "RUNNING", "FAILED", "SETUP", "BASING"], "FALLING")
    out = pd.DataFrame({"stage": st, "pivot": piv, "gap": gap, "trig_age": age, "trig_level": trig_level,
                        "trigger": trig}, index=f.index)
    out.loc[sma50.isna() | piv.isna(), "stage"] = "n/a"
    return out


def move_size(close: pd.Series, horizon: int) -> pd.Series:
    """The stock's normal swing over `horizon` sessions, from the prior 60 sessions' daily volatility."""
    return (np.log(close).diff().rolling(60, min_periods=40).std().clip(0.008, 0.05) * np.sqrt(horizon))


def outcomes(close: pd.Series, label: dict) -> pd.DataFrame:
    """Triple-barrier outcomes with the target and stop sized to each day's volatility."""
    size = move_size(close, label["horizon"])
    out = barrier_labels(close, label["horizon"], (size * label["up"]).to_numpy(), (size * label["down"]).to_numpy())
    return out.assign(target=size * label["up"], stop=size * label["down"])


def barrier_labels(close: pd.Series, horizon: int, up, down) -> pd.DataFrame:
    """Triple-barrier outcome of buying each day's close.

    y = 1 if the close gains `up` before it loses `down` within `horizon` sessions, else 0. `up` and
    `down` are fractions, either one number or one per row (NaN rows stay unlabelled).
    ret is the return at whichever barrier came first (or at the horizon). Rows whose outcome
    is not known yet are NaN; a barrier hit inside the observed window resolves a row early.
    """
    c = close.to_numpy(float)
    n = len(c)
    if n == 0:
        return pd.DataFrame(columns=["y", "ret", "bars"], index=close.index)
    pad = np.concatenate([c, np.full(horizon, np.nan)])
    win = np.lib.stride_tricks.sliding_window_view(pad[1:], horizon)[:n]
    up = np.broadcast_to(np.asarray(up, float), (n,))[:, None]
    down = np.broadcast_to(np.asarray(down, float), (n,))[:, None]
    with np.errstate(invalid="ignore"):
        fwd = win / c[:, None] - 1
        hit_up, hit_dn = fwd >= up, fwd <= -down
    first_up = np.where(hit_up.any(1), hit_up.argmax(1), horizon)
    first_dn = np.where(hit_dn.any(1), hit_dn.argmax(1), horizon)
    first = np.minimum(first_up, first_dn)
    resolved = ((first < horizon) | ~np.isnan(win).any(1)) & ~np.isnan(up[:, 0]) & ~np.isnan(down[:, 0])
    y = np.where(resolved, (first_up < first_dn).astype(float), np.nan)
    ret = np.where(resolved, fwd[np.arange(n), np.minimum(first, horizon - 1)], np.nan)
    bars = np.where(resolved, np.minimum(first + 1, horizon), np.nan)
    return pd.DataFrame({"y": y, "ret": ret, "bars": bars}, index=close.index)

"""Offline tests for the breakout monitor (synthetic bars, no network)."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from monitor import config as C
from monitor import watchlist as W
from monitor.model import auc, fit_logit, predict, readiness, sigmoid
from monitor.signals import barrier_labels, features, stages, trigger_events


def bars(close, volume=None):
    idx = pd.bdate_range("2022-01-03", periods=len(close))
    c = pd.Series(close, index=idx, dtype=float)
    v = pd.Series(volume if volume is not None else 1e6, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": v})


def base_then_breakout(n_up=40):
    fall = np.linspace(100, 60, 200)                              # long decline
    base = 60 + np.sin(np.linspace(0, 4 * np.pi, 60))              # lows set, then held
    shelf = np.full(20, 59.5)                                      # drifts just under the 50-day average
    up = np.linspace(62, 75, n_up)                                 # breakout on doubled volume
    close = np.concatenate([fall, base, shelf, up])
    vol = np.full(len(close), 1e6)
    vol[280:] = 2e6
    return bars(close, vol)


@pytest.fixture
def spy():
    return pd.Series(np.linspace(400, 450, 400), index=pd.bdate_range("2022-01-03", periods=400))


def test_features_complete(spy):
    f = features(base_then_breakout(), spy)
    assert set(C.FEATURES) <= set(f.columns)
    assert f[C.FEATURES].iloc[-1].notna().all()


def test_breakout_is_detected_and_runs(spy):
    b = base_then_breakout()
    f = features(b, spy)
    p = dict(C.TRIGGER, vol_mult=1.2, min_below=5, min_low_age=10)
    trig = trigger_events(f, p)
    first = trig[trig].index[0]
    assert first == b.index[280]  # the breakout day, not a whipsaw inside the base
    st = stages(f, p)
    assert st.loc[first, "stage"] == "TRIGGERED"
    assert st["stage"].iloc[-1] in ("RUNNING", "EXTENDED")
    assert st["stage"].iloc[150] == "FALLING"


def test_failed_breakout(spy):
    b = base_then_breakout(n_up=6)
    c = np.concatenate([b["close"].to_numpy(), np.linspace(60, 52, 10)])
    v = np.concatenate([b["volume"].to_numpy(), np.full(10, 1e6)])
    f = features(bars(c, v), spy)
    st = stages(f, dict(C.TRIGGER, vol_mult=1.2, min_below=5, min_low_age=10))
    assert st["stage"].iloc[-1] == "FAILED"


def test_barrier_labels():
    c = pd.Series([100, 101, 104, 111, 90, 95], dtype=float)
    lab = barrier_labels(c, horizon=3, up=0.10, down=0.06)
    assert lab["y"].iloc[0] == 1 and lab["bars"].iloc[0] == 3  # 111 is +11% on day 3
    assert lab["y"].iloc[3] == 0 and lab["ret"].iloc[3] < -0.06  # 111 -> 90 stops out
    assert np.isnan(lab["y"].iloc[5])  # nothing after the last bar: unresolved
    flat = barrier_labels(pd.Series([100.0] * 10), horizon=3, up=0.10, down=0.06)
    assert flat["y"].iloc[0] == 0 and flat["ret"].iloc[0] == 0  # timeout


def test_logit_learns_signal_and_prior_shrinks():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 3))
    y = (rng.random(5000) < sigmoid(1.5 * X[:, 0] - 0.2)).astype(float)
    w, b = fit_logit(X, y, np.zeros(3), lam=1.0)
    assert w[0] == pytest.approx(1.5, abs=0.15) and abs(w[1]) < 0.1
    w_strong, _ = fit_logit(X, y, np.array([0.0, 0.5, 0.0]), lam=1e7)
    assert w_strong[1] == pytest.approx(0.5, abs=0.01)
    assert auc(y, X[:, 0]) > 0.75 and auc(y, -X[:, 0]) < 0.25


def test_readiness_percentiles():
    m = {"features": ["a"], "coef": [1.0], "intercept": 0.0, "mean": [0.0], "std": [1.0],
         "quantiles": list(np.linspace(0.1, 0.9, 101))}
    p = predict(m, np.array([[0.0]]))
    assert p[0] == pytest.approx(0.5)
    assert readiness(m, [0.05, 0.5, 0.95]).tolist() == [0, 50, 100]


def test_watchlist_merge_adds_refreshes_and_drops(monkeypatch):
    def scan(names, tiers, scores):
        n = len(names)
        return pd.DataFrame({"name": names, "description": names, "sector": "S", "industry": "I", "exchange": "NYSE",
                             "close": 10.0, "score": scores, "tier": tiers, "n_checks": 7, "timing": "Near lows",
                             "fcf_yield": 9.0, "sh_yield": 6.0, "fwd_pe": 8.0, "dd_52w": -40.0, "target_upside": 30.0,
                             "nd_ebitda": 1.0, "why": "a | b", "risk": ""})
    monkeypatch.setattr(C, "ADMIT_TOP_N", 0)
    wl, d = {}, dt.date(2026, 9, 1)
    added, _ = W.merge_scan(wl, scan(["AAA", "BBB", "CCC"], ["A", "B", "C"], [70, 65, 80]), d)
    assert added == ["AAA", "BBB"] and "CCC" not in wl
    for i in range(C.DROP_AFTER_MISSES):
        _, dropped = W.merge_scan(wl, scan(["AAA"], ["A"], [70]), d + dt.timedelta(days=7 * (i + 1)))
    assert dropped == ["BBB"] and wl["AAA"]["status"] == "active"
    added, _ = W.merge_scan(wl, scan(["BBB"], ["A"], [70]), d + dt.timedelta(days=60))
    assert added == ["BBB"] and wl["BBB"]["status"] == "active"


def test_since_trigger_tracks_the_trade(spy):
    from monitor.run import since_trigger
    b = base_then_breakout()
    f = features(b, spy)
    p = dict(C.TRIGGER, vol_mult=1.2, min_below=5, min_low_age=10, min_ready=0)
    st = stages(f, p)
    m = {"features": C.FEATURES, "coef": [0.0] * len(C.FEATURES), "intercept": 0.0,
         "mean": [0.0] * len(C.FEATURES), "std": [1.0] * len(C.FEATURES), "quantiles": list(np.linspace(0, 1, 101))}
    t = since_trigger(f, st, spy, {"label": C.LABEL, "model": m, "trigger": p})
    assert t["date"] == str(b.index[280].date()) and t["sessions"] == 39
    assert t["entry"] == 62.0 and t["change"] == pytest.approx((75 / 62 - 1) * 100, abs=0.1)
    assert t["worst"] == 0.0 and t["best"] == t["change"]
    assert t["status"] == "hit target" and t["stop_px"] < 62 < t["target_px"]

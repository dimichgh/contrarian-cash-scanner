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


def test_run_without_recent_trigger_uses_todays_level(spy):
    # an old breakout, a sell-off to new lows, then a quiet recovery that runs with no new trigger
    b = base_then_breakout()
    c = np.concatenate([b["close"].to_numpy(), np.linspace(75, 50, 60), np.linspace(50, 51, 40), np.linspace(51, 58, 30)])
    v = np.concatenate([b["volume"].to_numpy(), np.full(130, 1e6)])
    spy = pd.Series(np.linspace(400, 450, len(c)), index=pd.bdate_range("2022-01-03", periods=len(c)))
    f = features(bars(c, v), spy)
    p = dict(C.TRIGGER, vol_mult=1.2, min_below=5, min_low_age=10)
    st = stages(f, p)
    assert not st["trigger"].iloc[320:].any()
    last = st.iloc[-1]
    assert last["stage"] in ("RUNNING", "EXTENDED")
    assert last["level"] == last["pivot"] and last["trig_level"] != last["pivot"]
    # while the original breakout ran, it was measured against the level it broke through
    assert st.loc[f.index[300], "stage"] in ("RUNNING", "EXTENDED")
    assert st.loc[f.index[300], "level"] == st.loc[f.index[280], "pivot"]


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


def _eng(p, timing=None):
    m = {"features": C.FEATURES, "coef": [0.0] * len(C.FEATURES), "intercept": 0.0,
         "mean": [0.0] * len(C.FEATURES), "std": [1.0] * len(C.FEATURES), "quantiles": list(np.linspace(0, 1, 101))}
    return {"label": C.LABEL, "model": m, "trigger": p, "version": 1,
            "metrics": {"trigger": {"timing": timing}} if timing else {}}


def test_trade_is_followed_and_frozen_at_the_exit(spy):
    from monitor import trades as T
    b = base_then_breakout()
    f = features(b, spy)
    p = dict(C.TRIGGER, vol_mult=1.2, min_below=5, min_low_age=10, min_ready=0)
    st = stages(f, p)
    eng = _eng(p)
    trig = b.index[280]
    t = T.follow("X", f, st, spy, trig, eng)
    assert t["date"] == str(trig.date()) and t["entry"] == 62.0 and t["status"] == "hit target"
    k = t["resolved_in"]
    exit_px = float(b["close"].iloc[280 + k])
    assert t["exit_date"] == str(b.index[280 + k].date()) and t["exit_px"] == round(exit_px, 2)
    assert exit_px >= t["target_px"] and float(b["close"].iloc[280 + k - 1]) < t["target_px"]  # first close through it
    assert t["change"] == pytest.approx((exit_px / 62 - 1) * 100, abs=0.1) and t["best"] == t["change"] and t["worst"] == 0.0
    assert t["stop_px"] < 62 < t["target_px"]

    # the stock then gives everything back: the closed trade keeps its exit result
    c = np.concatenate([b["close"].to_numpy(), np.linspace(75, 50, 30)])
    v = np.concatenate([b["volume"].to_numpy(), np.full(30, 1e6)])
    spy2 = pd.Series(np.linspace(400, 450, len(c)), index=pd.bdate_range("2022-01-03", periods=len(c)))
    f2 = features(bars(c, v), spy2)
    st2 = stages(f2, p)
    ledger, seen = T.update([], {"X": (f, st)}, spy, eng, {})
    assert [r["date"] for r in ledger] == [str(trig.date())] and seen == {"X": str(b.index[-1].date())}
    ledger2, _ = T.update(ledger, {"X": (f2, st2)}, spy2, eng, seen)
    assert ledger2 == ledger


def test_open_trade_keeps_its_plan_and_new_triggers_only_after_seen(spy):
    from monitor import trades as T
    def slow(n):  # triggers on day 280, then drifts up too slowly to reach its target
        c = base_then_breakout(n_up=1)["close"].to_numpy()[:280]
        c = np.concatenate([c, np.linspace(62, 62 + 0.2 * (n - 1), n)])
        v = np.full(len(c), 1e6)
        v[280:] = 2e6
        return bars(c, v)
    b = slow(6)
    f = features(b, spy)
    p = dict(C.TRIGGER, vol_mult=1.2, min_below=5, min_low_age=10, min_ready=0)
    st = stages(f, p)
    eng = _eng(p, timing={"p25": 11, "median": 18, "p75": 26})
    ledger, seen = T.update([], {"X": (f, st)}, spy, eng, {})
    (t,) = ledger
    assert t["status"] == "open" and t["sessions"] == 5 and t["exit_date"] is None
    w = T.window(t, eng)
    trig = pd.Timestamp(t["date"])
    assert w["from"] == str((trig + pd.offsets.BDay(11)).date()) and w["phase"] == "early"
    # a later run re-follows it with the target and stop it opened with
    b2 = slow(10)
    f2 = features(b2, spy)
    st2 = stages(f2, dict(p, pivot="hh40"))  # even if the trigger rule changed meanwhile
    ledger2, _ = T.update([dict(t, up=0.5)], {"X": (f2, st2)}, spy, eng, seen)
    (t2,) = ledger2
    assert t2["up"] == 0.5 and t2["target_px"] == round(62 * 1.5, 2) and t2["sessions"] == 9
    assert t2["level"] == t["level"]  # and the level it broke through
    assert len(ledger2) == 1  # the new rule's past triggers are not backfilled for a name already seen


def test_one_open_trade_per_name(spy):
    from monitor import trades as T
    b = base_then_breakout()
    f = features(b, spy)
    p = dict(C.TRIGGER, vol_mult=1.2, min_below=5, min_low_age=10, min_ready=0)
    st = stages(f, p)
    st["trigger"] = False
    st.iloc[[280, 283, 315], st.columns.get_loc("trigger")] = True  # 283: while the 280 trade is open
    ledger, _ = T.update([], {"X": (f, st)}, spy, _eng(p), {})
    first = next(r for r in ledger if r["date"] == str(b.index[280].date()))
    assert first["exit_date"] < str(b.index[315].date())
    assert sorted(r["date"] for r in ledger) == [str(b.index[280].date()), str(b.index[315].date())]


def test_target_hit_shows_as_target_not_running(spy):
    from monitor import trades as T
    from monitor.run import attach_trade
    b = base_then_breakout()
    f = features(b, spy)
    p = dict(C.TRIGGER, vol_mult=1.2, min_below=5, min_low_age=10, min_ready=0)
    st = stages(f, p)
    eng = _eng(p)
    ledger, _ = T.update([], {"X": (f, st)}, spy, eng, {})
    row = {"stage": st["stage"].iloc[-1], "prev_stage": st["stage"].iloc[-2], "date": str(b.index[-1].date())}
    assert row["stage"] in ("RUNNING", "EXTENDED")
    attach_trade(row, ledger[0], eng)
    assert row["stage"] == "TARGET" and row["prev_stage"] == "TARGET"  # it got there days ago: no new alert
    row = {"stage": "BASING", "prev_stage": "BASING", "date": row["date"]}
    attach_trade(row, ledger[0], eng)
    assert row["stage"] == "BASING"  # a trend that has since rolled over says so


def test_price_cache_updates_detects_splits_and_works_offline(monkeypatch, tmp_path):
    from monitor import prices as P
    monkeypatch.setattr(C, "BARS", tmp_path)
    monkeypatch.setattr(P, "INDEX", tmp_path / "_index.json")
    idx = pd.bdate_range("2024-06-03", "2026-03-31")
    full = pd.DataFrame({"open": 10.0, "high": 10.5, "low": 9.5, "close": np.linspace(10, 20, len(idx)), "volume": 1e6},
                        index=idx)
    calls = []

    def fake(t, range_="2y"):
        calls.append(range_)
        return feed[range_].copy() if range_ in feed else None
    monkeypatch.setattr(P, "fetch_ohlcv", fake)
    today = dt.date(2026, 3, 31)
    feed = {"2y": full.iloc[:-5]}
    got = P.fetch_many(["X"], "2y", today=today)
    assert calls == ["2y"] and len(got["X"]) == len(full) - 5
    # next run: only the last month is downloaded and merged in
    feed = {"1mo": full.iloc[-22:]}
    got = P.fetch_many(["X"], "2y", today=today)
    assert calls[-1] == "1mo" and got["X"]["close"].iloc[-1] == full["close"].iloc[-1]
    assert len(got["X"]) == len(full[full.index >= pd.Timestamp(today - dt.timedelta(days=731))])
    # a 2:1 split rescales the overlap: download everything again
    split = full.copy()
    split[["open", "high", "low", "close"]] /= 2
    feed = {"1mo": split.iloc[-22:], "2y": split}
    got = P.fetch_many(["X"], "2y", today=today)
    assert calls[-2:] == ["1mo", "2y"] and got["X"]["close"].iloc[0] == pytest.approx(full["close"].iloc[0] / 2, rel=1e-5)
    # Yahoo down: cached bars come back, flagged stale so they are not signalled on
    feed = {}
    stats = {}
    got = P.fetch_many(["X"], "2y", today=today, stats=stats)
    assert "X" in got and stats["stale"] == ["X"]
    # offline: no download at all
    n = len(calls)
    monkeypatch.setenv("MONITOR_OFFLINE", "1")
    stats = {}
    got = P.fetch_many(["X"], "2y", today=today, stats=stats)
    assert len(calls) == n and "X" in got and stats["stale"] == [] and stats["offline"]


def test_same_day_rescans_count_one_miss(monkeypatch):
    monkeypatch.setattr(C, "ADMIT_TOP_N", 0)
    def scan(names, tiers):
        return pd.DataFrame({"name": names, "description": names, "sector": "S", "industry": "I", "exchange": "NYSE",
                             "close": 10.0, "score": 70.0, "tier": tiers, "n_checks": 7, "timing": "Near lows",
                             "fcf_yield": 9.0, "sh_yield": 6.0, "fwd_pe": 8.0, "dd_52w": -40.0, "target_upside": 30.0,
                             "nd_ebitda": 1.0, "why": "", "risk": ""})
    wl, d = {}, dt.date(2026, 10, 4)
    W.merge_scan(wl, scan(["AAA", "BBB"], ["A", "A"]), d)
    for _ in range(4):  # four rescans in one day: BBB misses once, not four times
        _, dropped = W.merge_scan(wl, scan(["AAA"], ["A"]), d + dt.timedelta(days=1))
    assert dropped == [] and wl["BBB"]["misses"] == 1 and wl["BBB"]["status"] == "active"

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


P = dict(C.TRIGGER, vol_mult=1.2, min_below=5, min_low_age=10, min_ready=0)


def _eng(p=P, timing=None):
    m = {"features": C.FEATURES, "coef": [0.0] * len(C.FEATURES), "intercept": 0.0,
         "mean": [0.0] * len(C.FEATURES), "std": [1.0] * len(C.FEATURES), "quantiles": list(np.linspace(0, 1, 101))}
    return {"label": C.LABEL, "model": m, "trigger": p, "version": 1,
            "metrics": {"trigger": {"timing": timing}} if timing else {}}


def _spy_for(b):
    return pd.Series(np.linspace(400, 450, len(b)), index=b.index)


def _stages_with_trades(b, p=P):
    """Every day's final stage and level, the way the daily run reads them."""
    from monitor import trades as T
    spy = _spy_for(b)
    f = features(b, spy)
    st = stages(f, p)
    keep, T.LOOKBACK = T.LOOKBACK, 10_000  # follow every trigger in the series, however old
    try:
        ledger, _ = T.update([], {"X": (f, st)}, spy, _eng(p), {})
    finally:
        T.LOOKBACK = keep
    out = []
    for i in range(len(st)):
        day = str(st.index[i].date())
        out.append(T.stage_on(f, st, i, T.latest(ledger, "X", "1900-01-01", day), p))
    return pd.DataFrame(out, columns=["stage", "level"], index=st.index), ledger, f


def test_breakout_is_detected_and_runs(spy):
    b = base_then_breakout()
    f = features(b, spy)
    trig = trigger_events(f, P)
    first = trig[trig].index[0]
    assert first == b.index[280]  # the breakout day, not a whipsaw inside the base
    s, ledger, _ = _stages_with_trades(b)
    assert s.loc[first, "stage"] == "TRIGGERED"
    assert s["stage"].iloc[-1] == "TARGET" and ledger[0]["status"] == "hit target"  # the move reached its target
    assert s["stage"].iloc[150] == "FALLING"
    assert set(s["stage"].iloc[281:280 + ledger[0]["resolved_in"]]) <= {"TRIGGERED", "RUNNING", "EXTENDED"}


def test_failed_breakout(spy):
    b = base_then_breakout(n_up=1)
    c = np.concatenate([b["close"].to_numpy(), [61.5, 60.5, 59.0, 57.0, 55.0, 54.0]])
    v = np.concatenate([b["volume"].to_numpy(), np.full(6, 1e6)])
    s, ledger, _ = _stages_with_trades(bars(c, v))
    assert ledger[-1]["status"] == "stopped out" and s["stage"].iloc[-1] == "FAILED"
    assert c[-1] < s["level"].iloc[-1]


def test_run_without_recent_trigger_uses_todays_level(spy):
    # an old breakout, a sell-off to new lows, then a quiet recovery that runs with no new trigger
    b = base_then_breakout()
    c = np.concatenate([b["close"].to_numpy(), np.linspace(75, 50, 60), np.linspace(50, 51, 40), np.linspace(51, 58, 30)])
    v = np.concatenate([b["volume"].to_numpy(), np.full(130, 1e6)])
    b2 = bars(c, v)
    s, ledger, f = _stages_with_trades(b2)
    st = stages(f, P)
    assert not st["trigger"].iloc[320:].any() and len(ledger) == 1
    assert s["stage"].iloc[-1] == "ABOVE"  # above its level but no open trade: no entry signal
    assert s["level"].iloc[-1] == st["pivot"].iloc[-1] != ledger[0]["level"]
    # while the original trade was open, it was measured against the level it broke through
    assert s.loc[f.index[285], "stage"] in ("RUNNING", "EXTENDED") and s.loc[f.index[285], "level"] == ledger[0]["level"]


def test_stage_meanings_match_their_text(spy):
    # trigger at 62 over a ~59.8 level, then closes slipping back under it until the stop
    b = base_then_breakout(n_up=1)
    c = np.concatenate([b["close"].to_numpy(), [61.0, 59.7, 58.6, 57.9, 56.0, 55.0, 56.5, 58.0, 60.5, 61.5]])
    v = np.concatenate([b["volume"].to_numpy(), np.full(10, 1e6)])
    s, ledger, f = _stages_with_trades(bars(c, v))
    (t,) = ledger
    tail = s.iloc[-11:]
    lvl = t["level"]
    for (d, r), close in zip(tail.iterrows(), c[-11:]):
        day = str(d.date())
        open_ = t["exit_date"] is None or day < t["exit_date"]
        if r["stage"] in ("TRIGGERED", "RUNNING", "EXTENDED"):
            assert open_ and close >= lvl and r["level"] == lvl
        if r["stage"] == "RETEST":
            assert open_ and t["stop_px"] < close < lvl
        if r["stage"] == "FAILED":
            assert t["status"] == "stopped out" and day >= t["exit_date"] and close < lvl
    assert tail["stage"].iloc[0] == "TRIGGERED" and "FAILED" in set(tail["stage"])
    assert list(tail["stage"].iloc[-2:]) == ["ABOVE", "ABOVE"]  # back over the level, trade over: no live signal
    assert not stages(f, P)["trigger"].iloc[-10:].any()


def test_setup_is_always_under_the_level_and_falling_needs_new_lows(spy):
    st = stages(features(base_then_breakout(), spy), P)
    ok = st["stage"] != "n/a"
    assert (st.loc[ok & (st["stage"] == "SETUP"), "gap"] < 0).all()
    assert (st.loc[ok & (st["stage"] == "ABOVE"), "gap"] > 0).all()
    assert (st.loc[ok & (st["stage"] == "FALLING"), "low_age"] < 15).all()
    # a stock that never made a 120-day low after its first bars is not "making new lows"
    up = bars(np.linspace(50, 80, 300))
    st2 = stages(features(up, spy.iloc[:300]), P)
    assert "FALLING" not in set(st2["stage"].iloc[150:])


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


def test_author_calls_stay_and_removes_stick(monkeypatch):
    def scan(names):
        return pd.DataFrame({"name": names, "description": names, "sector": "S", "industry": "I", "exchange": "NYSE",
                             "close": 10.0, "score": 70, "tier": "A", "n_checks": 7, "timing": "Near lows",
                             "fcf_yield": 9.0, "sh_yield": 6.0, "fwd_pe": 8.0, "dd_52w": -40.0, "target_upside": 30.0,
                             "nd_ebitda": 1.0, "why": "a | b", "risk": ""})
    monkeypatch.setattr(C, "ADMIT_TOP_N", 0)
    wl, d = {}, dt.date(2026, 9, 1)
    W.merge_scan(wl, scan(["PYPL", "ZZZ"]), d)       # the scan finds an author call first
    W.ensure_author_calls(wl, d)
    assert wl["PYPL"]["source"] == "author" and wl["PYPL"]["pinned"]
    for i in range(C.DROP_AFTER_MISSES + 1):
        W.merge_scan(wl, scan(["AAA"]), d + dt.timedelta(days=7 * (i + 1)))
    assert wl["PYPL"]["status"] == "active" and wl["ZZZ"]["status"] == "dropped"
    W.remove(wl, "ZZZ", d)
    W.merge_scan(wl, scan(["ZZZ"]), d + dt.timedelta(days=70))
    assert wl["ZZZ"]["status"] == "dropped"           # a name removed by hand is not re-added
    W.remove(wl, "PYPL", d)
    W.ensure_author_calls(wl, d + dt.timedelta(days=71))
    assert wl["PYPL"]["status"] == "dropped"


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


def test_target_and_exit_rules(spy):
    from monitor import trades as T
    b = base_then_breakout()
    s, ledger, f = _stages_with_trades(b)
    (t,) = ledger
    exit_i = 280 + t["resolved_in"]
    assert s["stage"].iloc[exit_i] == "TARGET" and s["stage"].iloc[exit_i - 1] != "TARGET"
    # a closed winner never reads FAILED or RUNNING again, even if the price comes back down
    c = np.concatenate([b["close"].to_numpy(), np.linspace(74, 58, 12)])
    v = np.concatenate([b["volume"].to_numpy(), np.full(12, 1e6)])
    s2, ledger2, _ = _stages_with_trades(bars(c, v))
    assert ledger2[0]["status"] == "hit target"
    assert not {"FAILED", "RUNNING", "RETEST", "EXTENDED"} & set(s2["stage"].iloc[exit_i:])
    # a trade nobody can follow any more is closed after its horizon
    r = dict(t, status="open", exit_date=None, exit_px=None, resolved_in=None, date="2020-01-02", as_of="2020-01-10")
    out, _ = T.update([r], {}, _spy_for(b), _eng(), {}, today="2020-06-01")
    assert out[0]["status"] == "no data" and out[0]["exit_date"] == "2020-01-10"


def test_ledger_rechecks_recent_days_and_frees_closed_trades(spy):
    from monitor import trades as T
    b = base_then_breakout()
    f = features(b, _spy_for(b))
    st = stages(f, P)
    trig_day = b.index[280]
    # a run that saw day 281 before the trigger existed (e.g. the trigger-day volume came in late)
    st_before = st.copy()
    st_before["trigger"] = False
    ledger, seen = T.update([], {"X": (f.iloc[:282], st_before.iloc[:282])}, _spy_for(b), _eng(), {})
    assert ledger == [] and seen["X"] == str(b.index[281].date())
    ledger, _ = T.update(ledger, {"X": (f, st)}, _spy_for(b), _eng(), seen)
    assert [r["date"] for r in ledger] == [str(trig_day.date())]
    # an open trade that closed between runs no longer blocks the next trigger
    st2 = st.copy()
    st2.iloc[315, st2.columns.get_loc("trigger")] = True
    t_open = dict(ledger[0], status="open", exit_date=None, exit_px=None, resolved_in=None)
    out, _ = T.update([t_open], {"X": (f, st2)}, _spy_for(b), _eng(), {"X": str(b.index[290].date())})
    assert sorted(r["date"] for r in out) == [str(trig_day.date()), str(b.index[315].date())]


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


def test_alerts_are_news_only():
    from monitor.run import _alert_due
    idx = pd.bdate_range("2026-10-01", periods=30)
    day = lambda i: str(idx[i].date())
    st = lambda stages: pd.DataFrame({"stage": stages}, index=idx[:len(stages)])
    base = st(["BASING"] * 30)
    row = lambda s, trade=None: {"t": "X", "stage": s, "trade": trade, "date": day(29)}
    tr = {"date": day(27), "sessions": 2, "exit_date": None}
    assert _alert_due(row("TRIGGERED", tr), base, []) == (True, day(27))
    sent = [{"t": "X", "type": "TRIGGERED", "date": day(28), "trigger_date": day(27)}]
    assert not _alert_due(row("TRIGGERED", tr), base, sent)[0]                    # once per trigger
    assert _alert_due(row("RUNNING", dict(tr, sessions=6)), base, sent)[0]
    assert not _alert_due(row("RUNNING", dict(tr, sessions=30)), base, sent)[0]   # not every return to RUNNING
    won = {"date": day(2), "sessions": 20, "exit_date": day(27)}
    assert _alert_due(row("TARGET", won), base, [])[0]
    assert not _alert_due(row("TARGET", dict(won, exit_date=day(10))), base, [])[0]  # weeks ago: not news
    # a missed run delays a FAILED signal, it does not lose it
    lost = {"date": day(2), "sessions": 20, "exit_date": day(27)}
    assert _alert_due(row("FAILED", lost), base, [])[0]
    # SETUP: once when the run of SETUP days starts, not again while it hovers
    s1 = st(["BASING"] * 26 + ["SETUP"] * 4)
    assert _alert_due(row("SETUP"), s1, []) == (True, day(26))
    assert not _alert_due(row("SETUP"), s1, [{"t": "X", "type": "SETUP", "date": day(26), "trigger_date": day(26)}])[0]
    s2 = st(["BASING"] * 20 + ["SETUP"] * 3 + ["ABOVE"] * 4 + ["SETUP"] * 3)
    assert not _alert_due(row("SETUP"), s2, [{"t": "X", "type": "SETUP", "date": day(20), "trigger_date": day(20)}])[0]
    assert not _alert_due(row("ABOVE"), s2, [])[0]
    # an alert from before trade dates were recorded still counts for the trade it announced
    legacy = [{"t": "X", "type": "RUNNING", "date": day(28)}]
    assert not _alert_due(row("RUNNING", dict(tr, sessions=6)), base, legacy)[0]


def test_lagging_last_bar_takes_the_quote_volume(monkeypatch):
    import json as _json
    from monitor import prices as P
    days = pd.bdate_range("2026-09-01", periods=30)
    ts = [int(pd.Timestamp(d, tz="America/New_York").replace(hour=9, minute=30).timestamp()) for d in days]
    end = int(pd.Timestamp(days[-1], tz="America/New_York").replace(hour=16).timestamp())
    closes = [10.0] * 29 + [None]

    def payload(meta_extra):
        return _json.dumps({"chart": {"result": [{"timestamp": ts, "meta": {
            "exchangeTimezoneName": "America/New_York", "currentTradingPeriod": {"regular": {"end": end}},
            "regularMarketTime": end + 60, "regularMarketPrice": 11.0, **meta_extra},
            "indicators": {"quote": [{"open": [10.0] * 29 + [None], "high": [10.1] * 29 + [None],
                                      "low": [9.9] * 29 + [None], "close": closes, "volume": [1e6] * 29 + [None]}]}}]}}).encode()
    monkeypatch.setattr(P, "http_get", lambda url, timeout=20: payload({"regularMarketVolume": 3e6}))
    b = P.fetch_ohlcv("X")
    assert b["close"].iloc[-1] == 11.0 and b["volume"].iloc[-1] == 3e6 and len(b) == 30
    monkeypatch.setattr(P, "http_get", lambda url, timeout=20: payload({}))
    b = P.fetch_ohlcv("X")
    assert len(b) == 29 and (b["volume"] > 0).all()  # no volume yet: wait for the bar instead of a 0


def test_open_trade_under_its_level_is_a_retest(spy):
    from monitor import trades as T
    b = base_then_breakout()
    f = features(b, spy)
    st = stages(f, P)
    lvl = float(f["close"].iloc[290]) * 1.02  # the close on day 290 sits 2% under the level it broke
    trade = {"t": "X", "date": str(b.index[280].date()), "level": lvl, "exit_date": None, "status": "open"}
    assert T.stage_on(f, st, 290, trade, P) == ("RETEST", lvl)
    closed = dict(trade, exit_date=str(b.index[288].date()), status="timed out")
    assert T.stage_on(f, st, 290, closed, P)[0] == st["stage"].iloc[290]  # trade over: price stage, today's level

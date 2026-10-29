"""Offline tests for the alert scorer (synthetic bars, no network): pytest research/banana3"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracker as T  # noqa: E402

NY = {"tz": "America/New_York", "splits": []}


def path(rows):
    """Bars from (open, high, low, close) tuples, one hour apart."""
    t = pd.date_range("2026-01-05 15:00", periods=len(rows), freq="h", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"]).assign(t=t)


def daily(closes, start="2026-01-05", spread=0.01):
    idx = pd.bdate_range(start, periods=len(closes))
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c * (1 + spread), "low": c * (1 - spread), "close": c})


def test_snowflake_time():
    assert T.snowflake_time(2045900452613742677).strftime("%Y-%m-%d %H:%M:%S") == "2026-04-19 16:20:53"


def test_target_then_breakeven_stop():
    out = T.follow(path([(100, 103, 98, 102), (102, 106, 101, 105), (105, 105, 99, 100)]),
                   fill=100, long=True, stop=95, targets=[105, 110])
    assert out["t1_time"] is not None and out["stop_time"] is None
    assert out["ret_t1"] == pytest.approx(0.05)
    assert out["ret_parts"] == pytest.approx(0.025)    # half at 105, half back at the 100 entry
    assert out["mae"] == pytest.approx(-0.02)


def test_stop_and_target_on_one_bar_counts_as_stopped():
    out = T.follow(path([(100, 106, 94, 100)]), fill=100, long=True, stop=95, targets=[105])
    assert out["stop_time"] is not None and out["ambiguous"]
    assert out["ret_t1"] == pytest.approx(-0.05)


def test_gap_through_stop_fills_at_open():
    out = T.follow(path([(90, 91, 88, 89)]), fill=100, long=True, stop=95, targets=[105])
    assert out["ret_t1"] == pytest.approx(-0.10)


def test_short_side():
    out = T.follow(path([(100, 101, 89, 90)]), fill=100, long=False, stop=105, targets=[90])
    assert out["t1_time"] is not None and out["ret_t1"] == pytest.approx(0.10)


def test_no_targets_holds_to_the_end():
    out = T.follow(path([(100, 101, 99, 100), (100, 121, 99, 120)]), fill=100, long=True, stop=None, targets=[])
    assert out["ret_t1"] == pytest.approx(0.20) and out["ret_parts"] == pytest.approx(0.20)


def test_split_factor_undoes_reverse_split():
    meta = {"splits": [{"date": "2026-03-09", "num": 1, "den": 20}]}
    assert T.split_factor(meta, pd.Timestamp("2025-01-07", tz="UTC")) == 20
    assert T.split_factor(meta, pd.Timestamp("2026-04-01", tz="UTC")) == 1


def test_score_alert_posted_premarket_fills_at_open():
    bars = daily([100, 102, 104, 106, 108, 110])
    alert = {"alert_id": "a", "ticker": "X", "posted_at": "2026-01-05T12:00:00Z", "t1": "105", "stop": "95",
             "horizon_days": "30"}
    row = T.score_alert(alert, bars, None, NY, now=pd.Timestamp("2026-03-01", tz="UTC"))
    assert row["px_at_post"] == 100 and row["status"] == "t1" and row["win"] is True
    assert row["r_t1"] == pytest.approx(1.0)


def test_score_alert_target_missed_by_horizon_is_a_loss():
    bars = daily([100, 101, 100, 101, 100, 101])
    alert = {"alert_id": "a", "ticker": "X", "posted_at": "2026-01-05T12:00:00Z", "t1": "120", "horizon_days": "3"}
    row = T.score_alert(alert, bars, None, NY, now=pd.Timestamp("2026-03-01", tz="UTC"))
    assert row["status"] == "expired" and row["win"] is False


def test_target_behind_the_fill_is_flagged_not_scored():
    bars = daily([100, 102, 104, 106])
    alert = {"alert_id": "a", "ticker": "X", "posted_at": "2026-01-05T12:00:00Z", "t1": "90", "stop": "85"}
    row = T.score_alert(alert, bars, None, NY, now=pd.Timestamp("2026-03-01", tz="UTC"))
    assert row["status"] == "bad_levels" and "win" not in row


def test_entry_level_waits_for_fill():
    bars = daily([100, 101, 103, 106, 108, 110])
    alert = {"alert_id": "a", "ticker": "X", "posted_at": "2026-01-05T12:00:00Z", "entry": "104",
             "t1": "108", "stop": "100", "horizon_days": "30"}
    row = T.score_alert(alert, bars, None, NY, now=pd.Timestamp("2026-03-01", tz="UTC"))
    assert row["fill"] == pytest.approx(104) and row["win"] is True


def test_implied_vol_round_trip():
    p = T.bs_price(100, 105, 0.1, 0.5, call=True)
    assert T.implied_vol(p, 100, 105, 0.1, call=True) == pytest.approx(0.5, abs=1e-4)


def test_option_without_premium_scored_at_expiry():
    bars = daily([190, 200, 210, 220, 221, 222])
    alert = {"alert_id": "a", "ticker": "X", "posted_at": "2026-01-05T12:00:00Z", "level_basis": "option",
             "opt_type": "C", "strike": "195", "expiry": "2026-01-09"}
    row = T.score_alert(alert, bars, None, NY, now=pd.Timestamp("2026-03-01", tz="UTC"))
    assert row["status"] == "itm_at_expiry" and row["intrinsic_at_expiry"] == pytest.approx(26)


def test_option_premium_targets_use_model():
    bars = daily([100, 104, 108, 112, 116, 120, 124, 128])
    alert = {"alert_id": "a", "ticker": "X", "posted_at": "2026-01-05T12:00:00Z", "level_basis": "option",
             "opt_type": "C", "strike": "105", "expiry": "2026-02-20", "entry": "3", "stop": "1.5", "t1": "6"}
    row = T.score_alert(alert, bars, None, NY, now=pd.Timestamp("2026-03-01", tz="UTC"))
    assert row["win"] is True and row["ret_t1"] == pytest.approx(1.0)
    assert "modelled premium" in row["note"]


def test_random_entry_rate_on_a_steady_climb():
    bars = daily([100 * 1.01 ** i for i in range(120)])
    t0 = pd.Timestamp(bars.index[-1], tz="UTC") + pd.Timedelta(days=1)
    rate = T.random_entry_rate(bars, t0, "America/New_York", True, 0.05, -0.05, pd.Timedelta(days=20))
    assert rate == 1.0


def test_read_alerts_keeps_hashtags(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("# a note\nalert_id,ticker,notes\nx1,BTC-USD,#Bitcoin call\n")
    assert T.read_alerts(p) == [{"alert_id": "x1", "ticker": "BTC-USD", "notes": "#Bitcoin call"}]

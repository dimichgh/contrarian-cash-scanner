"""Paths, cadences and the engine defaults the learner starts from."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state"
REPORTS = ROOT / "reports"
WATCHLIST = STATE / "watchlist.json"
ENGINE = STATE / "engine.json"
ENGINE_LOG = STATE / "engine_history.jsonl"
ALERTS = STATE / "alerts.jsonl"
LAST_RUN = STATE / "last_run.json"
HISTORY = STATE / "history"
SCAN = STATE / "scan_latest.csv"

RESCAN_DAYS = 7       # full fundamental re-scan cadence
LEARN_DAYS = 7        # engine re-fit cadence
TRAIN_UNIVERSE = 300  # top-N names of the latest scan used as training data

# how a name gets onto the watchlist from a scan
ADMIT_TIERS = {"A"}
ADMIT_MIN_SCORE_B = 60.0  # tier B names need at least this score
ADMIT_TOP_N = 25          # plus the top N by score whatever their tier
DROP_AFTER_MISSES = 3     # consecutive rescans a scan-sourced name may miss before it is dropped

# "The real move": the close rises by `up` of the stock's normal 40-session swing (daily volatility
# over the prior 60 sessions × √40) before it falls by `down` of it, within 40 sessions. For a typical
# value stock (1.8% a day) that is about +10% before −6%; a volatile name needs a bigger move.
# Sizing by volatility keeps the engine hunting for direction instead of rewarding volatility.
# Fixed on purpose: changing the objective would make engine versions incomparable.
LABEL = {"horizon": 40, "up": 0.9, "down": 0.55}

# Starting trigger rule: the author's confirmation tranche, a close back above the 50-day average
# after a real stretch below it. The learner tunes these against history.
TRIGGER = {
    "pivot": "sma50",     # trigger level: sma50 | hh20 | hh40 | sma50_hh20
    "vol_mult": 1.2,      # trigger-day volume vs its 50-day average (0 = no volume check)
    "min_below": 10,      # sessions below the level out of the previous 20
    "min_low_age": 10,    # sessions since the last 120-day low
    "confirm": 5,         # sessions a fresh trigger stays TRIGGERED before it counts as RUNNING
    "fail_tol": 0.03,     # a close this far back under the trigger level = failed breakout
    "setup_gap": 0.04,    # within this distance under the level (with lows holding) = SETUP
}

TRIGGER_GRID = {
    "pivot": ["sma50", "hh20", "hh40", "sma50_hh20"],
    "vol_mult": [0.0, 1.2, 1.5],
    "min_below": [5, 10],
    "min_low_age": [0, 10],
}

# Expert prior for the readiness model, in standard deviations of each feature. The learner fits
# a logistic regression shrunk toward these, so thin or noisy data leaves the prior in charge.
PRIOR = {
    "f_sma50_gap": 0.15,    # above the 50-day average: the author's confirmation signal
    "f_sma200_gap": 0.05,
    "f_sma50_slope": 0.20,  # 50-day average turning up
    "f_hh40_gap": 0.20,     # close to (or through) the top of the recent range
    "f_low_age": 0.15,      # lows holding: time since the last 120-day low
    "f_higher_low": 0.15,
    "f_rs_slope": 0.20,     # outperforming the S&P 500 over a month
    "f_rs_pos": 0.10,
    "f_rsi": 0.0,
    "f_squeeze": -0.10,     # volatility contraction (coiling) before a break
    "f_updown_vol": 0.15,   # accumulation: volume on up days vs down days
    "f_vol_surge": 0.05,
    "f_ret20": 0.0,
    "f_dd_52w": 0.0,
    "f_mkt": 0.10,          # S&P 500 above its 200-day average
    "f_vol20": 0.0,
}
FEATURES = list(PRIOR)
LAMBDAS = [30.0, 300.0, 3000.0, 30000.0]  # prior strength candidates (pseudo-observations)
ADOPT_AUC_MARGIN = 0.003    # a challenger model must beat the champion's holdout AUC by this much
ADOPT_EXP_MARGIN = 0.0025   # a challenger trigger rule must beat the champion's expectancy by this much

# Contrarian Cash Scanner

Ranks US-listed stocks and ADRs on the stock-selection checklist used in
[arden_trading](https://www.tradingview.com/u/arden_trading/)'s long-term TradingView theses
(PEP, CMCSA, T, PYPL, SONY, F, INTC, UDMY).

The pattern it looks for: a hated, beaten-down company that still produces a lot of free cash,
returns it to shareholders, trades below its peers, carries manageable debt, still beats
estimates, sits on a Hold-heavy analyst consensus with a big target upside, and has insiders
buying or short sellers leaving.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

No API keys are needed.

## Usage

```bash
# full scan (~2 min): writes output/arden_scan.csv and output/arden_scan.json
.venv/bin/python arden_scanner.py

# build the interactive HTML report from the latest scan: output/scan_report.html
.venv/bin/python make_report.py
```

Options:

```bash
.venv/bin/python arden_scanner.py --min-mcap 10e9       # large caps only (default $2B)
.venv/bin/python arden_scanner.py --enrich 200          # deep-check more candidates (default 120)
.venv/bin/python arden_scanner.py --tickers PEP T VZ    # score specific names against the universe
.venv/bin/python arden_scanner.py --top 50              # rows printed to the terminal
```

## How it scores

1. **Universe**: US common stocks and ADRs on NYSE/NASDAQ/AMEX above the market-cap floor,
   one line per company.
2. **Gates**: positive trailing free cash flow, some price damage (≥20% off the 52-week high,
   ≥30% off the all-time high, 1-year return ≤ −10% or 5-year ≤ −30%), and no banks, insurers
   or funds (FCF and EBITDA don't mean much there).
3. **Base score (0–100)**: seven factors, each a percentile within the gated universe.

   | Factor | Weight | Metric |
   |---|---|---|
   | Cash machine | 22 | FCF yield (FCF ÷ market cap), FCF margin |
   | Cheap vs peers | 18 | Fwd P/E, EV/EBITDA, P/FCF, P/S, P/B relative to sector median |
   | Paid to wait | 15 | Dividend yield + net share-count reduction; dividend cover |
   | Capitulation | 15 | Drop from 52w high, 5-year return, distance from 52w low |
   | Balance sheet | 10 | Net debt ÷ EBITDA, current ratio |
   | Business still works | 10 | Revenue growth, operating margin, last EPS surprise, ROE |
   | Contrarian consensus | 10 | Analyst target upside × share of Hold/Sell ratings |

   Penalties: beta above 1.5, net debt/EBITDA above 3.5 (except captive-finance
   auto/equipment makers, whose lending arm inflates net debt).
4. **Smart-money points (up to +15)**, for the top `--enrich` names only: executive/director
   open-market purchases in the last 90 days, short interest down 15%+ over ~6 weeks, a 5%+
   drop after an EPS beat, volume accumulation near the lows, a reclaim of the 50-day average.
5. **Final score** = 0.85 × base + smart-money points.
6. **Tier**: how many of the author's ten checks pass. A = 7+, B = 5–6, C = 4 or fewer.
7. **Timing**: *Near lows* (first-tranche zone), *Turning up* (back above the 50-day average),
   *Still falling* (RSI < 35 and −8% in a month), *Mid-range*.

## Breakout monitor

The scanner finds the names; the monitor (`monitor/`) watches them every trading day for the moment
the price turns and starts the move up, rates how close each one is to that move, and re-learns its
own rules every week.

```bash
.venv/bin/python -m monitor daily            # score the watchlist → reports/summary.md + reports/dashboard.html
                                             # (rescan days also rebuild reports/scan_report.html)
.venv/bin/python -m monitor daily --rescan   # force a full re-scan first (runs weekly on its own)
.venv/bin/python -m monitor daily --learn    # force a re-fit of the engine (runs weekly on its own)
.venv/bin/python -m monitor add PEP --note "buy zone 120–125"
.venv/bin/python -m monitor remove XOP
.venv/bin/python -m monitor list
.venv/bin/python -m pytest tests             # offline tests
```

**Watchlist.** Seeded from the scan: every Tier A name, Tier B names scoring 60+, and the top 25
whatever their tier, plus the author's own calls. A weekly re-scan adds new names that qualify and
drops scan names that miss three re-scans in a row, unless their move is already under way.

**Stages**, from daily bars alone:

| Stage | Meaning |
|---|---|
| Falling | Made a new 120-day low in the last 15 sessions |
| Basing | Lows holding 15+ sessions, not near the trigger level yet |
| Setup | Under today's trigger level by 4% or less, lows holding: closest to a trigger |
| Triggered | Triggered in the last 5 sessions (a close through the trigger level, by default the 50-day average, the author's confirmation tranche, on 1.2x volume after 10+ of the prior 20 sessions below it); trade open, price at or above the level |
| Running | Trade open, price at or above the level it broke through |
| Extended | Trade open and stretched: 20%+ above the 50-day average or RSI over 75 |
| Retest | Trade open, but the price is back under the level it broke through (stop not hit) |
| Target hit | The trade reached its target and the price is still above its level (up to 60 sessions after the trigger) |
| Failed | The trade hit its stop and the price is still under its level (up to 40 sessions after the trigger) |
| Above, no trigger | Above today's trigger level with no open trade: crossed without the volume or the base, or the trade is over. No entry signal |

The first six come from the name's trade (see *Since the trigger*), so a stage can never disagree
with the trade shown next to it, and they are measured against the level the trade broke through.
The last four come from the price alone and are measured against today's level.

Signals are news only, and decided from what was actually sent, so a missed run delays a signal
instead of losing it: a trigger once per trade (with the trigger day's numbers and readiness), Running
once per trade when its first sessions are over, Target hit and Failed once per trade within a week
of the exit, and Setup when a run of setup days begins, not again within 10 sessions. A trigger is
**strong** when its readiness clears a bar the engine learns.

**Readiness (0–100)** is the engine's probability that the stock reaches its target before its stop
within 40 trading days, as a percentile of every setup in the training history. Target and stop are
sized to the stock's own volatility (0.9× and 0.55× its normal 40-day swing: about +10% / −6% for a
typical value stock), so the engine looks for direction rather than for volatility.

**How it improves.** Each week it replays five years of daily bars for the watchlist and the scan's
top 300 names, labels every day by what happened next, and:

1. refits the readiness weights (a logistic regression on 16 price and volume features, shrunk toward
   hand-set expert weights; how far it may move from them is chosen on a held-out final year);
2. replays 48 variants of the trigger rule (level, volume, time below the level, age of the low) and
   switches rule only when the challenger also wins on the held-out year;
3. picks the readiness bar for a strong trigger on the held-out year (at least 30 events).

Every daily call is stored, and the dashboard scores those calls against what the price actually did,
counting a call only once its full 40 sessions have passed (counting early resolutions first would
over-weight quick stops and targets).

**Since the trigger.** Every trigger is followed as a trade from the trigger-day close, one trade per
name at a time, until a close reaches its target or its stop, or 40 sessions pass. The trades live in
`state/trades.jsonl`: an open trade is re-followed each run with the target and stop it opened with,
and a closed one keeps its exit date, exit price, result, best and worst close up to the exit and the
S&P 500 over the same days for good, whatever the watchlist or the trigger rule does later. Each run
also re-checks the last 5 sessions, so a trigger revealed by a revised bar still gets its trade. Names
that leave the watchlist with a trade open are still followed until it closes; one that cannot be
followed any more (delisted, renamed) is closed as "no data" 15 business days after its horizon. The dashboard lists
open trades and recent exits (older exits behind a button) with today's change beside each closed one,
and the totals so far. Open trades also get a target window: the dates between which past triggers
that reached their target usually got there (the middle half of their times, re-measured at every
learning cycle), and whether the trade is early, in that window or late. Setup names show the stop
and target they would get if they triggered at their level.

**Prices** are cached in `state/bars/` (one CSV per name, committed with the state). A run downloads
only the last month of each name from Yahoo and merges it in; a name is downloaded in full the first
time, every 60 days, when a longer history is needed, or when the overlap shows a split. If Yahoo
fails for a name, its cached bars still feed the learner but it gets no signal that day. To work from
the cache alone, without any download: `MONITOR_OFFLINE=1 .venv/bin/python -m monitor daily`.

**State** lives in `state/` and is committed after each run: `watchlist.json`, `engine.json` (the
current rule and weights), `engine_history.jsonl` (one line per learning cycle), `alerts.jsonl`,
`history/YYYY-MM.csv` (daily calls), `trades.jsonl` (every trade), `bars/` (cached prices) and
`scan_latest.csv`. The pages are committed too:
`reports/dashboard.html` (the monitor, every run) and `reports/scan_report.html` (the full scan,
rebuilt at each weekly rescan). Both are self-contained: download one and open it in a browser.

**Scheduled runs.** A GitHub Actions workflow (`.github/workflows/monitor.yml`) runs the monitor
every weekday at 21:23 UTC (5:23pm ET in summer, 4:23pm ET in winter), with no Claude usage. It
calls `scripts/cloud_run.sh`, which runs the monitor, commits `state/` and the pages to `main` and
prints the summary. The summary goes to the run's page and, when there is a new close, to a comment
on the pinned "Breakout monitor: daily summaries" issue, so GitHub notifies you by email or on its
mobile app. A failed run comments there too. Start a run by hand from the Actions tab (Run
workflow), with optional re-scan or re-fit.

The pages are also published to GitHub Pages after every run:
`https://dimichgh.github.io/contrarian-cash-scanner/` (the monitor) and `…/scan.html` (the full
scan). `scripts/build_site.py` wraps the committed `reports/*.html` into the site. One-time setup:
Settings → Pages → Source: "GitHub Actions". The site is public, like the repo.

To run it yourself instead: `bash scripts/cloud_run.sh` (or `python -m monitor daily` without the
git steps). Skipped days only delay signals: each run compares against the last recorded one.
`scripts/routine_prompt.md` keeps the prompt for running it as a Claude cloud routine instead.

## Data sources

| Source | Used for |
|---|---|
| TradingView screener API | Universe, fundamentals, estimates, analyst ratings, technicals |
| Yahoo Finance chart API | Daily bars (earnings reaction, volume, 50-day reclaim) |
| OpenInsider | Form 4 open-market purchases |
| FINRA consolidated short interest | Short-interest history |

## What it can't see

- **Catalysts.** Every one of the author's picks had one (spin-off, new CEO, activist,
  product event). Check this by hand.
- **Sum-of-the-parts** valuations of individual segments.
- **One-off cash flows.** FCF yields above ~25% often include disposals; compare with guidance.
- **Macro drivers** such as rates, currency or commodity prices.

This is a screening tool built from public articles, not investment advice.

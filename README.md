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
| Falling | Still making new 120-day lows |
| Basing | Lows holding 15+ sessions, 50-day average flattening |
| Setup | Within 4% under the trigger level, lows holding: closest to a breakout |
| Triggered | Fresh close through the trigger level (default: the 50-day average, the author's confirmation tranche) on 1.2x volume after 10+ of the prior 20 sessions below it |
| Running | Held above the level since the trigger, or above a rising 50-day average |
| Extended | Running but 20%+ above the 50-day average or RSI over 75 |
| Failed | Closed 3%+ back under the level within 20 sessions of a trigger |

A move into Triggered, Setup, Running or Failed raises a signal. A trigger is **strong** when its
readiness clears a bar the engine learns.

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

Every daily call is stored, and the dashboard scores those calls against what the price actually did.

**Since the trigger.** Each name's latest trigger from the last 90 sessions is followed as a trade from
the trigger-day close: sessions and days since, entry and last price, change, best and worst close
since, the S&P 500 over the same days and the difference, its stop and target prices, and whether it
is still open, hit its target, was stopped out or timed out after 40 sessions. Open trades also get a
target window: the dates between which past triggers that reached their target usually got there
(the middle half of their times, re-measured at every learning cycle; about 11–26 sessions today),
and whether the trade is early, in that window or late. Setup names show the stop and target they
would get if they triggered at their level.

**State** lives in `state/` and is committed after each run: `watchlist.json`, `engine.json` (the
current rule and weights), `engine_history.jsonl` (one line per learning cycle), `alerts.jsonl`,
`history/YYYY-MM.csv` (daily calls) and `scan_latest.csv`. The pages are committed too:
`reports/dashboard.html` (the monitor, every run) and `reports/scan_report.html` (the full scan,
rebuilt at each weekly rescan). Both are self-contained: download one and open it in a browser.

**Scheduled runs.** `scripts/cloud_run.sh` runs the monitor, commits `state/` and pushes it to the
current branch (or `$MONITOR_BRANCH`), then prints the summary. A Claude cloud routine calls it after
each US close; its prompt is in `scripts/routine_prompt.md`.

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

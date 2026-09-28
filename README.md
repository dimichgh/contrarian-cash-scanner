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

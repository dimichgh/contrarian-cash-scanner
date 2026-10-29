# Banana3 one-month subscription audit

Handoff for a Claude session on your own computer. That session can use your logged-in Chrome and
X account, which a cloud session can't. The goal: subscribe to [@Banana3Stocks](https://x.com/Banana3Stocks)
for one month, record **every** trade alert, losers included, score each one against market prices,
and decide whether the subscription is worth keeping.

**Prompt to start the local session** (Claude Code or the Claude desktop app, with the Claude in
Chrome extension connected and Chrome signed in to X):

> Read `research/banana3/README.md` and run the audit it describes, using my logged-in Chrome.
> Start with Phase 0 and Phase 1. Ask me before anything that costs money or changes my accounts.

Files in this folder:

| File | What it is |
|---|---|
| `README.md` | This runbook |
| `tracker.py` | Scores alerts against Yahoo prices: `date`, `bars`, `score` commands |
| `prior_calls.csv` | The 21 public calls from the first pass, in the alerts format (reproduces the numbers below) |
| `test_tracker.py` | Offline tests: `pytest research/banana3` |
| `data/` | Everything you capture. **Gitignored**: it holds paid content, never commit it |

---

## 1. What we already know (first pass, 2026-10-29)

Built from about 50 search-indexed public posts, because X blocks reading the account without a
login. Details are in the appendix.

**The account:** "Professional Trader/Investor, 25 yrs experience, Real Estate Developer", Los
Angeles, about 153K followers, joined Feb 2019. Paid tier through X Subscriptions (posts marked 🍇
"for my SUBSCRIBERS"). Their [Linktree](https://linktr.ee/banana3stocks) also links a Discord server.

**Strategy as seen from outside:**
- **Options on big tech:** short-dated options on NVDA, AMD, ARM, TSLA, META, AMZN, CRWD, SNOW, IBM,
  BA, CRDO and APP. Alert format: "Entry, Stop Loss, 1st 🎯, 2nd 🎯, 3rd 🎯 and more", plus
  "runner targets". Some are "HIGH RISK 1DTE lotto" plays. Positions are large and concentrated
  ("bought $100 grand of the BA May 15 $195 calls").
- **Chart patterns:** inverse and "failed" head-and-shoulders, W double bottoms and gap fills, on
  charts from 4-hour to quarterly. Supporting lines: the 13 EMA, the 50-week SMA, and their own
  "Banana325SMA" (a 325-period simple moving average with extended hours on, any timeframe).
- **Market-wide signals:** McClellan breadth (NAMO), "breadth thrust", Fed cuts and liquidity.
  Almost always bullish: no bearish calls found, and they mock bears.
- **Micro-cap biotech "baskets":** ALGS, BIVI, CLDI, NXL and RVPH, sold on catalyst stories.
- **A "$30K → $500K challenge" account**, traded live for subscribers. They claimed about $200K in
  Jul 2025 and about $220K in Oct 2025, with $500K as the year-end goal. No later update was found.

**Public price calls checked against prices:**
- **Large caps, indexes and BTC:** 9 of 11 judged calls won (first target hit, or the option expired
  in the money). A random entry with the same distance
  to target, on the same asset in the prior two years, hit about 93% of the time over the same
  horizon. That is no evidence of skill beyond being long in a bull market. Several hits came only
  after 24–35% drawdowns that short-dated options would not have survived.
- **Small-cap biotech:** 0 of 4 targets hit. The 4-name "Biotech Basket" is down about 94% while
  the S&P 500 rose 27%.

**Their own performance claims** ("ARM +240% in a day", "IBM +150%", "all three targets hit") are
cherry-picked winners. No public post reports a stopped-out alert.

**What this audit tests:**

| # | Question | Measured by |
|---|---|---|
| H1 | What is the real win rate of subscriber alerts (T1 before the stop)? | `win rate` in the summary |
| H2 | Is it better than chance? | `win rate` against `baseline` (random entries, same distances) |
| H3 | Are the public brags cherry-picked? | Win rate of `advertised=1` against `advertised=0` |
| H4 | Do 1DTE lottos and small-cap ideas lose money overall? | `bucket=lotto` and `bucket=small` rows |
| H5 | Are losers closed out openly, or quietly deleted? | Phase 4 re-check of every captured post |
| H6 | Do the challenge-account balances match the posted trades? | `challenge.csv` against the alerts |

## 2. Ground rules

- **Personal research only.** Subscriber posts are paid content: don't publish, quote at length or
  commit them. `data/` is gitignored, and this repository may be public. Only the aggregate
  findings (`findings.md`, no alert details) may be committed.
- **Read like a person.** Use only the logged-in Chrome session, at a normal browsing pace. Don't
  write scripts that pull from X or Discord: X's terms forbid automated scraping, and it can get
  the account locked. If X shows a rate limit or an "unusual activity" prompt, stop and tell the user.
- **Read only.** No likes, replies, DMs, follows or reposts from the user's account.
- **Money and accounts are the user's call.** The agent never subscribes, pays, cancels, or joins
  a server without the user's go-ahead; it prepares the step and asks.
- **No hindsight.** Transcribe levels exactly as posted. If a number is unreadable, write `?` in
  `notes` and leave the field blank rather than guessing from the chart that followed.

## 3. Setup (once)

```bash
git pull
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt pytest
.venv/bin/pytest -q research/banana3                     # 16 offline tests
.venv/bin/python research/banana3/tracker.py score research/banana3/prior_calls.csv --baseline
mkdir -p research/banana3/data
```

The last command should reproduce the first-pass table: big caps 9 of 11 wins (82%), small caps
0 of 4. It writes `data/outcomes.csv` and `data/summary.md`; move those aside before scoring your
own alerts.

## 4. Timeline

| When | Phase | Who |
|---|---|---|
| Days −7 to 0 | **0.** Note the subscription offer. **1.** Back-fill the public timeline (free) | Agent, then the user decides whether to pay |
| Day 1 | Subscribe. Note the renewal date and set a reminder for day 27 | User |
| Days 1–3 | **2.** Back-fill subscriber posts (and the Discord, if included) | Agent |
| Days 1–30 | **3.** Live capture: check 2–3 times each trading day; bank bars every session | Agent (and a daily scheduled job) |
| Days 26–27 | **4.** Re-check every post, final score, report | Agent |
| Day 27 | Cancel before the renewal, unless the report says keep | User |

## 5. Phase 0: before paying

1. Open the profile's **Subscribe** page and record in `data/offer.md`: price, billing period,
   promised perks (alerts, Discord, challenge account, write-ups), and whether past subscriber
   posts are visible after joining.
2. Note how many subscribers the page shows, if any, and the date.
3. If the Discord invite on the Linktree is public, record its channel list without joining.
   Joining is the user's call.

## 6. Phase 1: public back-fill (free, logged in)

Use X search on the **Latest** tab, one month at a time, going back 12 months:

```
from:Banana3Stocks since:2025-10-01 until:2025-11-01
```

Then run targeted passes over the whole account history:

```
from:Banana3Stocks "target hit"        from:Banana3Stocks stopped
from:Banana3Stocks "stop loss"         from:Banana3Stocks closing
from:Banana3Stocks loss                from:Banana3Stocks 30kto500k
from:Banana3Stocks filter:replies 🎯
```

Scroll each window to the end. **Log only posts that matter for scoring:**

| kind | What it is | Log to |
|---|---|---|
| `alert` | A trade idea with levels, or a "just bought" position | `posts.jsonl`, then one or more rows in `alerts.csv` |
| `update` | Closed, stopped, rolled, "taking profits" | `posts.jsonl`, plus `notes` on the alert |
| `claim` | "Target hit", "+150%", "what my subscribers got" | `posts.jsonl` and `claims.csv` |
| `challenge` | A challenge-account balance or position | `posts.jsonl` and `challenge.csv` |
| `call` | A public price target without entry or stop (like the ones in `prior_calls.csv`) | `posts.jsonl` and `alerts.csv` (`source=x_public`) |

Skip pure commentary and promotion; just count them per month in `data/activity.csv`
(`month,posts_seen,logged`) so the report can say how much of the feed was trading content.

Charts carry most of the numbers. Open each image full-size and transcribe every level printed or
drawn on it into `media`. Get each post's exact time from its id:
`python research/banana3/tracker.py date <post_id>`.

## 7. Phase 2: subscriber back-fill (days 1–3)

1. Find where subscriber posts live: a profile tab for subscriber posts if X shows one, otherwise
   the main timeline, where they carry a subscriber label. Record which in `offer.md`.
2. Log **every** subscriber post of kind `alert`, `update`, `claim` or `challenge` back to the start
   of the visible history, with `source=x_sub`. Older posts the account later bragged about publicly
   are already in `claims.csv` from Phase 1: link them through `alert_id`.
3. If the subscription includes the Discord and the user has joined: log the alerts channel
   history the same way with `source=discord`. Message ids give exact times:
   `tracker.py date --discord <message_id>`.
4. Look for a published track record, trade log or spreadsheet. If one exists, save a copy to
   `data/` and score it as a separate alerts file.

## 8. Phase 3: live month (days 1–30)

- **Check 2–3 times per trading day:** before the open, around midday, and after the close. Log new
  alerts within hours, along with every later update to them.
- **Record option quotes where possible:** when an alert names an option contract, record the
  contract's bid/ask/last and the stock price at capture time in `option_quotes.csv`. Use the
  user's broker or any quote page open in the browser; this calibrates the option model.
- **Bank bars every session**, after 4:15pm ET:
  ```bash
  .venv/bin/python research/banana3/tracker.py bars --alerts research/banana3/data/alerts.csv
  ```
  Yahoo keeps 5-minute bars for only 60 days. The cache keeps them for good, and with them most
  alerts can be ordered within the day instead of flagged as ambiguous. A local scheduled job
  (cron, or a Claude scheduled task on this machine) is the reliable way to run it.
- **Score weekly:**
  ```bash
  .venv/bin/python research/banana3/tracker.py score research/banana3/data/alerts.csv --baseline
  ```
  Send the user a short weekly status: alerts logged, judged, win rate, baseline, and anything odd.
- **Watch for deletions.** Every few days, re-open the alerts logged so far. A post that no longer
  loads gets `"deleted": true` and a note on when it was last seen.

## 9. Phase 4: wrap-up (days 26–27)

1. Re-open every logged post URL and set `seen_again_utc` or `deleted` (H5).
2. Fill in `advertised` on each alert: `1` if a public, non-subscriber post bragged about it.
3. Check every `claims.csv` row against `outcomes.csv`, and record in the `verified` column whether
   the claim matches the scored outcome (`yes`, `no` or `partly`).
4. Reconcile the challenge account (H6): do the posted balance changes fit the trades it alerted?
5. Final score with `--baseline`, then write `data/report.md` (outline in section 12).
6. Write `research/banana3/findings.md`: aggregate numbers only, no alert text or levels. That file
   is safe to commit.
7. Remind the user of the renewal date and give the keep or cancel recommendation.

## 10. Data formats (all in `data/`)

**`posts.jsonl`** has one JSON object per line:

```json
{"id": "2037230514718040499", "url": "https://x.com/Banana3Stocks/status/2037230514718040499",
 "created_utc": "2026-03-26 18:09:13", "source": "x_sub", "kind": "alert", "tickers": ["AMD"],
 "text": "full post text", "media": "chart: entry 205-207, stop 198, T1 215, T2 222, T3 230",
 "reply_to": null, "quoted": null, "alert_ids": ["2026-03-26-amd-1"],
 "captured_utc": "2026-11-02 14:05", "seen_again_utc": null, "deleted": false}
```

**`alerts.csv`** is what `tracker.py score` reads. One row per trade idea; the columns follow
`prior_calls.csv`, and lines starting with `#` are comments.

| Column | Meaning |
|---|---|
| `alert_id` | Unique, e.g. `2026-11-03-nvda-1` |
| `post_id` | X post id (or Discord message id). Gives the time when `posted_at` is blank |
| `posted_at` | ISO UTC time; only needed when there's no usable post id |
| `source` | `x_public`, `x_sub`, `discord` |
| `ticker` | Yahoo symbol: `NVDA`, `^GSPC` for SPX, `^NDX`, `BTC-USD`. For SPY/QQQ options use `SPY`/`QQQ` |
| `side` | `long` or `short`; a bought put is `long` with `opt_type=P` |
| `level_basis` | `underlying` when the entry, stop and targets are stock prices; `option` when they are option premiums |
| `entry` | The price as posted. Blank for "now/market". For a range, use the midpoint and note the range |
| `stop` | As posted. If it's close-based ("close below 198"), still enter 198 and note it: the scorer uses touches, which is stricter |
| `t1`, `t2`, `t3` | Targets as posted. Runner targets beyond the third go to `notes` |
| `opt_type`, `strike`, `expiry` | For option alerts (`C`/`P`, number, `YYYY-MM-DD`). Fill them in even when the levels are stock prices |
| `horizon_days` | How long to follow it when there's no expiry. Use 1 for day trades, about 30 for swings; the default is 90 |
| `bucket` | `large` (big caps, ETFs, indexes, BTC), `small` (small caps and biotech), `lotto` (0–1DTE) |
| `advertised` | `1` if a public post later bragged about it, `0` if not; filled in Phase 4 |
| `notes` | Anything else: original wording, ranges, rolls, legibility doubts |

When an alert gives both stock levels and option premiums, use the stock levels (`underlying`) and
put the premiums in `notes`. They score more reliably.

**`claims.csv`**: `claim_id,post_id,alert_id,claim_text,claimed_return,verified`

**`challenge.csv`**: `post_id,date,claimed_balance,positions,notes`

**`option_quotes.csv`**: `alert_id,captured_utc,contract,bid,ask,last,underlying_price,source`

## 11. How `tracker.py` scores

- **Fill:** an alert with no entry fills at the first price after the post: the next 5-minute bar's
  open, or with daily bars the session's open (posted before the open) or close (posted during
  the session). An alert with an entry level fills when the price trades through it within 5 days;
  otherwise it is `no_fill`. Filling at the first possible price is generous to the trader:
  subscribers react later.
- **Outcome:** each bar is checked for the stop and the targets by touch. **Win** means T1 was hit
  before the stop. A target still unreached at the horizon or expiry is a loss. Alerts with no
  target get a return but no win or loss.
- **Ambiguous bars:** a bar that touches both the stop and T1 counts as stopped and is flagged
  `ambiguous`. That only happens on daily bars, which is why the daily `bars` job matters.
- **Two exit plans are scored:** all-out at T1 (`ret_t1`, `r_t1` = return divided by the
  entry-to-stop risk), and equal parts at each target with the stop moved to the entry after T1
  (`ret_parts`).
- **Splits:** levels are taken as quoted on the post date; later splits are undone automatically.
- **Options:** when the levels are premiums, the contract is repriced from the stock path with
  Black-Scholes, using the volatility implied by the entry premium (`note: modelled premium, iv
  ..%`). This ignores volatility crush after earnings and skew, so check it against
  `option_quotes.csv`. Option alerts without a premium are judged only on whether they expired
  in the money.
- **Baseline:** random entries in the 500 sessions before the post, on the same asset, with the same
  % distance to T1 and to the stop and the same horizon. It shows what luck plus the market's drift
  would have delivered. A real edge shows as a win rate clearly above it.
- **Not modelled:** commissions, bid/ask spreads (often 3–10% of an option's premium per round
  trip) and partial fills. The report should take a realistic cost off the average return.
- **Status values:** `t1`/`t2`/`t3` (highest target reached; `+open` if still running), `stopped`,
  `expired`, `open`, `no_fill`, `waiting`, `itm_at_expiry`/`otm_at_expiry`, `bad_levels` (a target
  or stop on the wrong side of the fill, almost always a transcription slip: fix the row),
  `bad_option`, `no_data`.

## 12. Report outline (`data/report.md`)

1. **Verdict** in three lines: keep or cancel, and why.
2. **Coverage:** posts seen, alerts logged by source, days monitored, posts deleted.
3. **Headline table:** alerts, judged, win rate, baseline, average return under both exit plans,
   average R, after an assumed cost.
4. **By bucket** (large, small, lotto) and **by source** (public, subscriber, Discord).
5. **Cherry-picking (H3):** the advertised win rate against the rest; the claims checked, and how
   many held up.
6. **Worst ten and best ten alerts** with their outcome and maximum drawdown.
7. **Challenge-account reconciliation (H6).**
8. **Limits:** option-model error measured against the recorded quotes, ambiguous bars, and posts
   that couldn't be read.

**Suggested keep rule** (the user decides): keep only if, over at least 30 judged alerts, the win
rate beats the baseline by 10 points or more, the average R after costs is above +0.3, the
advertised and non-advertised win rates are close, and no losing alerts were deleted.

## 13. Pitfalls

- **Edited posts.** Only the latest version is visible. Record the capture time, and treat a level
  that changed between checks as a red flag.
- **"Entry" given as a zone or a chart line.** Transcribe what is written; if it's only drawn, read
  the price axis and note `approximate`.
- **Rolls and adds.** "Rolled to the June 200 calls" or "added more" is a new row with the same
  ticker and a note linking it to the original alert.
- **Index options.** SPX and NDX alerts score on `^GSPC`/`^NDX`; SPY and QQQ alerts on `SPY`/`QQQ`.
- **Crypto** trades 24/7. `BTC-USD` daily bars are UTC days.
- **Daily bars before the subscription.** Back-filled alerts older than 60 days only have daily
  bars, so expect more `ambiguous` flags there than in the live month.

---

## Appendix A: first-pass call log

Prices were checked through Oct 27–28, 2026. `prior_calls.csv` re-runs all of this.

| Posted (UTC) | Call | Price then | What happened |
|---|---|---|---|
| 2024-07-25 | TSLA → 235 | 220 | Hit in 42 days |
| 2024-09-06 | NVDA → 146 | 108 | Hit in 62 days |
| 2024-09-28 | NIO → 9 / 14 | 7.70 | Missed; −60% at the worst |
| 2024-12-16 | NVDA → 155.55 ("ATH coming") | 134 | Hit after 192 days, after a −35% drawdown |
| 2025-02-09 | TSLA → 500 / 700 / 800 | 356 | Missed; peak 498.83 |
| 2025-07-02 | AMZN → 235 | 222 | Hit in 22 days |
| 2026-01-15 | S&P → 7015 | 6944 | Hit in 90 days, after a −9% dip |
| 2026-04-15 | S&P → 7500 / 8000 | 6978 | 7500 hit May 14; 8000 not yet reached |
| 2026-04-19 | META → 740 / 820 | 681 | 740 hit after 155 days and a −23% drawdown; 820 missed |
| 2026-04-30 | BTC → 85k / 90k | 76.3k | 85k hit after 144 days and a −24% drawdown; 90k missed |
| 2026-05-06 | S&P → 8300 (pinned) | 7365 | Open; peak 7845 |
| 2026-03-30 | BA May 15 $195 calls ("$100 grand") | 189 | Expired about $25 in the money (premium not posted) |
| 2024-12-09 | RVPH Dec 20 $5 calls | 2.12 | Expired out of the money |
| 2025-01-07 | RVPH "up 400%… to 8.50" | 2.00 | Missed; −88% at the worst within a year |
| 2025-01-22 | RVPH "$3–5 in 60 days" | 1.94 | Missed; peak 2.15 |
| 2025-01-08 | ALGS "see you at 42" | 38.39 | Missed; −90% at the worst |
| 2024-12-05 | Biotech Basket: ALGS, BIVI, CLDI, NXL | — | −94% to date (S&P +27%) |
| 2025-01-31 | CLDI "best risk/reward in the market" | 0.76 | −99.6% to date |

Self-reported, not verifiable: ARM "240% in a day" (2026-03-25), AMZN "+100% in 6 hrs"
(2026-05-13), IBM "+150%" (2026-05-21), CRWD "all three targets" (2026-03-30), CRDO "$50 in 3 days".
TipRanks lists 4 ratings (NVDA, AMZN; late 2023 to early 2024) with 100% success, too few to mean
anything. The only loss they posted publicly: a −$24K day (2024-03-27). One undated post says an
account went from 30K to 700K "in one day" and was closed at 580K after breaking their own rules.

## Appendix B: reference posts

- Method: [Banana325SMA how-to](https://x.com/Banana3Stocks/status/1809608448969232453),
  [alert format (AMD)](https://x.com/Banana3Stocks/status/2037230514718040499),
  [1DTE lotto (TSLA)](https://x.com/Banana3Stocks/status/2075218750236950543),
  [$100K BA calls](https://x.com/Banana3Stocks/status/2038643430092673419),
  [NAMO breadth](https://x.com/Banana3Stocks/status/1742734085142048896)
- Challenge account: [~$200K, Jul 2025](https://x.com/Banana3Stocks/status/1942969344423510222),
  [~$220K, Oct 2025](https://x.com/Banana3Stocks/status/1975618820711821481)
- Claims: [ARM](https://x.com/Banana3Stocks/status/2036800027994321298),
  [CRWD](https://x.com/Banana3Stocks/status/2038618643597324692),
  [AMZN](https://x.com/Banana3Stocks/status/2054650136102662279),
  [IBM](https://x.com/Banana3Stocks/status/2057498347074981952),
  [CRDO](https://x.com/Banana3Stocks/status/2044044980260614423)
- Loss admission: [−$24K day](https://x.com/Banana3Stocks/status/1773116019550368122)
- Third party: [TipRanks profile](https://www.tipranks.com/experts/bloggers/banana3)

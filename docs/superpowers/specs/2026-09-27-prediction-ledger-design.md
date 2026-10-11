# Prediction Ledger — Design Spec

**Date:** 2026-09-27
**Status:** awaiting approval
**Implements:** `PRODUCTION_MASTER_PLAN.md` §5.5 (The Ledger & Learning Loop) and
`AGENT_LEARNING_PLAN.md` (Recommendation schema), extended to cover broker-report
predictions as well as the agent's own.

---

## 1. Goal

Give the system a memory it can be graded against.

Today every `demo.py` run produces a prose report and forgets it. No row anywhere
records *"on 2026-09-27 I said buy HINDALCO at ₹640"*, so nothing can ever be checked.
`market.db` has no `recommendations` table; `traces/` holds 5 files of prose.

This spec builds the spine: **write down every prediction with a date and a price,
then later go back and measure what actually happened.**

### Success criteria

1. Every pipeline run writes exactly one structured prediction row, or fails loudly.
2. Broker reports already in `Broker_Reports/` can be backfilled as prediction rows.
3. A nightly job computes forward abnormal returns at +1/+5/+20/+60 trading days.
4. `scripts/ledger.py --scorecard` answers *"of the calls made 60+ days ago, what
   share beat the market?"* — for agent calls and broker calls separately.
5. Nothing in the ledger is ever silently rewritten.

### Non-goals (explicitly deferred)

- **The retro/learning loop.** `PRODUCTION_MASTER_PLAN.md` Sprint 4 is explicit:
  *"Do NOT build the retro loop yet — first accumulate data to retro over."*
- **Backtesting / walk-forward simulation.** Needs a populated ledger first.
- **Real money, broker integration, order placement.** Paper only.
- **Statistical inference** (purged CV, deflated Sharpe). Those belong to the
  analysis phase, not the recording phase.

---

## 2. Research findings that drive this design

Every number below was measured against live data on 2026-09-27, not assumed.

### 2.1 The past gets rewritten (look-ahead bias)

Fetching the same historical date twice gives different prices, because Yahoo applies
dividends and splits *backwards*:

| Stock | Close on 2023-09-25 (raw) | Same day, adjusted | Gap |
|---|---|---|---|
| ITC | 426.19 | 379.91 | **−10.9%** |
| INFY | 1474.15 | 1341.13 | **−9.0%** |
| HINDALCO | 469.35 | 460.87 | −1.8% |
| RELIANCE | 1170.22 | 1156.28 | −1.2% |

**Consequence:** an entry price that is re-fetched later is not the price that was on
the screen. A ledger that re-derives its own entry prices will quietly drift by ~10%
on dividend-paying stocks and report returns that never happened.

**Design rule:** the entry price is **snapshotted at prediction time and never
recomputed**. It is stored as the raw (unadjusted) close — the number a human saw.

### 2.2 Symbols disappear (survivorship bias)

`TATAMOTORS.NS` returned `404 — possibly delisted` during testing. It is gone because
of the Tata Motors demerger into TMCV/TMPV, which this project already knows about
(see `ticker-disambiguation` memory).

**Consequence:** any design that stores only a symbol and re-fetches everything later
loses its history whenever a company demerges, delists or is renamed.

**Design rule:** the prediction row carries its own price and benchmark snapshot. A
vanished symbol degrades one row's future outcomes; it never erases the past.

### 2.3 Trading days are not calendar days

Measured on RELIANCE from 2026-03-25:

| Horizon | Calendar days |
|---|---|
| +1 trading day | +2 |
| +5 trading days | +12 |
| +20 trading days | +34 |
| +60 trading days | +90 |

**Design rule:** horizons are counted in **trading days**, by indexing into the actual
price series. Never by date arithmetic.

### 2.4 Stock and benchmark trade on different days

Over 6 months: RELIANCE had 129 trading days, Nifty 500 had 125. Four days did not match.

**Design rule:** stock and benchmark series are **inner-joined on date** before any
return is computed. Unmatched days are dropped, not interpolated.

### 2.5 Raw return is not the signal

RELIANCE over one 20-day window:

```
RELIANCE      -1.71%     "down slightly"
Nifty 500     +5.99%
ABNORMAL      -7.71%     "badly underperformed"
```

**Design rule:** the ledger's headline number is the **abnormal return**
(stock − benchmark). Raw return is stored too, but never used for grading.

### 2.6 Market-adjusted is good enough

[Event-study literature](https://www.eventstudytools.com/expected-return-models)
finds that for **short event windows**, the market model (beta-adjusted regression over
a 120-day estimation window) and the simpler market-adjusted model perform similarly;
the real gains come from handling event clustering and variance estimation.

**Design rule:** use the **market-adjusted model** — `abnormal = r_stock − r_benchmark`.
No beta estimation, no estimation window. This removes a whole subsystem for no
measurable loss at our horizons. Revisit only if horizons extend past ~6 months.

### 2.7 Most Indian sector indices are unusable on this feed

Tested over one month:

| Index | Rows returned |
|---|---|
| `^CRSLDX` Nifty 500 | 23 ✅ |
| `^NSEBANK` Bank | 23 ✅ |
| `^CNXIT` IT | 23 ✅ |
| `^CNXPHARMA` Pharma | 23 ✅ |
| `^CNXAUTO`, `^CNXFMCG`, `^CNXMETAL`, `^CNXENERGY`, `^CNXREALTY`, `^CNXINFRA`, `^CNXMEDIA`, `^CNXPSUBANK` | **1 each** ❌ |

**Design rule:** Nifty 500 is the **primary benchmark for every prediction**. A sector
benchmark is computed **additionally** only for the three sectors that have a working
index. The benchmark symbol used is stored on the outcome row, so a reader always knows
what a number was measured against. `SECTOR_BENCHMARKS` lives in `outcomes.py` as a `rulebook -> index symbol` lookup keyed
on `sectors.profile_for()`, seeded with the three that work (`financials -> ^NSEBANK`,
`technology -> ^CNXIT`, `healthcare_pharma -> ^CNXPHARMA`). It is expected to grow as
data sources improve — it is not a promise that every sector has one. A rulebook absent
from the lookup simply gets no sector column, which is a normal outcome, not an error.

---

## 3. Architecture

```
  demo.py run ─────┐
                   ├──> predictions table (append-only)
  broker PDFs ─────┘          │
                              │  nightly
                              ▼
                        outcomes table
                     (one row per prediction × horizon)
                              │
                              ▼
                    scripts/ledger.py --scorecard
```

### 3.1 Why one table for both sources

The agent and the brokers make the *same kind of statement*: "this symbol, on this
date, at this price, will do well or badly." The machinery that grades them —
fetching prices, aligning the benchmark, counting trading days, computing abnormal
returns, labelling — is **identical**. Only the fields describing *why* differ.

Splitting into `recommendations` and `report_events` would mean writing and testing
that machinery twice. Instead: one spine, one `source` column, one JSON `payload` for
the fields unique to each source.

The cost of this choice is one column (`source`) and one indirection (`payload`).
The alternative cost is a duplicated outcome tracker.

### 3.2 Why outcomes are a separate table

Outcomes arrive at different times. The +1d outcome for a prediction made today is
knowable tomorrow; its +60d outcome is knowable in three months. They are also
**recomputable** — if the abnormal-return method changes, outcomes are rebuilt while
predictions stay untouched.

One row per `(prediction_id, horizon_days)`. Predictions are immutable; outcomes are
derived and may be recomputed freely.

---

## 4. Data model

### 4.1 `predictions` (append-only)

| Column | Type | Notes |
|---|---|---|
| `prediction_id` | TEXT PK | `ULID`-style: `{as_of}-{symbol}-{hash8}`, sortable by time |
| `source` | TEXT | `agent` \| `broker` |
| `symbol` | TEXT | NSE symbol, no suffix |
| `exchange` | TEXT | `NSE` |
| `as_of_date` | TEXT | ISO date. **The date the call could first have been acted on.** |
| `as_of_ts` | TEXT | ISO timestamp of the write |
| `direction` | TEXT | `bullish` \| `bearish` \| `neutral` — normalised across sources |
| `conviction` | INTEGER | 1–5 |
| `horizon` | TEXT | `1w` \| `1m` \| `3m` |
| `entry_price` | REAL | **Raw close as known on `as_of_date`. Never recomputed.** |
| `entry_price_asof` | TEXT | The date `entry_price` was actually taken from |
| `invalidation_price` | REAL NULL | "I am wrong if it closes past this" |
| `sector` | TEXT NULL | For benchmark selection |
| `rulebook` | TEXT NULL | From `sectors.profile_for()` |
| `payload` | TEXT | JSON — source-specific fields |
| `provenance` | TEXT | JSON — `trace_id` + model versions, or broker + file |
| `created_at` | TEXT | |

**`payload` for `source='agent'`:** `thesis`, `catalysts[]`, `risks[]`, `query`
**`payload` for `source='broker'`:** `broker`, `analysts[]`, `rating`, `prev_rating`,
`target_price`, `prev_target_price`, `valuation_multiple`, `estimates{}`,
`earnings_quality_flag`, `report_type`

**Immutability:** enforced in code — `save_predictions()` uses plain `INSERT`, never
`INSERT ... ON CONFLICT DO UPDATE`. A duplicate `prediction_id` raises.

### 4.2 `outcomes` (derived, recomputable)

| Column | Type | Notes |
|---|---|---|
| `prediction_id` | TEXT | FK |
| `horizon_days` | INTEGER | 1, 5, 20, 60 — **trading days** |
| `outcome_date` | TEXT | The actual trading date reached |
| `stock_return_pct` | REAL | From adjusted closes |
| `benchmark_symbol` | TEXT | `^CRSLDX`, or the sector index |
| `benchmark_return_pct` | REAL | |
| `abnormal_return_pct` | REAL | **stock − benchmark. The graded number.** |
| `sector_benchmark_symbol` | TEXT NULL | Only where one exists |
| `sector_abnormal_return_pct` | REAL NULL | |
| `label` | TEXT | `hit` \| `miss` \| `flat` \| `invalidated` \| `open` \| `unavailable` |
| `computed_at` | TEXT | |

PK: `(prediction_id, horizon_days)`.

### 4.3 Labelling rules

Applied to `abnormal_return_pct` against the broad benchmark:

| Condition | Label |
|---|---|
| Horizon not yet reached | `open` |
| Price crossed `invalidation_price` before the horizon | `invalidated` |
| `direction=bullish` and abnormal > +`FLAT_BAND` | `hit` |
| `direction=bullish` and abnormal < −`FLAT_BAND` | `miss` |
| `direction=bearish`, signs reversed | |
| `\|abnormal\| <= FLAT_BAND` | `flat` |
| `direction=neutral` | `flat` if within band, else `miss` |
| Price data missing (delisted, halted) | `unavailable` |

`FLAT_BAND = 2.0` (percentage points). Rationale: below ~2% over 20 days, the result
is indistinguishable from noise and bid-ask friction; calling it a hit would flatter
the record. The band is a named constant so it can be tuned once and everywhere.

`invalidated` is checked on **daily closes between entry and horizon**, not just the
endpoint — an invalidation that was hit and recovered still counts as hit.

---

## 5. Modules

| File | Responsibility | ~lines |
|---|---|---|
| `pennystock/data/ledger.py` | `Prediction` dataclass, both tables, save/load/query. No network, no maths. | 200 |
| `pennystock/data/outcomes.py` | Price + benchmark fetching, trading-day arithmetic, abnormal return, labelling. | 220 |
| `pennystock/agents/recommendation.py` | `Recommendation` pydantic schema + extraction from the analyst's report. | 130 |
| `scripts/track_outcomes.py` | The nightly job. CLI. | 90 |
| `scripts/ledger.py` | Inspect the ledger: `--scorecard`, `--recent`, `--explain <id>`. | 120 |
| `tests/test_ledger.py` | Schema, immutability, round-trip, id generation. | ~180 |
| `tests/test_outcomes.py` | Trading days, alignment, abnormal return, every label branch. | ~220 |

Boundaries follow the existing house style (`metrics.py` = storage, `screener.py` =
logic, `sectors.py` = reference data). `ledger.py` never touches the network;
`outcomes.py` never writes predictions.

### 5.1 Integration points (files modified, not created)

- **`pennystock/pipeline.py`** — after the graph completes, extract a `Recommendation`
  and write one prediction row. Wrapped in try/except: **a ledger failure must never
  fail the user's analysis.** Logged loudly.
- **`pennystock/config.py`** — `ledger_enabled`, `ledger_horizons = (1, 5, 20, 60)`,
  `broad_benchmark = "^CRSLDX"`, `flat_band_pct = 2.0`.
- **`pennystock/data/maintenance.py`** — register `outcomes` as a dataset so
  `refresh_data.py --status` reports how many predictions are awaiting grading.

---

## 6. How the agent produces a structured call

The analyst writes prose today. Two candidate approaches were considered:

**(a) Force structured output on the analyst node.** Rejected — it would change the
analyst's whole job and risk degrading the report a human reads.

**(b) Add an extraction step after the report.** *Chosen.* A small, cheap call takes
the finished report and fills the `Recommendation` schema. The report stays prose for
the human; the ledger gets structure.

The mechanism is already proven three times in this codebase
(`nodes.py:76`, `query_expand.py:29`, `build_eval_set.py:118`):

```python
llm.with_structured_output(Recommendation, method="json_schema")
```

**Failure handling:** if extraction fails or returns an unparseable object, write a
prediction with `direction='neutral'`, `conviction=0` and
`payload.extraction_failed=True`. Never guess a direction. A run that produced no
gradeable call is itself a fact worth recording.

---

## 7. Point-in-time discipline

The single rule that makes this ledger trustworthy:

> **A prediction row may only contain information that existed on `as_of_date`.**

Enforced by:

1. `entry_price` is captured at write time, never re-derived (§2.1).
2. `as_of_date` for broker predictions is the **date you could have acted**, not the
   report date. Trendlyne publishes after the broker's clients have traded, so this is
   the *next trading day's* open by default.
3. Outcomes are written only when the horizon date is `<= today`, never projected.
4. `outcomes` rows carry `computed_at`, so a rebuild is distinguishable from an original.

---

## 8. Error handling

| Failure | Behaviour |
|---|---|
| Ollama down during extraction | Prediction saved with `extraction_failed`, run unaffected |
| Ledger write fails | Logged at WARNING, analysis still returned to the user |
| Symbol delisted at outcome time | `label='unavailable'`, row kept, not deleted |
| Benchmark fetch fails | Outcome skipped, retried next run. Never fall back to raw return silently |
| Horizon not yet reached | `label='open'`, recomputed on the next run |
| Duplicate `prediction_id` | Raises — immutability violation is a bug, not a warning |

---

## 9. Testing

Following the house pattern: every test pins a behaviour that was **measured**, with
the reason in the docstring.

- Trading-day arithmetic: +20 trading days is not +20 calendar days (§2.3)
- Benchmark alignment drops unmatched days rather than interpolating (§2.4)
- Abnormal return uses the RELIANCE worked example: −1.71% raw, −7.71% abnormal (§2.5)
- Entry price is never recomputed — a re-save with a changed feed price must not alter
  the stored row (§2.1)
- Every label branch, including `invalidated` triggering on an intermediate close
- A delisted symbol yields `unavailable`, not a crash (§2.2)
- Extraction failure still writes a row
- Immutability: a duplicate id raises

Network is mocked in tests; one live smoke test sits in `scripts/`, not `tests/`.

`scripts/track_outcomes.py` and `scripts/ledger.py` are **not** unit-tested, matching
the existing house convention (`screen.py`, `ingest_metrics.py` and the other scripts
have no test files either). Their logic lives in `outcomes.py` and `ledger.py`, which
are tested; the scripts are thin CLI wrappers. `track_outcomes.py --dry-run` is the
manual check.

---

## 10. Open decisions for the reviewer

1. **Broker `as_of_date`.** Spec defaults to *the next trading day after the report
   date*, because Trendlyne publishes after institutional clients have traded. The
   conservative alternative is +2 days. This choice materially affects measured edge.
2. **Backfill scope.** Ledger will accept the 12 reports in `Broker_Reports/` now. The
   earlier recommendation was to backfill 60–80 names deeply rather than broadly;
   that is a data-collection decision, not a schema one.
3. **`FLAT_BAND = 2.0`** percentage points — tunable, currently a judgement call.

---

## 11. What this explicitly does not prove

A populated ledger does not mean the system works. It means the question becomes
answerable. With ~1 run/day, the agent's own track record needs **12+ months** before
any conclusion is safe; broker predictions get there sooner only because they can be
backfilled.

The honest output of this subsystem, for a long time, will be *"not enough data yet"* —
and printing that truthfully is the point.

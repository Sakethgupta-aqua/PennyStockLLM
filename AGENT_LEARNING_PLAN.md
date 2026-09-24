# Agent Learning Plan — making PennyStockLLM improve from the market

Companion to `ENTERPRISE_ROADMAP.md`. That doc makes the system *enterprise-grade*;
this one makes it *self-improving*.

## The honest foundation

LLM agents cannot "learn to predict the market" — nobody reliably can, least of all
in penny stocks (your own corpus is largely about why: manipulation, pump-and-dumps,
illiquidity). What agents CAN learn:

1. **Better evidence** — fresher data, more sources, less hallucination.
2. **Better process** — structured theses, explicit risks, consistent scoring.
3. **Accountability** — every recommendation logged, priced, and graded later
   against what the market actually did.

The system improves because it **remembers what it said, checks what happened,
and feeds the lesson back**. That loop — not a smarter model — is the learning.

---

## The core architecture: the Learning Loop

```
                    ┌────────────────────────────────────────────┐
                    ▼                                            │
  [Pipeline run] → Recommendation Ledger (Postgres)              │
                    every rec: date, ticker, action, conviction, │
                    entry price, thesis, risks, invalidation px  │
                    ▼                                            │
  [Nightly job]  → Outcome Tracker                               │
                    fetch prices at +1d/+1w/+1m/+3m → label recs │
                    (hit / miss / invalidated / open)            │
                    ▼                                            │
  [Weekly job]   → Retro Analyst (new agent)                     │
                    "Of last month's calls: what worked, what    │
                    didn't, what pattern of errors repeats?"     │
                    ▼                                            │
  Feedback injection ────────────────────────────────────────────┘
    a) Track-record memo → prepended to Analyst context each run
    b) Failure patterns → prompt/rubric updates (in git)
    c) Logged tool-call failures → next penny-llama fine-tune dataset
```

**Key design rule: recommendations must be STRUCTURED, not prose.** A pydantic
schema the Analyst must fill:

```python
class Recommendation(BaseModel):
    ticker: str; exchange: str
    action: Literal["buy","avoid","watch"]
    conviction: int          # 1-5
    horizon: Literal["1w","1m","3m"]
    thesis: str              # 2-3 sentences
    catalysts: list[str]
    risks: list[str]
    invalidation_price: float  # "I'm wrong if it closes below X"
```

Without this, outcomes can't be graded and nothing can learn. This schema IS the
learning system.

---

## Per-agent design (who learns what, and how)

### 1. Supervisor — do NOT let it learn
Routing must be boring and deterministic. Keep the general model, tighten rules in
code (not prompt) where possible. Reliability > cleverness. (You've already seen
what a degraded router does: the 9× analyst loop.)

### 2. MarketDataAgent (penny-llama) — learns from its own logs
- **Widen its senses (tools):** `get_price_history` (OHLCV for trend/volatility),
  `get_fundamentals` (P/E, debt, promoter holding %), `get_news_headlines`.
- **Harvest real usage:** log every tool call (args + success/error) to the ledger.
  Misfires become new training rows. When ~100 real examples accumulate →
  fine-tune v3 with the exact pipeline you already own (dataset → LoRA → fuse →
  GGUF → Ollama). Your synthetic data bootstrapped it; real traffic perfects it.

### 3. FinancialAnalystAgent — learns from its track record
- **Structured output** (the schema above) — enforced via `with_structured_output`.
- **Fresh retrieval:** nightly ingest of news/filings for tracked tickers into a
  second vector collection ("live context") alongside the book ("timeless wisdom").
  RAG then draws on both: principles + current events.
- **Track-record memory:** each run, prepend a short memo:
  "Your last 20 calls: 9 hit, 7 missed, 4 open. Recurring error: overweighting
  volume spikes in illiquid names (3 misses). Compensate accordingly."
  This is genuine in-context learning — no retraining needed.

### 4. EvaluatorAgent (Prometheus) — learns from outcomes
- Keep as pre-publication gate (faithfulness/relevance, as today).
- **Outcome-aware rubric upgrades:** when the Retro Analyst finds a recurring
  failure ("theses citing 'high retail interest' underperform"), add it to the
  rubric as a red flag. The judge becomes the immune system that remembers
  past infections.

### 5. NEW — TechnicalAgent
Computes indicators from `get_price_history` (SMA/EMA, RSI, volume profile,
52-wk position) with plain pandas — deterministic math the LLM narrates, not
invents. LLMs are bad at arithmetic; never let them compute indicators.

### 6. NEW — SentimentAgent (= Roadmap Phase 6 FinBERT, now with a purpose)
Your fine-tuned FinBERT scores news headlines → a sentiment feature the Analyst
cites. Its learning loop: retrain when sentiment-vs-outcome correlation drifts.

### 7. NEW — RiskManagerAgent (the final gate)
Rule-based first, LLM-assisted second. Checks every "buy" before it enters the
ledger: liquidity (volume × price floor), volatility band, red-flag screen from
the book's pump-and-dump heuristics, max conviction sizing. Can VETO. The most
valuable agent in penny-stock land.

---

## Management & operations

| Cadence | Job |
|---|---|
| Nightly | price/news ingest → outcome labels update |
| Weekly | Retro Analyst report → track-record memo refresh |
| Monthly | eval suite re-run (RAGAS + tool-call accuracy); prompt/rubric tuning PR |
| Quarterly | consider penny-llama v(n+1) fine-tune from harvested logs |

- **Version everything:** prompts and rubrics live in git (a prompt change is a PR
  with eval results); models are versioned in Ollama/MLflow; the ledger records
  which versions produced each rec — so you can A/B agent versions on real outcomes.
- **Kill-switch metrics:** if hit-rate or judge-pass-rate drops N points after a
  change → revert (your Phase 4 CI eval gates, applied to finance).

## Guardrails (non-negotiable)

- **Paper only.** The ledger is a research journal, not an order book. No
  auto-trading; a human reads every recommendation.
- Outputs are research hypotheses with explicit invalidation prices — never
  financial advice. Penny stocks add manipulation risk your own book documents;
  the RiskManager's veto exists for exactly that.

## Build order (maps to the roadmap)

1. **Now (with Phase 3 Postgres):** Recommendation schema + ledger + nightly
   outcome tracker. *The single highest-value piece — start measuring.*
2. **Phase 3.5:** price-history & news tools + TechnicalAgent + live-context
   vector collection.
3. **Phase 4:** retro loop + track-record memo + outcome-aware rubric updates.
4. **Phase 6:** SentimentAgent (FinBERT) + RiskManagerAgent + first
   log-harvested penny-llama v3.

Rule of thumb: **measure before you optimize** — the ledger comes first; every
other improvement is judged by whether it moves the ledger's numbers.

---

# Full-Market Expansion (all NSE/BSE stocks, not just penny)

Scope change decided July 2026. Key consequences:

1. **Memorize → Lookup.** penny-llama cannot (and should not) memorize ~6,000
   company→ticker mappings. Build a `resolve_symbol` lookup backed by the official
   NSE/BSE equity master lists (fuzzy name matching); embed it inside
   `get_indian_stock_data` so any company name or approximate ticker resolves
   correctly. The fine-tuned model's job narrows to what fine-tuning is FOR:
   reliable tool-calling behavior, not knowledge storage.
2. **Universe table** (Postgres): every listed symbol + sector + market-cap bucket
   (AMFI classification: large = top 100, mid = 101–250, small = 251+, plus a
   penny/micro flag). Every agent becomes bucket-aware.
3. **Full-market EOD ingest:** NSE bhavcopy (official free daily file) → one
   download covers the whole market; yfinance stays for on-demand live quotes.
4. **RAG becomes multi-collection:** penny book = one collection ("penny wisdom"),
   general investing/fundamentals corpus = second, live news/filings = third.
   Retrieval filters by the stock's bucket.
5. **New ScreenerAgent:** SQL over universe+prices ("small-cap IT stocks with
   rising volume") — the discovery engine for the whole market.
6. **RiskManager rules become bucket-specific** (penny rules stay strict;
   large-cap rules differ).
7. Fine-tune v3 goal shifts: tool-chaining + passing unfamiliar names through
   verbatim — NOT more ticker memorization.

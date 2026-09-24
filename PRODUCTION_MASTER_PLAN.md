# PennyStockLLM — Production Master Plan

> **Status:** Canonical reference. This document supersedes `ENTERPRISE_ROADMAP.md` and
> `AGENT_LEARNING_PLAN.md` as the single source of truth for *what we are building and in
> what order*. Those two remain useful appendices (job-skill mapping; learning-loop design
> rationale) but this file is the plan of record.
>
> **Version:** 1.0 · **Date:** 2026-07-29 · **Owner:** Saketh · **Cadence assumption:** ~10 focused hrs/week
> **Horizon:** ~6 months to a defensible production v1 (compresses with more hours).
>
> **Author's stance:** Written as a senior principal engineer would write a plan he has to
> defend in a design review. It is blunt on purpose. Where a claim is grounded in external
> research it is cited in §14. Where it is my judgment, it says so.

---

## 0. How to use this document

- **§1–§4** are the *why* — read once, re-read when you lose the plot or feel like chasing a shiny feature.
- **§5** is the target architecture — the thing we are converging toward.
- **§6** is the engineering bar — the definition of "done" that applies to every ticket.
- **§7 is the plan of record** — epics → sprints → stories with acceptance criteria. This is where you live day to day.
- **§8–§13** are the guardrails: metrics, risks, stack decisions, roles, glossary.
- Treat every sprint's **"Do NOT do"** line as binding. Scope creep is the failure mode that kills this project, not lack of ideas.

---

## 1. North Star & Product Definition

### 1.1 The one-sentence definition
**PennyStockLLM is a production-grade, accountable research & decision-support copilot for Indian small/micro/penny-cap equities (NSE/BSE) — paper-only, human-in-the-loop, and graded by the market after the fact.**

### 1.2 What it IS
- A system that produces **structured, falsifiable research theses** on Indian small-caps (not prose essays).
- A system that **remembers every call, tracks the outcome, and grades itself** — the accountability loop is the product, not a feature.
- A system whose credibility rests on **honest, look-ahead-free evaluation**, not on a cherry-picked backtest.
- **Indian-market-first.** This is the moat (see §4). Every US-centric repo treats India as an afterthought; we treat it as the whole point.

### 1.3 What it is NOT (write this on the wall)
- ❌ **Not an auto-trader.** No order placement, ever, in v1. Paper only.
- ❌ **Not an alpha-generation engine.** We do not claim to beat the market (see §2.1). We claim to be a rigorous, accountable analyst that a human uses to make *better-informed* decisions.
- ❌ **Not financial advice.** Legally this matters in India (see §2.2). Outputs are research hypotheses with explicit invalidation prices, framed as education/research.
- ❌ **Not a chatbot.** Conversational polish is the last 5%, not the product.

### 1.4 The dual goal, reconciled
The `ENTERPRISE_ROADMAP.md` framing ("get hired as an AI Engineer") and this framing ("build a production system") are **the same path if we build the hard, real things** — a data spine, an accountability ledger, a look-ahead-free backtest, an eval-gated CI pipeline, an observable serving stack. Those are exactly what a senior interviewer respects *and* what a real product needs. The résumé takes care of itself if the system is real. **We optimize for the real system; the job is a side effect.**

---

## 2. The Non-Negotiable Reality (read before every planning session)

### 2.1 LLMs do not beat the market, and honest benchmarks prove it
This is not pessimism; it is the current state of the evidence, and it defines our entire strategy.

- **StockBench** (contamination-free, evaluated Mar–Jun 2025, *after* model training cutoffs, scored on cumulative return / max drawdown / Sortino): **most state-of-the-art LLM agents fail to beat a simple buy-and-hold baseline.** A few edge it in risk-adjusted terms; none do so reliably across regimes. [§14]
- **Look-ahead bias / memorization** contaminates almost every impressive published LLM-trading backtest: GPT-class models can recall historical prices within their training window with <1% error, so a backtest overlapping that window is *retrieval dressed up as prediction*. [§14]
- Consequence: **any claim we make about performance must come from a walk-forward, point-in-time backtest on data after our models' training cutoff.** Anything else is self-deception (see §5.6).

**Therefore our thesis is not "predict better." It is "be accountable better."** We win by:
1. **Better evidence** — fresher, sourced, deterministic where math is involved.
2. **Better process** — structured theses, adversarial debate, explicit risk gates.
3. **Accountability** — every call logged, priced, and graded by reality later.

`AGENT_LEARNING_PLAN.md` already states this. It is the most correct idea in the repo. This plan's job is to *build* it.

### 2.2 SEBI: the legal reality most GitHub projects ignore (and that a production Indian bot cannot)
This is a real, non-optional constraint for anything India-facing that touches recommendations.

- SEBI operates an **AI Accountability Framework: using AI does not reduce your responsibility, it increases it.** You are fully legally accountable for AI-generated output. [§14]
- **"Educational purposes only" disclaimers do not excuse unregistered advisory activity.** If the system issues buy/sell/target recommendations to *other people*, that is regulated activity requiring SEBI Research Analyst (RA) or Investment Adviser (IA) registration. [§14]
- SEBI's proposed **AI Governance Framework** for regulated entities expects: model explainability, audit trails, fallback plans, continuous monitoring, and periodic accuracy reporting. [§14]

**What this means for us (design constraints, not legal advice — consult a professional before any public launch):**
1. **v1 is single-user, personal research.** You analyzing stocks for yourself is not regulated advisory. Keep it that way until/unless you register.
2. **Build the audit trail and explainability from day one** (the Ledger does most of this) — it is both good engineering and the exact thing a regulator wants to see.
3. **If you ever expose it to others**, the compliance epic (§7 Epic I) — disclosures, disclaimers, AI-usage disclosure, registration — becomes a hard gate *before* launch, not an afterthought.
4. **Never** wire in order execution without a separate, deliberate legal review. This is why "no auto-trading" is architectural, not just cautious.

---

## 3. Current-State Audit (brutally honest, verified against the repo on 2026-07-29)

### 3.1 What actually exists and works
| Component | Reality |
|---|---|
| 4-agent LangGraph pipeline | ✅ Works end-to-end: Supervisor (`llama3-groq-tool-use:8b`) → MarketDataAgent (`penny-llama`) → Analyst (`qwen3.5:35b`) → Prometheus-2 judge |
| Hybrid RAG | ✅ Genuinely good: bge-m3 dense + BM25s sparse + custom RRF over a 700-pg book |
| `resolve_symbol` fuzzy matcher | ✅ Works; 2,405 NSE symbols in `market.db` (`universe` table), min_score 88 |
| QLoRA fine-tune → GGUF → Ollama | ✅ `penny-llama` v2 deployed; v3 dataset changed, **not yet retrained/redeployed** |
| Doc ingestion pipeline | ✅ `scripts/ingest_documents.py` is solid (OCR fallback, idempotent chunk IDs) |

### 3.2 The gaps that make "production" a fantasy today
| Gap | Severity | Note |
|---|---|---|
| **No historical price data** | 🔴 Critical | `get_indian_stock_data` returns a *single-day snapshot* (price/prevclose/high/low/vol/mktcap). The demo query asks to "assess momentum" — impossible from one day. The docstring says "historical" and it is a lie. |
| **No accountability Ledger** | 🔴 Critical | `market.db` has *only* the `universe` table. Zero recommendations stored, zero outcomes tracked. The entire "learning loop" is unbuilt. |
| **No backtesting / point-in-time eval** | 🔴 Critical | Nothing. We literally cannot state whether the system helps. |
| **No numeric evals** | 🔴 Critical | `test_adapter.py` is a 5-ticker smoke test. The "97% accuracy" in the roadmap is a *template placeholder*, not a measurement. |
| **Fragile data source** | 🟠 High | `yfinance .info` is scraped, rate-limited, breaks on Yahoo HTML changes — industry consensus: unfit for continuous/production use. [§14] |
| **Fine-tune is redundant in-pipeline** | 🟠 High | With the resolver mapping names→tickers, `penny-llama`'s only job is to copy text into a field. A prompted 7B + structured output matches it. Keep the fine-tune as an asset for the vLLM multi-LoRA future (§5.2), not as a load-bearing prompt-replacement. |
| **Judge measures the wrong thing** | 🟠 High | Prometheus-2 scores faithfulness/relevance to retrieved text, *not* financial correctness or calibration. A 5/5 report can be useless. Real judge = the market, later (the Ledger). |
| **BSE unvalidated** | 🟡 Med | Resolver loads NSE only; "BSE" path just appends `.BO`. Half the advertised coverage is untested. |
| **Single-stock, single-shot** | 🟡 Med | No screening, ranking, or portfolio view. |
| **Notebook, not a package** | 🟡 Med | Not importable, not tested, not deployable. |

**Verdict:** We have a strong *demo of a research assistant for one stock*, plus excellent thinking that is 0% built. The plan below builds the excellent thinking first.

---

## 4. Competitive Landscape & Positioning (grounded, mid-2026)

| Project | Scale | Strength | What we learn / take |
|---|---|---|---|
| [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) | ~87k★, paper-backed | **Firm-structured agents + bull/bear debate + risk oversight**; backtesting | Adopt the **debate topology** (§5.3) — it's their most elegant, bias-reducing idea |
| [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) | ~51k★ | 13 investor-persona agents; **backtester**; portfolio/risk manager; Ollama-local | Adopt the **backtester-first discipline** and portfolio/risk roles |
| [AI4Finance/FinRobot](https://github.com/ai4finance-foundation/finrobot) + FinGPT | Large | **Valuation via deterministic code paths with provenance** (DCF/WACC in code, not LLM) | Adopt: **LLMs narrate, code computes.** Never let the LLM do arithmetic |
| microsoft/qlib, OpenBB | Very large | Production data/quant infra | Reference for data-layer rigor |

**Where we are behind:** backtesting and portfolio/risk agents — we have neither; even hobby forks do.
**Where we are ahead:** serious hybrid retrieval over a domain corpus, and an honestly-scoped fine-tune with a real debugging narrative. Most star-heavy repos are prompt-wired API calls.

**Our moat (defend it):** **None of the famous projects are Indian-small/micro-cap-first.** Indian micro-caps have worse data, thinner coverage, and — per your own book corpus — *structural manipulation risk* (pump-and-dump). A copilot built specifically for that adversarial, under-served niche is not a commodity. "Another US large-cap LLM trader" is. **Stay in the niche where you have an edge.**

---

## 5. Target Production Architecture (the end state we converge toward)

```
                       ┌────────────────────────────────────────────────────────────┐
                       │                      COMPLIANCE / AUDIT LAYER                │
                       │  AI-usage disclosure · disclaimers · full audit trail ·      │
                       │  human-in-the-loop gate · no order execution                 │
                       └────────────────────────────────────────────────────────────┘
   User query
       │
       ▼
 ┌───────────┐   ┌──────────────────────── AGENT ORCHESTRATION (LangGraph) ─────────────────────────┐
 │ FastAPI   │   │  Supervisor (deterministic router)                                                │
 │ (SSE)     │──▶│    │                                                                               │
 └───────────┘   │    ├─▶ Analyst Team (parallel):  Fundamentals · Technical(det. math) · News/Sentiment · RAG-Context │
                 │    ├─▶ Research Team:  Bull Researcher  ⇄  Bear Researcher  (N-round debate)         │
                 │    ├─▶ Risk Manager (rule-based veto, bucket-specific)                               │
                 │    └─▶ Portfolio/Decision (emits STRUCTURED Recommendation)                          │
                 └───────────────────────────────────────────────────────────────────────────────────┘
                        │ tools                        │ writes
       ┌────────────────┼───────────────┐              ▼
       ▼                ▼               ▼      ┌──────────────────────┐
 ┌───────────┐   ┌────────────┐  ┌──────────┐ │  RECOMMENDATION LEDGER│──┐
 │ Data Layer│   │ RAG (multi-│  │ Technical│ │  (Postgres)           │  │
 │ PIT store │   │ collection │  │ (pandas) │ └──────────────────────┘  │
 │ vendor API│   │ + reranker)│  └──────────┘         │                 │
 └───────────┘   └────────────┘                       ▼                 │
       ▲                                       ┌──────────────┐         │
       │  nightly EOD ingest (bhavcopy + vendor)│ Outcome      │◀────────┘
       └───────────────────────────────────────│ Tracker      │  grades calls at +1d/1w/1m/3m
                                                └──────────────┘
                                                       │
                                     ┌─────────────────┼──────────────────┐
                                     ▼                 ▼                  ▼
                              Walk-forward       RAGAS evals        Retro Analyst
                              PIT Backtest       (CI gate)          (weekly memo → in-context learning)
                                     │
                                     ▼
                        Observability: Langfuse traces · MLflow experiments · Grafana ops
```

### 5.1 Data layer (the foundation everything else stands on)
- **Decision:** move off `yfinance` for anything load-bearing. For India:
  - **EOD / universe:** official **NSE bhavcopy** (free daily file) + BSE bhavcopy → own Point-in-Time store. This is the backbone.
  - **Intraday / live quotes / historical candles:** a **broker API**. Grounded options [§14]:
    | Provider | Data cost (approx) | Notes |
    |---|---|---|
    | **Angel One SmartAPI** | Free API | Good starting point; free is a real advantage for a personal project |
    | **Zerodha Kite Connect** | ~₹500/mo data (₹2000/mo w/ trading) | Most battle-tested, best docs; historical candles paid |
    | **Upstox** | Low per-order; API data | Solid streaming |
    | **Dhan** | ~₹499/mo data pack | Automation-focused |
  - **Recommendation:** start with **Angel One SmartAPI (free)** for historical/live candles + **NSE bhavcopy** for universe/EOD. Abstract behind a `MarketDataProvider` interface so the vendor is swappable (never hard-code a vendor again — that's how we got trapped in `yfinance`).
- **Point-in-Time (PIT) store is mandatory** (§5.6): store data *as it was known on each date*, keep **delisted symbols** (survivorship-bias mitigation), apply **corporate-action adjustments** and **reporting lags**. [§14]

### 5.2 Model serving (the honest migration path)
- **Now → v1:** Ollama is fine. It's single-user, great DX. Don't over-engineer.
- **When you need concurrency / an eval suite running many requests / a public demo:** migrate to **vLLM**. Grounded reasons [§14]: ~6× throughput at concurrency, PagedAttention, and — critically for us — **native multi-LoRA: hot-swap fine-tuned adapters on one base model without a restart.**
- **This reframes the fine-tune positively:** under vLLM, `penny-llama` (and future FinBERT-style adapters) become swappable adapters on a shared base — cheap to serve, versionable. Keep investing in the fine-tune *for that future*, not as an in-pipeline prompt replacement today.

### 5.3 Agent topology (adopt the TradingAgents pattern, adapted for Indian micro-caps)
Grounded on the TradingAgents 5-phase design [§14], deliberately trimmed for our scope:
1. **Analyst Team (parallel):** Fundamentals · **Technical (deterministic pandas math)** · News/Sentiment · RAG-Context (the book = "timeless wisdom"). *LLMs narrate; code computes.*
2. **Research Team — Bull ⇄ Bear debate (N rounds):** the single best idea to steal. Forcing an adversarial case *before* deciding is the cheapest, most effective bias reducer for single-perspective hallucination. Especially valuable in pump-and-dump territory where the bull case is often manufactured.
3. **Risk Manager (rule-based, bucket-specific, can VETO):** liquidity floor (volume×price), volatility band, the book's pump-and-dump red-flag screen, position sizing. For micro-caps this is the **most valuable agent** — it's the immune system.
4. **Portfolio/Decision agent:** emits the **structured Recommendation** (the Ledger schema). No prose-only output ever.
- **Supervisor stays deterministic and boring** (rules in code, not prompt cleverness). We already learned this the hard way (the 9× analyst loop from a degraded router).

### 5.4 RAG (upgrade to 2026 production standard)
Grounded on 2026 RAG best-practice research [§14]:
- **Multi-collection:** (a) penny/investing book = "timeless wisdom"; (b) **live context** = nightly-ingested filings/news for tracked tickers; (c) optionally a general-investing corpus. Retrieval filters by the stock's cap bucket.
- **Semantic chunking** (split where meaning shifts) — reported to beat fixed-size materially on financial corpora.
- **Cross-encoder reranker** (e.g., bge-reranker) on top of the hybrid retriever — **the single highest-ROI RAG upgrade** (+~17pp MRR@3 in studies). Rule of thumb: **retrieve 20 → rerank to 5 → send 3–5 to the LLM.**
- **Keep BM25** — for financial text (tickers, precise terms, numbers) lexical matching is a strength, not legacy. Our existing hybrid+RRF is already on the right track.
- **Do not** use HyDE for numeric financial QA — research shows it's counterproductive there.
- **Gate on RAGAS** (§5.7).

### 5.5 The Ledger & Learning Loop (the spine — build first)
The `AGENT_LEARNING_PLAN.md` design, now the #1 priority:
- **Structured `Recommendation`** (pydantic, enforced via structured output): `ticker, exchange, action{buy/avoid/watch}, conviction(1–5), horizon{1w/1m/3m}, thesis, catalysts[], risks[], invalidation_price`. **Without this schema nothing can be graded and nothing learns.**
- **Every rec persisted** with the model/prompt versions that produced it (so we can A/B on real outcomes).
- **Outcome Tracker (nightly):** fetch prices at +1d/+1w/+1m/+3m → label hit/miss/invalidated/open.
- **Retro Analyst (weekly):** "of last month's calls, what pattern of errors repeats?" → a **track-record memo prepended to the analyst each run** (genuine in-context learning, no retrain).
- **Learning ≠ a smarter model. Learning = remember → check → feed back.**

### 5.6 Evaluation & Backtesting (what separates us from the 50k-star toys)
This is the credibility engine. Grounded on quant backtesting methodology [§14]:
- **Point-in-Time data + walk-forward only.** At every simulated date, the agent sees *only* what was knowable then. No peeking. Carry the paper portfolio forward to the next rebalance.
- **Survivorship-bias mitigation:** include delisted names in the historical universe.
- **Contamination-free windows:** evaluate on periods *after* the models' training cutoffs (the StockBench discipline). Expect to lose to buy-and-hold; **report it honestly** — that honesty is the portfolio centerpiece and the thing a senior engineer respects.
- **Metrics:** cumulative return, max drawdown, Sortino/Sharpe, and — most importantly for us — **recommendation hit-rate and calibration** (does conviction 5 actually outperform conviction 2?).

### 5.7 RAG & agent evaluation gates (CI)
- **RAGAS thresholds (production, grounded [§14]):** faithfulness > 0.85, answer relevancy > 0.8, context precision > 0.75, context recall > 0.8. Below → CI fails.
- **Golden set:** ~50 hand-written Q/A from the book; regression suite forever.
- **Agent tool-call accuracy:** turn `test_adapter.py` into a scored 100-query suite → a real number.
- **CI gate:** a PR that drops any metric below threshold **fails the build.** This is the heart of LLMOps.

### 5.8 Observability & MLOps
Grounded [§14]:
- **Langfuse** (open-source, self-hostable, framework-agnostic, integrates RAGAS, has a native guardrail observation type) — trace every agent/tool/retrieval span, tokens, latency, cost.
- **MLflow** — log fine-tune runs (params/metrics/artifacts), register model versions (v1 "BANK bug" → v2 fix → v3).
- **Grafana/Prometheus** — ops metrics (request rate, latency percentiles, error rate).
- **Docker Compose** — one command brings up API + serving + Postgres + vector DB + Langfuse.

### 5.9 Compliance layer (see §2.2)
- AI-usage disclosure, research/education framing, explicit invalidation prices, no execution, human-in-the-loop, full audit trail (the Ledger *is* most of this). A hard gate before any multi-user exposure.

---

## 6. Engineering Standards — Definition of Done (applies to every ticket)

A story is **not done** until:
1. Code lives in the `pennystock/` package (not a notebook), typed, `ruff`-clean, `mypy`-clean.
2. It has tests; core logic (retrieval fusion, resolver, ledger grading, backtest engine) targets ~70%+ coverage.
3. Secrets come from config/env (`pydantic-settings`) — **never** hard-coded (we already had an HF-token incident).
4. If it touches an agent/prompt/rubric: the prompt is versioned in git, and the change ships with **eval numbers** (before/after).
5. If it touches data: the source is behind the `MarketDataProvider` interface and has a fallback.
6. It's observable: emits a Langfuse span (once §5.8 lands) or a structured log.
7. The end-to-end app still runs. **The pipeline is never broken at the end of a sprint.**

**Global rules:**
- One epic in flight at a time. The app always works.
- Numbers beat adjectives. Every improvement ships with a measured delta.
- Cut ruthlessly. The "Do NOT do" lines are binding.

---

## 7. The Scrum Plan (plan of record)

**Structure:** 3 phases (~quarters) → 12 two-week sprints. Each sprint has a **Goal**, **Stories w/ acceptance criteria**, **Definition of Done**, the **one metric it moves**, and an explicit **Do NOT do**. Epics (A–J) are the workstreams; sprints sequence them so differentiators land first and résumé-stunts land last.

### Epic map
| Epic | Name | Why it matters |
|---|---|---|
| A | Foundation & Truth | Can't build on a notebook or on lies about data |
| B | Data Spine (PIT) | Nothing works without real historical, survivorship-safe data |
| C | Ledger & Outcome Tracking | The product spine; the only real "learning" |
| D | Evaluation & Backtesting | The credibility engine; what beats the toys |
| E | Agent Redesign (analyst team + debate + risk + portfolio) | Better process, bias reduction |
| F | RAG Excellence | Reranker + semantic chunk + live context |
| G | Serving & API | Turn a script into a product surface |
| H | MLOps & Observability | LLMOps: traces, experiments, CI eval gates |
| I | Compliance & Safety | Legal reality for an India-facing rec system |
| J | Cloud & Scale | Last, optional, time-boxed |

---

### PHASE 1 — Make it real and honest (Sprints 1–4)

#### Sprint 1 — Foundation & Truth (Epic A)
- **Goal:** notebook → tested package; stop the data tool from lying; freeze the fine-tune dependency.
- **Stories:**
  - As a dev, I can `pip install -e .` and import `pennystock.agents`, `pennystock.retrieval`, `pennystock.tools`, `pennystock.data`. *AC: the thin demo notebook only imports from the package.*
  - `resolve_symbol` and the RRF fusion have unit tests. *AC: `pytest` green; ~70% coverage on those two.*
  - `config.py` via `pydantic-settings`; zero hard-coded secrets/paths. *AC: grep finds no secrets; paths come from config.*
  - Swap the in-pipeline MarketDataAgent to plain `ChatOllama` + structured output and prove parity vs `penny-llama` on the smoke set. *AC: same tickers/exchanges emitted; fine-tune preserved as an artifact, not a dependency.*
- **DoD:** package installs, tests pass, pipeline runs from the package.
- **Metric moved:** test coverage 0 → ~70% on core logic.
- **Do NOT:** touch FastAPI, cloud, or new agents. Just make the foundation honest.

#### Sprint 2 — Data Spine part 1: EOD + universe (Epic B)
- **Goal:** own the data. NSE bhavcopy EOD ingest + universe with delisted flags.
- **Stories:**
  - `MarketDataProvider` interface with two impls: `NseBhavcopyProvider` (EOD) and `BrokerProvider` stub. *AC: `yfinance` sits behind the interface as a fallback only.*
  - Nightly job ingests bhavcopy → `prices` table (OHLCV, adjusted). *AC: 1y of daily OHLCV retrievable offline for any active NSE symbol.*
  - Universe keeps **delisted** symbols (`is_active=0`) with date — survivorship mitigation. *AC: a known-delisted symbol is present and flagged.*
- **DoD:** cron/APScheduler job runs; `prices` populated; documented.
- **Metric moved:** historical coverage 0 → full NSE EOD, 1y+ depth.
- **Do NOT:** build intraday/live streaming yet. EOD first.

#### Sprint 3 — Data Spine part 2: broker candles + `get_price_history` (Epic B)
- **Goal:** real historical candles via a broker API; fix the momentum lie.
- **Stories:**
  - Integrate **Angel One SmartAPI** (free) behind `BrokerProvider` for historical candles + live quote. *AC: fetch daily/intraday candles for a symbol; credentials via config.*
  - New tool `get_price_history(symbol, interval, lookback)` returns OHLCV series. *AC: analyst can now actually see a trend.*
  - Rewrite `get_indian_stock_data` docstring/contract to match reality (or deprecate in favor of explicit tools). *AC: no tool claims data it doesn't return.*
- **DoD:** the demo "assess momentum" query is answerable from real history.
- **Metric moved:** the analyst goes from 1-day snapshot → real time series.
- **Do NOT:** add technical indicators yet (that's Sprint 6). Just deliver clean history.

#### Sprint 4 — The Ledger (Epic C)
- **Goal:** build the spine. Structured recommendations + persistence.
- **Stories:**
  - `Recommendation` pydantic schema enforced via structured output on the decision agent. *AC: pipeline emits a valid Recommendation object, never prose-only.*
  - `recommendations` table (Postgres, or promote SQLite for now) storing the rec + **model/prompt versions**. *AC: every run writes one graded-later row with provenance.*
  - Run the pipeline 20× → inspect the table. *AC: 20 structured recs with invalidation prices persisted.*
- **DoD:** a real Ledger exists with real rows.
- **Metric moved:** recommendations stored 0 → N (the product now has memory).
- **Do NOT:** build the retro loop yet — first accumulate data to retro *over*.

**End of Phase 1 checkpoint:** tested package, own data spine with history, and a Ledger recording structured calls. *This is already ahead of most GitHub repos in rigor.*

---

### PHASE 2 — Make it credible (Sprints 5–8)

#### Sprint 5 — Outcome Tracker + Backtest engine skeleton (Epics C, D)
- **Goal:** grade the calls; stand up walk-forward machinery.
- **Stories:**
  - Nightly Outcome Tracker labels each rec at +1d/1w/1m/3m from the `prices` store → hit/miss/invalidated/open. *AC: hit-rate query returns a real number.*
  - Walk-forward backtest harness: replays the pipeline over historical dates using **PIT data only** (no look-ahead). *AC: a run on a past window produces a paper track record; an assertion test proves no future data leaked in.*
- **DoD:** first honest hit-rate + first walk-forward run exist.
- **Metric moved:** outcomes graded 0 → all; first backtest baseline.
- **Do NOT:** optimize anything yet. Measure first (rule of thumb: measure before you optimize).

#### Sprint 6 — Technical agent (deterministic) + Risk Manager veto (Epic E)
- **Goal:** LLMs narrate, code computes; add the micro-cap immune system.
- **Stories:**
  - `TechnicalAgent`: SMA/EMA/RSI/volatility/52-wk position in **pandas**; LLM only narrates. *AC: indicators match a hand-computed check within tolerance.*
  - `RiskManagerAgent`: rule-based, bucket-specific, can **VETO** a buy pre-Ledger (liquidity floor, volatility band, pump-and-dump red flags from the book). *AC: a deliberately illiquid micro-cap gets vetoed.*
- **DoD:** no arithmetic done by an LLM anywhere; veto demonstrably fires.
- **Metric moved:** add risk-veto rate; kill a class of bad calls.
- **Do NOT:** add the bull/bear debate yet (next sprint).

#### Sprint 7 — Research Team: Bull ⇄ Bear debate (Epic E)
- **Goal:** adversarial process to cut single-perspective bias.
- **Stories:**
  - Bull and Bear researcher nodes debate N configurable rounds over the analyst evidence before the decision agent commits. *AC: `max_debate_rounds` config; both cases logged.*
  - Backtest with vs without debate. *AC: a measured delta on hit-rate/calibration (even if small/negative — report it).*
- **DoD:** debate wired; A/B measured on the backtest.
- **Metric moved:** hit-rate/calibration delta from debate, measured.
- **Do NOT:** add more personas (13-investor cosplay) — resist the temptation; measure value first.

#### Sprint 8 — RAG Excellence: reranker + semantic chunking + RAGAS gate (Epics F, D)
- **Goal:** bring RAG to 2026 production standard, and gate it.
- **Stories:**
  - Add cross-encoder **reranker** (retrieve 20 → rerank 5 → send 3–5). *AC: RAGAS context-precision improves, measured.*
  - Re-ingest corpus with **semantic chunking**. *AC: before/after RAGAS table.*
  - 50-Q golden set + RAGAS harness; wire thresholds (faithfulness>0.85, relevancy>0.8, ctx-precision>0.75, recall>0.8) as a **CI gate**. *AC: a PR that regresses RAG fails CI.*
- **DoD:** `evals/` folder + results table in README; CI gate live.
- **Metric moved:** RAG faithfulness/precision to threshold; the interview-gold before/after table.
- **Do NOT:** migrate vector DBs (Chroma→Qdrant) unless an eval says you must. Don't move for résumé reasons mid-credibility-phase.

**End of Phase 2 checkpoint:** graded calls, a look-ahead-free backtest, deterministic technicals, a risk veto, adversarial debate, and eval-gated RAG. *This is a genuinely credible research system with numbers to prove it.*

---

### PHASE 3 — Make it a product (Sprints 9–12)

#### Sprint 9 — API & persistence (Epic G)
- **Goal:** a product surface.
- **Stories:**
  - FastAPI: `POST /analyze` (SSE streaming of agent progress), `GET /reports/{id}`, `GET /health`; API-key auth; pydantic models. *AC: `curl` streams a live analysis.*
  - Postgres migration (SQLAlchemy + Alembic) for universe/prices/recommendations/outcomes. *AC: schema versioned; migrations run clean.*
- **DoD:** the pipeline is a service, not a script.
- **Metric moved:** add API latency percentiles.
- **Do NOT:** build a fancy frontend. API + `curl`/thin UI is enough for v1.

#### Sprint 10 — Observability & MLOps (Epic H)
- **Goal:** see everything; make quality a gate.
- **Stories:**
  - **Langfuse** tracing on every agent/tool/retrieval span (tokens, latency, cost). *AC: open a full trace of one run.*
  - **MLflow** for fine-tune runs + model registry (v1 BANK-bug → v2 → v3). *AC: versions comparable with charts.*
  - GitHub Actions: PR runs ruff+mypy+pytest; nightly runs the eval suite and **fails on metric regression**. *AC: green badge; a deliberately-bad PR is blocked.*
- **DoD:** `docker compose up` → API + serving + Postgres + Langfuse; CI gates live.
- **Metric moved:** MTTR on agent bugs (traces); regression caught automatically.
- **Do NOT:** hand-roll observability. Use Langfuse.

#### Sprint 11 — Retro loop + live-context RAG + Compliance (Epics C, F, I)
- **Goal:** close the learning loop and make it lawful to show anyone.
- **Stories:**
  - **Retro Analyst (weekly):** track-record memo prepended to the analyst (in-context learning). *AC: memo reflects real recent hit/miss patterns.*
  - **Live-context collection:** nightly ingest of news/filings for tracked tickers into a second vector collection. *AC: analyst cites current events, not just the book.*
  - **Compliance gate:** disclaimers, AI-usage disclosure, research/education framing, audit-trail export, human-in-the-loop confirmation. *AC: no output ships without disclosure; audit trail exportable. (Consult a professional before any multi-user launch — see §2.2.)*
- **DoD:** the loop is closed; the system is defensible to show.
- **Metric moved:** track-record-memo effect on forward hit-rate (measured over following weeks).
- **Do NOT:** expose to real users/clients without the §2.2 legal review. Personal-use only until then.

#### Sprint 12 — Serving upgrade + Cloud (optional, time-boxed) (Epics G, J)
- **Goal:** scale readiness + a public demo, *only if* the product is real.
- **Stories:**
  - If concurrency/eval-throughput demands it: migrate serving Ollama → **vLLM** (multi-LoRA hot-swap for `penny-llama` + future adapters). *AC: measured throughput gain; adapters swappable.*
  - Deploy the compose stack to one cloud instance (RDS free tier for Postgres); S3/ECR for artifacts; HTTPS. *AC: a public demo URL (personal, disclaimed).*
- **DoD:** a live, observable, disclaimed demo.
- **Metric moved:** throughput; a real deployment.
- **Do NOT:** do Kubernetes/SageMaker for their own sake. If you want the résumé line, time-box it to a weekend and tear it down — but only *after* the product stands. The product is the résumé.

---

### Sprint cadence summary
| Phase | Sprints | Theme | Exit artifact |
|---|---|---|---|
| 1 | 1–4 | Real & honest | Tested package + PIT data spine + Ledger |
| 2 | 5–8 | Credible | Walk-forward backtest + risk/debate + eval-gated RAG |
| 3 | 9–12 | Product | API + observability + closed loop + compliant demo |

---

## 8. Metrics, KPIs & Kill-switches

**Product KPIs (the ones that matter):**
- **Recommendation hit-rate** by horizon and by conviction bucket (calibration: does conviction 5 beat conviction 2?).
- **Paper track record vs buy-and-hold** on walk-forward, contamination-free windows (expect to lose; measure honestly).
- **Risk-veto rate** and how many vetoed names later blew up (the veto's value).

**Model/quality KPIs:**
- RAGAS: faithfulness >0.85, relevancy >0.8, ctx-precision >0.75, recall >0.8.
- Tool-call accuracy on the 100-query held-out suite.

**Ops KPIs:** p50/p95 latency, error rate, tokens/cost per analysis.

**Kill-switches (auto-revert triggers):**
- Any RAGAS metric drops below threshold in CI → block merge.
- Hit-rate or judge-pass-rate drops N points after a prompt/model change → revert; the Ledger records which version produced what, so A/B is on real outcomes.

---

## 9. Risk Register
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Chasing alpha / believing a rosy backtest | High | Fatal to credibility | §2.1, PIT walk-forward, contamination-free windows, report losses honestly |
| SEBI/regulatory exposure on any multi-user launch | Med | Legal/fatal | §2.2 compliance gate; personal-use until registered; no execution |
| Data vendor breakage / rate limits | High | High | `MarketDataProvider` interface + fallback; own the EOD store via bhavcopy |
| Scope creep (13 personas, K8s, new DBs) | High | Schedule death | Binding "Do NOT do" lines; one epic at a time |
| Micro-cap data quality (thin, manipulated) | High | Medium | Risk Manager veto; the whole point is modeling this adversarial niche |
| Fine-tune maintenance burden | Med | Medium | Keep as swappable vLLM adapter, not in-pipeline dependency |
| LLM arithmetic errors | High if allowed | High | Deterministic pandas for all math; LLM narrates only |
| Look-ahead bias sneaking into evals | Med | High | Assertion tests in the backtest engine that fail on future-data access |

---

## 10. Tech Stack Decisions (with rationale)
| Layer | Choice | Rationale | Alternative considered |
|---|---|---|---|
| Orchestration | LangGraph | Already invested; supervisor+debate maps cleanly | AutoGen, CrewAI |
| Serving (now) | Ollama | Best single-user DX | — |
| Serving (scale) | vLLM | ~6× throughput, multi-LoRA hot-swap [§14] | TGI, TensorRT-LLM |
| Data (EOD) | NSE/BSE bhavcopy | Official, free, survivorship-safe backbone | Paid vendors |
| Data (candles/live) | Angel One SmartAPI (free) → Kite if needed | Free start; swap via interface [§14] | Upstox, Dhan |
| Vector DB | Chroma now; Qdrant only if evals demand | Don't migrate for résumé reasons | Qdrant, pgvector |
| Embeddings | bge-m3 (keep) + bge-reranker (add) | Reranker = biggest RAG win [§14] | e5, jina |
| Ledger/DB | Postgres (SQLAlchemy+Alembic) | Real data modeling; time-series of prices | SQLite (dev only) |
| Evals | RAGAS + custom backtest | Standard + our PIT harness [§14] | DeepEval, Braintrust |
| Observability | Langfuse (self-host) | OSS, framework-agnostic, RAGAS integration [§14] | LangSmith, Braintrust |
| Experiments | MLflow | Model registry + fine-tune runs | W&B |
| Packaging | Docker Compose | One-command stack | bare processes |

---

## 11. Roles (solo dev wearing hats — RACI-lite)
You're one person; make the hats explicit so you don't skip the boring ones:
- **Product owner hat:** guards §1.3 and the "Do NOT do" lines. Says no.
- **Data engineer hat:** owns the PIT store, ingest jobs, vendor abstraction.
- **ML/agent engineer hat:** agents, prompts (versioned), fine-tune.
- **Eval/quant hat:** the backtest engine and its honesty (the most senior hat — it's allowed to declare the system doesn't work).
- **Platform/MLOps hat:** serving, CI gates, observability, compliance.
- Rotate deliberately; the eval hat must never be silenced by the ML hat's optimism.

---

## 12. What we are explicitly NOT doing in v1 (the anti-scope)
- No order execution / auto-trading. No brokerage integration for *trades*.
- No intraday HFT, no options/F&O modeling.
- No 13-persona investor cosplay until a measured reason exists.
- No Kubernetes/SageMaker before the product stands (résumé stunts are Phase-3-optional, time-boxed).
- No multi-user/public advisory before the §2.2 legal gate.
- No claim, anywhere, that the system beats the market.

---

## 13. Glossary
- **PIT (Point-in-Time) data:** data stored as it was known on each historical date; prevents look-ahead.
- **Look-ahead bias:** using information not available at the simulated moment (incl. an LLM "remembering" prices from training).
- **Survivorship bias:** testing only on stocks that survived; fixed by keeping delisted names.
- **Walk-forward:** train on past, test on the immediately-following out-of-sample window, roll forward.
- **RRF:** Reciprocal Rank Fusion — how we merge dense+sparse retrieval.
- **RAGAS:** RAG evaluation framework (faithfulness, relevancy, context precision/recall).
- **Ledger:** our store of structured recommendations + graded outcomes — the product spine.
- **Bucket:** cap classification (large/mid/small/micro/penny) driving bucket-specific rules.

---

## 14. Sources (grounding for external claims; verified 2026-07-29)

**LLM trading reality / benchmarks**
- StockBench — contamination-free LLM trading benchmark; most agents fail to beat buy-and-hold: https://stockbench.github.io/ · https://openreview.net/forum?id=9tFRj7cmrS
- Look-ahead bias in LLM trading / memorization of prices: https://arxiv.org/html/2605.24564 · https://paperswithbacktest.com/course/look-ahead-bias-llm-trading
- InvestorBench (financial decision-making benchmark): https://arxiv.org/pdf/2412.18174

**SEBI / regulation (India)**
- SEBI AI accountability — regulated entities responsible for AI output: https://www.lexology.com/library/detail.aspx?g=f2e36044-cc40-4c8f-8e74-79141d32031e
- SEBI digital compliance 2026 for IAs/RAs: https://www.mondaq.com/india/securities/1759228/sebis-new-digital-compliance-rules-what-investment-advisers-must-know-in-2026
- "Educational only" disclaimers don't excuse unregistered advisory: https://lawyervikasgupta.com/blog/sebi-ria-ra-compliance-changes-in-2025/

**Indian market data APIs**
- Broker API comparison (Kite/SmartAPI/Upstox/Dhan) costs & limits: https://algotest.in/blog/best-brokers-for-algo-trading-in-india/ · https://stratzy.in/blog/cost-comparison-algo-trading-apis-india/
- Zerodha Kite Connect: https://zerodha.com/products/api/
- yfinance unreliable for production + alternatives: https://medium.com/@craakash/9-best-yfinance-alternatives-for-reliable-financial-data-in-2025-b48b4aaa9ac9

**Architecture (multi-agent)**
- TradingAgents paper (analyst→research debate→trader→risk→PM): https://arxiv.org/pdf/2412.20138 · https://github.com/TauricResearch/TradingAgents · https://deepwiki.com/TauricResearch/TradingAgents
- ai-hedge-fund (backtester, portfolio/risk agents): https://github.com/virattt/ai-hedge-fund
- FinRobot (deterministic valuation code paths): https://github.com/ai4finance-foundation/finrobot

**RAG (2026 best practice)**
- Reranker = biggest single win; retrieve-20→rerank-5; semantic chunking; RAGAS thresholds: https://www.callmissed.com/en/blog/rag-best-practices-2026 · https://www.stackai.com/insights/retrieval-augmented-generation-(rag)-best-practices-for-enterprise-ai-chunking-embeddings-reranking-and-hybrid-search-optimization
- Financial RAG retrieval (BM25 strength on financial text; HyDE counterproductive for numeric): https://arxiv.org/pdf/2404.07221 · https://arxiv.org/pdf/2504.15800

**Serving**
- vLLM vs Ollama (throughput, multi-LoRA hot-swap): https://www.spheron.network/blog/ollama-vs-vllm/ · https://developers.redhat.com/articles/2025/08/08/ollama-vs-vllm-deep-dive-performance-benchmarking

**Observability**
- Langfuse (OSS, self-host, RAGAS integration, guardrails): https://langfuse.com/blog/2024-07-ai-agent-observability-with-langfuse · https://mlflow.org/top-5-agent-observability-tools/

**Backtesting methodology**
- Point-in-time / walk-forward / survivorship bias mitigation: https://sharpely.in/blog/bias-free-backtesting-explained:-how-sharpely-uses-point-in-time-data-to-avoid-look-ahead-and-survivorship-bias · https://daviddtech.medium.com/the-three-deadly-sins-of-backtesting-overfitting-look-ahead-bias-and-p-hacking-a68c6345e668

---

## 15. Immediate next actions (start here)
1. **Sprint 1, Story 1:** scaffold `pennystock/` and move the notebook cells into modules (agents/retrieval/tools/data). Keep one demo notebook that imports the package.
2. **Sprint 1, Story 3:** move all paths/secrets into `config.py` (`pydantic-settings`).
3. **Sprint 2, Story 2:** write the NSE bhavcopy nightly ingest → `prices` table. This unblocks everything downstream.
4. Re-read §2.1 and §2.2 before you write a line of "prediction." The plan's whole value is that it's honest.

> **Remember the one rule that governs all others:** *measure before you optimize.* Build the Ledger and the walk-forward backtest first; every other feature is judged by whether it moves those numbers.

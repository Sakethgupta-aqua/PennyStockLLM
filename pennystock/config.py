"""Central configuration for the whole package.

Every path, model name, and tunable magic number lives here — nothing is hard-coded in
the modules. Values can be overridden by environment variables (prefix ``PENNY_``) or a
``.env`` file, so secrets/paths never live in code (PRODUCTION_MASTER_PLAN.md §6.3).

    from pennystock.config import settings
    settings.data_model            # "penny-llama"
    PENNY_DATA_MODEL=qwen3.5:latest python ...   # override without touching code
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parent.parent   # .../PennyStockLLM


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PENNY_", env_file=".env", extra="ignore")

    # --- paths ---
    project_root: Path = _ROOT
    market_db: Path = _ROOT / "market.db"
    vector_db_dir: Path = _ROOT / "penny_stock_vector_db"

    # --- vector store / embeddings (embed_model MUST match how the DB was ingested) ---
    collection: str = "knowledge_base"
    # Broker research lives in its OWN collection, never mixed into knowledge_base.
    # Sell-side notes are opinion with a structural Buy bias; the book corpus supplies
    # frameworks. Blending them would let broker optimism answer methodology questions.
    broker_collection: str = "broker_reports"
    embed_model: str = "BAAI/bge-m3"
    # longest text the embedder reads (tokens). Passages are cut to ~1,000 tokens; a garbled text layer can
    # run to one token per character, and the GPU's attention memory grows with the square of the length
    embed_max_tokens: int = 1024

    # --- Ollama models, one per agent role ---
    data_model: str = "penny-llama"                       # kept as an artifact; swappable via env
    analyst_model: str = "qwen3.5:35b"
    judge_model: str = "ggozad/prometheus2:latest"
    grounding_model: str = "qwen3.5:35b"               # CRAG auditor - needs a strong model; an 8b
                                                       # produced self-contradictory verdicts (false alarms)
    num_ctx: int = 16384                               # was 8192 - room for RAG context + a full report
    # The analyst and grounding roles (qwen3.5:35b) read every tool result. 16,384 overflowed on a 16-tool
    # run (2026-10-10) and Ollama silently dropped the oldest messages. Measured on this M4 Max: the model is
    # hybrid (10 of 40 layers keep a KV cache, ~20 KB/token), so 64K adds ~1 GB and stays 100% on the GPU;
    # 131,072 is the largest that does; the model's own maximum is 262,144. The 8B roles are capped at 8,192.
    large_num_ctx: int = 65536
    # a prompt this close to its window may already have been trimmed by Ollama (which raises no error)
    context_warn_share: float = 0.85
    # The analyst's prompt is fitted to large_num_ctx ON PURPOSE (oldest tool results set aside first)
    # rather than left to Ollama, which silently drops whole messages from the start - the question first.
    # Estimate = characters / chars_per_token_estimate (measured 1.5-4.7 chars per token here, so 2.5 errs
    # high) + the 16 tool descriptions (~3,900 tokens measured), leaving answer_reserve_tokens for the report.
    answer_reserve_tokens: int = 8192
    tool_schema_tokens: int = 4000
    chars_per_token_estimate: float = 2.5
    # How long Ollama keeps a model resident. "-1" (forever) pinned FOUR models at once and
    # drove this Mac into swap (3.2GB of 4GB swap used, ~500MB free) - one eval question took
    # 90 minutes. Small, frequently-reused models stay warm briefly; big ones unload at once.
    small_model_keep_alive: str = "5m"                 # decomposer / data agent
    large_model_keep_alive: str = "0"                  # analyst / judge / grounding (unload now)
    # Qwen3.5 "thinks" in hidden tokens before every answer unless told not to. Measured on this
    # M4 Max (2026-10-02): the same short answer took 1,381 generated tokens / 24.7 s with thinking
    # and 157 tokens / 2.1 s without - ~9x the GPU work, on every analyst and grounding call.
    # None = the model's own default; True/False = explicit. Only these roles run a thinking model.
    analyst_reasoning: bool | None = False
    grounding_reasoning: bool | None = False

    # --- local accelerator for the embedder and reranker (see pennystock/devices.py) ---
    device: str = "auto"                  # auto = cuda > mps (Apple GPU) > cpu, or force one
    half_precision: bool = True           # fp16 on a GPU: half the memory, same results (measured)

    # --- retrieval ---
    dense_k: int = 12                     # was 6 (retrieve broad; the reranker prunes)
    dense_fetch_k: int = 40               # was 30
    bm25_k: int = 10                      # was 4
    rrf_k: int = 4
    rrf_c: int = 60

    # --- advanced retrieval (multi-query -> fuse -> rerank -> reorder) ---
    research_subqueries: int = 3          # multi-query decomposition count
    candidate_k: int = 20                 # fused candidates kept per sub-query
    rerank_pool: int = 40                 # max candidates sent to the cross-encoder
    rerank_top_n: int = 8                 # final chunks handed to the analyst
    rerank_overfetch: int = 3             # rerank top_n*this, so the diversity cap has choices
    max_chunks_per_source: int = 3        # no single book may monopolise the context
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # tokens the reranker reads per (question, passage) pair: filing passages run to ~1,000; at 512 it never
    # saw the second half of annual-report passages (books and broker notes fit in 512 either way)
    rerank_max_tokens: int = 1024
    decomposer_model: str = "llama3-groq-tool-use:8b"   # small and fast; splits a request into sub-queries

    # --- orchestration ---
    recursion_steps_default: int = 25
    recursion_floor: int = 2
    max_tool_chars: int = 1200
    pass_score: int = 4
    max_revisions: int = 2                                # cap the evaluator->analyst revision loop
    max_grounding_retries: int = 1                        # cap the CRAG grounding->analyst loop
    # evidence sent to the grounding checker: 8,000 chars showed it only the first 8 of 23 tool results
    # (2026-10-10). Sized for its 64K window: 100K chars is ~30-50K tokens, leaving room for the report.
    grounding_max_tokens: int = 1024               # the verdict is one word or a short list; 51,211 tokens
    grounding_feedback_chars: int = 3000           # of a verdict sent back to the analyst (2026-10-11 run)
    judge_max_tokens: int = 1024                   # the judge's feedback goes back to the analyst as well
    max_grounding_chars: int = 100_000
    # --- NSE integrated filings (XBRL: audited figures + the half-yearly related-party disclosure) ---
    integrated_filings_max_age_days: int = 7       # the filing list is re-read weekly; a parsed file never is
    integrated_parser_version: int = 1             # bump to re-parse every cached XBRL file
    # a broker's reported-year figure within this % of the filed one is on the filed basis; else adjusted
    broker_filed_tolerance_pct: float = 3.0
    # contingent liabilities: the note's text read from its page (and the next page's top when it runs on)
    cl_note_chars: int = 7000
    cl_extractor_version: int = 5                  # bump to re-read every note (5: bank, unit rules)
    extraction_num_ctx: int = 16384                # one note (~2-4K tokens) and its rows
    extraction_keep_alive: str = "10m"             # warm across a batch of reports
    # --- the evidence pack: the structured tools run in code before the analyst writes ---
    evidence_concurrency: int = 4                  # tools at once (several of them call NSE)
    evidence_tool_timeout_s: float = 300.0         # one slow source never holds the pack
    evidence_max_chars: int = 14000                # the fact sheet's budget (~4K tokens)
    evidence_alternatives: int = 6                 # screen candidates when the user asks for alternatives
    judge_evidence_chars: int = 6000               # of the fact sheet shown to the judge (8,192-token window)

    # --- live market data (structured half of the data spine) ---
    market_provider: str = "yfinance"
    history_period: str = "5y"          # was "1y" - long enough to see the real multi-year peak
    history_recent_days: int = 252      # ~1 trading year for MAs / RSI / support-resistance
    history_interval: str = "1d"
    sr_lookback: int = 60          # support/resistance window (trading days)
    rsi_window: int = 14
    financials_years: int = 5
    structural_break_pct: float = 20.0      # single-day % move that flags a possible corporate action

    # --- ownership / promoter pledge (Phase 2: the penny-cap red flag) ---
    ownership_provider: str = "nse+screener"   # NSE = statutory pledge; screener adds FII/DII
    ownership_lookback_quarters: int = 4      # quarters used for trend deltas
    ownership_max_age_days: int = 30          # refetch cached shareholding older than this
    ownership_timeout_s: int = 15

    # --- corporate events (announcements / insider / credit rating) ---
    events_scan_limit: int = 120              # announcements pulled before keyword screening
    events_insider_limit: int = 30
    events_rating_limit: int = 10
    events_report_limit: int = 8              # material items actually shown to the analyst
    # event signals (stored, typed, graded): docs/superpowers/specs/2026-10-05-event-signals-design.md
    events_min_value_rs: float = 1_000_000    # insider trades below Rs 10 lakh are noise, not a signal
    events_cluster_days: int = 30             # two insiders buying within this window = a cluster
    events_veto_days: int = 365               # a severe event keeps a stock out of the screen this long
    events_call_spacing_days: int = 30        # one ledger call per company and event type per window
    events_since: str = "2024-10-01"          # first-run backfill start
    events_stale_days: int = 7

    # --- peer comparison / valuation history ---
    min_peers: int = 4                        # below this a "sector median" is meaningless
    peer_max_age_days: int = 45

    # --- earnings-call transcripts / language analysis ---
    lexicon_auto_download: bool = True        # fetch the full Loughran-McDonald dictionary once
    transcript_max_quarters: int = 4          # how many past calls to keep per company
    transcript_guidance_limit: int = 6        # forward-looking sentences surfaced per quarter
    transcript_evasion_limit: int = 4         # evasive Q&A exchanges surfaced
    transcript_timeout_s: int = 60
    transcript_non_answer_flag: float = 0.15  # share of questions explicitly not answered

    # --- annual-report notes (contingent liabilities / related-party) ---
    annual_report_timeout_s: int = 300        # reports run to 460 pages / ~17MB
    annual_report_window: int = 2500          # characters of note text to keep

    # --- screening ---
    turnover_lookback_days: int = 20          # ~1 trading month of median turnover
    metrics_max_age_days: int = 30            # screening cache refresh cadence

    # --- scheduled data maintenance ---
    refresh_delay_s: float = 0.4              # politeness gap between bulk network calls

    # --- symbol universe (NSE equity master) ---
    universe_auto_refresh: bool = True        # keep the resolver's symbol list current
    universe_max_age_days: int = 7            # the listing master only changes on list/delist
    universe_url: str = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    universe_timeout_s: int = 30

    # --- resolver ---
    resolver_min_score: float = 88.0

    # --- tracing & Layer-0 assertions (the evaluation foundation) ---
    trace_enabled: bool = True
    trace_dir: Path = _ROOT / "traces"
    trace_max_field_chars: int = 20000     # per-field cap so one huge tool result can't bloat a trace
    min_report_chars: int = 400            # shorter than this = the report did not really complete
    min_distinct_sources: int = 2          # retrieval must not collapse onto a single book
    min_quote_words: int = 8               # quoted spans this long must be traceable to a chunk

    # --- prediction ledger (the memory: remember -> check -> score) ---
    ledger_enabled: bool = True
    # Horizons in TRADING days. +20 trading days measured out at +34 calendar days on
    # 2026-09-27, which is why these are never converted with date arithmetic.
    ledger_horizons: tuple[int, ...] = (1, 5, 20, 60)
    # Nifty 500. The one Indian index that reliably returns full history on this feed -
    # eight of the twelve sector indices return a single row (see data/outcomes.py).
    broad_benchmark: str = "^CRSLDX"
    # Below this many points of abnormal return, a result is called `flat` rather than a
    # hit: under ~2 points over 20 sessions you cannot separate skill from bid-ask friction.
    flat_band_pct: float = 2.0
    # Trendlyne publishes a broker report after that broker's paying clients have already
    # traded on it, so a broker prediction is dated to the next session you could have
    # acted on. Raise to 2 for a more conservative measurement of edge.
    broker_action_lag_days: int = 1

    # --- broker research: sector KPIs (L1), consensus (L2), narrow RAG (L3) ---
    # A broker's view is "current" for this long. Brokers re-rate every results season (~90
    # days), so two missed seasons means the view is stale; I/B/E/S reviews estimates older
    # than ~100 days. Stale targets are what make a consensus look rosier than it is.
    broker_max_age_days: int = 180
    broker_revision_days: int = 90          # window for up/downgrades and consensus drift
    # Target range / median above which the consensus upside is flagged as unreliable: when
    # targets disagree this much, implied upside has historically pointed the wrong way.
    broker_high_dispersion: float = 0.25
    # Lenders that must carry a broker KPI before it may enter the score. Below this a
    # percentile is drawn from a handful of names and ranks noise.
    broker_kpi_min_companies: int = 20
    broker_search_top_n: int = 6            # passages search_broker_research returns
    broker_chunks_per_report: int = 2       # so one long report cannot fill the context

    # --- company documents library (docs/superpowers/specs/2026-10-04-company-documents-library-design.md)
    stock_reports_dir: Path = _ROOT / "Stock_Reports"
    filings_dir: Path = _ROOT / "Corporate_Filings"
    filings_days: int = 730                 # key filings from the last two years
    filings_delay_s: float = 1.0            # NSE is someone else's server: one request a second
    filings_timeout_s: int = 60
    filings_repair_timeout_s: int = 300     # large annual reports were cut off at 60 s
    # filings search: docs/superpowers/specs/2026-10-05-filings-search-design.md
    filings_db: Path = _ROOT / "filings.db"     # page text + FTS5 (kept out of market.db: ~1.5 GB)
    filings_db_timeout_s: float = 30.0          # how long a write waits for the database before giving up
    filings_ocr_min_chars: int = 40             # a page with less native text is OCR'd (not presentations)
    filings_table_numeric_share: float = 0.30   # more numeric tokens than this = a table page
    filings_passage_chars: int = 4000           # ~1,000 tokens per passage
    filings_overlap: float = 0.25               # tail carried into the next passage on a split
    filings_collection: str = "filings"         # Chroma collection beside the books and broker notes
    filings_search_top_n: int = 8
    filings_per_doc: int = 2                    # no single filing fills the results
    filings_fts_k: int = 50
    filings_dense_k: int = 50
    filings_search_mode: str = "fused_rerank"   # keyword | dense | fused | fused_rerank (set by eval_filings)
    # NSE's one-page intimations are ~200 KB and a real PDF is never under a few KB; below this
    # the body is an error page or a truncated download
    filing_min_pdf_bytes: int = 5_000
    stock_report_min_pdf_bytes: int = 50_000    # a Trendlyne Stock Report is 1.5-2.6 MB
    stock_report_daily_opens: int = 70          # Trendlyne's daily stock-report allowance


settings = Settings()

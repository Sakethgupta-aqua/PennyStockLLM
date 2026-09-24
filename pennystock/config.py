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

    # --- Ollama models, one per agent role ---
    data_model: str = "penny-llama"                       # kept as an artifact; swappable via env
    analyst_model: str = "qwen3.5:35b"
    judge_model: str = "ggozad/prometheus2:latest"
    supervisor_model: str = "llama3-groq-tool-use:8b"
    grounding_model: str = "qwen3.5:35b"               # CRAG auditor - needs a strong model; an 8b
                                                       # produced self-contradictory verdicts (false alarms)
    num_ctx: int = 16384                               # was 8192 - room for RAG context + a full report
    # How long Ollama keeps a model resident. "-1" (forever) pinned FOUR models at once and
    # drove this Mac into swap (3.2GB of 4GB swap used, ~500MB free) - one eval question took
    # 90 minutes. Small, frequently-reused models stay warm briefly; big ones unload at once.
    small_model_keep_alive: str = "5m"                 # supervisor / decomposer / data agent
    large_model_keep_alive: str = "0"                  # analyst / judge / grounding (unload now)

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
    decomposer_model: str = "llama3-groq-tool-use:8b"   # reuse the supervisor model

    # --- orchestration ---
    recursion_steps_default: int = 25
    recursion_floor: int = 2
    max_tool_chars: int = 1200
    pass_score: int = 4
    max_revisions: int = 2                                # cap the evaluator->analyst revision loop
    max_grounding_retries: int = 1                        # cap the CRAG grounding->analyst loop
    max_grounding_chars: int = 8000                       # cap the evidence sent to the grounding checker

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


settings = Settings()

# CLAUDE.md

This file gives Claude Code persistent context about this project. It is read
automatically at the start of every session. Edit it freely — add anything you
find yourself repeating to Claude.

## Project

PennyStockLLM — building/evaluating an LLM over penny-stock content using
retrieval-augmented generation (RAG), fine-tuning, and local inference.

## Layout

- `Penny-stock-LLM_2Agents.ipynb` — two-agent notebook.
- `Penny-stock-Multi-Agent-RAG.ipynb` — multi-agent RAG pipeline.
- `Stocks_PG_Code.ipynb`, `penny_llm.ipynb` — working/experiment notebooks.
- `penny_stock_vector_db/` — vector store for RAG.
- `Penny_stock_PDFS/`, `Penny_stock.pdf`, `Index_Page_PennyStock_Book/` — source corpus.
- `finetune/` — fine-tuning scripts/artifacts.
- `llama.cpp-master/` — local inference (llama.cpp).

## Conventions & preferences

<!-- Add things you want Claude to always do. Examples:
- Use `uv` (not pip) for installing packages.
- Activate the venv at ./.venv before running anything.
- Prefer concise answers; don't explain basic Python.
- When editing notebooks, keep cell outputs cleared.
-->

## Commands

Data caches live in `market.db`. `python scripts/refresh_data.py --status` shows what is stale.

```bash
# per-stock analysis
python demo.py "<query>"                                # run the agent graph
python scripts/check_traces.py                          # Layer-0 assertion pass rates

# screening (offline ingest, then instant ranking)
python scripts/ingest_metrics.py --index nifty50        # ~3 min / 50 names
python scripts/ingest_metrics.py --from-universe --resume
python scripts/screen.py --top 25                       # funnel + ranked shortlist
python scripts/screen.py --explain HINDALCO             # one stock's scorecard

# corpus + retrieval
python scripts/ingest_documents.py --no-ocr
python scripts/eval_rag.py --verified-only

# broker research: L1 sector KPIs, L2 consensus, L3 narrow RAG
python scripts/ingest_broker_reports.py --status        # downloaded / parsed / embedded / missing
python scripts/fetch_midcap_reports.py --cookies cookies.txt   # PDFs need a Trendlyne login cookie
python scripts/ingest_broker_reports.py                 # KPI tables + vector store, new reports only
python scripts/backfill_broker_predictions.py           # ratings + targets -> ledger (L2 input)
python scripts/screen.py --explain AUBANK               # KPI panel + broker consensus panel

# company documents library: Broker_Reports/ + Stock_Reports/ + Corporate_Filings/, one catalog
python scripts/catalog_documents.py                     # what the library holds (files, companies, size)
python scripts/catalog_documents.py --backfill --verify # register PDFs already on disk; re-hash every file
python scripts/fetch_filings.py --dry-run --symbol BHEL # NSE filings that would be kept (2 years, key types)
python scripts/fetch_filings.py                         # download them (no Trendlyne quota; resumable)
python scripts/file_stock_reports.py                    # file Trendlyne Stock Report downloads from ~/Downloads
python scripts/fetch_stock_reports.py --cookies cookies.txt   # monthly snapshot, <=70 opens, stops at the limit

# Trendlyne's view (third-party D/V/M scores): snapshots + ledger calls; both scripts above run this too
python scripts/parse_stock_reports.py                   # read new Stock Reports, record Trendlyne calls in the ledger
python scripts/parse_stock_reports.py --status          # reports read; which fields came back blank
python scripts/screen.py --explain APARINDS             # TRENDLYNE'S VIEW panel beside our pillar scores

# event signals: insider trades, promoter pledges, rating actions, severe announcements (NSE, no quota)
python scripts/ingest_events.py --status                # stored events by type; feed months that came back empty
python scripts/ingest_events.py                         # first run ~1.5-2 h; then only new months / stale companies
python scripts/ingest_events.py --symbol IRB            # one company's announcements only
python scripts/screen.py --explain IRB                  # EVENTS panel; severe events veto a stock for 12 months

# filings search: every filing page (keyword search) + the passages worth reading (meaning), one company at a time
python scripts/index_filings.py --status                # extracted / embedded / OCR'd pages
python scripts/index_filings.py --repair                # re-download truncated filings first, then index
python scripts/index_filings.py                         # new filings only: extract + OCR, then embed passages
python scripts/search_filings.py "NIM guidance" --symbol AUBANK --type earnings_call --as-of 2026-06-30
python scripts/build_filings_eval.py --n 100            # golden questions from real passages (local LLM)
python scripts/review_eval_set.py --file eval/filings_golden.jsonl --pending
python scripts/eval_filings.py                          # page recall@k + MRR per search mode, verified rows

# analyst evidence (statutory XBRL + verified notes): what the evidence pack hands the analyst as one fact sheet
python scripts/ingest_integrated_filings.py --status    # filed-year + related-party coverage (reads the cache only)
python scripts/ingest_integrated_filings.py             # NSE integrated filings: audited figures, Reg 23(9) related parties
python scripts/ingest_ownership.py --refresh DIXON      # statutory shareholding XBRL: promoter pledge + encumbrance
python scripts/extract_contingent_liabilities.py --status
python scripts/extract_contingent_liabilities.py --symbol DIXON --force   # re-read one note with the local model
python scripts/eval_contingent.py                       # hand-checked gold set; exits 1 on any verified-but-wrong
PENNY_LEDGER_ENABLED=false python demo.py "<query>"     # a test run that records no ledger call
```

<!-- Record the exact commands you run often, e.g.:
- Build llama.cpp:  cd llama.cpp-master && make
- Run notebook server:  jupyter lab
-->

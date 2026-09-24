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
```

<!-- Record the exact commands you run often, e.g.:
- Build llama.cpp:  cd llama.cpp-master && make
- Run notebook server:  jupyter lab
-->

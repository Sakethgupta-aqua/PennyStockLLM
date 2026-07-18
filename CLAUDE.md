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

<!-- Record the exact commands you run often, e.g.:
- Build llama.cpp:  cd llama.cpp-master && make
- Run notebook server:  jupyter lab
-->

"""PennyStockLLM — accountable research copilot for Indian small/micro-cap equities.

Package layout (see PRODUCTION_MASTER_PLAN.md §5):
  config     — central settings (pydantic-settings); no hard-coded paths/secrets elsewhere
  data       — symbol resolution (name/ticker -> official NSE symbol) + universe DB
  retrieval  — hybrid RAG: BM25 sparse + Chroma dense + reciprocal-rank fusion
  tools      — the agent tools (market data, RAG researcher)
  agents     — prompts, graph state, nodes, and the compiled LangGraph pipeline
"""
__version__ = "0.1.0"

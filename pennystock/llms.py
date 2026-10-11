"""Factory functions for the Ollama models - one per agent role.

Each reads its model name from ``config`` so a model can be swapped via env var without touching
code (e.g. ``PENNY_DATA_MODEL=qwen3.5:latest``). Construction is cheap and does not open a
connection, so importing this module is free.

KEEP-ALIVE matters more than it looks. These factories originally pinned models with
keep_alive=-1 ("forever"). With six roles - and after the grounding auditor was moved to a 23GB
model - that meant several large models resident simultaneously. Measured consequence on this
machine: 11.6GB held on the GPU, 3.2GB of 4GB swap in use, ~500MB free, and a single eval-set
question taking 90 minutes. Models now expire: small ones stay warm for a few minutes across a
run, large ones unload as soon as they are done.
"""
from functools import lru_cache

from langchain_ollama import ChatOllama

from pennystock.config import settings


def _make(model: str, temperature: float, keep_alive: str, reasoning: bool | None = None,
          num_ctx: int | None = None, num_predict: int | None = None) -> ChatOllama:
    # reasoning=None sends nothing and leaves the model's own default; only roles that run a
    # thinking-capable model pass True/False (see settings.analyst_reasoning)
    return ChatOllama(model=model, temperature=temperature, keep_alive=keep_alive,
                      num_ctx=num_ctx or settings.num_ctx, reasoning=reasoning, num_predict=num_predict)


def data_agent_llm() -> ChatOllama:
    return _make(settings.data_model, temperature=0.0, keep_alive=settings.small_model_keep_alive)


def analyst_llm() -> ChatOllama:
    return _make(settings.analyst_model, temperature=0.2, keep_alive=settings.large_model_keep_alive,
                 reasoning=settings.analyst_reasoning, num_ctx=settings.large_num_ctx)


def judge_llm() -> ChatOllama:
    return _make(settings.judge_model, temperature=0.0, keep_alive=settings.large_model_keep_alive,
                 num_predict=settings.judge_max_tokens)


def extraction_llm() -> ChatOllama:
    """The analyst's model, offline: lays a contingent-liability table out as rows. It computes nothing -
    every number it returns is verified against the page in code - so thinking is off; warm across a batch."""
    return _make(settings.analyst_model, temperature=0.0, keep_alive=settings.extraction_keep_alive,
                 reasoning=False, num_ctx=settings.extraction_num_ctx)


@lru_cache(maxsize=1)
def decomposer_llm() -> ChatOllama:
    """Fast model for splitting a research request into focused sub-queries (multi-query RAG)."""
    return _make(settings.decomposer_model, temperature=0.0, keep_alive=settings.small_model_keep_alive)


@lru_cache(maxsize=1)
def grounding_llm() -> ChatOllama:
    """Independent auditor for the CRAG grounding gate (deliberately NOT the analyst's own model).

    Uses the LARGE keep-alive because grounding_model is a 35B: pinning it would hold ~23GB."""
    return _make(settings.grounding_model, temperature=0.0, keep_alive=settings.large_model_keep_alive,
                 reasoning=settings.grounding_reasoning, num_ctx=settings.large_num_ctx,
                 num_predict=settings.grounding_max_tokens)    # a verdict, never an essay (2026-10-11)

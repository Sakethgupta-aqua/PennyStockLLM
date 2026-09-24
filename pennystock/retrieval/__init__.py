"""Hybrid retrieval: BM25 sparse + Chroma dense, merged by reciprocal-rank fusion."""
from pennystock.retrieval.fusion import reciprocal_rank_fusion

__all__ = ["reciprocal_rank_fusion"]

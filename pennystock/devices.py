"""Where the embedding and reranking models run, and at what precision - one place for both.

Measured on this M4 Max (2026-10-02): half precision halves each model's weights (bge-m3 and
bge-reranker-v2-m3: 2.12 GB -> 1.06 GB each, 2.12 GB freed in total) with the same results
(embedding cosine >= 0.9998 against fp32, reranker top-8 unchanged) and a 17% faster rerank. On a
36 GB Mac the binding constraint is free RAM, not GPU speed - Ollama logged "system_limited=true"
evictions while loading the 22.5 GB analyst model - so those 2 GB matter more than milliseconds.

fp16 is GPU-only: on a CPU it is slow or unsupported, so CPU always runs fp32.
"""
from __future__ import annotations

from pennystock.config import settings


def torch_device() -> str:
    """`settings.device`, or the best available accelerator: NVIDIA GPU, Apple GPU, then CPU."""
    if settings.device != "auto":
        return settings.device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def torch_dtype(device: str):
    import torch
    return torch.float16 if settings.half_precision and device in ("cuda", "mps") else torch.float32

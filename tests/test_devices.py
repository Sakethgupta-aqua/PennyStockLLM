"""Where the embedder and reranker run, and at what precision - decided in ONE place.

The rule being protected: half precision only on a GPU. Measured on the M4 Max it halves each model's
weights (2.12 -> 1.06 GB) with identical retrieval results; on a CPU fp16 is slow, so it must never be
picked there, and an explicit device setting always wins over auto-detection.
"""
import torch

from pennystock import devices


def test_auto_prefers_cuda_then_apple_gpu_then_cpu(monkeypatch):
    monkeypatch.setattr(devices.settings, "device", "auto")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert devices.torch_device() == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert devices.torch_device() == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert devices.torch_device() == "cuda"


def test_an_explicit_device_overrides_detection(monkeypatch):
    monkeypatch.setattr(devices.settings, "device", "cpu")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert devices.torch_device() == "cpu"


def test_half_precision_only_on_a_gpu(monkeypatch):
    monkeypatch.setattr(devices.settings, "half_precision", True)
    assert devices.torch_dtype("mps") is torch.float16
    assert devices.torch_dtype("cuda") is torch.float16
    assert devices.torch_dtype("cpu") is torch.float32
    monkeypatch.setattr(devices.settings, "half_precision", False)
    assert devices.torch_dtype("mps") is torch.float32


def test_the_embedder_never_reads_more_than_the_token_cap(monkeypatch):
    # a garbled text layer runs to ~1 token per character: 32 such passages asked the Mac's GPU for one
    # 29.9 GB buffer (32 x 16 heads x 3,819^2 tokens x 4 bytes) and aborted the filings index (2026-10-06)
    import langchain_huggingface

    from pennystock.retrieval import store

    class FakeClient:
        max_seq_length = 8192

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            self._client = FakeClient()
    monkeypatch.setattr(langchain_huggingface, "HuggingFaceEmbeddings", FakeEmbeddings)
    assert store._embeddings()._client.max_seq_length == store.settings.embed_max_tokens == 1024

# Using the Mac's GPU well — measured on this machine

Apple M4 Max · 32-core GPU · 36 GB unified memory · Metal 4 · macOS 26.6 · Ollama 0.32.15
All numbers below were measured here on 2026-10-02 unless a source is given.

## Where the GPU time actually goes

| Workload | Runs on | Measured |
|---|---|---|
| Analyst + grounding LLM (`qwen3.5:35b`, Qwen3.5-35B-A3B MoE, Q4_K_M GGUF) | Ollama → llama.cpp Metal, 42/42 layers on GPU, flash attention on | ~76 tokens/s generation; 6.4 s to load |
| Supervisor / data agent / judge (7–8B GGUF) | same | — |
| bge-m3 embeddings, bge-reranker-v2-m3 | PyTorch MPS (Apple GPU) | 38–41 chunks/s; 40 pairs reranked in ~1.2 s |

## What changed, and the evidence

1. **Hidden "thinking" is now OFF for the analyst and grounding roles** (`analyst_reasoning`,
   `grounding_reasoning` in `config.py`). Qwen3.5 reasons in hidden tokens unless told not to:
   the same short answer took 1,381 tokens / 24.7 s with thinking and 157 tokens / 2.1 s without.
   End to end, the same AU Small Finance Bank question went from **402.7 s to 106.5 s** with the same
   tools, the same judge score (5/5), and a cleaner report. Set either to `None` (model default) or
   `True` to turn it back on, e.g. `PENNY_ANALYST_REASONING=true`.
2. **The embedder and reranker run in half precision on the GPU** (`half_precision`,
   `pennystock/devices.py`). Each model's weights drop from 2.12 GB to 1.06 GB; retrieval on the
   golden set is identical (hit@1 0.900, hit@3 1.000, MRR 0.933 before and after) and reranking is
   ~17% faster. CPU always stays fp32.

## What was tested and deliberately NOT changed

- **Keeping the 35B model loaded between calls** (`PENNY_LARGE_MODEL_KEEP_ALIVE=2m`): the run took
  119.1 s vs 106.5 s, free memory fell to 23% and swap rose to 9.7 GB. Ollama's log shows why —
  `model predicted to exceed available memory, evicting ... system_limited=true`: on a 36 GB Mac
  with other apps open, **free RAM is the limit, not GPU speed**, so the model is evicted anyway.
  Keep the default `"0"`.
- **Raising the GPU memory cap** (`sudo sysctl iogpu.wired_limit_mb=...`): the default (~75%,
  27.6 GiB here) is not what binds — free system RAM is — and Apple advises staying under ~70%.

## Optional, your call (system-level, not changed for you)

**Smaller KV cache.** Halves the attention cache with negligible quality loss (Ollama docs); worth
~1 GB per 8B model at 16k context, ~160 MB for Qwen3.5 (it has few full-attention layers):

```bash
launchctl setenv OLLAMA_KV_CACHE_TYPE q8_0     # then quit and reopen the Ollama app
launchctl unsetenv OLLAMA_KV_CACHE_TYPE        # to undo
```

**Ollama's MLX engine.** Your models are GGUF, so they run on llama.cpp. Ollama has run MLX-format
models natively since v0.19 (preview, Mar 2026) and by default since v0.30 (May 2026, needs ≥32 GB).
Published M4 Max numbers for this exact model: Ollama/GGUF 41–48 tok/s, MLX 90–130 tok/s. Same
memory footprint (22 GB). To A/B it before switching:

```bash
ollama pull qwen3.5:35b-a3b-nvfp4                                    # 22 GB, MLX format
PENNY_ANALYST_MODEL=qwen3.5:35b-a3b-nvfp4 PENNY_GROUNDING_MODEL=qwen3.5:35b-a3b-nvfp4 \
  python demo.py "<the same question>"   # compare duration_s + check_traces with the GGUF run
```

Switch only if tool calling, the Layer-0 checks and the judge hold up.

## Sources

[Ollama blog — MLX on Apple Silicon](https://ollama.com/blog) ·
[Ollama v0.30 MLX release notes](https://runaihome.com/blog/ollama-v030-mlx-stable-upgrade-2026/) ·
[Qwen3.5-35B on M4 Max: Ollama vs llama.cpp vs MLX](https://antekapetanovic.com/blog/qwen3.5-apple-silicon-benchmark/) ·
[MLX vs llama.cpp on Apple Silicon](https://yage.ai/share/mlx-apple-silicon-en-20260331.html) ·
[KV-cache quantization and flash attention](https://mljourney.com/how-to-speed-up-ollama-gpu-quantization-and-performance-tuning/) ·
[Wired-memory limit on Apple Silicon](https://mehmetbaykar.com/posts/increase-vram-apple-silicon-local-llm/) ·
[Ollama keep-alive and reload cost](https://dev.to/jovan_chan_9500711396d4e6/ollama-keeps-reloading-the-model-fix-vram-unloading-cold-starts-and-model-swapping-2026-25b7)

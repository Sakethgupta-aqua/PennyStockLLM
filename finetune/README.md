# Fine-tuning Llama-3-8B to replace the MarketDataAgent (QLoRA on Apple Silicon)

Goal: a specialized model that reliably calls `get_indian_stock_data(symbol, exchange)`,
then drop it into `Stocks_PG_Code.ipynb` as `ChatOllama(model="penny-llama")`.

All commands use the **pennystock** conda env:
    PY=/opt/anaconda3/envs/pennystock/bin

Golden rule: **training format must match inference format.** We train with the Llama-3.1
tool-calling chat template and serve with the same template in Ollama.

---

## Phase 0 — Dataset (you are here: hand-writing)

Each line of `data/train.jsonl` / `data/valid.jsonl` is one example:
`{"messages": [system, user, assistant-with-tool_calls], "tools": [<schema>]}`

See the seed examples already in those files. Guidelines:
- Aim for **150–300 train** + **20–40 valid** examples.
- Vary: company names (full name vs ticker), NSE vs BSE, phrasings, typos, partial info.
- Keep the `tools` schema identical on every line.
- The assistant turn has empty `content` and a `tool_calls` list — that is the label.
- (Optional, advanced) add multi-turn examples: tool result -> assistant final summary.

Validate the format + that the template renders tool calls BEFORE training:

    $PY -c "
    import json
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('/Users/saketh/models/Llama-3-8B-mlx-q4')  # after Phase 1+2
    d = json.loads(open('data/train.jsonl').readline())
    print(tok.apply_chat_template(d['messages'], tools=d['tools'], tokenize=False))
    "
    # The printed text MUST contain the tool name + arguments. If it doesn't, the
    # chat template wasn't overridden (Phase 2).

---

## Phase 1 — Make the MLX 4-bit base (the real "quantize to fine-tune" step)

NOTE: this uses the **safetensors** at /Users/saketh/models/Llama-3-8B, NOT the .gguf.

    $PY/mlx_lm.convert \
      --hf-path /Users/saketh/models/Llama-3-8B \
      -q --q-bits 4 \
      --mlx-path /Users/saketh/models/Llama-3-8B-mlx-q4

## Phase 2 — Override the chat template (base Llama-3 template has NO tool support)

Replace the `chat_template` field in
`/Users/saketh/models/Llama-3-8B-mlx-q4/tokenizer_config.json`
with the Llama-3.1 tool-calling Jinja template.
>>> Ask Claude to drop the template in, or copy `chat_template` from a
    Meta-Llama-3.1-8B-Instruct tokenizer_config.json. Then re-run the Phase 0 validator.

## Phase 3 — LoRA fine-tune (QLoRA, since base is 4-bit)

    $PY/mlx_lm.lora \
      --model /Users/saketh/models/Llama-3-8B-mlx-q4 \
      --train \
      --data /Users/saketh/PennyStockLLM/finetune/data \
      --adapter-path /Users/saketh/PennyStockLLM/finetune/adapters \
      --iters 300 --batch-size 1 --num-layers 8 \
      --learning-rate 1e-4 --mask-prompt --steps-per-report 10

Watch train/val loss. ~36GB RAM is plenty for 8B 4-bit LoRA. Bump --iters if val loss
still falling; lower --num-layers if memory-tight.

## Phase 4 — Fuse adapter into fp16 weights

    $PY/mlx_lm.fuse \
      --model /Users/saketh/models/Llama-3-8B-mlx-q4 \
      --adapter-path /Users/saketh/PennyStockLLM/finetune/adapters \
      --save-path /Users/saketh/models/penny-llama-fused \
      --de-quantize

## Phase 5 — Convert to GGUF F16 (same script you already used)

    $PY/python /Users/saketh/PennyStockLLM/llama.cpp-master/convert_hf_to_gguf.py \
      /Users/saketh/models/penny-llama-fused \
      --outfile /Users/saketh/models/penny-llama-F16.gguf --outtype f16

## Phase 6 — Quantize to Q4_K_M (regenerates your .gguf, now fine-tuned)

    # verify the binary name first: ls /Users/saketh/PennyStockLLM/llama.cpp-master/build/bin | grep quantize
    /Users/saketh/PennyStockLLM/llama.cpp-master/build/bin/llama-quantize \
      /Users/saketh/models/penny-llama-F16.gguf \
      /Users/saketh/models/penny-llama-Q4_K_M.gguf Q4_K_M

## Phase 7 — Import into Ollama (reuse a known-good tools template)

    # grab a working llama3.1 tools template to start from:
    ollama show llama3-groq-tool-use:8b --modelfile > Modelfile
    # edit Modelfile: change the first line to
    #   FROM /Users/saketh/models/penny-llama-Q4_K_M.gguf
    # keep its TEMPLATE/PARAMETER lines (they already handle tool calls)
    ollama create penny-llama -f Modelfile
    ollama run penny-llama "Get me Suzlon's data on NSE"   # smoke test

## Phase 8 — Use it in the notebook (one line, cell 1)

    llm_data_agent = ChatOllama(model="penny-llama", temperature=0.0, keep_alive=-1, num_ctx=8192)

Then re-run the data-agent node and confirm it emits a get_indian_stock_data tool call.

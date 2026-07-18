"""
Step 6 — smoke-test the fine-tuned adapter.

Feeds the model the SAME shape it trained on (system prompt + tool schema + a user
question) and prints what it generates. We want to see a correct tool call like:
    {"name": "get_indian_stock_data", "parameters": {"symbol": "SUZLON", "exchange": "NSE"}}

Run:
    /opt/anaconda3/envs/pennystock/bin/python finetune/test_adapter.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlx_lm import load, generate
from generate_dataset import SYSTEM, TOOL   # reuse the EXACT system prompt + tool schema

BASE = "/Users/saketh/models/Llama-3-8B-mlx-q4"
ADAPTERS = "/Users/saketh/PennyStockLLM/finetune/adapters"

# load the frozen 4-bit base WITH the trained adapter on top
model, tokenizer = load(BASE, adapter_path=ADAPTERS)

# a spread of real-world phrasings, incl. one stock NOT in training (Adani Power) to
# check it learned the *skill*, not just memorized the list.
TESTS = [
    "Get me Suzlon's data on the NSE.",                 # name + NSE explicit
    "How is Yes Bank trading on the BSE?",              # name + BSE explicit
    "fetch IRFC",                                       # raw ticker, no exchange
    "What's Vodafone Idea doing today?",                # casual, no exchange -> NSE
    "Pull up Tata Power from the National Stock Exchange.",
    "How is Adani Power doing on the NSE?",             # UNSEEN stock -> generalization
]

for q in TESTS:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": q}]
    prompt = tokenizer.apply_chat_template(
        messages, tools=[TOOL], add_generation_prompt=True
    )
    out = generate(model, tokenizer, prompt=prompt, max_tokens=128, verbose=False)
    print("Q:", q)
    print("A:", out.strip())
    print("-" * 70)

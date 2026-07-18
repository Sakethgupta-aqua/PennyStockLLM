"""
Generate a BALANCED QLoRA fine-tuning dataset for the MarketDataAgent.

v2 fixes (after the first run learned the format but not the tickers):
  - WIDE, sector-diverse stock list (large-cap, IT, banks, PSU, metals, energy,
    auto, pharma, penny) instead of a bank-heavy list.
  - GUARANTEED BALANCE: every stock gets the exact same number of examples, so no
    single ticker fragment (e.g. "BANK") can dominate.
  - More reinforcement per mapping so the company -> ticker association actually sticks.

Output: data/train.jsonl, data/valid.jsonl in mlx-lm chat format
        {"messages": [...], "tools": [<schema>]}

Run: /opt/anaconda3/envs/pennystock/bin/python finetune/generate_dataset.py
"""
import json, random, os
from collections import Counter

random.seed(42)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")

# ---- EXACT system prompt from the notebook's MarketDataAgent (train == inference) ----
SYSTEM = (
    "You are a strictly constrained Market Data Agent. "
    "Your ONLY job is to fetch NSE/BSE data using the provided tools. "
    "CRITICAL: DO NOT analyze the data, do not compare it to PDFs, and do not write a report. "
    "If the tool returns an error, output the error message so the user knows it failed. "
    "Only if the tool returns actual JSON data, reply with: 'Data fetched successfully. "
    "Ready for the FinancialAnalystAgent to perform the analysis.'"
)
SUCCESS_REPLY = ("Data fetched successfully. Ready for the FinancialAnalystAgent "
                 "to perform the analysis.")

# ---- Tool schema (name + param names identical to the notebook's @tool) ----
TOOL = {
    "type": "function",
    "function": {
        "name": "get_indian_stock_data",
        "description": "Fetches the real-time and historical stock data for Indian equities from NSE or BSE.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string",
                           "description": "The ticker symbol of the Indian equity, e.g. RELIANCE, SUZLON, TCS."},
                "exchange": {"type": "string", "enum": ["NSE", "BSE"],
                             "description": "The exchange to query. Defaults to NSE."},
            },
            "required": ["symbol"],
        },
    },
}

# ---- WIDE, sector-diverse set of real NSE-listed stocks: (Company Name, ticker) ----
STOCKS = [
    # large-cap / blue-chip
    ("Reliance Industries", "RELIANCE"), ("Tata Consultancy Services", "TCS"),
    ("Infosys", "INFY"), ("HDFC Bank", "HDFCBANK"), ("ICICI Bank", "ICICIBANK"),
    ("State Bank of India", "SBIN"), ("Hindustan Unilever", "HINDUNILVR"), ("ITC", "ITC"),
    ("Larsen & Toubro", "LT"), ("Bharti Airtel", "BHARTIARTL"), ("Kotak Mahindra Bank", "KOTAKBANK"),
    ("Axis Bank", "AXISBANK"), ("Maruti Suzuki", "MARUTI"), ("Sun Pharma", "SUNPHARMA"),
    ("Titan Company", "TITAN"),
    # mid-cap / varied sectors
    ("Tata Motors", "TATAMOTORS"), ("Tata Steel", "TATASTEEL"), ("Tata Power", "TATAPOWER"),
    ("JSW Steel", "JSWSTEEL"), ("Hindalco Industries", "HINDALCO"), ("Adani Enterprises", "ADANIENT"),
    ("Adani Power", "ADANIPOWER"), ("Power Grid Corporation", "POWERGRID"), ("NTPC", "NTPC"),
    ("Coal India", "COALINDIA"), ("Oil and Natural Gas Corporation", "ONGC"), ("Cipla", "CIPLA"),
    ("Eicher Motors", "EICHERMOT"), ("UltraTech Cement", "ULTRACEMCO"), ("Wipro", "WIPRO"),
    # small-cap / PSU / penny
    ("Suzlon Energy", "SUZLON"), ("Yes Bank", "YESBANK"), ("Vodafone Idea", "IDEA"),
    ("Reliance Power", "RPOWER"), ("Jaiprakash Power Ventures", "JPPOWER"),
    ("Indian Railway Finance Corporation", "IRFC"), ("NHPC", "NHPC"), ("SJVN", "SJVN"),
    ("BHEL", "BHEL"), ("Steel Authority of India", "SAIL"), ("NMDC", "NMDC"),
    ("Indian Overseas Bank", "IOB"), ("Punjab National Bank", "PNB"), ("Rail Vikas Nigam", "RVNL"),
    ("Indian Renewable Energy Development Agency", "IREDA"),
]

# Delisted / suspended names -> realistic "delisted" errors for category C
BAD_STOCKS = [
    ("Reliance Communications", "RCOM"), ("Future Retail", "FRETAIL"),
    ("Sintex Industries", "SINTEX"), ("DHFL", "DHFL"), ("Jet Airways", "JETAIRWAYS"),
]

EXAMPLES_A_PER_STOCK = 8   # tool-call examples per stock (the core skill)
EXAMPLES_B_PER_STOCK = 3   # success-flow examples per stock
VALID_PER_STOCK      = 1   # held-out tool-call example per stock

# ---- Phrasings. {n}=name, {t}=ticker, {e}=exchange phrase. ----
NAME_TEMPLATES = [
    "Get me the latest market data for {n} on the {e}.",
    "Pull {n}'s live quote from the {e}.",
    "What is {n} doing on the {e} today?",
    "Fetch the current numbers for {n} ({e}).",
    "I need {n} stock data from the {e}.",
    "Can you check {n} on the {e}?",
    "Show me {n}'s price action on the {e}.",
    "Pull up {n} listed on the {e}.",
    "Grab the live data for {n}, {e} please.",
    "Market data request: {n}, {e}.",
    "How is {n} trading on the {e} right now?",
    "Give me a live quote for {n} on the {e}.",
]
NAME_TEMPLATES_NOEX = [  # no exchange mentioned -> default NSE, omit exchange arg
    "Latest on {n}?", "How is {n} trading today?", "Pull {n}'s current numbers.",
    "Get me data for {n}.", "What's the live quote for {n}?", "Check {n} for me.",
    "hey can u grab {n} numbers", "i want the current data for {n}", "Fetch {n}'s market data.",
]
TICKER_TEMPLATES = [
    "Get {t} data.", "{t} on the {e}, please.", "Pull live numbers for {t} ({e}).",
    "Quote for {t}.", "{t} {e} data now.", "fetch {t}",
]
EXCHANGE_PHRASE = {
    "NSE": ["NSE", "National Stock Exchange", "NSE exchange"],
    "BSE": ["BSE", "Bombay Stock Exchange", "BSE exchange"],
}

def tool_call(symbol, exchange):
    args = {"symbol": symbol}
    if exchange:
        args["exchange"] = exchange
    return {"type": "function",
            "function": {"name": "get_indian_stock_data", "arguments": args}}

def fake_success_json(name, ticker, exchange):
    px = round(random.uniform(3, 3200), 2)
    return json.dumps({
        "longName": name, "symbol": f"{ticker}.{'NS' if exchange!='BSE' else 'BO'}",
        "currentPrice": px, "previousClose": round(px*random.uniform(0.95,1.05),2),
        "dayHigh": round(px*random.uniform(1.0,1.06),2), "dayLow": round(px*random.uniform(0.94,1.0),2),
        "volume": random.randint(50_000, 25_000_000), "currency": "INR",
    })

def delisted_error(ticker, exchange):
    suf = "NS" if exchange != "BSE" else "BO"
    return random.choice([
        f"Error: No price data found for {ticker}.{suf} (symbol may be delisted or invalid).",
        f"Error: yfinance returned empty info for {ticker}.{suf}.",
        f"Error: HTTPError 404 Not Found for ticker {ticker}.{suf}.",
    ])

def transient_error(ticker, exchange):
    suf = "NS" if exchange != "BSE" else "BO"
    return random.choice([
        f"Error: Read timed out while fetching {ticker}.{suf}.",
        f"Error: Rate limited by data provider for {ticker}.{suf}, try again later.",
        f"Error: Connection reset fetching {ticker}.{suf}.",
    ])

def pick_exchange_mode():
    return random.choices(["none", "NSE", "BSE"], weights=[40, 30, 30])[0]

def make_user_and_call(name, ticker):
    mode = pick_exchange_mode()
    style = random.choices(["name", "name_noex", "ticker"], weights=[55, 25, 20])[0]
    if style == "name_noex" or mode == "none":
        if style == "ticker":
            text = random.choice([t for t in TICKER_TEMPLATES if "{e}" not in t]).format(t=ticker)
        else:
            text = random.choice(NAME_TEMPLATES_NOEX).format(n=name)
        return text, tool_call(ticker, None), None
    e = random.choice(EXCHANGE_PHRASE[mode])
    if style == "ticker":
        text = random.choice(TICKER_TEMPLATES).format(t=ticker, e=e)
    else:
        text = random.choice(NAME_TEMPLATES).format(n=name, e=e)
    return text, tool_call(ticker, mode), mode

def ex_A(name, ticker):
    text, call, _ = make_user_and_call(name, ticker)
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": text},
                         {"role": "assistant", "content": "", "tool_calls": [call]}],
            "tools": [TOOL]}

def ex_B(name, ticker):
    text, call, mode = make_user_and_call(name, ticker)
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": text},
                         {"role": "assistant", "content": "", "tool_calls": [call]},
                         {"role": "tool", "content": fake_success_json(name, ticker, mode)},
                         {"role": "assistant", "content": SUCCESS_REPLY}],
            "tools": [TOOL]}

def ex_C(name, ticker, kind="delisted"):
    text, call, mode = make_user_and_call(name, ticker)
    err = delisted_error(ticker, mode) if kind == "delisted" else transient_error(ticker, mode)
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": text},
                         {"role": "assistant", "content": "", "tool_calls": [call]},
                         {"role": "tool", "content": err},
                         {"role": "assistant", "content": err}],
            "tools": [TOOL]}

def _key(row):
    return (row["messages"][1]["content"] + "|" + row["messages"][-1].get("content", "")
            + "|" + json.dumps(row["messages"][2].get("tool_calls")))

def make_unique(maker, n, seen, *args):
    """Produce n unique examples from maker, skipping duplicates."""
    out, tries = [], 0
    while len(out) < n and tries < n * 40:
        row = maker(*args); tries += 1
        k = _key(row)
        if k not in seen:
            seen.add(k); out.append(row)
    return out

def build():
    seen = set()
    train, valid = [], []
    # BALANCED: every stock gets the same counts
    for name, ticker in STOCKS:
        train += make_unique(ex_A, EXAMPLES_A_PER_STOCK, seen, name, ticker)
        train += make_unique(ex_B, EXAMPLES_B_PER_STOCK, seen, name, ticker)
        valid += make_unique(ex_A, VALID_PER_STOCK, seen, name, ticker)
    # error flow: delisted names + a few transient errors on real stocks
    for name, ticker in BAD_STOCKS:
        train += make_unique(lambda n, t: ex_C(n, t, "delisted"), 5, seen, name, ticker)
    for name, ticker in random.sample(STOCKS, 20):
        train += make_unique(lambda n, t: ex_C(n, t, "transient"), 1, seen, name, ticker)
    random.shuffle(train); random.shuffle(valid)
    return train, valid

def cat(r):
    last = r["messages"][-1]
    if "tool_calls" in last: return "A"
    return "B" if last["content"] == SUCCESS_REPLY else "C"

def main():
    os.makedirs(OUT, exist_ok=True)
    train, valid = build()
    for fname, data in [("train.jsonl", train), ("valid.jsonl", valid)]:
        with open(os.path.join(OUT, fname), "w") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")
    print(f"train={len(train)}  valid={len(valid)}")
    print("train categories:", dict(Counter(cat(r) for r in train)))
    # balance check: how many times each ticker appears in train
    tickers = [r["messages"][2]["tool_calls"][0]["function"]["arguments"]["symbol"] for r in train]
    c = Counter(tickers)
    print(f"distinct tickers: {len(c)} | per-ticker min={min(c.values())} max={max(c.values())}")

if __name__ == "__main__":
    main()

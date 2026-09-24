"""
Generate a QLoRA fine-tuning dataset for the MarketDataAgent  (v3.1: NAME PASS-THROUGH, WIDE).

v3 philosophy (unchanged):
  Teach the model to put into the tool's `symbol` field EXACTLY what the user typed -
  the company NAME if they used a name, the TICKER if they used a ticker. A separate
  resolver (scripts/resolver.py) maps names -> tickers. Copying generalises to the whole
  market; memorizing tickers does not (v2's "KAYNESTEEL" hallucination).

v3.1 additions:
  - STOCKS expanded 45 -> 106 real NSE companies (every one VERIFIED against market.db:
    both the written name and the ticker resolve to the expected symbol).
  - Long analytical queries (like real production questions), lowercase/casual phrasing.
  - Edge cases, each verified at generation time against the resolver so we NEVER teach
    an unresolvable output:
      * typo'd names        ("Suzlon Enegry")   -> pass the typo verbatim (fuzzy handles it)
      * partial names       ("Hindalco")        -> pass the shorthand verbatim
      * name + ticker combo ("Suzlon Energy (SUZLON)") -> pass the TICKER (most precise)
      * lowercase tickers   ("fetch irfc")      -> pass verbatim
      * suffixed tickers    ("SUZLON.NS")       -> strip suffix, set exchange from it
  - main() runs a hard AUDIT: every trained symbol (except deliberate error rows) must
    resolve via scripts/resolver.py, else it fails loudly. One-time-job insurance.

Output: data/train.jsonl, data/valid.jsonl in mlx-lm chat format {"messages":[...],"tools":[...]}
Run:    /opt/anaconda3/bin/python finetune/generate_dataset.py

IMPORTANT: keep SYSTEM below identical to DATA_AGENT_PROMPT in Stocks_PG_Code.ipynb.
"""
import json, random, os, sys
from collections import Counter

random.seed(42)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")

# The dataset now depends on the resolver: edge cases are only emitted if they resolve.
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
try:
    from resolver import resolve_symbol
except Exception as e:
    sys.exit(f"Cannot import scripts/resolver.py (is market.db built?): {e}")

# ---- System prompt: MUST match the notebook's DATA_AGENT_PROMPT exactly ----
SYSTEM = (
    "You are a strictly constrained Market Data Agent. "
    "Your ONLY job is to fetch NSE/BSE stock data using the provided tool. "
    "SYMBOL RULE: Put the stock into the tool's `symbol` field EXACTLY as the user referred to it. "
    "If the user gave a company name, pass the company name; if the user gave a ticker, pass the ticker. "
    "NEVER convert a name into a ticker yourself, and NEVER invent or guess a ticker - the tool resolves "
    "company names to tickers automatically. "
    "CRITICAL: DO NOT analyze the data, do not compare it to PDFs, and do not write a report. "
    "If the tool returns an error, output the error message so the user knows it failed. "
    "Only if the tool returns actual JSON data, reply with: 'Data fetched successfully. "
    "Ready for the FinancialAnalystAgent to perform the analysis.'"
)
SUCCESS_REPLY = ("Data fetched successfully. Ready for the FinancialAnalystAgent "
                 "to perform the analysis.")

# ---- Tool schema: `symbol` accepts a ticker OR a company name (matches the notebook) ----
TOOL = {
    "type": "function",
    "function": {
        "name": "get_indian_stock_data",
        "description": "Fetches the real-time and historical stock data for Indian equities from NSE or BSE.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string",
                           "description": "The stock to fetch: either an exact ticker (e.g. RELIANCE) OR "
                                          "the company name exactly as the user wrote it (e.g. 'Kaynes "
                                          "Technology'). Company names are resolved to tickers automatically."},
                "exchange": {"type": "string", "enum": ["NSE", "BSE"],
                             "description": "The exchange to query. Defaults to NSE."},
            },
            "required": ["symbol"],
        },
    },
}

# ---- 106 real NSE stocks: (Company Name as a user writes it, ticker). ----
# Every pair verified against market.db: resolve(name)==ticker and resolve(ticker)==ticker.
STOCKS = [
    # large-cap / blue-chip
    ("Reliance Industries", "RELIANCE"), ("Tata Consultancy Services", "TCS"),
    ("Infosys", "INFY"), ("HDFC Bank", "HDFCBANK"), ("ICICI Bank", "ICICIBANK"),
    ("State Bank of India", "SBIN"), ("Hindustan Unilever", "HINDUNILVR"), ("ITC", "ITC"),
    ("Larsen & Toubro", "LT"), ("Bharti Airtel", "BHARTIARTL"), ("Kotak Mahindra Bank", "KOTAKBANK"),
    ("Axis Bank", "AXISBANK"), ("Maruti Suzuki", "MARUTI"), ("Sun Pharma", "SUNPHARMA"),
    ("Titan Company", "TITAN"), ("Bajaj Finance", "BAJFINANCE"), ("Asian Paints", "ASIANPAINT"),
    ("HCL Technologies", "HCLTECH"), ("Tech Mahindra", "TECHM"), ("Nestle India", "NESTLEIND"),
    ("Mahindra & Mahindra", "M&M"), ("Bajaj Auto", "BAJAJ-AUTO"),
    ("Adani Ports and Special Economic Zone", "ADANIPORTS"),
    ("Dr Reddy's Laboratories", "DRREDDY"), ("Divi's Laboratories", "DIVISLAB"),
    ("Apollo Hospitals Enterprise", "APOLLOHOSP"), ("IndusInd Bank", "INDUSINDBK"),
    ("InterGlobe Aviation", "INDIGO"), ("Zee Entertainment Enterprises", "ZEEL"),
    # mid-cap / varied sectors
    ("Tata Motors", "TATAMOTORS"), ("Tata Steel", "TATASTEEL"), ("Tata Power", "TATAPOWER"),
    ("JSW Steel", "JSWSTEEL"), ("Hindalco Industries", "HINDALCO"), ("Adani Enterprises", "ADANIENT"),
    ("Adani Power", "ADANIPOWER"), ("Power Grid Corporation", "POWERGRID"), ("NTPC", "NTPC"),
    ("Coal India", "COALINDIA"), ("Oil and Natural Gas Corporation", "ONGC"), ("Cipla", "CIPLA"),
    ("Eicher Motors", "EICHERMOT"), ("UltraTech Cement", "ULTRACEMCO"), ("Wipro", "WIPRO"),
    ("JSW Energy", "JSWENERGY"), ("Torrent Power", "TORNTPOWER"), ("Tata Elxsi", "TATAELXSI"),
    ("Persistent Systems", "PERSISTENT"), ("Coforge", "COFORGE"), ("Lupin", "LUPIN"),
    ("Aurobindo Pharma", "AUROPHARMA"), ("Zydus Lifesciences", "ZYDUSLIFE"),
    ("Bank of Baroda", "BANKBARODA"), ("Canara Bank", "CANBK"), ("IDFC First Bank", "IDFCFIRSTB"),
    ("Bandhan Bank", "BANDHANBNK"), ("Vedanta", "VEDL"), ("National Aluminium Company", "NATIONALUM"),
    ("GAIL", "GAIL"), ("Indian Oil Corporation", "IOC"), ("Bharat Petroleum Corporation", "BPCL"),
    ("Hindustan Petroleum Corporation", "HINDPETRO"), ("Exide Industries", "EXIDEIND"),
    ("KPIT Technologies", "KPITTECH"), ("PVR Inox", "PVRINOX"),
    # PSU / defence / rail
    ("Hindustan Aeronautics", "HAL"), ("Bharat Electronics", "BEL"), ("Bharat Dynamics", "BDL"),
    ("Mazagon Dock Shipbuilders", "MAZDOCK"), ("Cochin Shipyard", "COCHINSHIP"),
    ("Garden Reach Shipbuilders", "GRSE"), ("Container Corporation of India", "CONCOR"),
    ("Indian Railway Catering and Tourism Corporation", "IRCTC"),
    # small-cap / PSU / penny
    ("Suzlon Energy", "SUZLON"), ("Yes Bank", "YESBANK"), ("Vodafone Idea", "IDEA"),
    ("Reliance Power", "RPOWER"), ("Jaiprakash Power Ventures", "JPPOWER"),
    ("Indian Railway Finance Corporation", "IRFC"), ("NHPC", "NHPC"), ("SJVN", "SJVN"),
    ("BHEL", "BHEL"), ("Steel Authority of India", "SAIL"), ("NMDC", "NMDC"),
    ("Indian Overseas Bank", "IOB"), ("Punjab National Bank", "PNB"), ("Rail Vikas Nigam", "RVNL"),
    ("Indian Renewable Energy Development Agency", "IREDA"),
    ("Kaynes Technology", "KAYNES"), ("Olectra Greentech", "OLECTRA"),
    ("South Indian Bank", "SOUTHBANK"), ("UCO Bank", "UCOBANK"),
    ("Central Bank of India", "CENTRALBK"), ("Trident", "TRIDENT"), ("HFCL", "HFCL"),
    ("Vakrangee", "VAKRANGEE"), ("RattanIndia Power", "RTNPOWER"), ("Alok Industries", "ALOKINDS"),
    ("MMTC", "MMTC"), ("Hindustan Copper", "HINDCOPPER"), ("NBCC", "NBCC"), ("ITI", "ITI"),
    ("IFCI", "IFCI"), ("IRB Infrastructure Developers", "IRB"), ("Dish TV", "DISHTV"),
    ("GTL Infrastructure", "GTLINFRA"),
]

# Delisted / suspended names -> realistic "delisted" errors for category C
BAD_STOCKS = [
    ("Reliance Communications", "RCOM"), ("Future Retail", "FRETAIL"),
    ("Sintex Industries", "SINTEX"), ("DHFL", "DHFL"), ("Jet Airways", "JETAIRWAYS"),
]

EXAMPLES_A_PER_STOCK = 6   # tool-call examples per stock (the core skill)
EXAMPLES_B_PER_STOCK = 2   # success-flow examples per stock
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
    "whats happening with {n} today", "quick price check on {n} please",
]
LONG_TEMPLATES = [  # realistic analytical queries (how real users actually ask)
    "I'm evaluating a speculative long position in {n} on the {e}. Pull its current market data first.",
    "I'm thinking of buying {n} shares this week - fetch the live numbers from the {e} so I can review the price action.",
    "My portfolio needs diversification and I'm considering {n}. Get me its latest trading data from the {e}.",
    "Before tomorrow's open I need a full quote for {n} - current price, volume and day range from the {e}.",
    "A friend recommended {n} to me. Can you fetch what it's doing on the {e} right now?",
    "Doing due diligence on {n} before investing - start by pulling the live market data from the {e}.",
]
LONG_TEMPLATES_NOEX = [
    "I'm researching {n} for a swing trade - start by pulling the live numbers.",
    "Considering an entry into {n} this month. First, fetch the current market data.",
    "I keep seeing news about {n}. Grab its live trading data so I can take a look.",
]
COMBO_TEMPLATES = [  # user gives BOTH name and ticker -> pass the TICKER (most precise)
    "I'm evaluating a speculative long position in {n} ({t}) on the {e}. Pull its current market data.",
    "Fetch the live quote for {n} ({t}) from the {e}.",
    "How is {n} ({t}) moving on the {e} today?",
    "Get me the numbers for {n}, ticker {t}, on the {e}.",
]
TICKER_TEMPLATES = [
    "Get {t} data.", "{t} on the {e}, please.", "Pull live numbers for {t} ({e}).",
    "Quote for {t}.", "{t} {e} data now.", "fetch {t}", "What's {t} at right now?",
    "price check on {t}",
]
EXCHANGE_PHRASE = {
    "NSE": ["NSE", "National Stock Exchange", "NSE exchange"],
    "BSE": ["BSE", "Bombay Stock Exchange", "BSE exchange"],
}

STYLE_STATS = Counter()   # populated during generation, printed by main()

def tool_call(symbol, exchange):
    """Build one tool call. `symbol` is the entity AS IT APPEARS IN THE USER TEXT."""
    args = {"symbol": symbol}
    if exchange:
        args["exchange"] = exchange
    return {"type": "function",
            "function": {"name": "get_indian_stock_data", "arguments": args}}

def _verified(text_form: str, expected_ticker: str) -> bool:
    """True if the resolver maps this string to the expected ticker (>=88 confidence)."""
    m = resolve_symbol(text_form)
    return bool(m) and m["symbol"] == expected_ticker

def make_typo(name: str, ticker: str) -> str | None:
    """Return a typo'd version of the name that STILL resolves to the right ticker.
    Tries a few small, human-like slips (swap / drop / double a letter); if none of
    them resolve confidently, returns None so the caller falls back to the clean name."""
    words = name.split()
    cands = [i for i, w in enumerate(words) if len(w) >= 6]
    if not cands:
        return None
    for _ in range(6):
        i = random.choice(cands)
        w = words[i]
        p = random.randint(1, len(w) - 3)
        kind = random.choice(["swap", "drop", "double"])
        if kind == "swap":   t = w[:p] + w[p+1] + w[p] + w[p+2:]
        elif kind == "drop": t = w[:p] + w[p+1:]
        else:                t = w[:p] + w[p] + w[p:]
        typo = " ".join(words[:i] + [t] + words[i+1:])
        if typo != name and _verified(typo, ticker):
            return typo
    return None

def make_partial(name: str, ticker: str) -> str | None:
    """Return a shorthand (first word) like 'Hindalco' for 'Hindalco Industries',
    but ONLY if the resolver maps it to the right ticker (avoids ambiguous ones
    like 'Tata' or 'Reliance'). Returns None otherwise."""
    words = name.split()
    if len(words) < 2 or len(words[0]) < 5:
        return None
    return words[0] if _verified(words[0], ticker) else None

def pick_exchange_mode():
    return random.choices(["none", "NSE", "BSE"], weights=[40, 30, 30])[0]

def make_user_and_call(name, ticker):
    """Build one (user text, matching tool call) pair.

    v3 KEY RULE: the tool call's `symbol` is the stock EXACTLY as it appears in the
    user's text - name, typo, shorthand, or ticker. Never a looked-up conversion.
    (One deliberate exception: 'SUZLON.NS' -> symbol 'SUZLON' + exchange NSE, because
    stripping a suffix is deterministic copying, not a guess.)
    """
    style = random.choices(
        ["name", "name_long", "name_noex", "name_lower", "name_typo",
         "name_partial", "name_plus_ticker", "ticker", "ticker_lower", "ticker_suffix"],
        weights=[20, 14, 12, 6, 8, 6, 10, 13, 5, 6])[0]
    mode = pick_exchange_mode()
    e = random.choice(EXCHANGE_PHRASE[mode]) if mode != "none" else None

    if style == "name_typo":
        typo = make_typo(name, ticker)
        if typo is None:
            style = "name"                      # fallback: clean name
        else:
            STYLE_STATS["name_typo"] += 1
            if e: return random.choice(NAME_TEMPLATES).format(n=typo, e=e), tool_call(typo, mode), mode
            return random.choice(NAME_TEMPLATES_NOEX).format(n=typo), tool_call(typo, None), None

    if style == "name_partial":
        part = make_partial(name, ticker)
        if part is None:
            style = "name"                      # fallback: clean name
        else:
            STYLE_STATS["name_partial"] += 1
            if e: return random.choice(NAME_TEMPLATES).format(n=part, e=e), tool_call(part, mode), mode
            return random.choice(NAME_TEMPLATES_NOEX).format(n=part), tool_call(part, None), None

    if style == "name_plus_ticker":
        STYLE_STATS["name_plus_ticker"] += 1
        if not e: mode, e = "NSE", "NSE"
        text = random.choice(COMBO_TEMPLATES).format(n=name, t=ticker, e=e)
        return text, tool_call(ticker, mode), mode      # ticker present -> pass the ticker

    if style == "ticker_suffix":
        STYLE_STATS["ticker_suffix"] += 1
        suf, mode = random.choice([(".NS", "NSE"), (".BO", "BSE")])
        text = random.choice([t for t in TICKER_TEMPLATES if "{e}" not in t]).format(t=ticker + suf)
        return text, tool_call(ticker, mode), mode      # strip suffix, exchange from suffix

    if style == "ticker_lower":
        STYLE_STATS["ticker_lower"] += 1
        low = ticker.lower()
        text = random.choice([t for t in TICKER_TEMPLATES if "{e}" not in t]).format(t=low)
        return text, tool_call(low, None), None          # pass verbatim; resolver uppercases

    if style == "ticker":
        STYLE_STATS["ticker"] += 1
        # only use {e}-templates when an exchange is set, so the call never names an
        # exchange the user didn't mention
        if e: return random.choice([t for t in TICKER_TEMPLATES if "{e}" in t]).format(t=ticker, e=e), tool_call(ticker, mode), mode
        text = random.choice([t for t in TICKER_TEMPLATES if "{e}" not in t]).format(t=ticker)
        return text, tool_call(ticker, None), None

    if style == "name_long":
        STYLE_STATS["name_long"] += 1
        if e: return random.choice(LONG_TEMPLATES).format(n=name, e=e), tool_call(name, mode), mode
        return random.choice(LONG_TEMPLATES_NOEX).format(n=name), tool_call(name, None), None

    if style == "name_lower":
        STYLE_STATS["name_lower"] += 1
        low = name.lower()
        text = random.choice(NAME_TEMPLATES_NOEX).format(n=low)
        return text, tool_call(low, None), None          # pass the lowercase name verbatim

    if style == "name_noex" or not e:
        STYLE_STATS["name_noex"] += 1
        return random.choice(NAME_TEMPLATES_NOEX).format(n=name), tool_call(name, None), None

    STYLE_STATS["name"] += 1
    return random.choice(NAME_TEMPLATES).format(n=name, e=e), tool_call(name, mode), mode

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
    # error flow: delisted names + transient errors on real stocks
    for name, ticker in BAD_STOCKS:
        train += make_unique(lambda n, t: ex_C(n, t, "delisted"), 5, seen, name, ticker)
    for name, ticker in random.sample(STOCKS, 30):
        train += make_unique(lambda n, t: ex_C(n, t, "transient"), 1, seen, name, ticker)
    random.shuffle(train); random.shuffle(valid)
    return train, valid

def cat(r):
    last = r["messages"][-1]
    if "tool_calls" in last: return "A"
    return "B" if last["content"] == SUCCESS_REPLY else "C"

def audit(rows):
    """One-time-job insurance: every symbol we teach (outside deliberate error rows)
    MUST resolve through scripts/resolver.py. Returns list of failures."""
    fails = []
    for r in rows:
        if cat(r) == "C":
            continue                       # error rows are SUPPOSED to fail resolution
        sym = r["messages"][2]["tool_calls"][0]["function"]["arguments"]["symbol"]
        if resolve_symbol(sym) is None:
            fails.append((sym, r["messages"][1]["content"][:60]))
    return fails

def main():
    os.makedirs(OUT, exist_ok=True)
    train, valid = build()
    for fname, data in [("train.jsonl", train), ("valid.jsonl", valid)]:
        with open(os.path.join(OUT, fname), "w") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")
    print(f"stocks={len(STOCKS)}  train={len(train)}  valid={len(valid)}")
    print("train categories:", dict(Counter(cat(r) for r in train)))
    print("query styles used:", dict(STYLE_STATS.most_common()))
    fails = audit(train) + audit(valid)
    if fails:
        print(f"\n❌ AUDIT FAILED - {len(fails)} trained symbols do not resolve:")
        for sym, q in fails[:10]:
            print(f"   {sym!r}   from: {q!r}")
        sys.exit(1)
    print("✅ AUDIT PASSED: every trained symbol (excl. error rows) resolves via the resolver.")
    print("\nsample tool calls:")
    for r in random.sample(train, 5):
        u = r["messages"][1]["content"]
        a = r["messages"][2]["tool_calls"][0]["function"]["arguments"]
        print(f"   user: {u[:64]!r}")
        print(f"       -> symbol={a['symbol']!r}  exchange={a.get('exchange')!r}")

if __name__ == "__main__":
    main()

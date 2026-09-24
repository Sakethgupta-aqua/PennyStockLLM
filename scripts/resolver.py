"""resolve_symbol: turn any company name or ticker into the correct NSE symbol."""
import sqlite3, re
from functools import lru_cache
from rapidfuzz import process, fuzz

DB = "/Users/saketh/PennyStockLLM/market.db"

def _normalize(name: str) -> str:
    n = name.lower().strip()
    n = re.sub(r"\b(limited|ltd\.?|india|co\.?|company)\b", "", n)
    return re.sub(r"[^a-z0-9 ]", " ", n).strip()

@lru_cache(maxsize=1)
def _load_universe():
    con = sqlite3.connect(DB)
    rows = con.execute("SELECT symbol, company_name, name_normalized, series "
                       "FROM universe WHERE is_active=1").fetchall()
    con.close()
    return {r[0].upper(): r for r in rows}, {r[2]: r for r in rows}

def resolve_symbol(query: str, min_score: float = 88.0) -> dict | None:
    """Resolve a ticker or company name to an official NSE symbol.

    min_score=88 is deliberately strict: a hallucinated symbol like 'KAYNESTEEL'
    fuzzy-matched an unrelated stock at 85.5, while a legitimate name match
    ('Tata Consultancy' -> TCS) scores 90. Returning None on low confidence lets
    the caller fail loudly instead of analysing the wrong company.
    """
    by_symbol, by_name = _load_universe()
    q = query.strip()
    if q.upper() in by_symbol:                      # 1. exact ticker
        r = by_symbol[q.upper()]
        return {"symbol": r[0], "company_name": r[1], "series": r[3], "score": 100.0, "how": "exact_ticker"}
    qn = _normalize(q)
    if qn in by_name:                               # 2. exact name
        r = by_name[qn]
        return {"symbol": r[0], "company_name": r[1], "series": r[3], "score": 100.0, "how": "exact_name"}
    best = process.extractOne(qn, list(by_name.keys()), scorer=fuzz.WRatio)  # 3. fuzzy
    if best and best[1] >= min_score:
        r = by_name[best[0]]
        return {"symbol": r[0], "company_name": r[1], "series": r[3], "score": round(best[1], 1), "how": "fuzzy"}
    return None

if __name__ == "__main__":
    for t in ["Suzlon", "TCS", "Tata Consultancy", "kAyNes Technology", "asdfgh"]:
        m = resolve_symbol(t)
        print(f"{t:20} -> {m['symbol'] if m else 'NO MATCH'}")

"""Resolve a company name or ticker to an official NSE symbol, using market.db.

The model no longer needs to *memorize* tickers (PRODUCTION_MASTER_PLAN.md §3.2): it passes
the entity through, and this deterministic lookup maps it to the correct symbol.

  resolve_symbol("Suzlon Energy")  -> {"symbol": "SUZLON", ...}   (exact name)
  resolve_symbol("SUZLON")         -> {"symbol": "SUZLON", ...}   (exact ticker)
  resolve_from_query("Analyze Kaynes Technology on NSE") -> {"symbol": "KAYNES", ...}
"""
import re
import sqlite3
from functools import lru_cache

from rapidfuzz import fuzz, process

from pennystock.config import settings


def normalize(name: str) -> str:
    """Lower-case and strip common suffixes so names match regardless of formatting."""
    n = name.lower().strip()
    n = re.sub(r"\b(limited|ltd\.?|india|co\.?|company)\b", "", n)
    return re.sub(r"[^a-z0-9 ]", " ", n).strip()


@lru_cache(maxsize=1)
def _load_universe():
    """Load active symbols once and cache them: (by_ticker, by_normalized_name)."""
    con = sqlite3.connect(str(settings.market_db))
    rows = con.execute(
        "SELECT symbol, company_name, name_normalized, series FROM universe WHERE is_active=1"
    ).fetchall()
    con.close()
    return {r[0].upper(): r for r in rows}, {r[2]: r for r in rows}


def resolve_symbol(query: str, min_score: float | None = None) -> dict | None:
    """Best-matching stock, or None if nothing is confident enough."""
    if min_score is None:
        min_score = settings.resolver_min_score
    by_symbol, by_name = _load_universe()
    q = query.strip()

    if q.upper() in by_symbol:                                   # 1. exact ticker
        r = by_symbol[q.upper()]
        return {"symbol": r[0], "company_name": r[1], "series": r[3], "score": 100.0, "how": "exact_ticker"}

    qn = normalize(q)
    if qn in by_name:                                            # 2. exact company name
        r = by_name[qn]
        return {"symbol": r[0], "company_name": r[1], "series": r[3], "score": 100.0, "how": "exact_name"}

    best = process.extractOne(qn, list(by_name.keys()), scorer=fuzz.WRatio)  # 3. fuzzy
    if best and best[1] >= min_score:
        r = by_name[best[0]]
        return {"symbol": r[0], "company_name": r[1], "series": r[3], "score": round(best[1], 1), "how": "fuzzy"}
    return None


def resolve_from_query(query: str, max_ngram: int = 4) -> dict | None:
    """Find a stock named in the user's own words (exact matches only -> no false positives)."""
    words = normalize(query).split()
    best = None
    for n in range(min(max_ngram, len(words)), 0, -1):          # prefer longer phrases
        for i in range(len(words) - n + 1):
            gram = " ".join(words[i:i + n])
            if len(gram) < 3:
                continue
            m = resolve_symbol(gram, min_score=101)              # 101 => exact matches only
            if m and (best is None or (n, m["score"]) > best[0]):
                best = ((n, m["score"]), m)
    return best[1] if best else None

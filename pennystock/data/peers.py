"""Peer comparison - the "versus what?" a bare ratio can never answer.

"A P/E of 33" is meaningless in isolation: pharma trades near 30x, metals near 12x. Without a
sector reference the analyst is guessing, and its cheap/expensive verdicts are opinion dressed as
analysis. This module answers "how does this stock rank against companies in its own industry?"

Design: peer metrics are CACHED in market.db (table `peer_metrics`), populated offline by
scripts/ingest_peers.py. Comparing live would mean one network call per peer - far too slow inside
an agent run - so the cache is what makes this usable at all. It is honest about thin coverage:
with too few cached peers it says so rather than quoting a median of two companies.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import median

from pennystock.config import settings

logger = logging.getLogger(__name__)

_UA_SIMPLE = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"

_SCHEMA = """CREATE TABLE IF NOT EXISTS peer_metrics (
    symbol TEXT PRIMARY KEY, company TEXT, sector TEXT, industry TEXT,
    market_cap_cr REAL, pe REAL, pb REAL, roe_pct REAL, debt_to_equity_pct REAL,
    profit_margin_pct REAL, fetched_at TEXT)"""

# metric -> (label, lower_is_better)
_METRICS = {"pe": ("P/E", True), "pb": ("P/B", True), "roe_pct": ("ROE %", False),
            "debt_to_equity_pct": ("Debt/Equity %", True), "profit_margin_pct": ("Profit margin %", False)}


@dataclass
class PeerRow:
    symbol: str
    company: str | None = None
    sector: str | None = None
    industry: str | None = None
    market_cap_cr: float | None = None
    pe: float | None = None
    pb: float | None = None
    roe_pct: float | None = None
    debt_to_equity_pct: float | None = None
    profit_margin_pct: float | None = None


def _connect():
    con = sqlite3.connect(str(settings.market_db))
    con.executescript(_SCHEMA)
    return con


def save_peers(rows: list[PeerRow]) -> int:
    now = datetime.now().isoformat(timespec="seconds")
    with _connect() as con:
        con.executemany(
            "INSERT INTO peer_metrics (symbol, company, sector, industry, market_cap_cr, pe, pb,"
            " roe_pct, debt_to_equity_pct, profit_margin_pct, fetched_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(symbol) DO UPDATE SET"
            "   company=excluded.company, sector=excluded.sector, industry=excluded.industry,"
            "   market_cap_cr=excluded.market_cap_cr, pe=excluded.pe, pb=excluded.pb,"
            "   roe_pct=excluded.roe_pct, debt_to_equity_pct=excluded.debt_to_equity_pct,"
            "   profit_margin_pct=excluded.profit_margin_pct, fetched_at=excluded.fetched_at",
            [(r.symbol.upper(), r.company, r.sector, r.industry, r.market_cap_cr, r.pe, r.pb,
              r.roe_pct, r.debt_to_equity_pct, r.profit_margin_pct, now) for r in rows])
    return len(rows)


def _load(where: str, args: tuple) -> list[PeerRow]:
    with _connect() as con:
        rows = con.execute(
            "SELECT symbol, company, sector, industry, market_cap_cr, pe, pb, roe_pct,"
            f" debt_to_equity_pct, profit_margin_pct FROM peer_metrics WHERE {where}", args).fetchall()
    return [PeerRow(*r) for r in rows]


def cached_symbols() -> list[str]:
    with _connect() as con:
        return [r[0] for r in con.execute("SELECT symbol FROM peer_metrics ORDER BY symbol")]


def last_fetched():
    with _connect() as con:
        row = con.execute("SELECT MAX(fetched_at) FROM peer_metrics").fetchone()
    try:
        return datetime.fromisoformat(row[0]) if row and row[0] else None
    except ValueError:
        return None


def is_stale() -> bool:
    stamp = last_fetched()
    return stamp is None or datetime.now() - stamp > timedelta(days=settings.peer_max_age_days)


# A broad, liquid starter set so common queries have real peer groups from day one.
SEED_LIQUID = [
    "HINDALCO", "TATASTEEL", "JSWSTEEL", "SAIL", "VEDL", "NATIONALUM", "JINDALSTEL", "NMDC", "HINDZINC",
    "SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "LUPIN", "AUROPHARMA", "TORNTPHARM", "ALKEM",
    "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "PERSISTENT", "COFORGE",
    "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "INDUSINDBK", "BAJFINANCE", "BAJAJFINSV",
    "MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", "TVSMOTOR", "ASHOKLEY",
    "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO", "GODREJCP", "COLPAL",
    "RELIANCE", "ONGC", "NTPC", "POWERGRID", "COALINDIA", "IOC", "BPCL", "TATAPOWER",
    "ULTRACEMCO", "SHREECEM", "AMBUJACEM", "ACC", "ASIANPAINT", "PIDILITIND", "GRASIM", "BERGEPAINT",
]


# NSE's own index constituent lists - already ranked/filtered by market cap and liquidity, which
# is far better than taking the first N symbols alphabetically out of the full 2,570-name universe.
INDEX_LISTS = {
    "nifty50": "ind_nifty50list.csv",
    "nifty100": "ind_nifty100list.csv",
    "nifty200": "ind_nifty200list.csv",
    "nifty500": "ind_nifty500list.csv",
    "totalmarket": "ind_niftytotalmarket_list.csv",     # ~755 large+mid+small
    "midsmall400": "ind_niftymidsmallcap400list.csv",
    "smallcap250": "ind_niftysmallcap250list.csv",
    "microcap250": "ind_niftymicrocap250_list.csv",
}
_INDEX_URL = "https://nsearchives.nseindia.com/content/indices/{file}"


def index_symbols(names: list[str]) -> list[str]:
    """Symbols from one or more NSE index constituent lists, de-duplicated, order preserved."""
    import csv
    import io

    import requests

    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = name.strip().lower()
        if key not in INDEX_LISTS:
            raise ValueError(f"unknown index '{name}'. Choose from: {', '.join(INDEX_LISTS)}")
        url = _INDEX_URL.format(file=INDEX_LISTS[key])
        r = requests.get(url, headers={"User-Agent": _UA_SIMPLE}, timeout=settings.ownership_timeout_s)
        r.raise_for_status()
        for row in csv.DictReader(io.StringIO(r.text)):
            sym = (row.get("Symbol") or "").strip().upper()
            if sym and sym not in seen:
                seen.add(sym)
                out.append(sym)
        logger.info(f"{key}: {len(out)} cumulative symbols")
    return out


def fetch_one(symbol: str) -> PeerRow | None:
    """One symbol's grouping + valuation metrics. Returns None if it cannot be grouped."""
    import yfinance as yf
    try:
        i = yf.Ticker(f"{symbol.upper()}.NS").info or {}
    except Exception as e:
        logger.warning(f"peer fetch failed for {symbol}: {type(e).__name__}: {e}")
        return None
    if not i.get("sector") and not i.get("industry"):
        return None                                          # nothing to group on
    mc, roe, margin = i.get("marketCap"), i.get("returnOnEquity"), i.get("profitMargins")
    return PeerRow(
        symbol=symbol.upper(), company=i.get("longName"), sector=i.get("sector"),
        industry=i.get("industry"), market_cap_cr=round(mc / 1e7, 1) if mc else None,
        pe=i.get("trailingPE"), pb=i.get("priceToBook"),
        roe_pct=round(roe * 100, 2) if roe is not None else None,
        debt_to_equity_pct=i.get("debtToEquity"),
        profit_margin_pct=round(margin * 100, 2) if margin is not None else None)


def fetch_and_cache(symbol: str) -> bool:
    """Fetch ONE symbol and cache it - used to self-heal a cache miss during a live run."""
    row = fetch_one(symbol)
    if row:
        save_peers([row])
        return True
    return False


def coverage() -> dict:
    with _connect() as con:
        n = con.execute("SELECT COUNT(*) FROM peer_metrics").fetchone()[0]
        sectors = con.execute("SELECT COUNT(DISTINCT industry) FROM peer_metrics").fetchone()[0]
    return {"symbols_cached": n, "industries_cached": sectors}


def _percentile(value: float, population: list[float], lower_is_better: bool) -> int | None:
    """Where this stock sits in its industry, 0-100, always oriented so HIGHER = BETTER."""
    if value is None or len(population) < 2:
        return None
    below = sum(1 for v in population if v < value)
    pct = round(below / len(population) * 100)
    return 100 - pct if lower_is_better else pct


def compare_to_peers(symbol: str, min_peers: int | None = None) -> dict:
    """Rank a stock against cached companies in the same industry (sector as a fallback)."""
    min_peers = min_peers or settings.min_peers
    symbol = symbol.upper()
    me = _load("symbol = ?", (symbol,))
    if not me:                                   # self-heal: one network call for the stock itself
        logger.info(f"{symbol} missing from peer cache; fetching it once.")
        if fetch_and_cache(symbol):
            me = _load("symbol = ?", (symbol,))
    if not me:
        return {"peers": "unavailable",
                "note": f"{symbol} is not in the peer cache and could not be fetched. "
                        f"Run scripts/ingest_peers.py."}
    me = me[0]

    group_by, group_val = "industry", me.industry
    peers = _load("industry = ? AND symbol != ?", (me.industry, symbol)) if me.industry else []
    if len(peers) < min_peers and me.sector:                 # widen to sector if industry is thin
        group_by, group_val = "sector", me.sector
        peers = _load("sector = ? AND symbol != ?", (me.sector, symbol))
    if len(peers) < min_peers:
        return {"peers": "insufficient",
                "note": f"only {len(peers)} cached peer(s) for {me.industry or me.sector}; "
                        f"need {min_peers}. Widen the cache with scripts/ingest_peers.py.",
                "company": me.company, "sector": me.sector, "industry": me.industry}

    out = {"symbol": symbol, "company": me.company, "sector": me.sector, "industry": me.industry,
           "compared_against": f"{len(peers)} companies in the same {group_by} ({group_val})",
           "metrics": {}}
    for key, (label, lower_better) in _METRICS.items():
        mine = getattr(me, key)
        pop = [getattr(p, key) for p in peers if getattr(p, key) is not None]
        if mine is None or len(pop) < 2:
            out["metrics"][label] = {"value": mine, "peer_median": None, "note": "insufficient peer data"}
            continue
        med = round(median(pop), 2)
        out["metrics"][label] = {
            "value": round(mine, 2),
            "peer_median": med,
            "vs_median_pct": round((mine / med - 1) * 100, 1) if med else None,
            "percentile_higher_is_better": _percentile(mine, pop, lower_better),
            "peers_with_data": len(pop),
        }
    # the headline a human actually wants
    pe = out["metrics"].get("P/E", {})
    if pe.get("vs_median_pct") is not None:
        direction = "PREMIUM to" if pe["vs_median_pct"] > 0 else "DISCOUNT to"
        out["headline"] = (f"Trades at {abs(pe['vs_median_pct'])}% {direction} its {group_by} "
                           f"median P/E ({pe['value']} vs {pe['peer_median']}).")
    return out

"""The tradable-symbol universe: NSE's equity master, cached in market.db.

The resolver maps a company name to a ticker using this table, so a STALE universe means newly
listed companies cannot be resolved at all and delisted ones linger. The NSE master only changes
when companies list or delist, so we refresh on a staleness check (weekly by default) rather than
on every query - network I/O per question would be wasteful and slow.

`normalize` is imported from resolver.py deliberately. The names written here MUST be normalised
exactly the way the resolver normalises a lookup, or matching silently fails. Previously
scripts/ingest_universe.py kept its own copy of that function - identical today, but any drift
would have been an invisible, very hard-to-debug break. One definition, one source of truth.

ensure_fresh() NEVER raises: if NSE is unreachable we log and keep using the cached universe.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta
from io import StringIO

from pennystock.config import settings
from pennystock.data.resolver import normalize

logger = logging.getLogger(__name__)

_SCHEMA = """CREATE TABLE IF NOT EXISTS universe (
    symbol TEXT NOT NULL, exchange TEXT NOT NULL DEFAULT 'NSE',
    company_name TEXT NOT NULL, name_normalized TEXT NOT NULL,
    isin TEXT, series TEXT, listing_date TEXT, face_value REAL,
    sector TEXT, cap_bucket TEXT,
    is_active INTEGER NOT NULL DEFAULT 1, updated_at TEXT NOT NULL,
    PRIMARY KEY (symbol, exchange))"""

_checked_this_process = False          # ensure_fresh() does its work at most once per process


def last_updated() -> datetime | None:
    try:
        with sqlite3.connect(str(settings.market_db)) as con:
            con.executescript(_SCHEMA)
            row = con.execute("SELECT MAX(updated_at) FROM universe").fetchone()
        return datetime.fromisoformat(row[0]) if row and row[0] else None
    except Exception:
        return None


def is_stale(max_age_days: int | None = None) -> bool:
    stamp = last_updated()
    if stamp is None:
        return True
    limit = timedelta(days=max_age_days if max_age_days is not None else settings.universe_max_age_days)
    return datetime.now() - stamp > limit


def fetch_master():
    """Download NSE's equity master CSV. Raises on failure - callers decide how to react."""
    import pandas as pd
    import requests

    r = requests.get(settings.universe_url,
                     headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
                     timeout=settings.universe_timeout_s)
    r.raise_for_status()                                   # fail loudly if NSE blocks us
    df = pd.read_csv(StringIO(r.text))
    df.columns = [c.strip() for c in df.columns]           # NSE headers carry stray spaces
    df["DATE OF LISTING"] = pd.to_datetime(df["DATE OF LISTING"], format="%d-%b-%Y",
                                           errors="coerce").dt.date
    return df


def refresh(df) -> int:
    """Upsert the master into `universe`; symbols missing from today's file are flagged
    is_active=0 rather than deleted, so history and past references survive."""
    now = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(str(settings.market_db)) as con:
        con.executescript(_SCHEMA)
        con.executemany(
            "INSERT INTO universe (symbol, exchange, company_name, name_normalized, isin, series,"
            " listing_date, face_value, is_active, updated_at) VALUES (?, 'NSE', ?, ?, ?, ?, ?, ?, 1, ?)"
            " ON CONFLICT(symbol, exchange) DO UPDATE SET"
            "   company_name=excluded.company_name, name_normalized=excluded.name_normalized,"
            "   isin=excluded.isin, series=excluded.series, is_active=1,"
            "   updated_at=excluded.updated_at",
            [(r["SYMBOL"], r["NAME OF COMPANY"], normalize(r["NAME OF COMPANY"]), r["ISIN NUMBER"],
              r["SERIES"], str(r["DATE OF LISTING"] or ""), r["FACE VALUE"], now)
             for _, r in df.iterrows()])
        placeholders = ",".join("?" * len(df))
        con.execute(f"UPDATE universe SET is_active=0, updated_at=? "
                    f"WHERE exchange='NSE' AND symbol NOT IN ({placeholders})",
                    [now] + df["SYMBOL"].tolist())

    # the resolver caches the universe in memory - drop it so the new rows are visible at once
    try:
        from pennystock.data.resolver import _load_universe
        _load_universe.cache_clear()
    except Exception:
        pass
    return len(df)


def ensure_fresh(force: bool = False) -> str:
    """Refresh the universe if it is stale. Returns a status string; never raises."""
    global _checked_this_process
    if not force:
        if _checked_this_process:
            return "already checked this process"
        if not settings.universe_auto_refresh:
            return "auto-refresh disabled"
    _checked_this_process = True

    if not force and not is_stale():
        return f"fresh (last updated {last_updated():%Y-%m-%d})"
    try:
        n = refresh(fetch_master())
        return f"refreshed {n} NSE symbols"
    except Exception as e:                                 # a network blip must not break analysis
        stamp = last_updated()
        logger.warning(f"Universe refresh failed ({type(e).__name__}: {e}); "
                       f"continuing with the cached universe from {stamp}")
        return f"refresh FAILED ({type(e).__name__}) - using cached universe from {stamp}"

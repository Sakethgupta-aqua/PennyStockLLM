"""Where ownership data comes from, and where it is kept.

Architecture (deliberate, and shaped by what is actually reachable):

  fetch  ScreenerOwnershipProvider  -> quarterly promoter / FII / DII / public percentages
  store  OwnershipStore (SQLite)    -> cached in market.db, keyed (symbol, quarter)
  read   get_ownership()            -> cache-first, refetch only when stale

Two facts drove this design:
  1. Shareholding is filed QUARTERLY (SEBI LODR Reg 31), so it changes four times a year.
     Caching is therefore not merely a speed win - it means the analyst still gets data, with
     an explicit as-of quarter, even when the upstream source breaks or blocks us.
  2. Promoter PLEDGE comes from NSE's own corporate-filings API - statutory, free, authoritative.
     It only serves the LATEST quarter, so pledge HISTORY accumulates in our cache as the job is
     re-run each quarter. A manual entry (--set-pledge) still wins if someone sets one, and a
     refetch never overwrites it.

Sources, in precedence order: NSE (statutory: pledge + promoter/public, 20-25 quarters) then
screener.in (adds the FII / DII split NSE does not publish). Screener scraping is best-effort and
ToS-grey - fine for personal research; swap in a licensed feed by adding one more
OwnershipProvider subclass and listing it first. Nothing else in the codebase changes.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from functools import lru_cache

from pennystock.config import settings
from pennystock.data.ownership import Ownership, Quarter

logger = logging.getLogger(__name__)

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120 Safari/537.36")

# screener row label (letters only, lower-cased) -> Quarter field
_ROW_MAP = {
    "promoters": "promoter_pct",
    "fiis": "fii_pct",
    "diis": "dii_pct",
    "government": "government_pct",
    "public": "public_pct",
    "noofshareholders": "shareholders",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS shareholding (
    symbol         TEXT NOT NULL,
    quarter        TEXT NOT NULL,
    quarter_ord    INTEGER,               -- sortable YYYYMM, so 'Jun 2026' orders correctly
    promoter_pct   REAL,
    fii_pct        REAL,
    dii_pct        REAL,
    public_pct     REAL,
    government_pct REAL,
    shareholders   INTEGER,
    pledge_pct     REAL,                  -- % of PROMOTER holding; NULL = UNKNOWN (never zero)
    pledge_pct_of_total REAL,             -- % of total shares (NSE headline number)
    pledge_source  TEXT,
    fetched_at     TEXT,
    PRIMARY KEY (symbol, quarter)
);
"""

_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}


def quarter_ord(quarter: str) -> int:
    """'Jun 2026' -> 202606 so quarters sort chronologically instead of alphabetically."""
    m = re.match(r"([A-Za-z]{3})\w*\s+(\d{4})", str(quarter).strip())
    if not m:
        return 0
    return int(m.group(2)) * 100 + _MONTHS.get(m.group(1).title(), 0)


def _num(text: str) -> float | None:
    t = str(text).replace(",", "").replace("%", "").strip()
    try:
        return float(t)
    except ValueError:
        return None


# ------------------------------------------------------------------ providers
class OwnershipProvider(ABC):
    """Swappable source of quarterly shareholding data."""

    @abstractmethod
    def fetch(self, symbol: str) -> Ownership: ...


class ScreenerOwnershipProvider(OwnershipProvider):
    """Parses screener.in's quarterly shareholding table.

    Returns promoter / FII / DII / government / public percentages and the shareholder count
    for every quarter shown (typically 12+). pledge_pct is always None here - screener's public
    page does not expose it.
    """

    URL = "https://www.screener.in/company/{symbol}/"

    def fetch(self, symbol: str) -> Ownership:
        import requests
        from bs4 import BeautifulSoup

        own = Ownership(symbol=symbol.upper(), fetched_at=datetime.now().isoformat(timespec="seconds"))
        try:
            r = requests.get(self.URL.format(symbol=symbol.upper()),
                             headers={"User-Agent": _UA}, timeout=settings.ownership_timeout_s)
            if r.status_code != 200:
                logger.warning(f"screener returned {r.status_code} for {symbol}")
                return own
            section = BeautifulSoup(r.text, "html.parser").find("section", id="shareholding")
            table = section.find("table") if section else None
            if table is None:
                logger.warning(f"no shareholding table found for {symbol}")
                return own

            headers = [th.get_text(strip=True) for th in table.find_all("th")][1:]   # quarter labels
            rows: dict[str, list] = {}
            for tr in table.find_all("tr")[1:]:
                cells = [td.get_text(strip=True) for td in tr.find_all("td")]
                if not cells:
                    continue
                key = re.sub(r"[^a-z]", "", cells[0].lower())
                if key in _ROW_MAP:
                    rows[_ROW_MAP[key]] = cells[1:]

            for i, q in enumerate(headers):
                quarter = Quarter(quarter=q)
                for field_name, values in rows.items():
                    if i < len(values):
                        val = _num(values[i])
                        if val is not None:
                            setattr(quarter, field_name,
                                    int(val) if field_name == "shareholders" else val)
                own.quarters.append(quarter)
            own.quarters.sort(key=lambda q: quarter_ord(q.quarter))
            logger.info(f"screener: {symbol} -> {len(own.quarters)} quarter(s)")
        except Exception as e:                       # never let a scrape failure break a run
            logger.warning(f"ownership fetch failed for {symbol}: {type(e).__name__}: {e}")
        return own


# ------------------------------------------------------------------ storage
class OwnershipStore:
    """SQLite cache in market.db, and the home of manually-entered pledge percentages."""

    def __init__(self, db_path=None):
        self.db_path = str(db_path or settings.market_db)
        with sqlite3.connect(self.db_path) as con:
            con.executescript(_SCHEMA)
            try:                                   # migrate DBs created before this column existed
                con.execute("ALTER TABLE shareholding ADD COLUMN pledge_pct_of_total REAL")
            except sqlite3.OperationalError:
                pass

    def load(self, symbol: str) -> Ownership:
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                "SELECT quarter, promoter_pct, fii_pct, dii_pct, public_pct, government_pct,"
                " shareholders, pledge_pct, pledge_source, fetched_at, pledge_pct_of_total FROM shareholding"
                " WHERE symbol=? ORDER BY quarter_ord", (symbol.upper(),)).fetchall()
        own = Ownership(symbol=symbol.upper())
        for r in rows:
            own.quarters.append(Quarter(quarter=r[0], promoter_pct=r[1], fii_pct=r[2], dii_pct=r[3],
                                        public_pct=r[4], government_pct=r[5], shareholders=r[6],
                                        pledge_pct=r[7], pledge_source=r[8],
                                        pledge_pct_of_total=r[10]))
            own.fetched_at = r[9]
        return own

    def save(self, own: Ownership) -> int:
        """Upsert quarters. A refetch NEVER clobbers a manually entered pledge."""
        now = datetime.now().isoformat(timespec="seconds")
        with sqlite3.connect(self.db_path) as con:
            for q in own.quarters:
                con.execute(
                    "INSERT INTO shareholding (symbol, quarter, quarter_ord, promoter_pct, fii_pct,"
                    " dii_pct, public_pct, government_pct, shareholders, pledge_pct, pledge_source,"
                    " fetched_at, pledge_pct_of_total) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"
                    " ON CONFLICT(symbol, quarter) DO UPDATE SET"
                    "   promoter_pct=excluded.promoter_pct, fii_pct=excluded.fii_pct,"
                    "   dii_pct=excluded.dii_pct, public_pct=excluded.public_pct,"
                    "   government_pct=excluded.government_pct, shareholders=excluded.shareholders,"
                    "   pledge_pct=COALESCE(excluded.pledge_pct, shareholding.pledge_pct),"
                    "   pledge_source=COALESCE(excluded.pledge_source, shareholding.pledge_source),"
                    "   pledge_pct_of_total=COALESCE(excluded.pledge_pct_of_total,"
                    "                               shareholding.pledge_pct_of_total),"
                    "   fetched_at=excluded.fetched_at",
                    (own.symbol.upper(), q.quarter, quarter_ord(q.quarter), q.promoter_pct, q.fii_pct,
                     q.dii_pct, q.public_pct, q.government_pct, q.shareholders, q.pledge_pct,
                     q.pledge_source, now, q.pledge_pct_of_total))
        return len(own.quarters)

    def set_pledge(self, symbol: str, quarter: str, pct: float, source: str = "manual") -> None:
        """Record a promoter-pledge percentage for one quarter (the manual/paid-feed path)."""
        now = datetime.now().isoformat(timespec="seconds")
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT INTO shareholding (symbol, quarter, quarter_ord, pledge_pct, pledge_source,"
                " fetched_at) VALUES (?,?,?,?,?,?)"
                " ON CONFLICT(symbol, quarter) DO UPDATE SET"
                "   pledge_pct=excluded.pledge_pct, pledge_source=excluded.pledge_source",
                (symbol.upper(), quarter, quarter_ord(quarter), float(pct), source, now))

    def is_stale(self, own: Ownership) -> bool:
        if not own.quarters or not own.fetched_at:
            return True
        try:
            age = datetime.now() - datetime.fromisoformat(own.fetched_at)
        except ValueError:
            return True
        return age > timedelta(days=settings.ownership_max_age_days)



def norm_quarter(raw: str) -> str:
    """'30-Jun-2026' / '30-JUN-2026' -> 'Jun 2026' so NSE and screener labels line up."""
    m = re.search(r"(\d{1,2})[-\s]([A-Za-z]{3})[A-Za-z]*[-\s](\d{4})", str(raw))
    return f"{m.group(2).title()} {m.group(3)}" if m else str(raw).strip()


class NseOwnershipProvider(OwnershipProvider):
    """NSE's own corporate-filings APIs - the authoritative statutory source, and free.

    Two endpoints:
      corporate-pledgedata            -> latest quarter's pledge (shares pledged + promoter holding)
      corporate-share-holdings-master -> 20-25 quarters of promoter vs public %

    IMPORTANT unit trap: NSE's headline `percSharesPledged` is a % of TOTAL shares, but the market
    convention for "promoter pledging" is % of the PROMOTER'S OWN holding. Suzlon reports 6.84% of
    total, which is 58.5% of the promoter stake - the difference between "low" and "SEVERE". We
    therefore compute numSharesPledged / totPromoterHolding ourselves and keep NSE's figure beside it.

    NSE returns 403 to any client without a full browser header set AND a Referer pointing at the
    symbol's quote page, so we prime a session per symbol before calling the API.
    """

    BASE = "https://www.nseindia.com"
    HEADERS = {
        "User-Agent": _UA,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "sec-ch-ua": '"Chromium";v="122", "Not(A:Brand";v="24"',
        "sec-ch-ua-mobile": "?0", "sec-ch-ua-platform": '"macOS"',
        "Sec-Fetch-Dest": "empty", "Sec-Fetch-Mode": "cors", "Sec-Fetch-Site": "same-origin",
    }

    def _session(self, symbol: str):
        import requests
        s = requests.Session()
        s.headers.update(self.HEADERS)
        ref = f"{self.BASE}/get-quotes/equity?symbol={symbol}"
        s.headers["Referer"] = ref
        s.get(ref, timeout=settings.ownership_timeout_s)          # sets the cookies NSE requires
        return s

    def fetch(self, symbol: str) -> Ownership:
        symbol = symbol.upper()
        own = Ownership(symbol=symbol, fetched_at=datetime.now().isoformat(timespec="seconds"))
        by_quarter: dict[str, Quarter] = {}
        try:
            s = self._session(symbol)

            # --- pledge (latest quarter only; history accumulates in our cache over time) ---
            try:
                rows = s.get(f"{self.BASE}/api/corporate-pledgedata?index=equities&symbol={symbol}",
                             timeout=settings.ownership_timeout_s).json().get("data", []) or []
                for r in rows:
                    q = norm_quarter(r.get("shp"))
                    pledged = _num(r.get("numSharesPledged"))
                    promoter_shares = _num(r.get("totPromoterHolding"))
                    quarter = by_quarter.setdefault(q, Quarter(quarter=q))
                    if pledged is not None and promoter_shares:
                        quarter.pledge_pct = round(pledged / promoter_shares * 100, 2)
                        quarter.pledge_source = "nse"
                    quarter.pledge_pct_of_total = _num(r.get("percSharesPledged"))
                    quarter.promoter_pct = quarter.promoter_pct or _num(r.get("percPromoterHolding"))
            except Exception as e:
                logger.warning(f"NSE pledge fetch failed for {symbol}: {type(e).__name__}: {e}")

            # --- promoter / public history ---
            try:
                for r in s.get(f"{self.BASE}/api/corporate-share-holdings-master"
                               f"?index=equities&symbol={symbol}",
                               timeout=settings.ownership_timeout_s).json() or []:
                    q = norm_quarter(r.get("date"))
                    quarter = by_quarter.setdefault(q, Quarter(quarter=q))
                    if quarter.promoter_pct is None:
                        quarter.promoter_pct = _num(r.get("pr_and_prgrp"))
                    if quarter.public_pct is None:
                        quarter.public_pct = _num(r.get("public_val"))
            except Exception as e:
                logger.warning(f"NSE shareholding fetch failed for {symbol}: {type(e).__name__}: {e}")

            own.quarters = sorted(by_quarter.values(), key=lambda q: quarter_ord(q.quarter))
            logger.info(f"NSE: {symbol} -> {len(own.quarters)} quarter(s)")
        except Exception as e:
            logger.warning(f"NSE ownership unavailable for {symbol}: {type(e).__name__}: {e}")
        return own


_MERGE_FIELDS = ("promoter_pct", "fii_pct", "dii_pct", "public_pct", "government_pct",
                 "shareholders", "pledge_pct", "pledge_pct_of_total", "pledge_source")


class CompositeOwnershipProvider(OwnershipProvider):
    """Query several sources and merge them, EARLIER providers winning each field.

    Order matters: NSE is the statutory source and is authoritative for pledge and promoter %;
    screener only fills in what NSE does not publish (the FII / DII split). If a source is down
    the others still deliver, so no single upstream can take ownership data offline.
    """

    def __init__(self, providers: list[OwnershipProvider]):
        self.providers = providers

    def fetch(self, symbol: str) -> Ownership:
        merged: dict[str, Quarter] = {}
        for prov in self.providers:
            try:
                fetched = prov.fetch(symbol)
            except Exception as e:
                logger.warning(f"{type(prov).__name__} failed for {symbol}: {e}")
                continue
            for q in fetched.quarters:
                target = merged.setdefault(q.quarter, Quarter(quarter=q.quarter))
                for f in _MERGE_FIELDS:
                    if getattr(target, f) is None and getattr(q, f) is not None:
                        setattr(target, f, getattr(q, f))
        own = Ownership(symbol=symbol.upper(),
                        fetched_at=datetime.now().isoformat(timespec="seconds"))
        own.quarters = sorted(merged.values(), key=lambda q: quarter_ord(q.quarter))
        return own


def tracked_symbols() -> list[str]:
    """Symbols already in the shareholding cache - what the scheduled refresh keeps current."""
    with sqlite3.connect(str(settings.market_db)) as con:
        con.executescript(_SCHEMA)
        return [r[0] for r in con.execute("SELECT DISTINCT symbol FROM shareholding ORDER BY symbol")]


def last_fetched():
    with sqlite3.connect(str(settings.market_db)) as con:
        con.executescript(_SCHEMA)
        row = con.execute("SELECT MAX(fetched_at) FROM shareholding").fetchone()
    try:
        return datetime.fromisoformat(row[0]) if row and row[0] else None
    except (ValueError, TypeError):
        return None


def cache_is_stale() -> bool:
    stamp = last_fetched()
    return stamp is None or datetime.now() - stamp > timedelta(days=settings.ownership_max_age_days)


@lru_cache(maxsize=1)
def _provider() -> OwnershipProvider:
    name = settings.ownership_provider
    if name == "nse+screener":                      # default: statutory source + FII/DII detail
        return CompositeOwnershipProvider([NseOwnershipProvider(), ScreenerOwnershipProvider()])
    if name == "nse":
        return NseOwnershipProvider()
    if name == "screener":
        return ScreenerOwnershipProvider()
    raise ValueError(f"Unknown ownership_provider: {name}")


@lru_cache(maxsize=1)
def _store() -> OwnershipStore:
    return OwnershipStore()


def get_ownership(symbol: str, refresh: bool = False) -> Ownership:
    """Cache-first read. Fetches only when the cache is empty, stale, or refresh is forced.

    If the fetch yields nothing (source blocked/changed), we fall back to whatever is cached -
    stale data with a stated as-of quarter beats no data at all.
    """
    store = _store()
    cached = store.load(symbol)
    if not refresh and not store.is_stale(cached):
        return cached
    fetched = _provider().fetch(symbol)
    if fetched.quarters:
        store.save(fetched)
        return store.load(symbol)            # re-read so manual pledge values are merged back in
    return cached

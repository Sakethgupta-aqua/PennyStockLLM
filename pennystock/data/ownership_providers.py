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
from urllib.parse import quote

from pennystock.config import settings
from pennystock.data.ownership import Ownership, Quarter
from pennystock.data.xbrl import facts, number

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
            for column in ("pledge_pct_of_total REAL", "encumbrance_note TEXT"):   # migrate older DBs
                try:
                    con.execute(f"ALTER TABLE shareholding ADD COLUMN {column}")
                except sqlite3.OperationalError:
                    pass

    def load(self, symbol: str) -> Ownership:
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                "SELECT quarter, promoter_pct, fii_pct, dii_pct, public_pct, government_pct,"
                " shareholders, pledge_pct, pledge_source, fetched_at, pledge_pct_of_total, encumbrance_note"
                " FROM shareholding"
                " WHERE symbol=? ORDER BY quarter_ord", (symbol.upper(),)).fetchall()
        own = Ownership(symbol=symbol.upper())
        for r in rows:
            own.quarters.append(Quarter(quarter=r[0], promoter_pct=r[1], fii_pct=r[2], dii_pct=r[3],
                                        public_pct=r[4], government_pct=r[5], shareholders=r[6],
                                        pledge_pct=r[7], pledge_source=r[8],
                                        pledge_pct_of_total=r[10], encumbrance_note=r[11]))
            own.fetched_at = r[9]
        return own

    def save(self, own: Ownership) -> int:
        """Upsert quarters. A refetch NEVER clobbers a manually entered pledge (source 'manual')."""
        now = datetime.now().isoformat(timespec="seconds")
        with sqlite3.connect(self.db_path) as con:
            for q in own.quarters:
                con.execute(
                    "INSERT INTO shareholding (symbol, quarter, quarter_ord, promoter_pct, fii_pct,"
                    " dii_pct, public_pct, government_pct, shareholders, pledge_pct, pledge_source,"
                    " fetched_at, pledge_pct_of_total, encumbrance_note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                    " ON CONFLICT(symbol, quarter) DO UPDATE SET"
                    "   promoter_pct=excluded.promoter_pct, fii_pct=excluded.fii_pct,"
                    "   dii_pct=excluded.dii_pct, public_pct=excluded.public_pct,"
                    "   government_pct=excluded.government_pct, shareholders=excluded.shareholders,"
                    "   pledge_pct=CASE WHEN shareholding.pledge_source = 'manual'"
                    "       THEN shareholding.pledge_pct"
                    "       ELSE COALESCE(excluded.pledge_pct, shareholding.pledge_pct) END,"
                    "   pledge_source=CASE WHEN shareholding.pledge_source = 'manual' THEN 'manual'"
                    "       ELSE COALESCE(excluded.pledge_source, shareholding.pledge_source) END,"
                    "   pledge_pct_of_total=CASE WHEN shareholding.pledge_source = 'manual'"
                    "       THEN shareholding.pledge_pct_of_total"
                    "       ELSE COALESCE(excluded.pledge_pct_of_total,"
                    "                     shareholding.pledge_pct_of_total) END,"
                    "   encumbrance_note=COALESCE(excluded.encumbrance_note, shareholding.encumbrance_note),"
                    "   fetched_at=excluded.fetched_at",
                    (own.symbol.upper(), q.quarter, quarter_ord(q.quarter), q.promoter_pct, q.fii_pct,
                     q.dii_pct, q.public_pct, q.government_pct, q.shareholders, q.pledge_pct,
                     q.pledge_source, now, q.pledge_pct_of_total, q.encumbrance_note))
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


_PROMOTER_CTX = "ShareholdingOfPromoterAndPromoterGroup_ContextI"
_TOTAL_CTX = "ShareholdingPattern_ContextI"
_ENCUMBERED = "WhetherAnySharesHeldByPromotersAreEncumbered"


def parse_shp_pledge(xml: str) -> dict | None:
    """Promoter pledge from one quarter's statutory shareholding pattern (SEBI LODR Reg 31, XBRL).

    NSE's corporate-pledgedata API returned no rows for ANY company on 2026-10-11 (including ones with 39
    pledge filings on record), while every quarter's SHP XBRL carries yes/no encumbrance flags and the pledged
    share count.
    Pledge is a % of the promoters' OWN holding (the market convention) and, beside it, of all shares."""
    fs = facts(xml)

    def at(name: str, ctx: str):
        return next((f for f in fs if f.name == name and f.context.id == ctx), None)

    def flag(suffix: str) -> bool | None:
        f = next((f for f in fs if f.name == _ENCUMBERED + suffix), None)
        return None if f is None else f.value.strip().lower() == "true"

    promoter, total = number(at("NumberOfShares", _PROMOTER_CTX)), number(at("NumberOfShares", _TOTAL_CTX))
    pledged = flag("UnderPledged")
    if promoter is None and total and pledged is not True:
        # no promoter group (YESBANK, FEDERALBNK, PAYTM, SWIGGY ...): the filing has no promoter holding, so
        # there is no promoter pledge; the provider takes this only when NSE lists 0% promoter as well
        return {"promoter_shares": 0.0, "total_shares": total, "pledged_shares": 0.0, "pledge_pct": 0.0,
                "pledge_pct_of_total": 0.0, "encumbrance_note": None, "no_promoter_group": True}
    if promoter is None or pledged is None:
        return None
    count = number(at("NumberOfSharesEncumberedUnderPledged", _PROMOTER_CTX))
    if count is None:
        if pledged:                                    # says pledged, gives no count: unreadable, never zero
            return None
        count = 0.0
    other = [name for name, on in (("a non-disposal undertaking", flag("UnderNonDisposalUndertaking")),
                                   ("another encumbrance", flag("OtherThanByWayOfPledgeOrNDU"))) if on]
    return {"promoter_shares": promoter, "total_shares": total, "pledged_shares": count,
            "pledge_pct": round(count / promoter * 100, 2) if promoter else 0.0,
            "pledge_pct_of_total": round(count / total * 100, 2) if total else None,
            "encumbrance_note": ("promoter shares are also encumbered by " + " and ".join(other))
            if other else None, "no_promoter_group": False}


class NseOwnershipProvider(OwnershipProvider):
    """NSE's own corporate-filings APIs - the authoritative statutory source, and free.

    corporate-share-holdings-master -> 20-25 quarters of promoter vs public %, each with its statutory XBRL;
    the pledge is read from that XBRL (parse_shp_pledge). The corporate-pledgedata API returned no rows for
    any company on 2026-10-11, so it is no longer called.

    IMPORTANT unit trap: NSE's headline `percSharesPledged` is a % of TOTAL shares, but the market
    convention for "promoter pledging" is % of the PROMOTER'S OWN holding. Suzlon reports 6.84% of
    total, which is 58.5% of the promoter stake - the difference between "low" and "SEVERE". We
    therefore compute pledged shares / promoter shares ourselves and keep the % of all shares beside it.

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

    def __init__(self, store: "OwnershipStore | None" = None):
        self.store = store                       # quarters already read from their XBRL are not re-fetched

    def _session(self, symbol: str):
        import requests
        s = requests.Session()
        s.headers.update(self.HEADERS)
        ref = f"{self.BASE}/get-quotes/equity?symbol={quote(symbol, safe='')}"
        s.headers["Referer"] = ref
        s.get(ref, timeout=settings.ownership_timeout_s)          # sets the cookies NSE requires
        return s

    def fetch(self, symbol: str) -> Ownership:
        symbol = symbol.upper()
        own = Ownership(symbol=symbol, fetched_at=datetime.now().isoformat(timespec="seconds"))
        by_quarter: dict[str, Quarter] = {}
        try:
            s = self._session(symbol)
            filings: list[dict] = []
            try:                                        # --- promoter / public history, 20-25 quarters ---
                # encoded: 'symbol=M&MFIN' reads as symbol=M plus a stray parameter
                url = (f"{self.BASE}/api/corporate-share-holdings-master?index=equities"
                       f"&symbol={quote(symbol, safe='')}")
                filings = s.get(url, timeout=settings.ownership_timeout_s).json() or []
                for r in filings:
                    label = norm_quarter(r.get("date"))
                    q = by_quarter.setdefault(label, Quarter(quarter=label))
                    q.promoter_pct, q.public_pct = _num(r.get("pr_and_prgrp")), _num(r.get("public_val"))
            except Exception as e:
                logger.warning(f"NSE shareholding fetch failed for {symbol}: {type(e).__name__}: {e}")

            # --- pledge, from each recent quarter's statutory XBRL, read once per quarter ---
            done = {q.quarter for q in (self.store or _store()).load(symbol).quarters
                    if q.pledge_source == "nse_shp_xbrl"}
            recent = sorted((r for r in filings if r.get("xbrl")),
                            key=lambda r: quarter_ord(norm_quarter(r.get("date"))), reverse=True)
            for r in recent[: settings.ownership_lookback_quarters + 1]:
                quarter = norm_quarter(r.get("date"))
                if quarter in done:
                    continue
                try:
                    pledge = parse_shp_pledge(s.get(r["xbrl"], timeout=settings.ownership_timeout_s).text)
                except Exception as e:
                    logger.warning(f"SHP XBRL unreadable for {symbol} {quarter}: {type(e).__name__}: {e}")
                    continue
                if pledge is not None and pledge["no_promoter_group"] and by_quarter[quarter].promoter_pct:
                    logger.warning(f"SHP XBRL for {symbol} {quarter} shows no promoter holding but NSE lists "
                                   f"{by_quarter[quarter].promoter_pct}% - pledge left unknown")
                    continue
                if pledge is not None:
                    q = by_quarter[quarter]
                    q.pledge_pct, q.pledge_pct_of_total = pledge["pledge_pct"], pledge["pledge_pct_of_total"]
                    q.pledge_source, q.encumbrance_note = "nse_shp_xbrl", pledge["encumbrance_note"]
            own.quarters = sorted(by_quarter.values(), key=lambda q: quarter_ord(q.quarter))
            logger.info(f"NSE: {symbol} -> {len(own.quarters)} quarter(s)")
        except Exception as e:
            logger.warning(f"NSE ownership unavailable for {symbol}: {type(e).__name__}: {e}")
        return own


_MERGE_FIELDS = ("promoter_pct", "fii_pct", "dii_pct", "public_pct", "government_pct",
                 "shareholders", "pledge_pct", "pledge_pct_of_total", "pledge_source", "encumbrance_note")


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

"""NSE integrated filings - a company's audited figures and its half-yearly related-party disclosure, as XBRL.

Since December 2024 every listed company files "Integrated Filing - Financials" with NSE as XBRL
(/api/integrated-filing-results). Measured on DIXON (2026-10-11): the March consolidated filing carries the
year's revenue (Rs 48,872.80 Cr), profit to owners (Rs 1,438.64 Cr), equity to owners (Rs 4,676.67 Cr), basic
EPS and the auditor; the September and March filings carry the SEBI Reg 23(9) related-party disclosure -
itemised, with each counterparty's relationship. Two traps, both measured: segment figures reuse the same
names in dimensional contexts, and related-party amounts cover the HALF-YEAR although their contexts carry the
quarter's dates (Airtel: Sep 1,271.99 + Mar 1,224.67 = 2,496.66 Cr, exactly the AGM notice's FY26 figure)."""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from urllib.parse import quote

from pennystock.config import settings
from pennystock.data.nse_client import NseFeed
from pennystock.data.xbrl import Context, facts, number

logger = logging.getLogger(__name__)

FINANCIALS = "Integrated Filing- Financials"
CRORE = 1e7
# the first concept a filing carries wins: standalone filings have no owners/NCI split
_CONCEPTS = {
    "revenue_cr": ("RevenueFromOperations",),
    "profit_to_owners_cr": ("ProfitOrLossAttributableToOwnersOfParent", "ProfitLossForPeriod", "ProfitLoss"),
    "equity_to_owners_cr": ("EquityAttributableToOwnersOfParent",),
    "equity_total_cr": ("Equity",),
    "basic_eps": ("BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
                  "BasicEarningsLossPerShareFromContinuingOperations", "BasicEarningsLossPerShare"),
    "exceptional_cr": ("ExceptionalItemsBeforeTax", "ExceptionalItems"),
}
_RPT_FIELDS = {
    "NameOfListedEntityOrSubsidiaryEnteringIntoTheTransaction": "entity",
    "NameOfCounterParty": "counterparty",
    "RelationshipOfTheCounterpartyWithTheListedEntityOrItsSubsidiary": "relationship",
    "TypeOfRelatedPartyTransaction": "type",
    "DetailsOfOtherRelatedPartyTransaction": "details",
    "AmountOfRelatedPartyTransactionDuringTheReportingPeriod": "amount",
    "ValueOfTheRelatedPartyTransactionAsApprovedByTheAuditCommittee": "approved",
}


@dataclass(frozen=True)
class RptRow:
    entity: str
    counterparty: str
    relationship: str
    type: str
    details: str
    amount_cr: float | None
    approved_cr: float | None


def _span(ctx: Context) -> str | None:
    if ctx.instant:
        return "I"
    d = ctx.days
    if d is None:
        return None
    return "FY" if 330 <= d <= 380 else "H" if 170 <= d <= 190 else "Q" if 80 <= d <= 100 else None


def financial_facts(xml: str) -> dict:
    """{key: {span: {"end", "value"}}} for the company's own figures (undimensioned contexts only; newest end
    per span wins), money in Rs crore, EPS in rupees; plus "auditor"."""
    found = facts(xml)
    out: dict = {}
    for key, names in _CONCEPTS.items():
        for name in names:
            picks: dict = {}
            for f in found:
                if f.name != name or f.context.dimensional:
                    continue
                span, v = _span(f.context), number(f)
                if span is None or v is None:
                    continue
                end = (f.context.instant or f.context.end)[:10]
                if span not in picks or end > picks[span]["end"]:
                    picks[span] = {"end": end, "value": v if key == "basic_eps" else round(v / CRORE, 2)}
            if picks:
                out[key] = picks
                break
    out["auditor"] = next((f.value for f in found if f.name == "AuditorsFirmName" and f.value), None)
    return out


def _money(raw: str | None) -> float | None:
    try:
        return round(float(str(raw).replace(",", "")) / CRORE, 2)
    except (TypeError, ValueError):
        return None


def rpt_rows(xml: str) -> list[RptRow]:
    """Every related-party transaction in a filing, in document order, amounts in Rs crore."""
    by_ctx: dict[str, dict] = {}
    for f in facts(xml):
        key = _RPT_FIELDS.get(f.name)
        if key:
            by_ctx.setdefault(f.context.id, {})[key] = f.value
    return [RptRow(d.get("entity", ""), d["counterparty"], d.get("relationship", ""), d.get("type", ""),
                   d.get("details", ""), _money(d.get("amount")), _money(d.get("approved")))
            for d in by_ctx.values() if d.get("counterparty")]


def quarter_end(qe: str) -> date | None:
    try:
        return datetime.strptime(str(qe).strip().title(), "%d-%b-%Y").date()
    except ValueError:
        return None


def half_label(end: date) -> tuple[str, str, str] | None:
    """The half-year a related-party disclosure covers, named from the filing's quarter end."""
    if end.month == 3:
        return f"H2 FY{end.year % 100:02d}", f"{end.year - 1}-10-01", end.isoformat()
    if end.month == 9:
        return f"H1 FY{(end.year + 1) % 100:02d}", f"{end.year}-04-01", end.isoformat()
    return None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS integrated_listings (
    symbol TEXT PRIMARY KEY, rows TEXT NOT NULL, fetched_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS integrated_xbrl (
    url TEXT PRIMARY KEY, symbol TEXT NOT NULL, qe_date TEXT, basis TEXT, broadcast_at TEXT,
    financials TEXT NOT NULL, rpt_rows TEXT NOT NULL, parser_version INTEGER NOT NULL,
    parsed_at TEXT NOT NULL);
"""


@dataclass(frozen=True)
class FiledYear:
    symbol: str
    fiscal_year: str
    period_end: str
    basis: str
    revenue_cr: float | None
    profit_to_owners_cr: float | None
    equity_to_owners_cr: float | None
    equity_total_cr: float | None
    basic_eps: float | None
    exceptional_cr: float | None
    auditor: str | None
    broadcast_at: str | None
    url: str


@dataclass(frozen=True)
class RptDisclosure:
    symbol: str
    half: str
    period_start: str
    period_end: str
    rows: list
    broadcast_at: str | None
    url: str
    revenue_cr: float | None
    revenue_note: str


def _value(fin: dict, key: str, span: str) -> float | None:
    return ((fin.get(key) or {}).get(span) or {}).get("value")


class IntegratedFilings:
    """The cache in market.db: a listing per company (refreshed weekly) and each XBRL parsed once - a
    filing's URL never changes (a revision is a new URL). `offline` reads the cache only, whatever its age -
    for a status report, which must never go to NSE."""

    def __init__(self, db_path=None, feed_factory=None, offline: bool = False):
        self.db_path = str(db_path or settings.market_db)
        self._feed_factory = feed_factory or (lambda symbol: NseFeed(symbol))
        self.offline = offline
        with self._connect() as con:
            con.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30)

    def listing(self, symbol: str, refresh: bool = False) -> list[dict]:
        symbol = symbol.upper()
        with self._connect() as con:
            row = con.execute("SELECT rows, fetched_at FROM integrated_listings WHERE symbol = ?",
                              (symbol,)).fetchone()
        if self.offline:
            return json.loads(row[0]) if row else []
        if row and not refresh and datetime.now() - datetime.fromisoformat(row[1]) < timedelta(
                days=settings.integrated_filings_max_age_days):
            return json.loads(row[0])
        path = f"integrated-filing-results?index=equities&symbol={quote(symbol)}"
        answer = self._feed_factory(symbol).api(path)
        data = answer.get("data") if isinstance(answer, dict) else None
        if data is None:                                  # NSE down: a stale listing beats none
            return json.loads(row[0]) if row else []
        rows = [r for r in data if r.get("type") == FINANCIALS and r.get("xbrl")]
        with self._connect() as con:
            con.execute("INSERT INTO integrated_listings VALUES (?,?,?) ON CONFLICT(symbol) DO UPDATE SET "
                        "rows = excluded.rows, fetched_at = excluded.fetched_at",
                        (symbol, json.dumps(rows), datetime.now().isoformat(timespec="seconds")))
        return rows

    def parsed(self, symbol: str, row: dict) -> dict | None:
        url = row["xbrl"]
        with self._connect() as con:
            hit = con.execute("SELECT financials, rpt_rows FROM integrated_xbrl WHERE url = ? "
                              "AND parser_version = ?", (url, settings.integrated_parser_version)).fetchone()
        if hit:
            return {"financials": json.loads(hit[0]), "rpt_rows": [RptRow(**r) for r in json.loads(hit[1])]}
        if self.offline:
            return None
        body = self._feed_factory(symbol).get(url)
        if body is None:
            logger.warning(f"integrated filing unavailable: {url}")
            return None
        xml = body.decode("utf-8", "replace")
        fin, rows = financial_facts(xml), rpt_rows(xml)
        with self._connect() as con:
            con.execute("INSERT OR REPLACE INTO integrated_xbrl VALUES (?,?,?,?,?,?,?,?,?)",
                        (url, symbol.upper(), row.get("qe_Date"), row.get("consolidated"),
                         row.get("broadcast_Date"), json.dumps(fin), json.dumps([asdict(r) for r in rows]),
                         settings.integrated_parser_version, datetime.now().isoformat(timespec="seconds")))
        return {"financials": fin, "rpt_rows": rows}

    def _pick(self, rows: list[dict], end: date, basis: str) -> dict | None:
        return next((r for r in rows
                     if quarter_end(r.get("qe_Date")) == end and r.get("consolidated") == basis), None)

    def _ends(self, rows: list[dict], months: tuple[int, ...]) -> list[date]:
        ends = {quarter_end(r.get("qe_Date")) for r in rows}
        return sorted((e for e in ends if e and e.month in months), reverse=True)

    def filed_year(self, symbol: str) -> FiledYear | None:
        rows = self.listing(symbol)
        for end in self._ends(rows, (3,)):
            for basis in ("Consolidated", "Standalone"):
                row = self._pick(rows, end, basis)
                fin = ((self.parsed(symbol, row) or {}).get("financials") or {}) if row else {}
                if _value(fin, "revenue_cr", "FY") is None:
                    continue
                return FiledYear(symbol.upper(), f"FY{end.year % 100:02d}", end.isoformat(), basis.lower(),
                                 _value(fin, "revenue_cr", "FY"), _value(fin, "profit_to_owners_cr", "FY"),
                                 _value(fin, "equity_to_owners_cr", "I"), _value(fin, "equity_total_cr", "I"),
                                 _value(fin, "basic_eps", "FY"), _value(fin, "exceptional_cr", "FY"),
                                 fin.get("auditor"), row.get("broadcast_Date"), row["xbrl"])
        return None

    def _half_revenue(self, symbol: str, rows: list[dict], end: date) -> tuple[float | None, str]:
        """The same half-year's consolidated revenue: a September filing's six-month figure; for March, the
        year minus that September half; half the year when only the year is on file (said so)."""
        def revenue(at: date, span: str) -> float | None:
            row = self._pick(rows, at, "Consolidated") or self._pick(rows, at, "Standalone")
            if row is None:
                return None
            return _value((self.parsed(symbol, row) or {}).get("financials") or {}, "revenue_cr", span)
        if end.month == 9:
            h = revenue(end, "H")
            return (h, f"six months to {end.isoformat()}") if h else (None, "half-year revenue not on file")
        fy, h1 = revenue(end, "FY"), revenue(date(end.year - 1, 9, 30), "H")
        if fy and h1:
            return round(fy - h1, 2), f"FY{end.year % 100:02d} revenue minus the April-September half"
        if fy:
            return round(fy / 2, 2), "approximate: half of the full year's revenue"
        return None, "half-year revenue not on file"

    def rpt_disclosure(self, symbol: str) -> RptDisclosure | None:
        rows = self.listing(symbol)
        for end in self._ends(rows, (3, 9))[:2]:          # the latest half-year, else the one before it
            for basis in ("Standalone", "Consolidated"):
                row = self._pick(rows, end, basis)
                items = ((self.parsed(symbol, row) or {}).get("rpt_rows") or []) if row else []
                if items:
                    label, start, stop = half_label(end)
                    revenue, note = self._half_revenue(symbol, rows, end)
                    return RptDisclosure(symbol.upper(), label, start, stop, items, row.get("broadcast_Date"),
                                         row["xbrl"], revenue, note)
        return None


@lru_cache(maxsize=1)
def _store() -> IntegratedFilings:
    return IntegratedFilings()


def filed_year(symbol: str) -> FiledYear | None:
    try:
        return _store().filed_year(symbol)
    except Exception as e:                                  # a yardstick is a bonus, never a crash
        logger.warning(f"filed year unavailable for {symbol}: {type(e).__name__}: {e}")
        return None


def rpt_disclosure(symbol: str) -> RptDisclosure | None:
    try:
        return _store().rpt_disclosure(symbol)
    except Exception as e:
        logger.warning(f"related-party disclosure unavailable for {symbol}: {type(e).__name__}: {e}")
        return None

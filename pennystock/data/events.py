"""Corporate events - the news flow that moves Indian stocks before the numbers ever show it.

Three statutory streams, all free from NSE:
  ANNOUNCEMENTS   auditor resignations, defaults, rating actions, litigation, board exits.
                  Research on Indian markets is blunt about this: a mid-term auditor resignation
                  is never routine - auditors do not walk away from a paying client unless they
                  have found something they cannot sign. It is one of the highest-signal events
                  available, and it appears here days or weeks before it appears in the numbers.
  INSIDER TRADES  SEBI PIT disclosures. Promoters and management buying with their own money is
                  the most honest bullish signal there is; heavy insider selling is the reverse.
  CREDIT RATING   agency, rating and action (upgrade / downgrade / reaffirm).

Announcements are noisy - a large company files hundreds a year, most of them routine investor
meets. So we SCREEN by keyword into material vs routine, and only the material ones reach the
analyst. That keyword list is the product here; getting the data is the easy part.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from pennystock.config import settings
from pennystock.data.nse_client import nse_api, nse_session

logger = logging.getLogger(__name__)

# Routine filings are checked FIRST and suppressed. A large company files hundreds of these a
# year and they all mention words like "SEBI" and "Regulation" - without this the material screen
# cries wolf constantly, which is worse than no screen at all. (Real example caught in testing:
# "Certificate under SEBI (Depositories and Participants) Regulations" was being flagged as fraud.)
ROUTINE_PATTERNS = r"""
    certificate\s+under | compliance\s+certificate | trading\s+window | newspaper\s+publication
  | investor\s+(meet|presentation) | analysts?/institutional | con\.?\s*call | earnings\s+call
  | record\s+date | reg\.?\s*7[46] | regulation\s+7[46] | disclosure\s+under\s+regulation\s+3[12]
  | intimation\s+of | loss\s+of\s+share\s+certificate | duplicate\s+share | transcript
  | schedule\s+of | copy\s+of\s+newspaper | annual\s+report | notice\s+of\s+(agm|egm)
"""
_ROUTINE = re.compile(ROUTINE_PATTERNS, re.IGNORECASE | re.VERBOSE)

# Ordered by severity - the first pattern that matches decides the label. Patterns are deliberately
# SPECIFIC: bare "SEBI" or "investigate" appear in routine text, so they are not used alone.
MATERIAL_PATTERNS: list[tuple[str, str]] = [
    ("auditor resignation / removal",
     r"auditor.{0,40}(resign|remov|cessation)|resign.{0,40}auditor|qualified\s+opinion|adverse\s+opinion|disclaimer\s+of\s+opinion"),
    ("default / insolvency",
     r"default\s+in\s+(payment|repayment)|\bNCLT\b|insolven|winding\s+up|bankrupt|moratorium|debt\s+restructur"),
    ("fraud / investigation / enforcement",
     r"\bfraud\b|forensic\s+audit|show\s+cause|adjudicat|search\s+and\s+seizure|prosecution"
     r"|penalt(y|ies)\s+(imposed|levied)|SEBI\s+(order|notice|summons|enforcement)|whistle\s*blower"),
    ("credit rating action",
     r"credit\s+rating|rating\s+(downgrade|upgrade|revision|action|withdraw)|outlook\s+revis"),
    ("pledge / encumbrance",
     r"pledge|encumbr|invocation\s+of"),
    ("senior management change",
     r"(resign|cessation|vacat).{0,70}(CFO|CEO|managing\s+director|whole.?time\s+director|chairman|chief\s+financial)"),
    ("litigation / dispute",
     r"litigation|arbitrat|tribunal|lawsuit|writ\s+petition|order\s+of\s+(the\s+)?(high\s+)?court"),
    ("fund raise / dilution",
     r"preferential\s+(issue|allot)|\bQIP\b|rights\s+issue|convertible\s+(debenture|warrant)|fund\s+rais"),
    ("M&A / restructuring",
     r"amalgamat|\bmerger\b|demerger|acquisition\s+of|slump\s+sale|divest|scheme\s+of\s+arrangement"),
]
_COMPILED = [(label, re.compile(pat, re.IGNORECASE)) for label, pat in MATERIAL_PATTERNS]


@dataclass
class Announcement:
    date: str
    category: str
    text: str
    material_type: str | None = None


@dataclass
class InsiderTrade:
    date: str
    name: str
    transaction: str          # Buy / Sell / Pledge ...
    shares: float | None = None
    value_rs: float | None = None
    category: str | None = None


@dataclass
class CreditRating:
    agency: str | None = None
    rating: str | None = None
    action: str | None = None
    outlook: str | None = None
    date: str | None = None


@dataclass
class Events:
    symbol: str
    announcements: list[Announcement] = field(default_factory=list)
    insider_trades: list[InsiderTrade] = field(default_factory=list)
    credit_ratings: list[CreditRating] = field(default_factory=list)


def classify(text: str) -> str | None:
    """Material-event label for an announcement, or None if it is routine/noise.

    Routine filings are suppressed first - see ROUTINE_PATTERNS for why that matters.
    """
    blob = str(text or "")
    if _ROUTINE.search(blob):
        return None
    for label, pattern in _COMPILED:
        if pattern.search(blob):
            return label
    return None


def _num(v):
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


class NseEventsProvider:
    """Fetches announcements, insider trades and credit ratings from NSE. Fails soft throughout."""

    def fetch(self, symbol: str) -> Events:
        symbol = symbol.upper()
        ev = Events(symbol=symbol)
        try:
            s = nse_session(symbol)
        except Exception as e:
            logger.warning(f"NSE session failed for {symbol}: {type(e).__name__}: {e}")
            return ev

        rows = nse_api(s, f"corporate-announcements?index=equities&symbol={symbol}") or []
        for r in rows[: settings.events_scan_limit]:
            text = f"{r.get('desc', '')} {r.get('attchmntText', '')}"
            ev.announcements.append(Announcement(
                date=str(r.get("an_dt", "")), category=str(r.get("desc", "")),
                text=" ".join(text.split())[:400], material_type=classify(text)))

        pit = nse_api(s, f"corporates-pit?index=equities&symbol={symbol}") or {}
        for r in (pit.get("data") or [])[: settings.events_insider_limit]:
            ev.insider_trades.append(InsiderTrade(
                date=str(r.get("date", "")), name=str(r.get("acqName", "")),
                transaction=str(r.get("tdpTransactionType", "")),
                shares=_num(r.get("secAcq")), value_rs=_num(r.get("secVal")),
                category=str(r.get("personCategory", "")) or None))

        ratings = nse_api(s, f"corporate-credit-rating?index=equities&symbol={symbol}") or []
        for r in ratings[: settings.events_rating_limit]:
            ev.credit_ratings.append(CreditRating(
                agency=r.get("NameOfCRAgency"), rating=r.get("CreditRating"),
                action=r.get("RatingAction"), outlook=r.get("Outlook"), date=r.get("Date")))
        return ev


def summarize_events(ev: Events) -> dict:
    """Compact, LLM-ready event picture: material items only, plus an insider buy/sell tally."""
    material = [a for a in ev.announcements if a.material_type]
    by_type: dict[str, int] = {}
    for a in material:
        by_type[a.material_type] = by_type.get(a.material_type, 0) + 1

    buys = [t for t in ev.insider_trades if "buy" in t.transaction.lower()]
    sells = [t for t in ev.insider_trades if "sell" in t.transaction.lower()]
    buy_val = sum(t.value_rs or 0 for t in buys)
    sell_val = sum(t.value_rs or 0 for t in sells)

    flags: list[str] = []
    for label in ("auditor resignation / removal", "default / insolvency",
                  "fraud / investigation / enforcement"):
        if by_type.get(label):
            flags.append(f"{by_type[label]} announcement(s) matching '{label}' - "
                         "one of the highest-signal warnings in Indian markets. Read them.")
    if by_type.get("credit rating action"):
        flags.append(f"{by_type['credit rating action']} credit-rating announcement(s) - "
                     "check whether it was an upgrade or a downgrade.")
    if sell_val > buy_val * 3 and sell_val > 0:
        flags.append(f"Insiders sold far more than they bought (Rs {sell_val:,.0f} sold vs "
                     f"Rs {buy_val:,.0f} bought across the disclosures fetched).")
    if not ev.announcements:
        flags.append("No announcement data returned - treat event risk as UNCHECKED, not clean.")

    latest_rating = ev.credit_ratings[0] if ev.credit_ratings else None
    return {
        "symbol": ev.symbol,
        "announcements_scanned": len(ev.announcements),
        "material_announcements": len(material),
        "material_by_type": by_type or "none matched the material-event screen",
        "recent_material": [{"date": a.date, "type": a.material_type, "text": a.text[:220]}
                            for a in material[: settings.events_report_limit]],
        "insider_summary": {
            "disclosures": len(ev.insider_trades),
            "buys": len(buys), "sells": len(sells),
            "buy_value_rs": round(buy_val), "sell_value_rs": round(sell_val),
            "net_signal": ("net BUYING" if buy_val > sell_val else
                           "net SELLING" if sell_val > buy_val else "balanced/none"),
        },
        "latest_credit_rating": ({"agency": latest_rating.agency, "rating": latest_rating.rating,
                                  "action": latest_rating.action, "outlook": latest_rating.outlook}
                                 if latest_rating else None),
        "event_flags": flags or ["No material corporate-event red flags in the window scanned."],
    }

"""Annual-report notes - the two oldest promoter checks in the Indian playbook.

Long before anyone computed a Z-score here, Indian analysts turned to two notes at the back of the
annual report, and they still matter more than most ratios:

  CONTINGENT LIABILITIES  Obligations that are not on the balance sheet yet. Guarantees given to
                          group companies live here, and so do disputed tax demands. A company can
                          look unlevered on the face of the balance sheet and be guaranteeing the
                          debts of half the promoter group in note 31. The traditional rule of
                          thumb: contingent liabilities running large against NET WORTH deserve an
                          explanation.
  RELATED-PARTY TXNS      The classic route by which value leaves a listed company and arrives at a
                          promoter-owned one. Rule of thumb: total RPT should be a modest share of
                          revenue (commonly cited: under ~10%), and the REASON matters as much as
                          the size.

Neither is in any free structured feed - they exist only in the annual-report PDF. NSE publishes
those at /api/annual-reports, so we fetch, extract, and cache JUST THE TWO NOTES (a full report is
~460 pages / 2.7M characters; storing it whole would bloat market.db for no benefit).

DELIBERATE LIMITATION: we do NOT parse the tables into a precise ratio. PDF table extraction
mangles columns, and a confidently wrong "contingent liabilities are 340% of net worth" is far more
dangerous than an honest passage. So we extract the NOTE TEXT plus candidate figures, and let the
analyst read them - the same compute-in-code / judge-in-the-model split used elsewhere here.
"""
from __future__ import annotations

import io
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from pennystock.config import settings
from pennystock.data.nse_client import nse_api, nse_session

logger = logging.getLogger(__name__)

_SCHEMA = """CREATE TABLE IF NOT EXISTS annual_report_notes (
    url TEXT PRIMARY KEY, symbol TEXT NOT NULL, from_yr TEXT, to_yr TEXT,
    contingent_text TEXT, related_party_text TEXT, pages INTEGER, fetched_at TEXT)"""

CONTINGENT = re.compile(r"contingent\s+liabilit", re.I)
# the note sits near this language; the accounting-policy paragraph does not
CONTINGENT_DETAIL = re.compile(r"claims?\s+against|not\s+acknowledged\s+as\s+debt|guarantee", re.I)
# ...and THIS is the phrase the actual note is built around. Indian annual reports have opened the
# contingent-liability schedule with "claims against the Company not acknowledged as debt" for
# decades. Pages carrying it are the note; pages that merely say "guarantee" are cross-references.
CONTINGENT_STRONG = re.compile(r"not\s+acknowledged\s+as\s+debt", re.I)

RELATED_PARTY = re.compile(r"related\s+part(?:y|ies)", re.I)
# the AGM notice is often bound in front of the annual report; its proposed-RPT resolutions read like the
# related-party note and won the search (DIXON FY26: pp. 4-55, 'Part A: Minimum information of the proposed
# RPT')
AGM_HEADER = re.compile(r"\bAGM\s+Notice\b|notice\s+of\s+(the\s+)?\d*\w*\s*annual\s+general\s+meeting", re.I)
AGM_BODY = re.compile(r"minimum\s+information\s+of\s+the\s+proposed\s+rpt|notice\s+is\s+hereby\s+given", re.I)
RELATED_DETAIL = re.compile(r"transactions?\s+with\s+related|key\s+manage(?:ment|rial)\s+personnel"
                            r"|holding\s+company|fellow\s+subsidiar", re.I)
# the transactions TABLE, as opposed to the page that merely lists related parties by name
RELATED_STRONG = re.compile(r"transactions?\s+(?:were\s+)?(?:carried\s+out|entered\s+into)\s+with"
                            r"|following\s+transactions?\s+with\s+related"
                            r"|nature\s+of\s+transaction", re.I)
# Indian reports write money either comma-grouped in the lakh-crore style (12,740 / 1,23,456) or
# with an explicit unit. Requiring one or the other is what keeps years ("2026") and mangled table
# identifiers ("4260848") out of the figure list - both were in the first live run.
FIGURE = re.compile(
    r"(?:INR|Rs\.?|₹)?\s?("
    r"\d{1,3}(?:,\d{2,3})+(?:\.\d+)?"                                  # 12,740  /  1,23,456.50
    r"|\d[\d,]*(?:\.\d+)?\s*(?:crore|crores|cr|lakh|lakhs|million|billion)\b"  # 250 crore
    r")", re.I)


@dataclass
class AnnualReportNotes:
    symbol: str
    from_yr: str | None = None
    to_yr: str | None = None
    url: str | None = None
    contingent_text: str = ""
    related_party_text: str = ""
    pages: int | None = None


def _cached(symbol: str, url: str | None = None) -> AnnualReportNotes | None:
    """Newest cached notes for a symbol (or for one exact report)."""
    with _connect() as con:
        sql = ("SELECT symbol, from_yr, to_yr, url, contingent_text, related_party_text, pages"
               " FROM annual_report_notes WHERE ")
        row = (con.execute(sql + "url=?", (url,)).fetchone() if url else
               con.execute(sql + "symbol=? ORDER BY fetched_at DESC LIMIT 1", (symbol,)).fetchone())
    return AnnualReportNotes(*row) if row else None


def _connect():
    con = sqlite3.connect(str(settings.market_db))
    con.executescript(_SCHEMA)
    return con


def list_reports(symbol: str) -> tuple[list[dict], object]:
    """Annual reports NSE has for a symbol, newest first, plus the primed session."""
    session = nse_session(symbol)
    data = nse_api(session, f"annual-reports?index=equities&symbol={symbol.upper()}") or {}
    return (data.get("data") or []), session


def _section(pages: list[str], head: re.Pattern, detail: re.Pattern, window: int,
             strong: re.Pattern | None = None) -> str:
    """Find the PAGE carrying the real note (heading + detail language) and return a window of it.

    Two heuristics, both learned from live runs on real reports:
      1. Require the DETAIL language as well as the heading. Matching the heading alone lands on the
         accounting-policy paragraph, which explains what a contingent liability *is* and carries no
         numbers at all.
      2. Rank candidate windows by HOW MANY MONEY FIGURES follow the heading, not by length. A
         related-party note spans several pages - one lists the parties by name, another holds the
         transactions table. Only the table is worth reading, and counting figures finds it.

      3. Rank on (has any figure, carries the note's canonical phrase, figure count, length), in
         that order. HAS-A-FIGURE has to come first: letting the canonical phrase win outright made
         the Hindalco run pick a figureless narrative paragraph over the actual schedule. The phrase
         then breaks ties, which is what stops the Vedanta run picking a credit-risk paragraph that
         merely points at "Note 40 Contingent liability".

    The window starts AT the heading and reads forward - nothing before it. Hindalco puts a
    segment-liabilities table ("Total liabilities 211,200 142,270") immediately above "30. Related
    party disclosures"; including any lookback both skewed the ranking and, worse, leaked that
    table's numbers into `figures_mentioned`, which is the one field that has to be trustworthy.
    """
    best, best_score = "", (-1, -1, -1, -1)
    for text in pages:
        if AGM_HEADER.search(text[:400]) or AGM_BODY.search(text):
            continue
        # `strong` is a stricter signal than `detail`, so it has to SATISFY the gate as well as win
        # the ranking. Without this, Hindalco's own wording - "the following transactions were
        # carried out with the related parties" - failed the narrower detail pattern and the real
        # transactions table was skipped.
        if not (detail.search(text) or (strong and strong.search(text))):
            continue
        for m in head.finditer(text):                  # a page can mention the note more than once
            chunk = " ".join(text[m.start(): m.start() + window].split())
            n_fig = len(FIGURE.findall(chunk))
            score = (1 if n_fig else 0, 1 if strong and strong.search(chunk) else 0,
                     n_fig, len(chunk))
            if score > best_score:
                best, best_score = chunk, score
    return best[:window]


def candidate_figures(text: str, limit: int = 8) -> list[str]:
    """Large numbers appearing in the note - HINTS for the analyst, not a parsed total."""
    out: list[str] = []
    for m in FIGURE.finditer(text or ""):
        raw = " ".join(m.group(1).split())
        digits = re.sub(r"[^\d.]", "", raw.split(" ")[0])
        try:
            if float(digits) < 100:         # note numbers, percentages, small counts
                continue
        except ValueError:
            continue
        if raw not in out:
            out.append(raw)
        if len(out) >= limit:
            break
    return out


def fetch_notes(symbol: str, refresh: bool = False) -> AnnualReportNotes:
    """Latest annual report's two notes. Cache-first - the PDF is downloaded at most once."""
    symbol = symbol.upper()
    try:
        reports, session = list_reports(symbol)
    except Exception as e:
        logger.warning(f"annual-report listing failed for {symbol}: {type(e).__name__}: {e}")
        reports, session = [], None
    latest = reports[0] if reports else None
    url = latest.get("fileName") if latest else None

    if url and not refresh and (hit := _cached(symbol, url)):
        return hit
    if not url or session is None:                  # NSE listing unavailable - serve what we have
        return _cached(symbol) or AnnualReportNotes(symbol=symbol)

    pages: list[str] = []
    for attempt in (1, 2):
        # These downloads are 15-20MB and DO come back truncated: a live VEDL run returned a body
        # pypdf could not open ("Cannot find Root object"). One retry clears it; a stale cached note
        # beats no note at all, so a second failure falls back rather than returning blank.
        try:
            from pypdf import PdfReader
            r = session.get(url, timeout=settings.annual_report_timeout_s)
            r.raise_for_status()
            reader = PdfReader(io.BytesIO(r.content))
            pages = [(p.extract_text() or "") for p in reader.pages]
            break
        except Exception as e:
            logger.warning(f"annual report for {symbol} failed (attempt {attempt}): "
                           f"{type(e).__name__}: {e}")
    if not pages:
        return _cached(symbol) or AnnualReportNotes(symbol=symbol)

    notes = AnnualReportNotes(
        symbol=symbol, from_yr=latest.get("fromYr"), to_yr=latest.get("toYr"), url=url,
        pages=len(pages),
        contingent_text=_section(pages, CONTINGENT, CONTINGENT_DETAIL,
                                 settings.annual_report_window, CONTINGENT_STRONG),
        related_party_text=_section(pages, RELATED_PARTY, RELATED_DETAIL,
                                    settings.annual_report_window, RELATED_STRONG))
    with _connect() as con:
        con.execute("INSERT INTO annual_report_notes (url, symbol, from_yr, to_yr, contingent_text,"
                    " related_party_text, pages, fetched_at) VALUES (?,?,?,?,?,?,?,?)"
                    " ON CONFLICT(url) DO UPDATE SET contingent_text=excluded.contingent_text,"
                    " related_party_text=excluded.related_party_text, fetched_at=excluded.fetched_at",
                    (url, symbol, notes.from_yr, notes.to_yr, notes.contingent_text,
                     notes.related_party_text, notes.pages,
                     datetime.now().isoformat(timespec="seconds")))
    return notes


def notes_from_pages(symbol: str, pages: list[str], url: str | None, from_yr: str | None,
                     to_yr: str | None) -> AnnualReportNotes:
    """The two notes from page texts already extracted (filings.db) - no download."""
    return AnnualReportNotes(symbol=symbol, from_yr=from_yr, to_yr=to_yr, url=url, pages=len(pages),
                             contingent_text=_section(pages, CONTINGENT, CONTINGENT_DETAIL,
                                                      settings.annual_report_window, CONTINGENT_STRONG),
                             related_party_text=_section(pages, RELATED_PARTY, RELATED_DETAIL,
                                                         settings.annual_report_window, RELATED_STRONG))


def summarize_notes(notes: AnnualReportNotes, net_worth_cr: float | None = None,
                    revenue_cr: float | None = None) -> dict:
    """Bounded, LLM-ready view of the two notes, with the yardsticks to judge them against."""
    if not notes.contingent_text and not notes.related_party_text:
        return {"annual_report_notes": "unavailable",
                "note": "No annual report note could be extracted. Treat off-balance-sheet exposure "
                        "and related-party dealings as UNCHECKED - not as clean."}
    return {
        "symbol": notes.symbol,
        "financial_year": f"{notes.from_yr}-{notes.to_yr}" if notes.from_yr else None,
        "source_url": notes.url,
        "report_pages": notes.pages,
        "contingent_liabilities": {
            "note_excerpt": notes.contingent_text[:1600],
            "figures_mentioned": candidate_figures(notes.contingent_text),
            "net_worth_cr_for_comparison": net_worth_cr,
            "how_to_judge": "Guarantees to group companies and disputed tax demands sit here. "
                            "Compare the totals against net worth: a contingent liability that is "
                            "large relative to net worth is real risk sitting off the balance sheet.",
        },
        "related_party_transactions": {
            "note_excerpt": notes.related_party_text[:1600],
            "figures_mentioned": candidate_figures(notes.related_party_text),
            "revenue_cr_for_comparison": revenue_cr,
            "how_to_judge": "This is the classic route by which value leaves a listed company for a "
                            "promoter-owned one. Judge SIZE against revenue (a commonly used "
                            "threshold is ~10%) and, just as important, the REASON given.",
        },
        "caveat": "Figures are extracted from a PDF and the tables do not survive extraction "
                  "cleanly. Treat 'figures_mentioned' as hints to verify in the linked report, "
                  "NEVER as parsed totals, and say so if you quote them.",
    }

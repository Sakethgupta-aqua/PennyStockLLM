"""Corporate filings: the company's own statutory documents, straight from NSE.

NSE's corporate-announcements feed returns ~1,400 rows per company, most of them routine (AU SF Bank:
520 investor-meet schedules, 60 newspaper copies, 50 trading-window notices). The library keeps only
the documents an analyst actually reads, from the last two years:

  results                 quarterly results, the results board-meeting outcome, integrated filings,
                          the results press release
  investor_presentation   the deck management presents
  earnings_call           the transcript of the results call
  credit_rating           rating-agency actions
  material_event          auditor change, fraud or forensic audit, regulatory action, litigation,
                          senior-management exits, M&A and restructuring, guarantees, related parties
  annual_report           from NSE's annual-reports feed (the newest two)

NSE's own subject line (`desc`) is a fixed regulatory taxonomy, so it decides first; free-text rules
(the transcript hint, events.classify) only fill the gaps the taxonomy leaves.

Files land in Corporate_Filings/<FOLDER>/<category>/<YYYY-MM-DD>_<category>_<slug>_<id>.pdf, date
first so a folder reads oldest to newest, NSE's seq_id last so two filings on one day never collide.
Spec: docs/superpowers/specs/2026-10-04-company-documents-library-design.md
"""
from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable
from urllib.parse import quote

from pennystock.config import settings
from pennystock.data import documents
from pennystock.data.events import classify as material_type
from pennystock.data.transcripts import TRANSCRIPT_HINT

logger = logging.getLogger(__name__)

CATEGORIES = ("results", "investor_presentation", "earnings_call", "annual_report", "credit_rating",
              "material_event")

_RESULTS_DESC = {"financial result updates", "integrated filing- financial", "integrated filing - financial"}
_RESULTS_TEXT = re.compile(r"financial\s+results", re.I)
_INVESTOR_DECK = re.compile(r"investor\s+presentation|earnings\s+presentation", re.I)
# NSE subject lines that ARE a material event, whatever the free text around them says
_MATERIAL_DESC = re.compile(
    r"^(change in auditors|frauds?/default|final forensic|action\(s\) (taken|initiated)|pendency of litigation"
    r"|resignation|cessation|change in management|demise|acquisition|scheme of arrangement"
    r"|other restructuring|offer for sale|giving guarantees|related party transactions|granting/withdrawal"
    r"|default)")
# "<Company> has informed the Exchange about ..." - the part of NSE's text that says nothing
_BOILERPLATE = re.compile(r"^.*?\bhas\s+(?:informed|submitted\s+to)\s+the\s+exchange\s*(?:,|about|regarding|that)?\s*",
                          re.I)


@dataclass(frozen=True)
class Filing:
    """One exchange document worth keeping."""
    symbol: str
    source_id: str          # NSE seq_id, or AR<fromYr>-<toYr> for an annual report
    date: str               # ISO date it was filed
    category: str
    subject: str            # NSE's subject line (desc)
    text: str               # NSE's attachment text, trimmed
    url: str

    @property
    def doc_key(self) -> str:
        if self.category == "annual_report":
            return f"filing:nse:AR:{self.symbol}:{self.source_id.removeprefix('AR')}"
        return f"filing:nse:{self.source_id}"


def _norm(s) -> str:
    return " ".join(str(s or "").split())


def classify_filing(desc: str, text: str, file_name: str) -> str | None:
    """The library category of one NSE announcement, or None when it is not worth keeping."""
    d, t = _norm(desc).lower(), _norm(text)
    blob = f"{_norm(desc)} {t}"
    if TRANSCRIPT_HINT.search(blob) and "transcript" in file_name.lower():
        return "earnings_call"
    if d.startswith("investor presentation") or (_INVESTOR_DECK.search(t) and "schedule" not in t.lower()):
        return "investor_presentation"
    if d in _RESULTS_DESC or (d.startswith(("outcome of board meeting", "press release"))
                              and _RESULTS_TEXT.search(t)):
        return "results"
    if d.startswith("credit rating"):
        return "credit_rating"
    if _MATERIAL_DESC.match(d):
        return "material_event"
    kind = material_type(blob)
    if kind == "credit rating action":
        return "credit_rating"
    return "material_event" if kind else None


def _parse_day(raw) -> str | None:
    """'2026-10-03 19:39:55' or '03-Oct-2026 19:39:55' -> '2026-10-03'."""
    s = str(raw or "").strip()
    if re.match(r"\d{4}-\d{2}-\d{2}", s):
        return s[:10]
    try:
        return datetime.strptime(s[:11], "%d-%b-%Y").date().isoformat()
    except ValueError:
        return None


def select_filings(symbol: str, rows: list[dict], today: date | None = None,
                   days: int | None = None) -> list[Filing]:
    """The announcements worth keeping: a PDF, inside the window, in a library category. Newest first."""
    today = today or date.today()
    since = (today - timedelta(days=days or settings.filings_days)).isoformat()
    out = []
    for r in rows:
        url = str(r.get("attchmntFile") or "")
        when = _parse_day(r.get("sort_date")) or _parse_day(r.get("an_dt"))
        # seq_id becomes part of a file name: NSE's ids are numbers, and nothing else is trusted there
        if not url.lower().endswith(".pdf") or not when or when < since or not str(r.get("seq_id", "")).isdigit():
            continue
        category = classify_filing(r.get("desc", ""), r.get("attchmntText", ""), url.rsplit("/", 1)[-1])
        if category:
            out.append(Filing(symbol=symbol.upper(), source_id=str(r["seq_id"]), date=when, category=category,
                              subject=_norm(r.get("desc")), text=_norm(r.get("attchmntText"))[:300], url=url))
    return sorted(out, key=lambda f: (f.date, f.source_id), reverse=True)


def select_annual_reports(symbol: str, rows: list[dict], keep: int = 2) -> list[Filing]:
    """The newest `keep` annual reports that are PDFs (some are filed as ZIPs; those are skipped)."""
    out, seen = [], set()
    for r in sorted(rows, key=lambda r: (str(r.get("toYr", "")), str(r.get("fromYr", ""))), reverse=True):
        url, fy, ty = str(r.get("fileName") or ""), str(r.get("fromYr", "")), str(r.get("toYr", ""))
        if not url.lower().endswith(".pdf") or (fy, ty) in seen:
            continue
        seen.add((fy, ty))
        when = _parse_day(r.get("disseminationDateTime")) or _parse_day(r.get("broadcast_dttm")) or f"{ty}-03-31"
        out.append(Filing(symbol=symbol.upper(), source_id=f"AR{fy}-{ty}", date=when, category="annual_report",
                          subject=f"Annual Report {fy}-{ty}", text="", url=url))
        if len(out) >= keep:
            break
    return out


def slug(filing: Filing, words: int = 6) -> str:
    """A few readable words for the file name, NSE's boilerplate removed."""
    text = _BOILERPLATE.sub("", filing.text) or filing.subject
    s = re.sub(r"[^a-z0-9]+", "-", " ".join(text.lower().split()[:words])).strip("-")
    return s[:50].rstrip("-") or filing.category


def filing_path(filing: Filing, root: Path | None = None) -> Path:
    """Corporate_Filings/<FOLDER>/<category>/<YYYY-MM-DD>_<category>_<slug>_<id>.pdf"""
    root = Path(root or settings.filings_dir)
    name = f"{filing.date}_{filing.category}_{slug(filing)}_{filing.source_id}.pdf"
    return root / documents.folder_name(filing.symbol) / filing.category / name


# ------------------------------------------------------------------ the network edge
def discover(symbol: str, api: Callable[[str], object], today: date | None = None,
             days: int | None = None) -> list[Filing]:
    """Every filing worth keeping for one company: announcements newest first, then annual reports.
    `api(path)` returns NSE's parsed JSON (or None). The symbol is URL-encoded - M&MFIN would
    otherwise split the query string at its ampersand."""
    q = quote(symbol.upper(), safe="")
    rows = api(f"corporate-announcements?index=equities&symbol={q}") or []
    reports = (api(f"annual-reports?index=equities&symbol={q}") or {})
    reports = reports.get("data") or [] if isinstance(reports, dict) else []
    return select_filings(symbol, rows if isinstance(rows, list) else [], today, days) \
        + select_annual_reports(symbol, reports)


def nse_getter(session, timeout: int | None = None) -> Callable[[str], bytes | None]:
    """A GET that hands back the body only for HTTP 200 - an error page is never a document."""
    def get(url: str) -> bytes | None:
        r = session.get(url, timeout=timeout or settings.filings_timeout_s)
        return r.content if r.status_code == 200 else None
    return get


def repair(get: Callable[[str], bytes | None], keys: list[str] | None = None, attempts: int = 3) -> dict:
    """Re-download the filings stored incomplete (cut off mid-transfer) and swap them in whole.
    Only NSE filings are refetched - broker and Trendlyne files need a login. NSE's CDN sometimes serves
    a cut copy whose Content-Length matches the cut, so each file gets up to `attempts` tries; a file
    still cut after them keeps its old copy for the next repair."""
    import time
    from collections import Counter
    keys = [k for k in (keys or documents.verify()["incomplete"]) if k.startswith("filing:")]
    tally: Counter = Counter()
    for key in keys:
        doc = documents.get(key)
        if doc is None or not doc.source_url:
            tally["no_source"] += 1
            continue
        outcome = "failed"
        for attempt in range(attempts):
            if attempt:
                time.sleep(settings.filings_delay_s)
            try:
                data = get(doc.source_url)
            except Exception as e:
                logger.warning(f"repair {key}: {type(e).__name__}: {e}")
                data = None
            if data:
                if documents.replace_file(key, data):
                    outcome = "repaired"
                    break
                outcome = "still_incomplete"
        tally[outcome] += 1
        time.sleep(settings.filings_delay_s)
    return dict(tally)


def download(filing: Filing, get: Callable[[str], bytes | None], root: Path | None = None) -> str:
    """Fetch one filing into the library: 'have' | 'saved' | 'not_pdf' | 'failed'.
    One retry on an exception (a reset connection); a non-200 answer is not retried."""
    if documents.has(filing.doc_key):
        return "have"
    data = None
    for attempt in (1, 2):
        try:
            data = get(filing.url)
            break
        except Exception as e:
            logger.warning(f"{filing.symbol} {filing.source_id} attempt {attempt}: {type(e).__name__}: {e}")
    if data is None:
        return "failed"
    title = f"{filing.subject}: {_BOILERPLATE.sub('', filing.text)[:150]}".rstrip(": ")
    path = documents.save_file(doc_key=filing.doc_key, symbol=filing.symbol, kind="filing",
                               category=filing.category, doc_date=filing.date, title=title, source="nse",
                               source_url=filing.url, data=data, dest=filing_path(filing, root),
                               min_bytes=settings.filing_min_pdf_bytes)
    return "saved" if path else "not_pdf"


def write_manifest(symbol: str, root: Path | None = None) -> Path | None:
    """Corporate_Filings/<FOLDER>/manifest.csv - the company's filings for a person to read, newest first."""
    rows = sorted(documents.list_documents(symbol, "filing"), key=lambda d: (d.doc_date, d.doc_key), reverse=True)
    if not rows:
        return None
    folder = Path(root or settings.filings_dir) / documents.folder_name(symbol)
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / "manifest.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "category", "title", "file", "bytes", "source_url"])
        for d in rows:
            w.writerow([d.doc_date, d.category, d.title, Path(d.local_path).name, d.bytes, d.source_url])
    return out

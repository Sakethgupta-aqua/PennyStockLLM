"""Which broker reports exist, where their bytes are, and which company each one is about.

Two listings describe the corpus, and they overlap:

  metadata.jsonl             the flat Trendlyne downloads (4 Sep 2026): price at the call, target,
                             rating - but NO ticker, only Trendlyne's display name ("AU SF Bank")
  <TICKER>/manifest.csv      one per company folder (99 Midcap-100 names, 297 reports): ticker
                             included, most PDFs not yet downloaded

The same report can sit in both - AU SF Bank's note 100115 is a flat file AND report #1 of AUBANK/
- so reports are merged on Trendlyne's document id, then on a hash of the PDF bytes. Counting one
note twice would enter one broker's opinion into a consensus as two brokers agreeing.

Every downstream job - the KPI tables, the vector store, the ledger - reads the corpus through
`discover()`, so all three see the same reports with the same identity.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

RE_TRENDLYNE_ID = re.compile(r"(?:/pdf/|trendlyne_)(\d+)")
RE_LEADING_NUM = re.compile(r"-?[\d,]*\.?\d+")
RE_BROKER_SUFFIX = re.compile(r"\b(?:limited|ltd|pvt|private|llp)\b\.?", re.I)
_DATE_FORMATS = ("%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%d-%b-%Y", "%d-%m-%Y", "%d/%m/%Y")


def broker_key(name: str | None) -> str:
    """'ICICI Securities Limited' and 'ICICI Securities Ltd.' are one broker, and one vote.
    ICICI Direct stays separate: it is a different research desk with its own ratings."""
    s = RE_BROKER_SUFFIX.sub(" ", str(name or "").lower())
    return " ".join(re.sub(r"[^a-z0-9& ]+", " ", s).split())


@dataclass
class ReportRecord:
    """One broker report, whether or not its PDF has been downloaded yet."""
    report_id: str | None = None       # Trendlyne document id - the stable external key
    symbol: str | None = None          # NSE symbol; None when the company could not be resolved
    company: str | None = None
    broker: str | None = None
    report_date: str | None = None     # ISO date
    rating: str | None = None
    target: float | None = None
    price: float | None = None         # price AT the recommendation, as the listing states it
    change_vs_prev: str | None = None
    headline: str | None = None
    pdf: Path | None = None            # None = listed, not downloaded
    doc_id: str | None = None          # content hash of the PDF - the document's identity

    def listing(self) -> dict:
        """The row shape `broker_reports.merge_listing_metadata` consumes."""
        return {"stock": self.company or "", "broker": self.broker or "",
                "date": self.report_date or "", "reco_type": self.rating or "",
                "ltp": self.price or "", "target": self.target or ""}


def content_hash(path: Path) -> str:
    """16 hex characters of SHA-256 over the bytes: the same PDF under two names is one document."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def iso_date(raw) -> str | None:
    """'04 Sep 2026' / '13 February 2026' / '2026-09-04' -> '2026-09-04'; None if unreadable."""
    s = " ".join(str(raw or "").split())
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _num(raw) -> float | None:
    """'1072.00 (-0.65%)' -> 1072.0. Trendlyne appends the day's move to the price it lists."""
    m = RE_LEADING_NUM.match(str(raw or "").strip())
    try:
        return float(m.group(0).replace(",", "")) if m else None
    except ValueError:
        return None


def _text(raw) -> str | None:
    return str(raw).strip() or None if raw is not None else None


def _trendlyne_id(*candidates) -> str | None:
    for c in candidates:
        if c and (m := RE_TRENDLYNE_ID.search(str(c))):
            return m.group(1)
    return None


def resolve_company(query: str | None) -> str | None:
    """NSE symbol for a company name or ticker, or None - never a guess below the resolver's bar."""
    if not query:
        return None
    from pennystock.data.resolver import resolve_symbol
    try:
        hit = resolve_symbol(query)
    except Exception as e:                     # the universe table may not exist yet
        logger.warning(f"could not resolve {query!r}: {type(e).__name__}: {e}")
        return None
    return hit["symbol"] if hit else None


def _locate(folder: Path, stored: str | None, fallback_name: str | None = None) -> Path | None:
    """The PDF on disk, tolerating a project that has moved since the listing was written."""
    for cand in (Path(stored) if stored else None,
                 folder / Path(stored).name if stored else None,
                 folder / fallback_name if fallback_name else None):
        if cand is not None and cand.is_file():
            return cand
    return None


def _manifest_records(folder: Path, resolve: Callable[[str], str | None]
                      ) -> tuple[list[ReportRecord], dict[str, str]]:
    """The listed reports, plus which ticker each company folder belongs to."""
    out, owners = [], {}
    for path in sorted(folder.glob("*/manifest.csv")):
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                ticker = (row.get("ticker") or "").strip().upper()
                # The sheet's ticker is trusted only if the NSE universe knows it EXACTLY; a
                # near-miss resolving to some other listed company would be worse than nothing.
                symbol = ticker if ticker and resolve(ticker) == ticker else None
                if symbol:
                    owners[path.parent.name] = symbol
                out.append(ReportRecord(
                    report_id=_trendlyne_id(row.get("url"), row.get("filename")),
                    symbol=symbol, company=_text(row.get("company")),
                    broker=_text(row.get("broker")), report_date=iso_date(row.get("report_date")),
                    rating=_text(row.get("rating")), target=_num(row.get("target")),
                    change_vs_prev=_text(row.get("change_vs_prev")),
                    headline=_text(row.get("headline")),
                    pdf=_locate(path.parent, None, (row.get("filename") or "").strip() or None)))
    return out, owners


def _jsonl_records(folder: Path, resolve: Callable[[str], str | None]) -> list[ReportRecord]:
    path = folder / "metadata.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            logger.warning(f"skipping unreadable line in {path.name}")
            continue
        company = _text(row.get("stock"))
        out.append(ReportRecord(
            report_id=_text(row.get("report_id")) or _trendlyne_id(row.get("trendlyne_url"),
                                                                   row.get("local_path")),
            symbol=resolve(company) if company else None, company=company,
            broker=_text(row.get("broker")), report_date=iso_date(row.get("date")),
            rating=_text(row.get("reco_type")), target=_num(row.get("target")),
            price=_num(row.get("price_at_reco")) or _num(row.get("ltp")),
            pdf=_locate(folder, row.get("local_path"))))
    return out


def _merge(into: ReportRecord, other: ReportRecord) -> None:
    """Fill the gaps in one record from another description of the same report. The first one
    wins on conflict, which is why manifests (with tickers) are read before metadata.jsonl."""
    for f in fields(ReportRecord):
        if getattr(into, f.name) in (None, "") and getattr(other, f.name) not in (None, ""):
            setattr(into, f.name, getattr(other, f.name))


def discover(folder: str | Path = "Broker_Reports",
             resolve: Callable[[str], str | None] | None = None) -> list[ReportRecord]:
    """Every report the corpus knows about, merged, hashed, with its company resolved."""
    folder = Path(folder)
    resolve = resolve or resolve_company
    manifests, owners = _manifest_records(folder, resolve)
    candidates = manifests + _jsonl_records(folder, resolve)

    listed = {c.pdf.resolve() for c in candidates if c.pdf}
    for pdf in sorted(folder.rglob("*.pdf")):
        if pdf.resolve() in listed:
            continue
        # an unlisted PDF in a company folder belongs to that company; the rest is read at ingest
        owner = owners.get(pdf.parent.name) if pdf.parent != folder else None
        candidates.append(ReportRecord(report_id=_trendlyne_id(pdf.name), symbol=owner, pdf=pdf))

    for c in candidates:
        if c.pdf:
            c.doc_id = content_hash(c.pdf)

    merged: dict[str, ReportRecord] = {}
    for c in candidates:
        key = (f"tl:{c.report_id}" if c.report_id else f"doc:{c.doc_id}" if c.doc_id
               else f"row:{c.symbol or c.company}|{c.report_date}|{c.broker}")
        if key in merged:
            _merge(merged[key], c)
        else:
            merged[key] = c
    # two different listings can still point at identical bytes
    by_doc: dict[str, ReportRecord] = {}
    out: list[ReportRecord] = []
    for r in merged.values():
        if r.doc_id and r.doc_id in by_doc:
            _merge(by_doc[r.doc_id], r)
            continue
        if r.doc_id:
            by_doc[r.doc_id] = r
        out.append(r)
    return sorted(out, key=lambda r: (r.symbol or "~", r.report_date or "", r.broker or ""))


def company_symbols(folder: str | Path = "Broker_Reports") -> list[str]:
    """Every listed company the broker corpus covers - the company list the document library serves."""
    return sorted({r.symbol for r in discover(folder) if r.symbol})

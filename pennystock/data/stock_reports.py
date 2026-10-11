"""Trendlyne Stock Reports: Trendlyne's own scored snapshot of a company, kept as dated PDFs.

Trendlyne builds a report at the moment it is requested - its file name carries the generation time -
so each one is a point-in-time snapshot: durability/valuation/momentum scores, SWOT and checklist,
valuation zone, technicals, peers, forecaster estimates, shareholding, management pay. The library
keeps one per company per month and never overwrites an older one:

  Stock_Reports/<FOLDER>/<FOLDER>-StockReport-<YYYYMMDD>-<HHMM>.pdf

Fetching needs the user's logged-in session and spends one of the account's 70 daily stock-report
opens per report (1,000 a month). In the browser the reports arrive as downloads named by Trendlyne
(<SYM>-StockReport-<YYYYMMDD>-<HHMM>.pdf, with __TL<id> appended by our capture) and are filed from
there; `fetch()` does the same over an exported cookie, stopping at Trendlyne's daily-runout page.
Spec: docs/superpowers/specs/2026-10-04-company-documents-library-design.md
"""
from __future__ import annotations

import csv
import logging
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from urllib.parse import quote

from pennystock.config import settings
from pennystock.data import documents

logger = logging.getLogger(__name__)

RE_NAME = re.compile(r"^(?P<sym>.+?)-StockReport-(?P<d>\d{8})-(?P<t>\d{4})(?:__TL(?P<id>\d+))?\.pdf$")
RE_DISPOSITION = re.compile(r'filename="?([^";]+)')
RUNOUT = "daily-runout"
BASE = "https://trendlyne.com"


@dataclass(frozen=True)
class SnapshotName:
    symbol: str
    stamp: str              # YYYYMMDD-HHMM: when Trendlyne generated the report
    tl_id: str | None       # Trendlyne's stock id, when the download name carries it


def parse_name(name: str) -> SnapshotName | None:
    m = RE_NAME.match(name)
    return SnapshotName(m.group("sym"), f"{m.group('d')}-{m.group('t')}", m.group("id")) if m else None


def _root(root: Path | None) -> Path:
    return Path(root or settings.stock_reports_dir)


def snapshot_path(symbol: str, stamp: str, root: Path | None = None) -> Path:
    folder = documents.folder_name(symbol)
    return _root(root) / folder / f"{folder}-StockReport-{stamp}.pdf"


def snapshot_key(symbol: str, stamp: str, tl_id: str | None) -> str:
    return f"stock_report:trendlyne:{tl_id or symbol.upper()}:{stamp}"


def _source_url(tl_id: str | None) -> str | None:
    return f"{BASE}/equity/stock-report/get-pdf/{tl_id}/" if tl_id else None


def file_snapshot(data: bytes, symbol: str, stamp: str, tl_id: str | None = None,
                  root: Path | None = None) -> Path | None:
    """Store one report (a real PDF only) and catalogue it. None when the bytes are not a report."""
    return documents.save_file(doc_key=snapshot_key(symbol, stamp, tl_id), symbol=symbol, kind="stock_report",
                               doc_date=f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}", title="Trendlyne Stock Report",
                               source="trendlyne", source_url=_source_url(tl_id), data=data,
                               dest=snapshot_path(symbol, stamp, root),
                               min_bytes=settings.stock_report_min_pdf_bytes)


def has_snapshot_in_month(symbol: str, today: date) -> bool:
    month = today.strftime("%Y-%m")
    return any(d.doc_date.startswith(month) and documents.has(d.doc_key)
               for d in documents.list_documents(symbol, "stock_report"))


def unpack_bundle(data: bytes) -> list[tuple[str, str, bytes, str]]:
    """Legacy capture bundles: repeated 'SYMBOL\\tTL_ID\\tSIZE\\tSERVER_NAME\\n' + SIZE bytes."""
    out, i = [], 0
    while i < len(data):
        j = data.index(b"\n", i)
        sym, tl_id, size, name = data[i:j].decode().split("\t")
        size = int(size)
        out.append((sym, tl_id, data[j + 1:j + 1 + size], name))
        i = j + 1 + size
    return out


def register_existing(root: Path | None = None) -> int:
    """Catalogue reports already in Stock_Reports/ from its index.csv (which knows the real symbol,
    e.g. GVT&D for folder GVT_D). Returns how many rows were added."""
    index = _root(root) / "index.csv"
    if not index.exists():
        return 0
    added = 0
    for r in csv.DictReader(index.open(encoding="utf-8")):
        path = Path(settings.project_root) / r["file"]
        name = parse_name(path.name)
        if not (name and path.is_file()):
            continue
        tl_id = (r.get("trendlyne_id") or "").strip() or None
        added += documents.register(doc_key=snapshot_key(r["symbol"], name.stamp, tl_id), symbol=r["symbol"],
                                    kind="stock_report", doc_date=f"{name.stamp[:4]}-{name.stamp[4:6]}-{name.stamp[6:8]}",
                                    title="Trendlyne Stock Report", source="trendlyne",
                                    source_url=_source_url(tl_id), path=path)
    return added


def write_index(root: Path | None = None) -> Path:
    """Stock_Reports/index.csv rebuilt from the catalog: every snapshot, newest first per company."""
    out = _root(root) / "index.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    stamp_of = lambda d: d.doc_key.rsplit(":", 1)[-1]                       # noqa: E731
    rows = sorted(documents.list_documents(kind="stock_report"), key=stamp_of, reverse=True)
    rows.sort(key=lambda d: d.symbol)                  # stable: newest first within each company
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["symbol", "trendlyne_id", "generated", "file", "bytes", "sha256", "source_url"])
        for d in rows:
            tl_id = d.doc_key.split(":")[2]
            stamp = stamp_of(d)
            w.writerow([d.symbol, tl_id if tl_id.isdigit() else "", f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} "
                        f"{stamp[9:11]}:{stamp[11:13]}", d.local_path, d.bytes, d.sha256, d.source_url or ""])
    return out


def fetch(symbols: list[str], session, today: date, limit: int | None = None, delay: float = 1.0) -> dict:
    """This month's report for each company over a logged-in session (cookie). At most `limit` opens
    (default: the daily allowance); stops at the first daily-runout page; skips companies that
    already have a snapshot this month. The research-page lookup is a page view, not an open."""
    limit = limit or settings.stock_report_daily_opens
    register_existing()     # files already on disk count, even if market.db was rebuilt since
    tally = {"saved": 0, "this_month": 0, "no_id": 0, "not_pdf": 0, "daily_limit": 0}
    opens = 0
    for sym in symbols:
        if has_snapshot_in_month(sym, today):
            tally["this_month"] += 1
            continue
        if opens >= limit:
            break
        page = session.get(f"{BASE}/research-reports/stock/{quote(sym, safe='')}/", allow_redirects=True)
        m = re.search(r"/research-reports/stock/(\d+)/", str(getattr(page, "url", "")))
        if not m:
            tally["no_id"] += 1
            continue
        tl_id = m.group(1)
        r = session.get(f"{BASE}/equity/stock-report/get-pdf/{tl_id}/", allow_redirects=True)
        opens += 1
        body = r.content or b""
        if RUNOUT in str(getattr(r, "url", "")) or (body[:5] != b"%PDF-" and RUNOUT.encode() in body[:80_000]):
            tally["daily_limit"] += 1
            logger.warning("Trendlyne's daily stock-report limit is used up - stopping")
            break
        server = (RE_DISPOSITION.search(r.headers.get("Content-Disposition", "")) or [None, ""])[1]
        name = parse_name(server)
        stamp = name.stamp if name else datetime.now().strftime("%Y%m%d-%H%M")
        if file_snapshot(body, sym, stamp, tl_id):
            tally["saved"] += 1
        else:
            tally["not_pdf"] += 1
        time.sleep(delay)
    return tally

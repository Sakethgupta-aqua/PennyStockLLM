"""The documents catalog: one row per file the project keeps, whatever its kind.

Three libraries live side by side (docs/superpowers/specs/2026-10-04-company-documents-library-design.md):

  Broker_Reports/<FOLDER>/...                     sell-side research (its own pipeline, unchanged)
  Stock_Reports/<FOLDER>/<FOLDER>-StockReport-<YYYYMMDD>-<HHMM>.pdf
  Corporate_Filings/<FOLDER>/<category>/<date>_<category>_<slug>_<id>.pdf

The folders are for people; this table is for programs. It answers "what do we hold for BHEL, from
where, since when" in one query, and it is what makes every fetcher idempotent: a document whose key
is catalogued, with its file still on disk at the recorded size, is never downloaded again.

Records are inserted once per key and never rewritten - a newer Stock Report is a new key, a new row
and a new file. The one exception is repair: a row whose file has vanished is replaced when the file
is fetched again. Writes are atomic (temp file in the target folder, fsync, rename), so a crash never
leaves half a PDF under a real name.
"""
from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pennystock.config import settings

KINDS = ("broker_report", "stock_report", "filing")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_key     TEXT PRIMARY KEY,
    symbol      TEXT NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('broker_report', 'stock_report', 'filing')),
    category    TEXT,
    doc_date    TEXT NOT NULL,
    title       TEXT,
    source      TEXT NOT NULL,
    source_url  TEXT,
    local_path  TEXT NOT NULL,
    bytes       INTEGER NOT NULL,
    sha256      TEXT NOT NULL,
    fetched_at  TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_documents_symbol ON documents(symbol, kind, doc_date);
"""

_COLS = ("doc_key", "symbol", "kind", "category", "doc_date", "title", "source", "source_url",
         "local_path", "bytes", "sha256", "fetched_at")


@dataclass(frozen=True)
class Document:
    """One catalogued file."""
    doc_key: str
    symbol: str
    kind: str
    category: str | None
    doc_date: str           # valid time: the date the document is dated
    title: str | None
    source: str
    source_url: str | None
    local_path: str         # relative to the project root
    bytes: int
    sha256: str
    fetched_at: str         # transaction time


def folder_name(symbol: str) -> str:
    """The filesystem-safe folder for a symbol - the rule Broker_Reports uses (M&MFIN -> M_MFIN)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", (symbol or "UNKNOWN").strip()).strip("_") or "UNKNOWN"


def is_pdf(data: bytes, min_bytes: int) -> bool:
    """Real PDF bytes - not an HTML error page, a login page or a truncated download."""
    return data[:5] == b"%PDF-" and len(data) >= min_bytes


def is_complete_pdf(data: bytes) -> bool:
    """A PDF that arrived whole: it starts with %PDF- and its end-of-file marker is in the last 2 KB.
    Checking only the header let 19 filings cut off mid-download (e.g. 4,030,464 bytes, a round
    multiple of the transfer block) into the library on 2026-10-04."""
    return data[:5] == b"%PDF-" and b"%%EOF" in data[-2048:]


def opens_cleanly(path: Path) -> bool:
    """Whether a PDF reader can open the file at all - the second half of "complete"."""
    import pypdfium2 as pdfium
    try:
        pdfium.PdfDocument(str(path)).close()
        return True
    except Exception:
        return False


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(str(settings.market_db))
    con.executescript(_SCHEMA)
    return con


def _relative(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path(settings.project_root).resolve()))
    except ValueError:
        return str(path)


def _absolute(local_path: str) -> Path:
    p = Path(local_path)
    return p if p.is_absolute() else Path(settings.project_root) / p


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


def get(doc_key: str) -> Document | None:
    with _connect() as con:
        row = con.execute(f"SELECT {', '.join(_COLS)} FROM documents WHERE doc_key = ?",
                          (doc_key,)).fetchone()
    return Document(*row) if row else None


def has(doc_key: str) -> bool:
    """Catalogued AND its file is still on disk at the recorded size: safe to skip a download."""
    doc = get(doc_key)
    if doc is None:
        return False
    p = _absolute(doc.local_path)
    return p.is_file() and p.stat().st_size == doc.bytes


def atomic_write(dest: Path, data: bytes) -> None:
    """Temp file beside the destination, fsync, rename: the file appears whole or not at all."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".part-", suffix=dest.suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _insert(con: sqlite3.Connection, doc: Document) -> bool:
    before = con.total_changes
    con.execute(f"INSERT INTO documents ({', '.join(_COLS)}) VALUES ({', '.join('?' * len(_COLS))}) "
                "ON CONFLICT(doc_key) DO NOTHING", tuple(getattr(doc, c) for c in _COLS))
    return con.total_changes > before


def save_file(*, doc_key: str, symbol: str, kind: str, doc_date: str, source: str, data: bytes,
              dest: Path, category: str | None = None, title: str | None = None,
              source_url: str | None = None, min_bytes: int = 1) -> Path | None:
    """Write `data` to `dest` atomically and catalogue it.

    Returns the stored path; the path already held when the key is catalogued with its file on disk
    (nothing is rewritten); or None when the bytes are not a real PDF (nothing is written)."""
    _check_kind(kind)
    if has(doc_key):
        return _absolute(get(doc_key).local_path)
    if not (is_pdf(data, min_bytes) and is_complete_pdf(data)):
        return None
    dest = Path(dest)
    atomic_write(dest, data)
    doc = Document(doc_key=doc_key, symbol=symbol.upper(), kind=kind, category=category,
                   doc_date=doc_date, title=title, source=source, source_url=source_url,
                   local_path=_relative(dest), bytes=len(data),
                   sha256=hashlib.sha256(data).hexdigest(), fetched_at=_now())
    with _connect() as con:
        con.execute("DELETE FROM documents WHERE doc_key = ?", (doc_key,))   # a row whose file vanished
        _insert(con, doc)
    return dest


def register(*, doc_key: str, symbol: str, kind: str, doc_date: str, source: str, path: Path,
             category: str | None = None, title: str | None = None,
             source_url: str | None = None) -> bool:
    """Catalogue a file that is already in place (broker PDFs, earlier downloads). True if added."""
    _check_kind(kind)
    data = Path(path).read_bytes()
    doc = Document(doc_key=doc_key, symbol=symbol.upper(), kind=kind, category=category,
                   doc_date=doc_date, title=title, source=source, source_url=source_url,
                   local_path=_relative(Path(path)), bytes=len(data),
                   sha256=hashlib.sha256(data).hexdigest(), fetched_at=_now())
    with _connect() as con:
        return _insert(con, doc)


def list_documents(symbol: str | None = None, kind: str | None = None) -> list[Document]:
    sql, where, args = f"SELECT {', '.join(_COLS)} FROM documents", [], []
    if symbol:
        where.append("symbol = ?")
        args.append(symbol.upper())
    if kind:
        where.append("kind = ?")
        args.append(kind)
    if where:
        sql += " WHERE " + " AND ".join(where)
    with _connect() as con:
        return [Document(*r) for r in con.execute(sql + " ORDER BY symbol, kind, doc_date, doc_key", args)]


def counts() -> dict[str, dict]:
    """{kind: {files, companies, bytes}} - what the library holds."""
    with _connect() as con:
        rows = con.execute("SELECT kind, COUNT(*), COUNT(DISTINCT symbol), COALESCE(SUM(bytes), 0) "
                           "FROM documents GROUP BY kind").fetchall()
    return {k: {"files": n, "companies": c, "bytes": b} for k, n, c, b in rows}


def verify() -> dict[str, list[str]]:
    """Re-hash every catalogued file: which are missing, which changed since they were stored, and which
    are incomplete (no end-of-file marker, or no reader can open them)."""
    missing, changed, incomplete = [], [], []
    for d in list_documents():
        p = _absolute(d.local_path)
        if not p.is_file():
            missing.append(d.doc_key)
            continue
        data = p.read_bytes()
        if hashlib.sha256(data).hexdigest() != d.sha256:
            changed.append(d.doc_key)
        if not is_complete_pdf(data) or not opens_cleanly(p):
            incomplete.append(d.doc_key)
    return {"missing": missing, "changed": changed, "incomplete": incomplete}


def replace_file(doc_key: str, data: bytes) -> Path | None:
    """Swap a catalogued file for a complete copy (the repair path): written atomically over the same
    path, with its size, fingerprint and fetch time updated. None when the new bytes are not complete."""
    doc = get(doc_key)
    if doc is None or not is_complete_pdf(data):
        return None
    path = _absolute(doc.local_path)
    atomic_write(path, data)
    with _connect() as con:
        con.execute("UPDATE documents SET bytes = ?, sha256 = ?, fetched_at = ? WHERE doc_key = ?",
                    (len(data), hashlib.sha256(data).hexdigest(), _now(), doc_key))
    return path

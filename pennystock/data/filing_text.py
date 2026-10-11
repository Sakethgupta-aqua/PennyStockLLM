"""Filing text: every NSE filing's pages extracted once into filings.db, keyword-searchable by company.

The storage of record for filings search (docs/superpowers/specs/2026-10-05-filings-search-design.md):

  filing_texts   one row per unique file content - 294 of 6,014 filings are the same PDF filed twice
  filing_docs    one row per catalogued filing, pointing at its content
  filing_pages   one row per page: text, whether it was OCR'd, its kind and (annual reports) section
  pages_fts      SQLite FTS5 over every page (porter stemming), kept in sync by triggers

Measured on the real library (2026-10-05): pypdfium2 reads a page in ~2 ms against 30-100 ms for
pdfplumber, so all ~152,600 pages take minutes, not hours; 0-6% of pages have no text layer.

Words, not numbers: a page whose tokens are mostly figures is a `table` page - searchable by keyword,
never embedded as prose. Annual reports keep only their key sections as prose.
"""
from __future__ import annotations

import logging
import re
import sqlite3
import unicodedata
from collections import Counter
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

from pennystock.config import settings

logger = logging.getLogger(__name__)

EXTRACTOR_VERSION = 1
KEY_SECTIONS = ("mdna", "board_report", "risk", "auditor", "contingent", "related_party")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS filing_texts (
    content_sha       TEXT PRIMARY KEY,
    pages             INTEGER NOT NULL,
    extractor_version INTEGER NOT NULL,
    extracted_at      TEXT NOT NULL,
    passages_version  INTEGER NOT NULL DEFAULT 0
) STRICT;
CREATE TABLE IF NOT EXISTS filing_docs (
    doc_key       TEXT PRIMARY KEY,
    content_sha   TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    doc_type      TEXT NOT NULL,
    doc_date      TEXT NOT NULL,
    available_at  TEXT NOT NULL,
    fiscal_period TEXT,
    title         TEXT,
    local_path    TEXT NOT NULL
) STRICT;
CREATE INDEX IF NOT EXISTS idx_filing_docs_sha ON filing_docs(content_sha);
CREATE INDEX IF NOT EXISTS idx_filing_docs_symbol ON filing_docs(symbol, available_at);
CREATE TABLE IF NOT EXISTS filing_pages (
    content_sha TEXT NOT NULL,
    page        INTEGER NOT NULL,
    text        TEXT NOT NULL,
    chars       INTEGER NOT NULL,
    ocr         INTEGER NOT NULL,
    kind        TEXT NOT NULL,
    section     TEXT,
    UNIQUE (content_sha, page)
) STRICT;
CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
    text, content='filing_pages', content_rowid='rowid', tokenize='porter unicode61');
CREATE TRIGGER IF NOT EXISTS filing_pages_ai AFTER INSERT ON filing_pages BEGIN
    INSERT INTO pages_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS filing_pages_ad AFTER DELETE ON filing_pages BEGIN
    INSERT INTO pages_fts(pages_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS filing_pages_au AFTER UPDATE ON filing_pages BEGIN
    INSERT INTO pages_fts(pages_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO pages_fts(rowid, text) VALUES (new.rowid, new.text);
END;
"""


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(str(settings.filings_db), timeout=settings.filings_db_timeout_s)
    # write-ahead log: a long read (a status scan, a search) never holds off the indexer's writes
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(_SCHEMA)
    return con


# ------------------------------------------------------------------ text rules (pure)
_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
                         "“": '"', "”": '"', "„": '"', "–": "-", "—": "-"})


def normalize(text: str) -> str:
    """Ligatures (ﬁ -> fi) and curly quotes folded - annual reports write "Board’s Report" - and runs of
    spaces collapsed, line breaks kept (they carry headings)."""
    text = unicodedata.normalize("NFKC", text or "").translate(_QUOTES)
    return "\n".join(" ".join(line.split()) for line in text.splitlines()).strip()


_NUMERIC = re.compile(r"^[(\-]?[₹$]?\d[\d,]*(\.\d+)?%?\)?$")


def page_kind(text: str, doc_type: str, section: str | None) -> str:
    """`table` when figures dominate (numeric tokens above `filings_table_numeric_share`), `boilerplate` for
    annual-report pages outside the key sections, else `prose`."""
    tokens = text.split()
    numeric = sum(1 for t in tokens if _NUMERIC.match(t))
    if tokens and numeric / len(tokens) > settings.filings_table_numeric_share:
        return "table"
    if doc_type == "annual_report" and section not in KEY_SECTIONS:
        return "boilerplate"
    return "prose"


# Calibrated on AU Small Finance Bank's 364-page report: a heading counts only where a page OPENS with it
# (the first ~100 characters) - "risk management" mentioned mid-sentence had hijacked ten MD&A pages -
# except notes to accounts, which start mid-page.
_AR_HEADINGS = [
    ("mdna", 100, re.compile(r"management\s*'?s?\s+discussion\s+(and|&)\s+analysis", re.I)),
    ("board_report", 100, re.compile(r"\b(board'?s|directors'?)\s+report", re.I)),
    # many Board's Reports carry no heading text, only the opening salutation
    ("board_report", 200, re.compile(r"dear\s+(members|shareholders)\b.{0,80}\b(board|directors)\b.{0,60}"
                                     r"\b(pleased|present)", re.I | re.S)),
    ("auditor", 100, re.compile(r"independent\s+auditor'?s?\s+report|key\s+audit\s+matters?", re.I)),
    ("risk", 100, re.compile(r"\brisk\s+management\b", re.I)),
    ("contingent", 300, re.compile(r"contingent\s+liabilit", re.I)),
    ("related_party", 300, re.compile(r"related\s+part(y|ies)\s+(disclosures?|transactions?)", re.I)),
]
# other major parts end a key section - but only when a page OPENS with them (60 characters)
_AR_STOP = re.compile(r"^\W*(\d+\s+)?((report\s+on\s+)?corporate\s+governance"
                      r"|notice\s+of\s+(the\s+)?(\d+\w*\s+)?annual\s+general\s+meeting"
                      r"|business\s+responsibility|sustainability\s+report"
                      r"|balance\s+sheet\s+as\s+at|statement\s+of\s+profit\s+and\s+loss|cash\s+flow\s+statement"
                      r"|(standalone|consolidated)\s+financial\s+statements|secretarial\s+audit"
                      r"|form\s+(no\.?\s*)?aoc"
                      r"|annexure|corporate\s+social\s+responsibility|corporate\s+information)", re.I)
# a section runs on from its heading; notes to accounts (contingent, related party) are short
_SPAN = {"mdna": 40, "board_report": 40, "auditor": 15, "risk": 6, "contingent": 2, "related_party": 2}


def ar_sections(pages: list[str]) -> list[str | None]:
    """Which key section each annual-report page belongs to: a page that opens with a key heading starts
    that section; it runs on until a page opens with another major heading, or its span ends."""
    out: list[str | None] = []
    current, left = None, 0
    for text in pages:
        top = normalize(text)
        hit = next((name for name, reach, pat in _AR_HEADINGS if pat.search(top[:reach])), None)
        if hit:
            current, left = hit, _SPAN[hit] - 1
            out.append(hit)
        elif _AR_STOP.search(top[:60]):
            current, left = None, 0
            out.append(None)
        elif current and left > 0:
            left -= 1
            out.append(current)
        else:
            current = None
            out.append(None)
    return out


_QUARTER_ENDS = ((3, 31, 4), (6, 30, 1), (9, 30, 2), (12, 31, 3))
_QUARTERLY = ("results", "earnings_call", "investor_presentation")


def fiscal_period(doc_type: str, doc_date: str) -> str | None:
    """India's April-March year: a quarterly filing within ~75 days of a quarter end belongs to that
    quarter (Q1 = Apr-Jun); an annual report covers the year that ended the March before it."""
    try:
        d = date.fromisoformat(doc_date[:10])
    except ValueError:
        return None
    if doc_type == "annual_report":
        return f"FY{(d.year if d.month >= 4 else d.year - 1) % 100:02d}"
    if doc_type not in _QUARTERLY:
        return None
    ends = [(date(y, m, dd), q) for y in (d.year - 1, d.year) for m, dd, q in _QUARTER_ENDS
            if date(y, m, dd) <= d]
    qe, q = max(ends)
    if (d - qe).days > 75:
        return None
    fy = qe.year if q == 4 else qe.year + 1
    return f"Q{q} FY{fy % 100:02d}"


_STAMP = re.compile(r"_(\d{2})(\d{2})(\d{4})(\d{2})(\d{2})(\d{2})(?=[_.])")


def available_at(source_url: str | None, doc_date: str) -> str:
    """When the market saw it: the exchange broadcast stamp NSE writes into every file name
    (_DDMMYYYYHHMMSS_), else the filing date at midnight."""
    for m in _STAMP.finditer(source_url or ""):
        dd, mm, yyyy, hh, mi, ss = (int(x) for x in m.groups())
        try:
            return datetime(yyyy, mm, dd, hh, mi, ss).strftime("%Y-%m-%dT%H:%M")
        except ValueError:
            continue
    return f"{doc_date[:10]}T00:00"


_STOP = set("""a an the is are was were be been being of to in on for and or not no nor what which who whom
how why when where did do does done about with by from at as it its this that these those any all has have had
will would can could should may might there their they them we our you your i me my so if than then into over
under also more most such very just tell give show please""".split())


def fts_query(question: str) -> str | None:
    """A natural question as a safe FTS5 query: its content words, each quoted (no operator, prefix or
    column syntax can slip through), joined by OR so pages matching more of them rank higher."""
    seen, words = set(), []
    for w in re.findall(r"[a-z0-9]+", (question or "").lower()):
        if len(w) >= 2 and w not in _STOP and w not in seen:
            seen.add(w)
            words.append(f'"{w}"')
    return " OR ".join(words) or None


# ------------------------------------------------------------------ extraction
def extract_pages(path: Path, doc_type: str, ocr=None) -> list[tuple[str, bool]]:
    """(text, was_ocrd) per page. pypdfium2 reads the text layer; a page with less than
    `filings_ocr_min_chars` goes to `ocr(path, page_index)` when one is given - never for investor
    presentations, whose text-less pages are chart images."""
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(path))
    try:
        out = []
        for i in range(len(doc)):
            page = doc[i]
            text = normalize(page.get_textpage().get_text_range())
            flag = False
            if ocr and len(text) < settings.filings_ocr_min_chars and doc_type != "investor_presentation":
                try:
                    scanned = normalize(ocr(path, i))
                    if len(scanned) > len(text):
                        text, flag = scanned, True
                except Exception as e:                  # OCR is a bonus, never a blocker
                    logger.warning(f"OCR failed on {path.name} p.{i + 1}: {type(e).__name__}: {e}")
            out.append((text, flag))
        return out
    finally:
        doc.close()


def index_texts(docs=None, reextract: bool = False, extract=None, ocr=None) -> dict:
    """Catalogue every filing in filings.db and extract each unique file once. A file already extracted
    at EXTRACTOR_VERSION is skipped (a re-run is a no-op); a file that will not open is counted."""
    from pennystock.data import documents
    docs = docs if docs is not None else documents.list_documents(kind="filing")
    tally: Counter = Counter(dict.fromkeys(("extracted", "current", "duplicates", "missing", "unreadable",
                                            "ocr_pages"), 0))
    with _connect() as con:
        done = dict(con.execute("SELECT content_sha, extractor_version FROM filing_texts"))
    seen: set[str] = set()
    for d in docs:
        sha, doc_type = d.sha256, d.category or "material_event"
        with _connect() as con:
            con.execute("INSERT OR REPLACE INTO filing_docs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (d.doc_key, sha, d.symbol.upper(), doc_type, d.doc_date,
                         available_at(d.source_url, d.doc_date), fiscal_period(doc_type, d.doc_date), d.title,
                         d.local_path))
        if sha in seen:
            tally["duplicates"] += 1
            continue
        seen.add(sha)
        if not reextract and done.get(sha) == EXTRACTOR_VERSION:
            tally["current"] += 1
            continue
        path = Path(settings.project_root) / d.local_path
        if extract is None and not path.is_file():
            tally["missing"] += 1
            continue
        try:
            pages = extract(path, doc_type) if extract else extract_pages(path, doc_type, ocr)
        except Exception as e:
            logger.warning(f"could not read {d.local_path}: {type(e).__name__}: {e}")
            tally["unreadable"] += 1
            continue
        sections = ar_sections([t for t, _ in pages]) if doc_type == "annual_report" else [None] * len(pages)
        rows = [(sha, i + 1, t, len(t), int(o), page_kind(t, doc_type, s), s)
                for i, ((t, o), s) in enumerate(zip(pages, sections))]
        with _connect() as con:
            con.execute("DELETE FROM filing_pages WHERE content_sha = ?", (sha,))
            con.executemany("INSERT INTO filing_pages VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
            con.execute("INSERT OR REPLACE INTO filing_texts VALUES (?, ?, ?, ?, 0)",
                        (sha, len(rows), EXTRACTOR_VERSION, datetime.now().isoformat(timespec="seconds")))
        tally["extracted"] += 1
        tally["ocr_pages"] += sum(1 for r in rows if r[4])
    return dict(tally)


# ------------------------------------------------------------------ reading
@lru_cache(maxsize=1)
def _company_names() -> dict[str, str]:
    try:
        from pennystock.data.metrics import load_metrics
        return {m.symbol: m.company for m in load_metrics() if m.company}
    except Exception:                               # a missing metrics cache only costs the name
        return {}


def doc_meta(content_sha: str) -> dict:
    """The filing a content belongs to (the earliest one the market saw), with the company's name."""
    with _connect() as con:
        con.row_factory = sqlite3.Row
        r = con.execute("SELECT * FROM filing_docs WHERE content_sha = ? "
                        "ORDER BY available_at, doc_key LIMIT 1", (content_sha,)).fetchone()
    if r is None:
        return {}
    meta = dict(r)
    meta["company"] = _company_names().get(meta["symbol"], meta["symbol"])
    return meta


def pages_of(content_sha: str) -> list[dict]:
    with _connect() as con:
        con.row_factory = sqlite3.Row
        return [dict(r) for r in con.execute("SELECT page, text, kind, section, ocr FROM filing_pages "
                                             "WHERE content_sha = ? ORDER BY page", (content_sha,))]


def keyword_pages(symbol: str, question: str, doc_types: list[str] | None = None, as_of: str | None = None,
                  limit: int = 50) -> list[dict]:
    """The best-matching pages of ONE company's filings by BM25, optionally limited to document types
    and to what the market had seen by `as_of` ("YYYY-MM-DDTHH:MM"). One hit per page of content."""
    query = fts_query(question)
    if not query:
        return []
    sql = ("SELECT p.content_sha, p.page, p.kind, p.section, p.text, "
           "snippet(pages_fts, 0, '[', ']', ' ... ', 24) AS snippet, bm25(pages_fts) AS score, "
           "d.doc_key, d.symbol, d.doc_type, d.doc_date, d.available_at, d.fiscal_period, d.local_path "
           "FROM pages_fts JOIN filing_pages p ON p.rowid = pages_fts.rowid "
           "JOIN filing_docs d ON d.content_sha = p.content_sha "
           "WHERE pages_fts MATCH ? AND d.symbol = ?")
    args: list = [query, symbol.upper()]
    if doc_types:
        sql += f" AND d.doc_type IN ({', '.join('?' * len(doc_types))})"
        args += list(doc_types)
    if as_of:
        sql += " AND d.available_at <= ?"
        args.append(as_of)
    sql += " ORDER BY score, d.available_at LIMIT ?"
    args.append(limit * 3)
    with _connect() as con:
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute(sql, args)]
    out, seen = [], set()
    for r in rows:
        key = (r["content_sha"], r["page"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out[:limit]


def indexed_count(symbol: str) -> int:
    """How many distinct filings of a company have their text extracted."""
    with _connect() as con:
        return con.execute("SELECT COUNT(DISTINCT d.content_sha) FROM filing_docs d JOIN filing_texts t "
                           "ON t.content_sha = d.content_sha WHERE d.symbol = ?",
                           (symbol.upper(),)).fetchone()[0]


def status() -> dict:
    with _connect() as con:
        docs = con.execute("SELECT COUNT(*), COUNT(DISTINCT symbol) FROM filing_docs").fetchone()
        texts = con.execute("SELECT COUNT(*), COALESCE(SUM(passages_version > 0), 0) "
                            "FROM filing_texts").fetchone()
        pages = dict(con.execute("SELECT kind, COUNT(*) FROM filing_pages GROUP BY kind"))
        ocr = con.execute("SELECT COUNT(*) FROM filing_pages WHERE ocr = 1").fetchone()[0]
        old = con.execute("SELECT COUNT(*) FROM filing_texts WHERE extractor_version != ?",
                          (EXTRACTOR_VERSION,)).fetchone()[0]
    return {"filings": docs[0], "companies": docs[1], "files": texts[0], "files_embedded": texts[1],
            "pages": sum(pages.values()), "pages_by_kind": pages, "ocr_pages": ocr, "older_extractor": old}

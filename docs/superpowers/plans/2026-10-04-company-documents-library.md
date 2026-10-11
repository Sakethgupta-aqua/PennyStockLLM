# Company Documents Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One catalog for every document the project keeps, an NSE corporate-filings library, and a
production Stock Report filer/fetcher — all idempotent, atomic and quota-safe.

**Architecture:** `pennystock/data/documents.py` owns the `documents` table and atomic file writes;
`filings.py` (NSE) and `stock_reports.py` (Trendlyne) decide WHAT to keep and WHERE, and save through
`documents.save_file`. Thin CLIs in `scripts/` do the I/O loops. Broker_Reports is only registered,
never moved.

**Tech Stack:** Python 3, SQLite (STRICT tables in `market.db`), `requests` (existing NSE session
recipe in `data/nse_client.py`), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-04-company-documents-library-design.md`

## Global Constraints

- Never work around a Trendlyne quota: stop at `/member/daily-runout/`; stock reports ≤ 70 opens per run; never use `post/pdf` or `cached_pdf` links for research reports.
- Keep a file only if it starts with `%PDF-` and meets the minimum size (filings 5,000 B; stock reports 50,000 B).
- Writes are atomic (temp file in the target folder, fsync, `os.replace`).
- Catalog rows are inserted once per `doc_key` (`ON CONFLICT DO NOTHING`); nothing is overwritten.
- `Stock_Reports/` and `Corporate_Filings/` stay git-ignored.
- Layout: `Stock_Reports/<FOLDER>/<FOLDER>-StockReport-<YYYYMMDD>-<HHMM>.pdf`; `Corporate_Filings/<FOLDER>/<category>/<YYYY-MM-DD>_<category>_<slug>_<id>.pdf`; `FOLDER` uses the Broker_Reports rule (`M&MFIN` → `M_MFIN`).
- Filing categories: `results | investor_presentation | earnings_call | annual_report | credit_rating | material_event`; window 730 days; annual reports: newest two.
- NSE: one request a second; tests never touch the network.
- Commits: none unless the user asks (repository rule); each task ends at a green full suite instead.

## Review Focus

- An NSE attachment that is not a PDF (`.zip`, `-`, empty) → skipped, never saved. (Task 2)
- NSE answering an HTML error page or a non-200 for a PDF URL → nothing saved under a `.pdf` name. (Task 3)
- Symbols containing `&` (`M&MFIN`, `GVT&D`) → URL-encoded in NSE queries, safe folder names. (Tasks 1–3)
- An original and a revised filing on the same day with the same subject → two distinct files. (Task 2)
- A crash in the middle of a write → no partial file; the next run completes it. (Task 1)

---

### Task 1: The documents catalog

**Files:**
- Modify: `pennystock/config.py` (new settings block before `settings = Settings()`)
- Create: `pennystock/data/documents.py`
- Test: `tests/test_documents.py`

**Interfaces:**
- Produces: `KINDS`, `Document` (frozen dataclass, fields `doc_key, symbol, kind, category, doc_date, title, source, source_url, local_path, bytes, sha256, fetched_at`), `folder_name(symbol) -> str`, `is_pdf(data, min_bytes) -> bool`, `get(doc_key) -> Document | None`, `has(doc_key) -> bool`, `atomic_write(dest, data) -> None`, `save_file(*, doc_key, symbol, kind, doc_date, source, data, dest, category=None, title=None, source_url=None, min_bytes=1) -> Path | None`, `register(*, doc_key, symbol, kind, doc_date, source, path, category=None, title=None, source_url=None) -> bool`, `list_documents(symbol=None, kind=None) -> list[Document]`, `counts() -> dict[str, dict]`, `verify() -> dict[str, list[str]]`.
- Settings: `stock_reports_dir`, `filings_dir`, `filings_days=730`, `filings_delay_s=1.0`, `filings_timeout_s=60`, `filing_min_pdf_bytes=5000`, `stock_report_min_pdf_bytes=50000`, `stock_report_daily_opens=70`.

- [ ] **Step 1: Write the failing tests** — `tests/test_documents.py`

```python
"""The documents catalog: one row per kept file, idempotent fetches, never half a PDF."""
import hashlib

import pytest

from pennystock.data import documents as docs

PDF = b"%PDF-1.7\n" + b"x" * 6000


@pytest.fixture(autouse=True)
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(docs.settings, "market_db", tmp_path / "test.db")
    monkeypatch.setattr(docs.settings, "project_root", tmp_path)
    return tmp_path


def save(root, key="filing:nse:1", data=PDF, name="a.pdf", **kw):
    args = dict(doc_key=key, symbol="aubank", kind="filing", doc_date="2026-09-24", source="nse",
                data=data, dest=root / "Corporate_Filings" / "AUBANK" / name,
                category="investor_presentation", min_bytes=5000)
    return docs.save_file(**(args | kw))


def test_a_saved_file_is_catalogued_with_its_fingerprint(project):
    path = save(project)
    d = docs.get("filing:nse:1")
    assert path.read_bytes() == PDF
    assert (d.symbol, d.local_path, d.bytes) == ("AUBANK", "Corporate_Filings/AUBANK/a.pdf", len(PDF))
    assert d.sha256 == hashlib.sha256(PDF).hexdigest()


def test_saving_the_same_key_again_never_rewrites_the_file(project):
    first = save(project)
    second = save(project, data=PDF + b"changed", name="b.pdf")
    assert second == first and first.read_bytes() == PDF
    assert not (project / "Corporate_Filings" / "AUBANK" / "b.pdf").exists()
    assert len(docs.list_documents()) == 1


def test_html_or_truncated_bytes_are_never_stored(project):
    assert save(project, data=b"<html><title>Trendlyne login</title></html>" * 200) is None
    assert save(project, key="filing:nse:2", data=b"%PDF-1.7 tiny") is None
    assert docs.list_documents() == []
    assert not list(project.rglob("*.pdf"))


def test_a_crash_mid_write_leaves_no_partial_file_and_a_rerun_completes(project, monkeypatch):
    real_replace = docs.os.replace

    def crash(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(docs.os, "replace", crash)
    with pytest.raises(OSError):
        save(project)
    folder = project / "Corporate_Filings" / "AUBANK"
    assert list(folder.iterdir()) == []                 # no a.pdf, no .part- leftovers
    assert docs.get("filing:nse:1") is None
    monkeypatch.setattr(docs.os, "replace", real_replace)
    assert save(project).read_bytes() == PDF


def test_a_file_that_vanished_is_fetched_again(project):
    save(project).unlink()
    assert not docs.has("filing:nse:1")
    assert save(project).read_bytes() == PDF and docs.has("filing:nse:1")


def test_an_unknown_kind_is_refused_before_anything_is_written(project):
    with pytest.raises(ValueError):
        save(project, kind="newsletter")
    assert not list(project.rglob("*.pdf"))


def test_registering_a_file_already_in_place_is_idempotent(project):
    f = project / "Broker_Reports" / "AUBANK" / "x.pdf"
    f.parent.mkdir(parents=True)
    f.write_bytes(PDF)
    kw = dict(doc_key="broker_report:trendlyne:100115", symbol="AUBANK", kind="broker_report",
              doc_date="2026-09-04", source="trendlyne", path=f)
    assert docs.register(**kw) is True
    assert docs.register(**kw) is False
    assert docs.counts() == {"broker_report": {"files": 1, "companies": 1, "bytes": len(PDF)}}


def test_verify_reports_missing_and_changed_files(project):
    a = save(project, key="k1", name="a.pdf")
    b = save(project, key="k2", name="b.pdf")
    a.unlink()
    b.write_bytes(PDF + b"tampered")
    assert docs.verify() == {"missing": ["k1"], "changed": ["k2"]}


@pytest.mark.parametrize("symbol, folder", [("M&MFIN", "M_MFIN"), ("GVT&D", "GVT_D"), ("360ONE", "360ONE")])
def test_folder_names_match_the_broker_reports_rule(symbol, folder):
    assert docs.folder_name(symbol) == folder
```

- [ ] **Step 2: Run to verify they fail**

Run: `/opt/anaconda3/bin/python -m pytest tests/test_documents.py -q`
Expected: collection error `ModuleNotFoundError: pennystock.data.documents` (then, once the module exists with stubs, assertion failures).

- [ ] **Step 3: Add the settings** — `pennystock/config.py`, after `broker_chunks_per_report`:

```python
    # --- company documents library (docs/superpowers/specs/2026-10-04-company-documents-library-design.md) ---
    stock_reports_dir: Path = _ROOT / "Stock_Reports"
    filings_dir: Path = _ROOT / "Corporate_Filings"
    filings_days: int = 730                 # key filings from the last two years
    filings_delay_s: float = 1.0            # NSE is someone else's server: one request a second
    filings_timeout_s: int = 60
    # NSE's one-page intimations are ~200 KB and a real PDF is never under a few KB; below this
    # the body is an error page or a truncated download
    filing_min_pdf_bytes: int = 5_000
    stock_report_min_pdf_bytes: int = 50_000    # a Trendlyne Stock Report is 1.5-2.6 MB
    stock_report_daily_opens: int = 70          # Trendlyne's daily stock-report allowance
```

- [ ] **Step 4: Implement** — `pennystock/data/documents.py` (full module as written in the repository
  after this step; key behaviours: `KINDS` check before any write; `has()` = row exists AND file
  exists at the recorded size; `save_file` returns the existing path for a held key, `None` for
  non-PDF bytes, else writes atomically, deletes a stale row whose file vanished, inserts with
  `ON CONFLICT DO NOTHING`; `register` hashes a file in place and returns whether a row was added;
  `verify` re-hashes every file).

```python
def atomic_write(dest: Path, data: bytes) -> None:
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
```

- [ ] **Step 5: Run to verify they pass**

Run: `/opt/anaconda3/bin/python -m pytest tests/test_documents.py -q` → all pass; then the full suite `-q` → no regressions.

---

### Task 2: Choosing which NSE filings to keep (pure logic)

**Files:**
- Create: `pennystock/data/filings.py` (selection half)
- Test: `tests/test_filings.py` (selection half)

**Interfaces:**
- Consumes: `documents.folder_name`, `events.classify`, `transcripts.TRANSCRIPT_HINT`, `settings.filings_days`, `settings.filings_dir`.
- Produces: `CATEGORIES`, `Filing` (frozen: `symbol, source_id, date, category, subject, text, url`; property `doc_key`), `classify_filing(desc, text, file_name) -> str | None`, `select_filings(symbol, rows, today=None, days=None) -> list[Filing]` (newest first), `select_annual_reports(symbol, rows, keep=2) -> list[Filing]`, `slug(filing, words=6) -> str`, `filing_path(filing, root=None) -> Path`.

- [ ] **Step 1: Write the failing tests** (real NSE subjects measured 2026-10-04)

```python
import pytest
from datetime import date
from pathlib import Path
from pennystock.data import filings as fl

CASES = [
    ("Investor Presentation", "AU Small Finance Bank Limited has informed the Exchange about Investor Presentation",
     "AUBANK_24092026230354_Revised_Investor_Presentation_SE.pdf", "investor_presentation"),
    ("Outcome of Board Meeting", "AU Small Finance Bank Limited has submitted to the Exchange, the financial results for the period ended Jun 30, 2026.",
     "AUBANK_25072026125625_BM_Outcome_July_25.pdf", "results"),
    ("Outcome of Board Meeting", "AU Small Finance Bank Limited has informed the Exchange regarding Outcome of Board Meeting held on September 28, 2026.",
     "SUMIT1_28092026212616_HIA_Re-appointment.pdf", None),
    ("Financial Result Updates", " AU Small Finance Bank Limited has submitted to the Exchange, the financial results for the period ended December 31, 2024.",
     "Results_24012025144132.pdf", "results"),
    ("Integrated Filing- Financial", "Unaudited Financial Results for & upto the quarter ended 31st December, 2024",
     "BHEL_28012025184917_31122024quickresults.pdf", "results"),
    ("Press Release", 'AU Small Finance Bank Limited has informed the Exchange regarding a press release dated July 25, 2026, titled "Press Release on Unaudited Financial Results of AU Small Fi',
     "AUBANK_25072026133046_Press_release_submission.pdf", "results"),
    ("Press Release", 'AU Small Finance Bank Limited has informed the Exchange regarding a press release dated May 19, 2026, titled "AU Small Finance Bank adopts Intellect s Purple Fabric to ac',
     "SUMIT1_19052026104108_Submission_PR_Purple_Fabric.pdf", None),
    ("Credit Rating", "AU Small Finance Bank Limited has informed the Exchange about Credit Rating",
     "SUMIT1_01102026201906_ICRA_Credit_Rating.pdf", "credit_rating"),
    ("Credit Rating- New", "IDFC First Bank Limited has informed the Exchange about Credit Rating- New",
     "IDFCFIRSTB_01102026165613_IDFCFIRSTB_IndRa_CARE_2026.pdf", "credit_rating"),
    ("Analysts/Institutional Investor Meet/Con. Call Updates", "AU Small Finance Bank Limited has informed the Exchange about Schedule of meet",
     "AUBANK_21092026183759_Investor_Meet_Intimation.pdf", None),
    ("Analysts/Institutional Investor Meet/Con. Call Updates", "AU Small Finance Bank Limited has informed the Exchange about Transcript of the Earnings Call for Q1 FY27",
     "AUBANK_30072026181500_Earnings_Call_Transcript_Q1FY27.pdf", "earnings_call"),
    ("Change in Auditors", "Bharat Heavy Electricals Limited has informed the Exchange regarding Change in Auditors of the company.",
     "BHEL_08092026154443_letter_to_SE_for_auditor.pdf", "material_event"),
    ("Frauds/Default by employees", "IDFC First Bank Limited has informed the Exchange about Frauds/Default by employees",
     "IDFCFIRSTB_22022026000738_IDFCFIRSTB_Reg30.pdf", "material_event"),
    ("Final  forensic  audit  report", "Please find enclosed Disclosure under Regulation 30 ... - Report on Chandigarh incident",
     "IDFCFIRSTB_05062026220239_Outcome_KPMG_Forensic_Review_Report.pdf", "material_event"),
    ("Resignation", "IDFC First Bank Limited has informed the Exchange regarding Resignation of Mr. Ashish Singh.",
     "IDFCFIRSTB_11082026185509_Change_in_SMP.pdf", "material_event"),
    ("Trading Window", "AU Small Finance Bank Limited has informed the Exchange regarding the Trading Window closure pursuant to SEBI (Prohibition of Insider Trading) Regulations, 2015",
     "SUMIT1_30092026173728_Q2_Window_closure_Intimation_signed.pdf", None),
    ("Copy of Newspaper Publication", "AU Small Finance Bank Limited has informed the Exchange about Copy of Newspaper Publication",
     "SUMIT1_14082026180538_SE_Intimation_Newspaper_Adv_14082026.pdf", None),
    ("ESOP/ESOS/ESPS", "AU Small Finance Bank Limited has informed the Exchange regarding Allotment of 378830  Shares.",
     "SUMIT1_22092026184338_ESOP_Allotment_Intimation.pdf", None),
]

@pytest.mark.parametrize("desc, text, name, want", CASES)
def test_nse_announcements_are_sorted_into_the_library_categories(desc, text, name, want):
    assert fl.classify_filing(desc, text, name) == want

def row(seq, when, desc="Credit Rating", text="X has informed the Exchange about Credit Rating", url=None):
    return {"seq_id": seq, "sort_date": f"{when} 18:00:00", "desc": desc, "attchmntText": text,
            "attchmntFile": url if url is not None else f"https://nsearchives.nseindia.com/corporate/X_{seq}.pdf"}

def test_only_pdfs_inside_the_two_year_window_are_kept_newest_first():
    rows = [row("1", "2024-10-05"), row("2", "2024-10-03"), row("3", "2026-09-01"),
            row("4", "2026-09-02", url="https://nsearchives.nseindia.com/corporate/X_ROID_9.zip"),
            row("5", "2026-09-03", url="-"), row("6", "2026-09-04", url="")]
    got = fl.select_filings("AUBANK", rows, today=date(2026, 10, 4), days=730)
    assert [f.source_id for f in got] == ["3", "1"]

def test_an_original_and_a_revised_filing_on_the_same_day_are_two_files():
    rows = [row("106800001", "2026-09-24", "Investor Presentation", "X has informed the Exchange about Investor Presentation"),
            row("106800002", "2026-09-24", "Investor Presentation", "X has informed the Exchange about Investor Presentation")]
    got = fl.select_filings("AUBANK", rows, today=date(2026, 10, 4))
    paths = {fl.filing_path(f, Path("/lib")) for f in got}
    assert len(paths) == 2

def test_a_filing_path_puts_the_date_first_and_drops_nse_boilerplate():
    f = fl.select_filings("M&MFIN", [row("106806399", "2026-10-01")], today=date(2026, 10, 4))[0]
    assert fl.filing_path(f, Path("/lib")) == Path("/lib/M_MFIN/credit_rating/2026-10-01_credit_rating_credit-rating_106806399.pdf")

def test_the_newest_two_annual_reports_are_kept_and_zips_skipped():
    reps = [{"fileName": "https://nsearchives.nseindia.com/annual_reports/AR_1_X_2025_2026.pdf", "fromYr": "2025", "toYr": "2026", "disseminationDateTime": "05-Aug-2026 18:34:31"},
            {"fileName": "https://nsearchives.nseindia.com/annual_reports/AR_2_X_2024_2025.zip", "fromYr": "2024", "toYr": "2025"},
            {"fileName": "https://nsearchives.nseindia.com/annual_reports/AR_3_X_2023_2024.pdf", "fromYr": "2023", "toYr": "2024"},
            {"fileName": "https://nsearchives.nseindia.com/annual_reports/AR_4_X_2022_2023.pdf", "fromYr": "2022", "toYr": "2023"}]
    got = fl.select_annual_reports("AUBANK", reps)
    assert [(f.source_id, f.date, f.category) for f in got] == [("AR2025-2026", "2026-08-05", "annual_report"),
                                                               ("AR2023-2024", "2024-03-31", "annual_report")]
    assert got[0].doc_key == "filing:nse:AR:AUBANK:2025-2026"
```

- [ ] **Step 2: Run to verify they fail** — `pytest tests/test_filings.py -q` → `ModuleNotFoundError`.
- [ ] **Step 3: Implement** the selection half of `pennystock/data/filings.py`: NSE `desc` decides first
  (`_RESULTS_DESC`, `_MATERIAL_DESC` regex over the taxonomy), then text rules (`_RESULTS_TEXT` for
  board-meeting outcomes and press releases, `TRANSCRIPT_HINT` + "transcript" in the file name),
  then `events.classify` for anything material (its credit-rating label maps to `credit_rating`).
  `_filing_date` reads `sort_date`, falling back to `an_dt` (`%d-%b-%Y`). `slug` strips
  "<Company> has informed/submitted to the Exchange (about|regarding|,)" and keeps six words.
- [ ] **Step 4: Run to verify they pass**, then the full suite.

---

### Task 3: Fetching filings from NSE + the CLI

**Files:**
- Modify: `pennystock/data/filings.py` (network half)
- Create: `scripts/fetch_filings.py`
- Test: `tests/test_filings.py` (network half, fakes only)

**Interfaces:**
- Consumes: Task 1 `documents.save_file/has/list_documents`, Task 2 selection.
- Produces: `discover(symbol, api, today=None, days=None) -> list[Filing]` (`api(path) -> json | None`; symbol URL-encoded), `nse_getter(session) -> Callable[[str], bytes | None]`, `download(filing, get, root=None) -> str` (`'have' | 'saved' | 'not_pdf' | 'failed'`, one retry on an exception), `write_manifest(symbol, root=None) -> Path | None`.

- [ ] **Step 1: Write the failing tests**

```python
PDF = b"%PDF-1.7\n" + b"y" * 9000

@pytest.fixture
def lib(tmp_path, monkeypatch):
    from pennystock.data import documents
    monkeypatch.setattr(documents.settings, "market_db", tmp_path / "t.db")
    monkeypatch.setattr(documents.settings, "project_root", tmp_path)
    return tmp_path / "Corporate_Filings"

def test_discover_encodes_ampersand_symbols_for_nse():
    seen = []
    fl.discover("M&MFIN", lambda p: seen.append(p) or None, today=date(2026, 10, 4))
    assert seen == ["corporate-announcements?index=equities&symbol=M%26MFIN",
                    "annual-reports?index=equities&symbol=M%26MFIN"]

def test_download_saves_once_then_skips(lib):
    f = fl.select_filings("AUBANK", [row("7", "2026-10-01")], today=date(2026, 10, 4))[0]
    calls = []
    get = lambda url: calls.append(url) or PDF
    assert fl.download(f, get, lib) == "saved"
    assert fl.download(f, get, lib) == "have"
    assert len(calls) == 1 and fl.filing_path(f, lib).read_bytes() == PDF

def test_an_html_error_page_is_never_saved_as_a_pdf(lib):
    f = fl.select_filings("AUBANK", [row("8", "2026-10-01")], today=date(2026, 10, 4))[0]
    assert fl.download(f, lambda url: b"<html>Access Denied</html>" * 500, lib) == "not_pdf"
    assert not fl.filing_path(f, lib).exists()

def test_a_failing_host_is_retried_once_then_reported(lib):
    f = fl.select_filings("AUBANK", [row("9", "2026-10-01")], today=date(2026, 10, 4))[0]
    calls = []
    def get(url):
        calls.append(url)
        raise ConnectionError("reset")
    assert fl.download(f, get, lib) == "failed"
    assert len(calls) == 2

def test_a_non_200_answer_is_a_failure_not_a_file(lib):
    class R:  # what requests returns for a 403
        status_code, content = 403, b"<html>Forbidden</html>"
    class S:
        def get(self, url, timeout=None):
            return R()
    f = fl.select_filings("AUBANK", [row("10", "2026-10-01")], today=date(2026, 10, 4))[0]
    assert fl.download(f, fl.nse_getter(S()), lib) == "failed"

def test_the_company_manifest_lists_every_filing_newest_first(lib):
    for seq, when in (("11", "2026-09-01"), ("12", "2026-10-01")):
        f = fl.select_filings("AUBANK", [row(seq, when)], today=date(2026, 10, 4))[0]
        fl.download(f, lambda url: PDF, lib)
    lines = fl.write_manifest("AUBANK", lib).read_text().splitlines()
    assert lines[0] == "date,category,title,file,bytes,source_url"
    assert [ln.split(",")[0] for ln in lines[1:]] == ["2026-10-01", "2026-09-01"]
```

- [ ] **Step 2: Run to verify they fail** (missing `discover`, `download`, `nse_getter`, `write_manifest`).
- [ ] **Step 3: Implement** the network half (`quote(symbol, safe="")`; `nse_getter` returns `r.content`
  only for HTTP 200, else `None`; `download` → `documents.save_file(..., min_bytes=settings.filing_min_pdf_bytes)`;
  `write_manifest` from `documents.list_documents(symbol, "filing")`), and `scripts/fetch_filings.py`
  (`--status`, `--dry-run`, `--symbol`, `--limit`, `--days`; companies = `broker_corpus.discover()`
  symbols; one `nse_session` per company; `settings.filings_delay_s` between downloads; manifest
  written per company; final tally).
- [ ] **Step 4: Run to verify they pass**, then the full suite, then `python scripts/fetch_filings.py --dry-run --symbol AUBANK` (network, read-only) to see real selections.

---

### Task 4: Stock reports — filer, cookie fetcher, index

**Files:**
- Create: `pennystock/data/stock_reports.py`
- Create: `scripts/file_stock_reports.py`, `scripts/fetch_stock_reports.py`
- Test: `tests/test_stock_reports.py`

**Interfaces:**
- Consumes: Task 1.
- Produces: `RE_NAME`, `SnapshotName(symbol, stamp, tl_id)`, `parse_name(name) -> SnapshotName | None`, `snapshot_path(symbol, stamp, root=None) -> Path`, `snapshot_key(symbol, stamp, tl_id) -> str`, `file_snapshot(data, symbol, stamp, tl_id=None, root=None) -> Path | None`, `has_snapshot_in_month(symbol, today) -> bool`, `unpack_bundle(data) -> list[tuple[str, str, bytes, str]]`, `register_existing(root=None) -> int`, `write_index(root=None) -> Path`, `fetch(symbols, session, today, limit=None, delay=1.0) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from datetime import date
from pennystock.data import documents, stock_reports as sr

PDF = b"%PDF-1.7\n" + b"z" * 60_000

@pytest.fixture(autouse=True)
def lib(tmp_path, monkeypatch):
    monkeypatch.setattr(documents.settings, "market_db", tmp_path / "t.db")
    monkeypatch.setattr(documents.settings, "project_root", tmp_path)
    monkeypatch.setattr(documents.settings, "stock_reports_dir", tmp_path / "Stock_Reports")
    return tmp_path / "Stock_Reports"

@pytest.mark.parametrize("name, want", [
    ("AUBANK-StockReport-20261003-2318.pdf", ("AUBANK", "20261003-2318", None)),
    ("GVT&D-StockReport-20261003-2355__TL67.pdf", ("GVT&D", "20261003-2355", "67")),
    ("AUBANK-StockReport-20261003-2318 (1).pdf", None),
    ("report.pdf", None)])
def test_trendlyne_download_names_are_parsed(name, want):
    got = sr.parse_name(name)
    assert (got and (got.symbol, got.stamp, got.tl_id)) == want if want else got is None

def test_a_snapshot_is_filed_under_its_safe_folder_and_catalogued(lib):
    p = sr.file_snapshot(PDF, "GVT&D", "20261003-2355", "67")
    assert p == lib / "GVT_D" / "GVT_D-StockReport-20261003-2355.pdf"
    d = documents.get("stock_report:trendlyne:67:20261003-2355")
    assert (d.symbol, d.doc_date, d.kind) == ("GVT&D", "2026-10-03", "stock_report")

def test_a_newer_snapshot_never_replaces_an_older_one(lib):
    sr.file_snapshot(PDF, "AUBANK", "20261003-2318", "54902")
    sr.file_snapshot(PDF + b"new", "AUBANK", "20261104-0910", "54902")
    assert sorted(p.name for p in (lib / "AUBANK").glob("*.pdf")) == [
        "AUBANK-StockReport-20261003-2318.pdf", "AUBANK-StockReport-20261104-0910.pdf"]

def test_the_monthly_rule_sees_this_months_snapshot(lib):
    sr.file_snapshot(PDF, "AUBANK", "20261003-2318", "54902")
    assert sr.has_snapshot_in_month("AUBANK", date(2026, 10, 30))
    assert not sr.has_snapshot_in_month("AUBANK", date(2026, 11, 1))

def test_a_legacy_bundle_splits_into_its_reports():
    blob = (b"ASTRAL\t122\t" + str(len(PDF)).encode() + b"\tASTRAL-StockReport-20261003-2345.pdf\n" + PDF
            + b"ATGL\t114071\t" + str(len(PDF)).encode() + b"\tATGL-StockReport-20261003-2346.pdf\n" + PDF)
    got = sr.unpack_bundle(blob)
    assert [(s, i, n) for s, i, _, n in got] == [("ASTRAL", "122", "ASTRAL-StockReport-20261003-2345.pdf"),
                                                 ("ATGL", "114071", "ATGL-StockReport-20261003-2346.pdf")]
    assert all(b == PDF for _, _, b, _ in got)

class Resp:
    def __init__(self, url, body=b"", headers=None):
        self.url, self.content, self.headers, self.status_code = url, body, headers or {}, 200

class FakeTrendlyne:
    """Research page redirects to /stock/<id>/...; get-pdf serves PDFs until `allowance`, then runout."""
    def __init__(self, allowance=1000):
        self.allowance, self.pdf_opens, self.headers = allowance, 0, {}
    def get(self, url, timeout=None, allow_redirects=True):
        if "/research-reports/stock/" in url:
            sym = url.rstrip("/").split("/")[-1]
            return Resp(f"https://trendlyne.com/research-reports/stock/{abs(hash(sym)) % 9999}/{sym}/x/")
        self.pdf_opens += 1
        if self.pdf_opens > self.allowance:
            return Resp("https://trendlyne.com/member/daily-runout/", b"<html>limit</html>")
        return Resp(url, PDF, {"Content-Disposition": 'inline; filename="X-StockReport-20261004-0100.pdf"'})

def test_the_fetcher_stops_at_the_runout_page(lib):
    site = FakeTrendlyne(allowance=2)
    tally = sr.fetch(["A", "B", "C", "D"], site, date(2026, 10, 4), delay=0)
    assert site.pdf_opens == 3 and tally["saved"] == 2 and tally["daily_limit"] == 1

def test_the_fetcher_never_opens_more_than_the_daily_allowance(lib):
    site = FakeTrendlyne()
    sr.fetch([f"S{i}" for i in range(90)], site, date(2026, 10, 4), delay=0)
    assert 0 < site.pdf_opens <= 70

def test_the_fetcher_skips_a_company_with_a_snapshot_this_month(lib):
    sr.file_snapshot(PDF, "A", "20261002-0900", "1")
    site = FakeTrendlyne()
    tally = sr.fetch(["A", "B"], site, date(2026, 10, 4), delay=0)
    assert site.pdf_opens == 1 and tally["this_month"] == 1
```

- [ ] **Step 2: Run to verify they fail** (`ModuleNotFoundError`).
- [ ] **Step 3: Implement** `stock_reports.py` (`RE_NAME = ^(?P<sym>.+?)-StockReport-(?P<d>\d{8})-(?P<t>\d{4})(?:__TL(?P<id>\d+))?\.pdf$`;
  key `stock_report:trendlyne:<tl_id or symbol>:<stamp>`; `fetch` resolves the id via the research
  page redirect, opens `get-pdf`, stops on `daily-runout` in the final URL or body, caps opens at
  `limit or settings.stock_report_daily_opens`, files through `file_snapshot`, reads the stamp from
  `Content-Disposition` (falls back to now); `register_existing` reads `Stock_Reports/index.csv`
  (symbol, trendlyne_id, file) and registers each file; `write_index` rebuilds `index.csv` from the
  catalog) and the two scripts (`file_stock_reports.py` scans `~/Downloads` or `--from DIR` for names
  and `tlsr_batch_*.bin`, files them, moves originals to `Stock_Reports/_filed_downloads/`, rewrites
  the index; `--register` catalogues what is already on disk. `fetch_stock_reports.py --cookies`
  loads a Netscape cookie jar exactly like `fetch_midcap_reports.py`, `--dry-run`, `--limit`).
- [ ] **Step 4: Run to verify they pass**, then the full suite.

---

### Task 5: Catalog backfill, status, docs — and the real run

**Files:**
- Create: `scripts/catalog_documents.py`
- Modify: `CLAUDE.md` (commands), `docs/superpowers/specs/2026-10-04-company-documents-library-design.md` (status line)
- Test: `tests/test_catalog_documents.py`

**Interfaces:**
- Consumes: Tasks 1, 4; `broker_corpus.ReportRecord`.
- Produces: `broker_document(record) -> dict | None` (kwargs for `documents.register`, None when the record has no PDF or symbol), CLI `--backfill`, `--verify`, default summary.

- [ ] **Step 1: Write the failing test**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from catalog_documents import broker_document  # noqa: E402
from pennystock.data.broker_corpus import ReportRecord

def test_a_downloaded_broker_report_becomes_a_catalog_entry(tmp_path):
    pdf = tmp_path / "AUBANK_1.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")
    got = broker_document(ReportRecord(report_id="100115", symbol="AUBANK", broker="ICICI Securities",
                                       report_date="2026-09-04", headline="Entering the big league", pdf=pdf, doc_id="abc"))
    assert got == {"doc_key": "broker_report:trendlyne:100115", "symbol": "AUBANK", "kind": "broker_report",
                   "doc_date": "2026-09-04", "source": "trendlyne", "path": pdf,
                   "title": "ICICI Securities: Entering the big league",
                   "source_url": "https://trendlyne.com/get-document/report/pdf/100115/"}

def test_listed_but_not_downloaded_reports_are_not_catalogued():
    assert broker_document(ReportRecord(report_id="1", symbol="X", report_date="2026-01-01")) is None
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement** `scripts/catalog_documents.py`.
- [ ] **Step 4: Full suite + CI lint gate**: `pytest -q`; `ruff check --select E9,F63,F7,F82 pennystock scripts tests`.
- [ ] **Step 5: Real run** (network to NSE only):
  `python scripts/catalog_documents.py --backfill` → `python scripts/file_stock_reports.py --register` →
  `python scripts/fetch_filings.py` (background, ~6,000 files at 1/s ≈ 2 h) → `python scripts/catalog_documents.py`
  and `--verify`. Update CLAUDE.md commands.

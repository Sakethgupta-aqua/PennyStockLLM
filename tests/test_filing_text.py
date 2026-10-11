"""Filing text: pages extracted once into filings.db, keyword-searchable, filtered by company, type and time.

Offline: extraction is injected, so no PDF is opened; page texts are shaped like real NSE filings.
"""
from __future__ import annotations

import pytest

from pennystock.data import documents
from pennystock.data import filing_text as ft


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(ft.settings, "filings_db", tmp_path / "filings.db")
    monkeypatch.setattr(ft.settings, "market_db", tmp_path / "market.db")
    monkeypatch.setattr(ft.settings, "project_root", tmp_path)
    return tmp_path


def doc(key, symbol, doc_type, sha, date, stamp):
    return documents.Document(doc_key=key, symbol=symbol, kind="filing", category=doc_type, doc_date=date,
                              title="t", source="nse",
                              source_url=f"https://nsearchives.nseindia.com/corporate/{symbol}{stamp}x.pdf",
                              local_path=f"Corporate_Filings/{symbol}/{doc_type}/{sha}.pdf", bytes=1,
                              sha256=sha, fetched_at="2026-10-05T00:00:00")


def test_curly_quotes_and_ligatures_are_folded():
    assert ft.normalize("Board’s Report ﬁnancial") == "Board's Report financial"

def test_a_statement_page_is_a_table_and_commentary_is_prose():
    table = "Revenue from operations 12,345.67 11,234.56 45,678.90 Other income 123.45 98.76 456.78 " * 6
    prose = "During the quarter the bank grew advances by 22% while asset quality held steady. " * 6
    assert ft.page_kind(table, "results", None) == "table"
    assert ft.page_kind(prose, "results", None) == "prose"
    assert ft.page_kind(prose, "annual_report", None) == "boilerplate"     # outside the key sections
    assert ft.page_kind(prose, "annual_report", "mdna") == "prose"

def test_annual_report_sections_run_from_their_heading_to_the_next_major_heading():
    pages = ["Notice of the Annual General Meeting", "Management Discussion and Analysis\nIndustry overview",
             "Our margins improved", "Board’s Report\nDear Members", "Dividend",
             "Report on Corporate Governance",
             "Independent Auditor’s Report\nKey Audit Matters", "Balance Sheet as at 31 March 2026",
             "Note 38 Contingent liabilities and commitments", "Note 39 Related party disclosures"]
    assert ft.ar_sections(pages) == [None, "mdna", "mdna", "board_report", "board_report", None,
                                     "auditor", None, "contingent", "related_party"]

def test_the_fiscal_period_follows_indias_april_to_march_year():
    assert ft.fiscal_period("results", "2026-07-25") == "Q1 FY27"
    assert ft.fiscal_period("earnings_call", "2026-02-10") == "Q3 FY26"
    assert ft.fiscal_period("annual_report", "2026-07-20") == "FY26"
    assert ft.fiscal_period("material_event", "2026-07-25") is None

def test_the_broadcast_stamp_in_the_url_is_when_the_market_saw_it():
    url = "https://nsearchives.nseindia.com/corporate/BDL_24122024171434_BDLIDCessation24Sep2024.pdf"
    assert ft.available_at(url, "2024-12-24") == "2024-12-24T17:14"
    assert ft.available_at("https://x/a.pdf", "2024-12-24") == "2024-12-24T00:00"

def test_a_question_full_of_search_syntax_becomes_a_safe_query():
    assert ft.fts_query('What did "management" say -- about NIM AND CASA* (Q1)?') == \
        '"management" OR "say" OR "nim" OR "casa" OR "q1"'
    assert ft.fts_query("what is the") is None

def test_duplicate_files_are_indexed_once_and_keyword_search_honours_company_type_and_date(db):
    pages = {"s1": [("The bank's NIM expanded to 5.9% on CASA growth.", False)],
             "s2": [("Credit cost guidance for FY27 is 1.5%.", False)]}
    docs = [doc("filing:nse:1", "AUBANK", "earnings_call", "s1", "2026-07-25", "_25072026183000_"),
            # the same file filed again
            doc("filing:nse:2", "AUBANK", "earnings_call", "s1", "2026-07-25", "_25072026183000_"),
            doc("filing:nse:3", "AUBANK", "investor_presentation", "s2", "2026-08-02", "_02082026100000_")]
    tally = ft.index_texts(docs, extract=lambda path, doc_type: pages[path.stem])
    assert tally["extracted"] == 2 and tally["duplicates"] == 1
    assert [h["page"] for h in ft.keyword_pages("AUBANK", "NIM and CASA")] == [1]
    assert ft.keyword_pages("AUBANK", "credit cost", doc_types=["earnings_call"]) == []
    assert ft.keyword_pages("AUBANK", "credit cost", as_of="2026-08-01T00:00") == []
    assert len(ft.keyword_pages("AUBANK", "credit cost", as_of="2026-08-02T10:00")) == 1
    assert ft.keyword_pages("OTHERCO", "NIM") == []
    assert ft.index_texts(docs, extract=lambda *a: 1 / 0)["extracted"] == 0               # unchanged: a no-op


# --- calibrated on AU Small Finance Bank's 364-page annual report (2026-10-05) ---------------------
def test_a_mid_sentence_mention_neither_starts_nor_interrupts_a_section():
    pages = ["Management Discussion & Analysis 94 AU Small Finance Bank Limited",
             "Financial Performance Highlights In FY 2024-25 the bank delivered strong growth while its risk "
             "management framework kept credit costs in check",
             "Board evaluation is a vital aspect of corporate governance that enhances the effectiveness",
             "Risk Management The Bank operates within an effective risk management framework"]
    assert ft.ar_sections(pages) == ["mdna", "mdna", "mdna", "risk"]


def test_a_board_report_that_opens_with_dear_members_is_found():
    pages = ["Corporate Information Registered Office Jaipur",
             'Dear Members, The Board of Directors ("the Board") is pleased to present the 30th Annual '
             'Report',
             "Change in Nature of Business During the year under review there were no changes"]
    assert ft.ar_sections(pages) == [None, "board_report", "board_report"]


def test_a_long_read_never_blocks_the_indexer_from_writing(db, monkeypatch):
    # a 21-second scan of every page stopped the embedding run with "database is locked" (2026-10-06):
    # in SQLite's default journal mode a reader holds off every writer
    import sqlite3
    monkeypatch.setattr(ft.settings, "filings_db_timeout_s", 0.2)
    ft.index_texts([doc("filing:nse:1", "AUBANK", "earnings_call", "s1", "2026-07-25", "_25072026183000_")],
                   extract=lambda path, doc_type: [("NIM guidance for FY27 is 5.8% to 6.0%.", False)])
    reader = sqlite3.connect(str(ft.settings.filings_db), timeout=0.2)
    reader.execute("BEGIN")
    reader.execute("SELECT COUNT(*) FROM filing_pages").fetchone()          # a read still in progress
    with ft._connect() as con:                                                 # the indexer marks a file done
        con.execute("UPDATE filing_texts SET passages_version = 1")
    reader.close()
    assert ft.status()["files_embedded"] == 1

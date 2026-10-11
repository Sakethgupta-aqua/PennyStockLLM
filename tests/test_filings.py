"""Corporate filings from NSE: which of ~1,400 announcements a company files are worth keeping, where
they go, and how they are fetched. Subjects below are real NSE rows (AUBANK, BHEL, IDFCFIRSTB),
measured 2026-10-04. Nothing here touches the network."""
from datetime import date
from pathlib import Path

import pytest

from pennystock.data import filings as fl

CASES = [
    ("Investor Presentation", "AU Small Finance Bank Limited has informed the Exchange about Investor Presentation",
     "AUBANK_24092026230354_Revised_Investor_Presentation_SE.pdf", "investor_presentation"),
    ("Outcome of Board Meeting",
     "AU Small Finance Bank Limited has submitted to the Exchange, the financial results for the period ended Jun 30, 2026.",
     "AUBANK_25072026125625_BM_Outcome_July_25.pdf", "results"),
    ("Outcome of Board Meeting",
     "AU Small Finance Bank Limited has informed the Exchange regarding Outcome of Board Meeting held on September 28, 2026.",
     "SUMIT1_28092026212616_HIA_Re-appointment.pdf", None),
    ("Financial Result Updates",
     " AU Small Finance Bank Limited has submitted to the Exchange, the financial results for the period ended December 31, 2024.",
     "Results_24012025144132.pdf", "results"),
    ("Integrated Filing- Financial", "Unaudited Financial Results for & upto the quarter ended 31st December, 2024",
     "BHEL_28012025184917_31122024quickresults.pdf", "results"),
    ("Press Release",
     'AU Small Finance Bank Limited has informed the Exchange regarding a press release dated July 25, 2026, '
     'titled "Press Release on Unaudited Financial Results of AU Small Fi',
     "AUBANK_25072026133046_Press_release_submission.pdf", "results"),
    ("Press Release",
     'AU Small Finance Bank Limited has informed the Exchange regarding a press release dated May 19, 2026, '
     'titled "AU Small Finance Bank adopts Intellect s Purple Fabric to ac',
     "SUMIT1_19052026104108_Submission_PR_Purple_Fabric.pdf", None),
    ("Credit Rating", "AU Small Finance Bank Limited has informed the Exchange about Credit Rating",
     "SUMIT1_01102026201906_ICRA_Credit_Rating.pdf", "credit_rating"),
    ("Credit Rating- New", "IDFC First Bank Limited has informed the Exchange about Credit Rating- New",
     "IDFCFIRSTB_01102026165613_IDFCFIRSTB_IndRa_CARE_2026.pdf", "credit_rating"),
    ("Analysts/Institutional Investor Meet/Con. Call Updates",
     "AU Small Finance Bank Limited has informed the Exchange about Schedule of meet",
     "AUBANK_21092026183759_Investor_Meet_Intimation.pdf", None),
    ("Analysts/Institutional Investor Meet/Con. Call Updates",
     "AU Small Finance Bank Limited has informed the Exchange about Transcript of the Earnings Call for Q1 FY27",
     "AUBANK_30072026181500_Earnings_Call_Transcript_Q1FY27.pdf", "earnings_call"),
    ("Change in Auditors",
     "Bharat Heavy Electricals Limited has informed the Exchange regarding Change in Auditors of the company.",
     "BHEL_08092026154443_letter_to_SE_for_auditor.pdf", "material_event"),
    ("Frauds/Default by employees", "IDFC First Bank Limited has informed the Exchange about Frauds/Default by employees",
     "IDFCFIRSTB_22022026000738_IDFCFIRSTB_Reg30.pdf", "material_event"),
    ("Final  forensic  audit  report",
     "Please find enclosed Disclosure under Regulation 30 ... - Report on Chandigarh incident",
     "IDFCFIRSTB_05062026220239_Outcome_KPMG_Forensic_Review_Report.pdf", "material_event"),
    ("Resignation", "IDFC First Bank Limited has informed the Exchange regarding Resignation of Mr. Ashish Singh.",
     "IDFCFIRSTB_11082026185509_Change_in_SMP.pdf", "material_event"),
    ("Trading Window",
     "AU Small Finance Bank Limited has informed the Exchange regarding the Trading Window closure pursuant to SEBI "
     "(Prohibition of Insider Trading) Regulations, 2015",
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
    deck = ("Investor Presentation", "X has informed the Exchange about Investor Presentation")
    rows = [row("106800001", "2026-09-24", *deck), row("106800002", "2026-09-24", *deck)]
    got = fl.select_filings("AUBANK", rows, today=date(2026, 10, 4))
    assert len({fl.filing_path(f, Path("/lib")) for f in got}) == 2


def test_a_filing_path_puts_the_date_first_and_drops_nse_boilerplate():
    f = fl.select_filings("M&MFIN", [row("106806399", "2026-10-01")], today=date(2026, 10, 4))[0]
    assert fl.filing_path(f, Path("/lib")) == Path(
        "/lib/M_MFIN/credit_rating/2026-10-01_credit_rating_credit-rating_106806399.pdf")
    assert f.doc_key == "filing:nse:106806399"


def test_the_newest_two_annual_reports_are_kept_and_zips_skipped():
    reps = [{"fileName": "https://nsearchives.nseindia.com/annual_reports/AR_1_X_2025_2026.pdf", "fromYr": "2025",
             "toYr": "2026", "disseminationDateTime": "05-Aug-2026 18:34:31"},
            {"fileName": "https://nsearchives.nseindia.com/annual_reports/AR_2_X_2024_2025.zip", "fromYr": "2024",
             "toYr": "2025"},
            {"fileName": "https://nsearchives.nseindia.com/annual_reports/AR_3_X_2023_2024.pdf", "fromYr": "2023",
             "toYr": "2024"},
            {"fileName": "https://nsearchives.nseindia.com/annual_reports/AR_4_X_2022_2023.pdf", "fromYr": "2022",
             "toYr": "2023"}]
    got = fl.select_annual_reports("AUBANK", reps)
    assert [(f.source_id, f.date, f.category) for f in got] == [("AR2025-2026", "2026-08-05", "annual_report"),
                                                               ("AR2023-2024", "2024-03-31", "annual_report")]
    assert got[0].doc_key == "filing:nse:AR:AUBANK:2025-2026"


# --- fetching (fakes only) ------------------------------------------------------------------------
PDF = b"%PDF-1.7\n" + b"y" * 9000 + b"\n%%EOF\n"


@pytest.fixture
def lib(tmp_path, monkeypatch):
    from pennystock.data import documents
    monkeypatch.setattr(documents.settings, "market_db", tmp_path / "t.db")
    monkeypatch.setattr(documents.settings, "project_root", tmp_path)
    return tmp_path / "Corporate_Filings"


def one(seq, when="2026-10-01"):
    return fl.select_filings("AUBANK", [row(seq, when)], today=date(2026, 10, 4))[0]


def test_discover_encodes_ampersand_symbols_for_nse():
    seen = []
    fl.discover("M&MFIN", lambda p: seen.append(p) or None, today=date(2026, 10, 4))
    assert seen == ["corporate-announcements?index=equities&symbol=M%26MFIN",
                    "annual-reports?index=equities&symbol=M%26MFIN"]


def test_download_saves_once_then_skips(lib):
    f, calls = one("7"), []

    def get(url):
        calls.append(url)
        return PDF
    assert fl.download(f, get, lib) == "saved"
    assert fl.download(f, get, lib) == "have"
    assert len(calls) == 1 and fl.filing_path(f, lib).read_bytes() == PDF


def test_an_html_error_page_is_never_saved_as_a_pdf(lib):
    f = one("8")
    assert fl.download(f, lambda url: b"<html>Access Denied</html>" * 500, lib) == "not_pdf"
    assert not fl.filing_path(f, lib).exists()


def test_a_failing_host_is_retried_once_then_reported(lib):
    f, calls = one("9"), []

    def get(url):
        calls.append(url)
        raise ConnectionError("reset")
    assert fl.download(f, get, lib) == "failed"
    assert len(calls) == 2


def test_a_non_200_answer_is_a_failure_not_a_file(lib):
    class Forbidden:                      # what requests returns for a 403
        status_code, content = 403, b"<html>Forbidden</html>"

    class Session:
        def get(self, url, timeout=None):
            return Forbidden()
    f = one("10")
    assert fl.download(f, fl.nse_getter(Session()), lib) == "failed"
    assert not fl.filing_path(f, lib).exists()


def test_the_company_manifest_lists_every_filing_newest_first(lib):
    for seq, when in (("11", "2026-09-01"), ("12", "2026-10-01")):
        fl.download(one(seq, when), lambda url: PDF, lib)
    lines = fl.write_manifest("AUBANK", lib).read_text().splitlines()
    assert lines[0] == "date,category,title,file,bytes,source_url"
    assert [ln.split(",")[0] for ln in lines[1:]] == ["2026-10-01", "2026-09-01"]


def test_a_seq_id_that_is_not_a_number_never_reaches_a_file_name():
    rows = [row("../../etc/passwd", "2026-10-01"), row("106806400", "2026-10-01")]
    got = fl.select_filings("AUBANK", rows, today=date(2026, 10, 4))
    assert [f.source_id for f in got] == ["106806400"]


def test_repair_refetches_only_incomplete_filings(tmp_path, monkeypatch):
    monkeypatch.setattr(fl.documents, "verify",
                        lambda: {"missing": [], "changed": [], "incomplete": ["filing:nse:1"]})
    monkeypatch.setattr(fl.documents, "get",
                        lambda k: type("D", (), {"source_url": "https://x/a.pdf", "doc_key": k})())
    replaced = []
    monkeypatch.setattr(fl.documents, "replace_file",
                        lambda k, data: replaced.append(k) or tmp_path / "a.pdf")
    assert fl.repair(lambda url: b"%PDF-1.7\nok\n%%EOF\n") == {"repaired": 1}
    assert replaced == ["filing:nse:1"]


def test_repair_tries_again_when_nse_serves_a_cut_copy(tmp_path, monkeypatch):
    # NSE's CDN sometimes serves a truncated body whose Content-Length matches the cut; minutes later the
    # same URL can come back whole (9 of 15 such annual reports did on a second try, 2026-10-05)
    monkeypatch.setattr(fl.documents, "verify",
                        lambda: {"missing": [], "changed": [],
                                 "incomplete": ["filing:nse:1", "filing:nse:2"]})
    monkeypatch.setattr(fl.documents, "get",
                        lambda k: type("D", (), {"source_url": f"https://x/{k[-1]}.pdf", "doc_key": k})())
    monkeypatch.setattr(fl.documents, "replace_file",
                        lambda k, data: tmp_path / "a.pdf" if fl.documents.is_complete_pdf(data) else None)
    monkeypatch.setattr(fl.settings, "filings_delay_s", 0)
    served = {"https://x/1.pdf": [b"%PDF-1.7\ncut", b"%PDF-1.7\nwhole\n%%EOF\n"],
              "https://x/2.pdf": [b"%PDF-1.7\ncut"]}
    calls: dict[str, int] = {}

    def get(url):
        calls[url] = calls.get(url, 0) + 1
        return served[url][min(calls[url], len(served[url])) - 1]
    assert fl.repair(get) == {"repaired": 1, "still_incomplete": 1}
    assert calls == {"https://x/1.pdf": 2, "https://x/2.pdf": 3}   # first whole copy wins; 3 tries at most

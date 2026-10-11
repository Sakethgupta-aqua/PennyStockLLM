"""Annual-report note extraction.

Every heuristic here was written in response to something a REAL report did: Hindalco's
segment-liabilities table sitting just above the related-party heading, Vedanta's credit-risk
paragraph that merely cross-references the contingent-liability note, and figure lists polluted
with years and mangled table identifiers. The tests pin those behaviours so a future tweak to the
patterns cannot silently undo them.
"""
import json
from pathlib import Path

from pennystock.data.annual_report import (AnnualReportNotes, CONTINGENT, CONTINGENT_DETAIL,
                                           CONTINGENT_STRONG, RELATED_PARTY, RELATED_DETAIL,
                                           RELATED_STRONG, _section, candidate_figures,
                                           summarize_notes)

WINDOW = 900


# --- candidate_figures: hints for the analyst, not parsed totals ----------------------------
def test_figures_keeps_indian_comma_grouped_amounts():
    assert candidate_figures("Claims not acknowledged as debt 12,740 5,679") == ["12,740", "5,679"]


def test_figures_keeps_amounts_written_with_a_unit():
    assert candidate_figures("guarantees of ` 150 Crore given") == ["150 Crore"]


def test_figures_rejects_years_and_mangled_table_ids():
    """The first live run returned ['2014', '2026', '4260848'] - years and a mangled identifier.
    Requiring comma-grouping or an explicit unit is what removes them."""
    got = candidate_figures("As at March 31, 2026 (2014) ref 4260848 total 8,181")
    assert got == ["8,181"]


def test_figures_ignores_small_note_numbers():
    """'(xvii) Demand ... 18 18' - note numbering, not money."""
    assert candidate_figures("(xvii) GST transition demand 18 18") == []


def test_figures_deduplicates_and_respects_the_limit():
    assert candidate_figures("1,111 1,111 2,222") == ["1,111", "2,222"]
    assert len(candidate_figures(" ".join(f"{i},000" for i in range(1, 20)), limit=3)) == 3


# --- _section: finding the note itself, not a page that mentions it -------------------------
def test_section_prefers_the_note_over_the_accounting_policy_paragraph():
    policy = ("Contingent liabilities are disclosed when an outflow of resources is not probable. "
              "A guarantee is recognised at fair value.")                   # heading + detail, no figures
    note = ("NOTE 45: CONTINGENT LIABILITIES not acknowledged as debt: licensing disputes 3,857 "
            "and spectrum dues 64,046 with guarantees of 4,598")
    assert "64,046" in _section([policy, note], CONTINGENT, CONTINGENT_DETAIL, WINDOW, CONTINGENT_STRONG)


def test_section_ignores_a_table_that_merely_sits_above_the_heading():
    """Hindalco: 'Total liabilities 211,200 142,270' immediately preceded '30. Related party
    disclosures'. Scoring the lookback let those numbers outvote the note's own."""
    page = ("Other corporate liabilities 106,684 72,508 Total liabilities 211,200 142,270 "
            "30. Related party disclosures. The following transactions were carried out with "
            "related parties: Nature of Transaction 5,381 4,849")
    got = _section([page], RELATED_PARTY, RELATED_DETAIL, WINDOW, RELATED_STRONG)
    assert "5,381" in candidate_figures(got)
    assert "211,200" not in candidate_figures(got)


def test_section_strong_signal_outranks_a_page_with_more_numbers():
    """Vedanta: a credit-risk paragraph pointing at 'Note 40 Contingent liability' beat the real
    note on figure count alone. The canonical phrase has to win regardless."""
    cross_ref = ("financial guarantees are described in Note 40 on Contingent liability. "
                 "8,404 12,381 1,571 1,242 1,086 3,337 5,647 5,151")
    real_note = "Contingent liabilities: claims against the Company not acknowledged as debt 2,500"
    got = _section([cross_ref, real_note], CONTINGENT, CONTINGENT_DETAIL, WINDOW, CONTINGENT_STRONG)
    assert "2,500" in got


def test_section_returns_empty_when_the_detail_language_is_absent():
    assert _section(["contingent liabilities"], CONTINGENT, CONTINGENT_DETAIL, WINDOW) == ""


# --- summarize_notes: what the analyst actually receives ------------------------------------
def test_summary_says_unchecked_rather_than_clean_when_nothing_was_extracted():
    """The whole point: silence about off-balance-sheet risk must never read as absence of it."""
    out = summarize_notes(AnnualReportNotes(symbol="ABC"))
    assert out["annual_report_notes"] == "unavailable"
    assert "UNCHECKED" in out["note"]


def test_summary_carries_the_yardsticks_the_notes_are_judged_against():
    out = summarize_notes(
        AnnualReportNotes(symbol="ABC", from_yr="2025", to_yr="2026", url="http://x/ar.pdf",
                          contingent_text="not acknowledged as debt 12,740",
                          related_party_text="Nature of Transaction 5,381"),
        net_worth_cr=95000, revenue_cr=272642)
    assert out["financial_year"] == "2025-2026"
    assert out["contingent_liabilities"]["net_worth_cr_for_comparison"] == 95000
    assert out["related_party_transactions"]["revenue_cr_for_comparison"] == 272642
    assert out["contingent_liabilities"]["figures_mentioned"] == ["12,740"]
    assert "NEVER as parsed totals" in out["caveat"]


# --- the AGM notice is not the related-party note -------------------------------------------------
def test_the_agm_notices_proposed_rpt_is_not_the_related_party_note():
    # DIXON FY26: the AGM notice (pp. 4-55) is bound in front and its 'proposed RPT' tables won the search
    agm = ("Dixon Technologies (India) Limited_33rd AGM Notice 2025-26\nPart A: Minimum information of the "
           "proposed RPT\nA(3) Details of previous transactions with the related party\n"
           "Nature of transaction: Sale of goods 2,496.66 1,159.78 4,500.00 3,200.00 2,100.00")
    note = ("32. Related party disclosures\nTransactions with related parties\n"
            "Nature of transaction 4,500 3,210")
    picked = _section([agm, note], RELATED_PARTY, RELATED_DETAIL, WINDOW, RELATED_STRONG)
    assert "4,500" in picked and "2,496.66" not in picked


# --- the tool: verified sources first, the excerpt only as a labelled fallback ---------------------
FIX = Path(__file__).parent / "fixtures"


def _verified_dixon_note() -> dict:
    from pennystock.data import contingent_liabilities as cl
    rows = json.loads((FIX / "notes" / "dixon_note46_rows.json").read_text())
    notes = FIX / "notes"
    text = (notes / "dixon_p474.txt").read_text() + "\n" + (notes / "dixon_p475.txt").read_text()
    kept, dropped = cl.verify(rows, text)
    return {"symbol": "DIXON", "fiscal_year": "FY26", "page": 474, "basis": "consolidated", "unit": "lakh",
            "status": cl.status_of(kept, dropped, "lakh", cl.reconcile(kept)), "rows": kept,
            "dropped": dropped, "totals": cl.totals(kept, "lakh"), "warnings": cl.reconcile(kept)}


def _dixon_disclosure():
    from pennystock.data.integrated_filings import RptDisclosure, rpt_rows
    return RptDisclosure("DIXON", "H2 FY26", "2025-10-01", "2026-03-31",
                         rpt_rows((FIX / "xbrl" / "dixon_rpt_h2fy26.xml").read_text()),
                         "12-May-2026 23:38:14", "u", 21182.1, "FY26 revenue minus the April-September half")


def _wire_tool(monkeypatch, cl_result=None, rpt=None, filed=None, provider=None):
    import ast
    import asyncio

    from pennystock.tools import market_data as md
    monkeypatch.setattr(md, "_resolve_ticker", lambda s, e: ("DIXON.NS", "Dixon Technologies"))
    monkeypatch.setattr(md, "stored_cl", lambda sym: cl_result)
    monkeypatch.setattr(md, "rpt_disclosure", lambda sym: rpt)
    monkeypatch.setattr(md, "filed_year", lambda sym: filed)
    monkeypatch.setattr(md, "_is_lender", lambda ticker: False)
    monkeypatch.setattr(md, "_excerpt_notes", lambda sym: AnnualReportNotes(
        symbol="DIXON", from_yr="2025", to_yr="2026", url="http://x/ar.pdf",
        contingent_text="not acknowledged as debt 39,841", related_party_text="Nature of Transaction 2,496"))
    if provider is not None:
        monkeypatch.setattr(md, "get_provider", lambda: provider)
    return lambda: ast.literal_eval(asyncio.run(md.get_annual_report_notes.ainvoke({"symbol": "DIXON"})))


def test_verified_claims_and_the_xbrl_disclosure_come_with_filed_yardsticks(monkeypatch):
    from pennystock.data.integrated_filings import FiledYear
    fy = FiledYear("DIXON", "FY26", "2026-03-31", "consolidated", 48872.8, 1438.64, 4676.67, 5386.67, 271.59,
                   0.0, "S.N. DHAWAN & CO LLP", "12-May-2026 23:38:17", "u")
    out = _wire_tool(monkeypatch, cl_result=_verified_dixon_note(), rpt=_dixon_disclosure(), filed=fy)()
    assert out["contingent_liabilities"]["machine_verified"] is True
    assert out["contingent_liabilities"]["disputed_claims_pct_of_net_worth"] == 11.3      # 530.27 / 4,676.67
    assert out["related_party_transactions"]["source"] == "NSE Reg 23(9) XBRL"
    assert out["yardsticks"] == {"net_worth_cr": 4676.67, "revenue_cr": 48872.8,
                                 "source": "NSE integrated filing FY26 (12-May-2026)"}


def test_no_structured_source_falls_back_to_the_excerpt_and_says_so(monkeypatch):
    from pennystock.data.providers import Financials

    class Provider:                     # lists run oldest -> newest: the yardstick is the LATEST year
        def get_financials(self, ticker, years):
            return Financials(fiscal_years=[2022, 2025, 2026], revenue_cr=[None, 38593.29, 48537.39],
                              equity_cr=[None, 3700.0, 4668.0])
    out = _wire_tool(monkeypatch, provider=Provider())()
    assert out["contingent_liabilities"]["machine_verified"] is False and out["contingent_liabilities"]["why"]
    assert "figures_mentioned" in out["contingent_liabilities"]
    assert out["related_party_transactions"]["machine_verified"] is False
    assert out["yardsticks"] == {"net_worth_cr": 4668.0, "revenue_cr": 48537.39, "source": "Yahoo statements"}

"""Annual-report note extraction.

Every heuristic here was written in response to something a REAL report did: Hindalco's
segment-liabilities table sitting just above the related-party heading, Vedanta's credit-risk
paragraph that merely cross-references the contingent-liability note, and figure lists polluted
with years and mangled table identifiers. The tests pin those behaviours so a future tweak to the
patterns cannot silently undo them.
"""
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

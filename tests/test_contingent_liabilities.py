"""Contingent liabilities: the annual report's note, located, read into rows and VERIFIED against the page."""
import json
from pathlib import Path

import pytest

from pennystock.data import contingent_liabilities as cl

NOTES = Path(__file__).parent / "fixtures" / "notes"


def pages(*names, start=100):
    return [{"page": start + i, "text": (NOTES / n).read_text()} for i, n in enumerate(names)]


def test_the_consolidated_note_is_found_with_its_unit():
    note = cl.locate_note(pages("policy.txt", "dixon_p474.txt", "dixon_p475.txt"))
    assert note.page == 101 and note.unit == "lakh"
    assert "Excise, custom duty and service tax" in note.text


@pytest.mark.parametrize("name, unit", [("astral_p211.txt", "million"), ("irctc_p344.txt", "lakh")])
def test_variant_headings_and_rupee_glyphs_are_read(name, unit):
    # Astral: no "not acknowledged as debt"; IRCTC: "not acknowledge as debt" and the rupee sign read as "J"
    note = cl.locate_note(pages(name))
    assert note is not None and note.unit == unit


def test_policy_prose_and_a_csr_table_are_not_the_note():
    assert cl.locate_note(pages("policy.txt", "csr.txt")) is None


def test_the_unit_falls_back_to_the_reports_all_amounts_line():
    assert cl.detect_unit("Claims not acknowledged as debt 1,234 1,100", "no unit here",
                          ["(All amounts in ₹ crores unless otherwise stated)"]) == "crore"


DIXON_ROWS = json.loads((NOTES / "dixon_note46_rows.json").read_text())
DIXON_TEXT = (NOTES / "dixon_p474.txt").read_text() + "\n" + (NOTES / "dixon_p475.txt").read_text()


def test_every_row_on_the_page_survives_verification():
    kept, dropped = cl.verify(DIXON_ROWS, DIXON_TEXT)
    assert len(kept) == len(DIXON_ROWS) and dropped == []


def test_a_number_not_on_the_page_is_dropped_and_the_note_is_not_verified():
    invented = {"label": "Invented", "category": "disputed_tax", "current": "77,777", "previous": None}
    bad = DIXON_ROWS + [invented]
    kept, dropped = cl.verify(bad, DIXON_TEXT)
    assert [r["label"] for r in dropped] == ["Invented"]
    assert cl.status_of(kept, dropped, "lakh") == "partial"


def test_totals_are_in_crore_by_bucket_with_protest_payments_and_subtotals_left_out():
    t = cl.totals(DIXON_ROWS, "lakh")
    assert t["disputed_claims"] == {"current": 530.27, "previous": 164.86}
    # trade finance = bank guarantees + letters of credit + bill discounting
    assert t["trade_finance"] == {"current": 799.35, "previous": 986.38}
    # the JV customs bond (100 lakh) + note (ii): guarantees to two suppliers on behalf of a subsidiary
    # (56,793 lakh; no comparative printed)
    assert t["guarantees_for_others"] == {"current": 568.93, "previous": 1.0}
    assert t["other"] == {"current": 185.55, "previous": 187.56}


def test_a_printed_subtotal_is_reconciled():
    assert cl.reconcile(DIXON_ROWS) == []
    broken = [dict(r, current="1,999") if r["category"] == "subtotal" else r for r in DIXON_ROWS]
    assert len(cl.reconcile(broken)) == 1


def _result(status="verified"):
    kept, dropped = cl.verify(DIXON_ROWS, DIXON_TEXT)
    return {"symbol": "DIXON", "fiscal_year": "FY26", "page": 474, "basis": "consolidated", "unit": "lakh",
            "status": status, "rows": kept, "dropped": dropped, "totals": cl.totals(kept, "lakh"),
            "warnings": cl.reconcile(kept)}


def test_the_summary_judges_claims_against_net_worth():
    s = cl.summarize_cl(_result(), 4676.67, lender=False)
    assert s["machine_verified"] and s["disputed_claims_pct_of_net_worth"] == 11.3
    assert s["source"] == "annual report FY26, page 474, consolidated"
    assert s["largest_items"][0]["label"] == "Letters of credit (outstanding)"


def test_a_lender_is_judged_on_claims_only():
    s = cl.summarize_cl(_result(), 4676.67, lender=True)
    assert "lender" in s["how_to_judge"] and "guarantees_for_others_pct_of_net_worth" not in s


def test_extract_runs_the_pipeline_and_stores_the_result(tmp_path, monkeypatch):
    class StubLLM:                          # the model's part is stubbed; everything after it is real
        pass
    monkeypatch.setattr(cl, "latest_report", lambda symbol: {"content_sha": "sha", "doc_date": "2026-09-03",
                                                              "fiscal_period": "FY26"})
    monkeypatch.setattr(cl, "pages_of", lambda sha: pages("policy.txt", "dixon_p474.txt", "dixon_p475.txt",
                                                            start=473))
    monkeypatch.setattr(cl, "model_rows", lambda text, llm: [dict(r) for r in DIXON_ROWS])
    monkeypatch.setattr(cl, "_store", lambda: cl.CLStore(tmp_path / "m.db"))
    r = cl.extract("DIXON", llm=StubLLM())
    assert r["status"] == "verified" and r["page"] == 474
    assert r["totals"]["disputed_claims"]["current"] == 530.27
    assert cl.stored("DIXON")["status"] == "verified"


def test_a_guarantee_for_others_must_name_someone_else():
    # DIXON FY26, real model output: bank guarantees 'on behalf of the Group' (its own) and its own customs
    # bond landed in the red-flag class; only the bond given 'on behalf of the joint venture company' belongs
    rows = [{"label": "i. Guarantees issued by bankers on behalf of the Group",
             "category": "guarantee_for_others", "current": "13,611", "previous": "2,647"},
            {"label": "a) Bond given to custom department on behalf of the joint venture company",
             "category": "guarantee_for_others", "current": "100", "previous": "100"},
            {"label": "b) Bond given to custom department under authorised economic operator",
             "category": "guarantee_for_others", "current": "18,555", "previous": "18,756"}]
    assert [r["category"] for r in cl.check_categories(rows)] == [
        "bank_guarantee_own", "guarantee_for_others", "other_contingent"]


def test_a_table_of_small_whole_numbers_is_still_a_table():
    # MARICO FY26: 'Income tax 324 336' - no commas, no decimals - read as no figures at all
    note = cl.locate_note(pages("marico_p241.txt"))
    assert note is not None and note.unit == "crore" and "Income tax 324 336" in note.text


def test_one_figure_is_enough_beside_the_canonical_phrase_but_nil_prose_is_no_table():
    # PERSISTENT: each claim is a paragraph ending in its amount; SWIGGY: 'amounted to Rs 0 Crore'
    note = cl.locate_note(pages("persistent_p393.txt"))
    assert note is not None and note.unit == "million"
    assert cl.locate_note(pages("swiggy_p142.txt")) is None


def test_the_latest_report_with_extracted_text_is_read(tmp_path, monkeypatch):
    # NTPC: its FY26 report is on file but was never extracted; the FY25 one was
    import sqlite3
    db = tmp_path / "f.db"
    with sqlite3.connect(db) as con:
        con.executescript("""
            CREATE TABLE filing_docs (doc_key TEXT, content_sha TEXT, symbol TEXT, doc_type TEXT,
                                      doc_date TEXT, fiscal_period TEXT);
            CREATE TABLE filing_texts (content_sha TEXT, pages INTEGER);
            INSERT INTO filing_docs VALUES ('a', 'fy26', 'NTPC', 'annual_report', '2026-08-04', 'FY26');
            INSERT INTO filing_docs VALUES ('b', 'fy25', 'NTPC', 'annual_report', '2025-08-07', 'FY25');
            INSERT INTO filing_texts VALUES ('fy25', 374);""")
    monkeypatch.setattr(cl.settings, "filings_db", db)
    assert cl.latest_report("NTPC")["content_sha"] == "fy25"


def test_a_guarantee_on_behalf_of_a_named_outside_body_stays_for_others():
    # MARICO FY26: corporate guarantees given to banks on behalf of Broadcast Audience Research Council
    barc = "Corporate guarantees given to banks on behalf of Broadcast Audience Research Council (BARC)"
    row = {"label": barc,
           "category": "guarantee_for_others", "current": "1", "previous": "1"}
    assert cl.check_categories([row])[0]["category"] == "guarantee_for_others"


def test_the_rupee_sign_read_as_I_still_names_the_unit():
    # BHEL FY26 prints '(I in Crore)' - one more glyph the rupee sign extracts as
    header = "Contingent Liabilities and Commitments\n(I in Crore)\nParticulars"
    assert cl.detect_unit(header, "", []) == "crore"


APL_ROWS = json.loads((NOTES / "aplapollo_rows.json").read_text())


def test_printed_group_subtotals_are_not_counted_twice():
    # APL APOLLO FY26, real model output: each group's subtotal came back as an unlabelled row and was added
    # in again (claims Rs 110.91 Cr against a printed total of Rs 58.30 Cr); its income-tax subtotal is a typo
    # (1.58 for 7.27), so only the empty label gives that one away
    rows = cl.mark_subtotals(APL_ROWS)
    subtotals = [r["current"] for r in rows if r["category"] == "subtotal"]
    assert subtotals == ["0.61", "15.81", "1.58", "34.61", "58.30"]
    assert cl.totals(rows, "crore")["disputed_claims"] == {"current": 58.3, "previous": 56.48}
    notes = cl.reconcile(rows)                           # the report's 1.58 typo is noted, not blocking:
    assert len(notes) == 1 and notes[0].startswith("note:")   # its grand total matches the items
    assert cl.status_of(rows, [], "crore", notes) == "verified"


def test_a_total_that_does_not_reconcile_is_never_verified():
    kept, dropped = cl.verify(APL_ROWS, (NOTES / "aplapollo_p392.txt").read_text())
    assert cl.reconcile(kept) and cl.status_of(kept, dropped, "crore", cl.reconcile(kept)) == "partial"


def test_a_share_of_joint_ventures_contingent_liabilities_is_not_a_claim():
    # ASHOKLEY FY26, real model output: 'b) Share of contingent liabilities of joint ventures 1.60 3.76' was
    # put among disputed claims - it is the group's slice of its JVs' contingents, not its own disputes
    row = {"label": "b) Share of contingent liabilities of joint ventures", "category": "disputed_other",
           "current": "1.60", "previous": "3.76"}
    assert cl.check_categories([row])[0]["category"] == "other_contingent"


# --- a guarantee for someone else that the note PRINTS must be among the rows ---------------------------
# The verifier checks that every number in a row is on the page; nothing checked the other direction. Two
# gold reports had a guarantee in a sentence below the table that the model left out, and still read as
# verified.
ASHOKLEY_TAIL = ("3.11.1 Others\nThe value of corporate guarantee given to bank for the loan taken by a "
                 "joint venture company for the Year ended March 31, 2026 is\n` Nil crores (March 31, 2025: "
                 "` 0.75 crores).\n3.12 COMMITMENTS\nAs at\nMarch 31, 2026\n")


def test_a_guarantee_printed_below_the_table_but_left_out_of_the_rows_blocks_verified():
    # DIXON FY26 p.475: '(ii) The Group had provided corporate guarantees of ` 56,793 lakhs to two suppliers
    # on behalf of subsidiary' - the model returned the table only
    rows = [r for r in DIXON_ROWS if r.get("current") != "56,793"]
    gaps = cl.guarantee_gaps(rows, DIXON_TEXT)
    assert len(gaps) == 1 and "56,793" in gaps[0]
    assert cl.status_of(rows, [], "lakh", gaps) == "partial"


def test_a_joint_venture_loan_guarantee_stated_in_a_sentence_must_be_a_row():
    row = {"label": "Corporate guarantee for a joint venture's loan", "category": "guarantee_for_others",
           "current": None, "previous": "0.75"}
    assert cl.guarantee_gaps([row], ASHOKLEY_TAIL) == []
    assert "0.75" in cl.guarantee_gaps([], ASHOKLEY_TAIL)[0]


def test_breakdown_footnotes_numbered_items_headings_and_guarantee_fees_are_not_missing_guarantees():
    # ABCAPITAL FY26: 'Includes ...' footnotes break down the row 'Guarantees given by the Group' (1,000.26)
    abcapital = ("i) Includes Corporate Guarantees given to the National Housing Bank on behalf of its "
                 "subsidiary Aditya Birla Housing Finance\nLimited (ABHFL) of ₹ 3,000 crore as at 31st March "
                 "2026 (Previous Year: as at 31st March 2025 is ₹ 3,500 crore), against which\nthe amount "
                 "outstanding in the books of ABHFL as at 31st March 2026 is ₹ 862.81 crore.\n")
    group_row = {"label": "Guarantees given by the Group (Refer Note below)", "category": "other_contingent",
                 "current": "1,000.26", "previous": "1,550.33"}
    assert cl.guarantee_gaps([group_row], abcapital) == []
    # JSWENERGY FY26: the item '3] Others:' starts a new sentence - its land area is not a guarantee amount
    jsw = ("adequacy test for likely deficiency / defaults by the entities on whose behalf the Group has "
           "given guarantees.\n3] Others:\nIn respect of land parcels admeasuring 30.73 hectares (as at 31st "
           "March, 2025, 30.73 hectares), acquired by\nthe Group, the claim by certain parties towards title "
           "disputes are not currently ascertainable.\n")
    assert cl.guarantee_gaps([], jsw) == []
    # a numbered heading is not an amount
    heading = "3.12 Guarantees given on behalf of subsidiaries 120.50 98.20\n"
    sub_row = {"label": "Guarantees given on behalf of subsidiaries", "category": "guarantee_for_others",
               "current": "120.50", "previous": "98.20"}
    assert cl.guarantee_gaps([sub_row], heading) == []
    # a fee earned or paid on a guarantee is not a guarantee given (illustrative wording)
    fee = "Service tax demand on guarantee commission received from subsidiaries ` 20.73 crores.\n"
    assert cl.guarantee_gaps([], fee) == []


def test_extract_keeps_a_note_with_a_left_out_guarantee_from_being_verified(tmp_path, monkeypatch):
    monkeypatch.setattr(cl, "latest_report", lambda symbol: {"content_sha": "sha", "doc_date": "2026-09-03",
                                                              "fiscal_period": "FY26"})
    monkeypatch.setattr(cl, "pages_of", lambda sha: pages("policy.txt", "dixon_p474.txt", "dixon_p475.txt",
                                                            start=473))
    table_only = [dict(r) for r in DIXON_ROWS if r.get("current") != "56,793"]
    monkeypatch.setattr(cl, "model_rows", lambda text, llm: table_only)
    monkeypatch.setattr(cl, "_store", lambda: cl.CLStore(tmp_path / "m.db"))
    r = cl.extract("DIXON", llm=object())
    assert r["status"] == "partial" and any("56,793" in w for w in r["warnings"])


def test_an_item_with_no_previous_year_figure_is_not_shown_as_zero():
    # DIXON's note (ii) states this year's guarantee only; 'previous Rs 0.00 Cr' would read as a new guarantee
    items = cl.summarize_cl(_result(), 4676.67, lender=False)["largest_items"]
    footnote = next(i for i in items if i["label"].startswith("(ii) Corporate guarantees"))
    assert footnote["current_cr"] == 567.93 and footnote["previous_cr"] is None


def test_a_banks_guarantees_for_its_constituents_are_business_not_the_red_flag_class():
    # RBI's schedule line for a bank's customer guarantees (AUBANK FY26, extractor v4); the spec treats a
    # lender's guarantees for customers as normal business
    row = {"label": "II. Guarantees given on behalf of Constituents (a) In India",
           "category": "guarantee_for_others", "current": "5,40,10,707", "previous": "4,13,63,420"}
    assert cl.check_categories([row])[0]["category"] == "other_contingent"


def test_guarantees_for_other_parties_stay_guarantees_for_others():
    # JSWENERGY FY26: 'Related party 599.80 / Other parties 5.00 / Total 604.80' - 'Other parties' are
    # someone else
    row = {"label": "Other parties", "category": "guarantee_for_others", "current": "5.00", "previous": None}
    assert cl.check_categories([row])[0]["category"] == "guarantee_for_others"


def test_an_amount_written_with_the_notes_own_unit_is_verified_without_the_word():
    # SRF FY26: the model returned '6.54 crores' for a sentence that prints '` 6.54 crores'
    text = "Amount deposited against contingent liability ` 6.54 crores (previous year ` 9.05 crore)."
    row = {"label": "Amount deposited", "category": "paid_under_protest", "current": "6.54 crores",
           "previous": "` 9.05 crore"}
    kept, dropped = cl.verify([row], text, "crore")
    assert len(kept) == 1 and dropped == [] and cl._value(kept[0]["current"]) == 6.54


def test_an_amount_written_in_another_unit_than_the_note_is_dropped():
    # in a lakh note, a crore figure taken as lakh would be 100 times too small
    text = "Corporate guarantee given on behalf of a subsidiary ` 6.54 crores."
    row = {"label": "Corporate guarantee for a subsidiary", "category": "guarantee_for_others",
           "current": "6.54 crores", "previous": None}
    kept, dropped = cl.verify([row], text, "lakh")
    assert kept == [] and len(dropped) == 1


def test_the_summary_gives_the_total_of_every_contingent_liability_against_net_worth():
    # the live DIXON report added the four lines itself and quoted the per-line 11-12% for the Rs 2,084 Cr sum
    s = cl.summarize_cl(_result(), 4676.67, lender=False)
    assert s["all_contingents_cr"] == {"current": 2084.1, "previous": 1339.8}
    assert s["all_contingents_pct_of_net_worth"] == 44.6

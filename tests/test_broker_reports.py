"""Broker research PDF extraction.

Every case here comes from something a real report did during development. Broker PDFs are
dashboards, not documents, and each of these behaviours was wrong on the first pass.
"""
import pytest

from pennystock.data.broker_reports import (BrokerReport, Section, Table, _clean_label,
                                            _header_years, is_boilerplate,
                                            merge_listing_metadata, parse_cover,
                                            parse_estimates_from_text, to_chunks)


# --- boilerplate: roughly half of every report is legalese --------------------------------
DISCLAIMER = ("ANALYST CERTIFICATION I/We hereby certify. Terms & conditions and other "
              "disclosures. SEBI Registration Number INH000000990. Past performance is not "
              "necessarily a guide to future performance. Registered Office Address: Mumbai.")


def test_a_disclosure_page_is_dropped():
    assert is_boilerplate(DISCLAIMER, page_no=4, n_pages=5) is True


def test_a_cover_page_mentioning_disclosures_is_kept():
    """"Please refer to important disclosures at the end of this report" appears on page 1 of most
    reports. That page holds the rating, the target price and the whole investment case - dropping
    it because of one footer line would gut the document."""
    cover = ("Please refer to important disclosures at the end of this report. HOLD (Maintain) "
             "CMP: INR 11,275 Target Price: INR 12,138. Research Analyst: Navin Sahadeo.")
    assert is_boilerplate(cover, page_no=1, n_pages=5) is False


def test_two_markers_count_only_at_the_back_of_the_document():
    text = "Research analyst. Past performance is not a guide."
    assert is_boilerplate(text, page_no=28, n_pages=28) is True
    assert is_boilerplate(text, page_no=1, n_pages=28) is False


def test_analysis_pages_survive():
    assert is_boilerplate("EBITDA margins should expand to 19.1% by FY28E on better mix.",
                          page_no=2, n_pages=5) is False


# --- the cover block ------------------------------------------------------------------------
COVER = """HOLD 
(Maintain)
CMP: INR 11,275 Target Price: INR 12,138 (INR 11,900) 8%
04 September 2026 India | Equity Research | Company Update
Navin Sahadeo
navin.sahadeo@icicisecurities.com
Market Cap (INR) 3,323bn
"""


def test_cover_reads_rating_price_and_target():
    got = parse_cover(COVER)
    assert got["rating"] == "HOLD"
    assert (got["cmp"], got["target_price"]) == (11275.0, 12138.0)


def test_the_bracketed_target_is_the_PREVIOUS_one():
    """"Target Price: INR 12,138 (INR 11,900)" means the target was RAISED. A target that moved,
    and by how much, is the most actionable line in the report."""
    assert parse_cover(COVER)["target_price_prev"] == 11900.0


def test_rating_action_survives_an_icon_font_glyph():
    """The banner is literally "HOLD \\uf020\\n(Maintain)" - a private-use-area glyph from the
    icon font sits between them, and it is not whitespace, so it broke the match."""
    assert parse_cover(COVER)["rating_action"] == "Maintain"


def test_upside_is_computed_not_trusted():
    assert parse_cover(COVER)["upside_pct"] == pytest.approx(7.7, abs=0.1)


def test_market_cap_is_normalised_to_crore():
    """Reports quote bn; everything else in this system speaks Rs crore."""
    assert parse_cover(COVER)["market_cap_cr"] == 332300.0


def test_analyst_names_come_from_the_line_above_the_email():
    assert parse_cover(COVER)["analysts"] == ["Navin Sahadeo"]


# --- estimates: the payload, and the place a units bug is catastrophic ---------------------
def _pages(text):
    return [(1, text)]


def test_estimates_are_read_per_fiscal_year():
    est = parse_estimates_from_text(_pages(
        "Y/E March (INR mn) FY25A FY26A FY27E FY28E\n"
        "Net Revenue 7,59,551 8,85,115 10,11,687 11,33,031\n"
        "Adjusted EPS (Diluted) 208.6 282.3 320.4 378.5\n"))
    assert est["FY28E"]["eps"] == 378.5
    assert est["FY28E"]["revenue_cr"] == 113303.1        # 11,33,031 mn -> Rs crore


@pytest.mark.parametrize("unit,value,expected_cr", [
    ("INR mn", "1,90,568", 19056.8),      # ICICI Securities industrials
    ("INR bn", "112.6", 11260.0),         # ICICI Securities banking notes
    ("Rs cr", "5,000", 5000.0),           # ICICI Direct
])
def test_the_quoted_unit_is_honoured(unit, value, expected_cr):
    """A 1000x error hides here. AU SF Bank quotes NII in INR bn while UltraTech quotes EBITDA in
    INR mn; assuming either one turns Rs 11,260 crore into Rs 11.26 crore or vice versa."""
    est = parse_estimates_from_text(_pages(f"Y/E March ({unit}) FY27E FY28E\n"
                                           f"Net Profit {value} {value}\n"))
    assert est["FY28E"]["net_profit_cr"] == expected_cr


def test_a_unit_written_into_the_row_label_wins():
    est = parse_estimates_from_text(_pages("Y/E March FY27E FY28E\nNII (INR bn) 112.6 139.4\n"))
    assert est["FY28E"]["net_interest_income_cr"] == 13940.0


def test_a_margin_row_does_not_overwrite_the_amount_row():
    """"EBITDA" and "EBITDA (%)" are different rows. Stripping "(%)" from labels made the margin
    land in ebitda_cr, turning Rs 19,057 crore of EBITDA into 1.65."""
    est = parse_estimates_from_text(_pages(
        "Y/E March (INR mn) FY27E FY28E\nEBITDA 1,90,568 2,16,358\nEBITDA (%) 18.8 19.1\n"))
    assert est["FY28E"]["ebitda_cr"] == 21635.8
    assert est["FY28E"]["ebitda_margin_pct"] == 19.1


def test_bracketed_numbers_are_negative():
    est = parse_estimates_from_text(_pages("Y/E March FY27E FY28E\nEPS (Rs) (14.6) 18.2\n"))
    assert est["FY27E"]["eps"] == -14.6


def test_bank_rows_are_recognised():
    est = parse_estimates_from_text(_pages(
        "Y/E March FY27E FY28E\nGNPA (%) 1.6 1.7\nRoE (%) 17.2 18.6\nP/BV (x) 3.4 2.8\n"))
    assert est["FY28E"] == {"gnpa_pct": 1.7, "roe_pct": 18.6, "pb": 2.8}


# --- header detection: the subtle one -------------------------------------------------------
def test_a_real_header_is_recognised():
    assert _header_years("Y/E March (INR mn) FY25A FY26A FY27E FY28E") == \
        ["FY25A", "FY26A", "FY27E", "FY28E"]


def test_a_data_row_mentioning_years_is_not_a_header():
    """Live failure on AU SF Bank: "EPS % change YoY 4.8 24.8 42.2 29.3 Earnings Revisions (%)
    FY27E FY28E" was read as a new header, silently re-pointing every later row at two columns."""
    assert _header_years("EPS % change YoY 4.8 24.8 42.2 29.3 Earnings Revisions (%) FY27E FY28E") is None


def test_a_single_year_is_not_a_header():
    assert _header_years("Y/E March FY28E") is None


def test_label_cleaning_separates_the_unit():
    assert _clean_label(["Net", "Profit", "(INR", "bn)"]) == ("net profit", "bn")
    assert _clean_label(["EBITDA", "(%)"]) == ("ebitda (%)", None)


# --- listing metadata is authoritative for identity -----------------------------------------
def test_listing_metadata_overrides_a_bespoke_cover_layout():
    """Every broker styles its cover differently and chasing each one breaks on every restyle.
    The Trendlyne row already carries stock, broker, rating and target for all of them."""
    rep = merge_listing_metadata(BrokerReport(path="x.pdf"), {
        "stock": "NTPC", "broker": "Axis Direct", "reco_type": "Buy",
        "ltp": "332.50", "target": "363.00", "date": "04 Sep 2026"})
    assert (rep.company, rep.broker, rep.rating) == ("NTPC", "Axis Direct", "BUY")
    assert rep.upside_pct == pytest.approx(9.2, abs=0.1)


def test_a_pdf_target_that_disagrees_becomes_the_previous_target():
    rep = BrokerReport(path="x.pdf", target_price=11900.0)
    rep = merge_listing_metadata(rep, {"stock": "UltraTech", "target": "12138", "ltp": "11408"})
    assert (rep.target_price, rep.target_price_prev) == (12138.0, 11900.0)


def test_merge_survives_a_row_with_empty_fields():
    rep = merge_listing_metadata(BrokerReport(path="x.pdf", company="From PDF"),
                                 {"stock": "", "target": "", "ltp": "", "reco_type": ""})
    assert rep.company == "From PDF"


# --- chunks --------------------------------------------------------------------------------
def _report():
    return BrokerReport(
        path="/tmp/x.pdf", company="UltraTech Cement", broker="ICICI Securities",
        report_date="04 September 2026", rating="HOLD", cmp=11275.0, target_price=12138.0,
        target_price_prev=11900.0, upside_pct=7.7, analysts=["Navin Sahadeo"],
        estimates={"FY28E": {"eps": 378.5, "revenue_cr": 113303.1}},
        sections=[Section(heading="Ambitious plan", text="x" * 300, page=1)],
        tables=[Table(page=3, title="Exhibit 5: Profit & Loss", rows=[["Net Sales", "7,59,551"],
                                                                     ["EBITDA", "1,25,575"]])])


def test_every_chunk_is_stamped_with_company_broker_and_date():
    """A chunk retrieved from the middle of a report otherwise arrives with no idea who wrote it
    or when - and a two-year-old SELL read as current advice is worse than no retrieval."""
    for c in to_chunks(_report()):
        assert "UltraTech Cement" in c["text"] and "ICICI Securities" in c["text"]
        assert "04 September 2026" in c["text"]


def test_chunk_types_cover_recommendation_estimates_narrative_and_tables():
    kinds = {c["metadata"]["chunk_type"] for c in to_chunks(_report())}
    assert kinds == {"recommendation", "estimates", "narrative", "table"}


def test_the_recommendation_chunk_states_the_target_move():
    rec = next(c for c in to_chunks(_report()) if c["metadata"]["chunk_type"] == "recommendation")
    assert "12,138" in rec["text"] and "11,900" in rec["text"]


def test_a_table_chunk_keeps_labels_beside_their_numbers():
    """A number is meaningless without its row label; flattening tables into prose destroys that."""
    tbl = next(c for c in to_chunks(_report()) if c["metadata"]["chunk_type"] == "table")
    assert "Net Sales | 7,59,551" in tbl["text"]


def test_metadata_carries_what_a_query_needs_to_filter_on():
    md = to_chunks(_report())[0]["metadata"]
    assert md["doc_type"] == "broker_report"
    assert md["company"] == "UltraTech Cement" and md["rating"] == "HOLD"


def test_a_long_section_is_split_but_nothing_is_lost():
    rep = _report()
    rep.sections = [Section(heading="H", text="y" * 5000, page=1)]
    narrative = [c for c in to_chunks(rep, max_chars=1000) if c["metadata"]["chunk_type"] == "narrative"]
    assert len(narrative) >= 5
    assert sum(c["text"].count("y") for c in narrative) == 5000

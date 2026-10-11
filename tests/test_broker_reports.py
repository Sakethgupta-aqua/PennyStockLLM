"""Broker research PDF extraction.

Every case here comes from something a real report did during development. Broker PDFs are
dashboards, not documents, and each of these behaviours was wrong on the first pass.
"""
import pytest

from pennystock.data.broker_reports import (BrokerReport, EstimateRow, Section, Table,
                                            _clean_label, _header_years, is_boilerplate,
                                            merge_listing_metadata, parse_cover,
                                            parse_estimate_rows, parse_estimates_from_text,
                                            period_info, split_sections, to_chunks)


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


# --- Layer 1: sector KPIs, the table shapes and units the first parser could not read --------
def test_quarter_columns_make_a_header():
    """AU SF Bank's richest table - NIM, CRAR, GNPA, NNPA, PCR - is headed by quarters. Quarter
    tokens were not recognised, so the whole table was silently skipped."""
    assert _header_years("Particulars Key Ratios Q1FY27 Q1FY26 Q4FY26 FY26") == \
        ["Q1FY27", "Q1FY26", "Q4FY26", "FY26"]


@pytest.mark.parametrize("line,expected", [
    ("1QFY27 2QFY27 3QFY27", ["Q1FY27", "Q2FY27", "Q3FY27"]),     # Motilal Oswal
    ("FY2025 FY2026", ["FY25", "FY26"]),                            # four-digit years
    ("FY22 FY25 FY30e", ["FY22", "FY25", "FY30E"]),                  # lower-case suffix
])
def test_period_spellings_are_normalised(line, expected):
    assert _header_years(line) == expected


def test_bare_years_under_a_march_year_end_are_fiscal_years():
    """Motilal Oswal heads every 'Financials and valuation' table with bare years. Not reading them
    left 7 of 15 Motilal reports in the first live batch with zero numbers - nine years of P&L,
    balance sheet and bank ratios each."""
    assert _header_years("Y/E March 2020 2021 2025 2026 2027E 2028E") == \
        ["FY20", "FY21", "FY25", "FY26", "FY27E", "FY28E"]


def test_bare_years_under_another_year_end_stop_the_previous_header():
    """A December company's '2026' is calendar 2026, not FY26, so its columns cannot be mapped -
    but the line is still a header, and the rows under it must not inherit the previous columns."""
    assert _header_years("Y/E December 2025 2026E 2027E") == []


def test_bare_years_without_a_year_end_label_are_not_a_header():
    assert _header_years("Shareholding pattern 2024 2025 2026") is None


def test_a_motilal_financials_table_is_read_with_its_caption_unit():
    """Live Bharat Dynamics note, 14 Aug 2026: nothing from this table reached the KPI store."""
    est = parse_estimates_from_text(_pages(
        "Income Statement (Standalone) (INR m)\n"
        "Y/E March 2020 2021 2022 2023 2024 2025 2026 2027E 2028E\n"
        "Net Sales 31,049 19,138 28,174 24,894 23,693 33,451 24,418 42,347 61,204\n"
        "Change (%) 1.2 -38.4 47.2 -11.6 -4.8 41.2 -27.0 73.4 44.5\n"))
    assert est["FY28E"]["revenue_cr"] == 6120.4
    assert est["FY26"]["revenue_cr"] == 2441.8


def test_two_labelled_tables_on_one_line_are_read_as_two_rows():
    """Live BOB Capital Ashok Leyland note, p7: the income statement and the per-share table share
    every line, each under its own 'Y/E 31 Mar' label, so no header was found and the whole report
    gave 3 numbers. Money stays out: one table is Rs mn, the other per share, on the same line."""
    est = parse_estimates_from_text(_pages(
        "Income Statement Per Share\n"
        "Y/E 31 Mar (Rs mn) FY25A FY26P FY27E FY28E FY29E Y/E 31 Mar (Rs) FY25A FY26P FY27E FY28E FY29E\n"
        "EBITDA 49,306 57,322 58,183 66,202 75,971 Adjusted EPS 11.3 6.1 6.6 7.6 8.9\n"
        "Depreciation 7,193 7,152 7,317 7,522 7,953 Dividend per share 5.0 3.5 3.5 3.5 3.5\n"
        "EBIT 44,615 53,322 53,970 61,938 71,818 Book value per share 39.2 22.3 25.6 29.3 34.1\n"
        "Adjusted P/E 15.3 28.3 25.9 22.5 19.3\n"
        "EV/EBITDA 11.0 9.2 8.9 7.7 6.9\n"))
    assert est["FY29E"] == {"eps": 8.9, "dps": 3.5, "book_value": 34.1, "pe": 19.3, "ev_ebitda": 6.9}


def test_a_left_row_is_not_lost_to_the_right_row_beside_it():
    """Under ICICI's repeated-years header, only the right-hand row of each line was read."""
    est = parse_estimates_from_text(_pages(
        "FY25A FY26A FY27E FY28E FY25A FY26A FY27E FY28E\n"
        "GNPA (%) 2.3 2.1 1.9 1.8 RoE (%) 12.0 13.5 15.2 16.0\n"))
    assert est["FY28E"] == {"gnpa_pct": 1.8, "roe_pct": 16.0}


def test_digits_split_by_the_text_layer_are_never_read_as_values():
    """ICICI Direct's text layer prints 15,123 as '1 5,123'. Eight tokens for four columns cannot
    be mapped, so that row is dropped - while the clean row beside it, and a label that merely
    contains a digit ('Tier 1'), are still read."""
    est = parse_estimates_from_text(_pages(
        "(Year-end March) FY25 FY26E FY27E FY28E (Year-end March) FY25 FY26E FY27E FY28E\n"
        "Total operating Income 1 5,123 1 6,812 1 9,674 2 2,691 EPS (₹) 19.1 22.8 34.6 47.2\n"
        "Tier 1 (%) 14.1 14.5 15.0 15.2 Others 342 290 -161 -168\n"))
    assert est["FY28E"] == {"eps": 47.2, "tier1_pct": 15.2}


def test_a_row_holding_an_impossible_value_is_dropped_whole():
    """Live Motilal Ashok Leyland note: the PDF's text layer pairs 'EBITDA Margins (%)' with the
    EBITDA GROWTH row (57,322 / 49,306 = +16.3%), so FY27E read a 2.9% margin for a 12% business.
    A 194.7% margin cannot exist, which proves the label is on the wrong numbers - all of them."""
    est = parse_estimates_from_text(_pages(
        "Income Statement (INR M)\n"
        "Y/E March 2024 2025 2026 2027E\n"
        "EBITDA 46,066 49,306 57,322 58,961\n"
        "EBITDA Margins (%) 194.7 7.0 16.3 2.9\n"
        "GNPA (%) 2.1 -12.4 1.8 1.7\n"))
    assert all("ebitda_margin_pct" not in v and "gnpa_pct" not in v for v in est.values())
    assert est["FY27E"]["ebitda_cr"] == 5896.1


def test_side_by_side_tables_with_the_same_years_share_one_header():
    """ICICI prints the P&L and the ratio table side by side, so the header line repeats the years.
    Both halves cover the same years, so one set describes every row."""
    assert _header_years("FY25A FY26A FY27E FY28E FY25A FY26A FY27E FY28E") == \
        ["FY25A", "FY26A", "FY27E", "FY28E"]


def test_a_header_with_scrambled_repeats_stops_the_previous_one():
    """'FY23 FY24 FY25 FY26 FY23 FY26' cannot be mapped to columns. Treating it as 'not a header'
    would leave the PREVIOUS header active and pin the next rows to the wrong years."""
    assert _header_years("FY23 FY24 FY25 FY26 FY23 FY26") == []
    text = ("Y/E March FY27E FY28E\nGNPA (%) 1.6 1.7\n"
            "FY23 FY24 FY25 FY26 FY23 FY26\nNNPA (%) 0.5 0.6\n")
    rows = parse_estimate_rows([(1, text)])
    assert {r.metric for r in rows} == {"gnpa_pct"}


KEY_RATIOS = """Key ratios for Margins, Cost, Balance Sheet and Asset Quality
Particulars Key Ratios Q1FY27 Q1FY26 Q4FY26 FY26
NIM 5.9% 5.4% 6.0% 5.7%
Margins Gross advance yield 13.7% 14.1% 13.8% 13.9%
Cost of funds 6.5% 7.1% 6.5% 6.8%
Cost to Income ratio (excl. CGFMU) 56.6% 53.7% 58.6% 57.4%
LCR 119.0% 123.0% 119.0% 120.0%
Tier-I ratio 17.1% 18.2% 16.9% 16.9%
CRAR 18.9% 20.0% 18.7% 18.7%
GNPA 2.1% 2.5% 2.0% 2.0%
NNPA 0.8% 0.9% 0.7% 0.7%
Asset Quality Credit cost (incl. CGFMU) 0.8% 1.4% 0.6% 1.0%
PCR (excl. technical w/off) 64.0% 65.0% 64.0% 64.0%
PCR (incl. technical w/off) 85.0% 83.0% 85.0% 85.0%
"""


def test_the_key_ratio_table_yields_every_lender_kpi():
    """Copied from AU SF Bank's report (ICICI Securities, 4 Sep 2026). Seven of the screen's
    declared lender blind spots are in this one table."""
    got = {(r.metric, r.period): r.value for r in parse_estimate_rows([(11, KEY_RATIOS)])}
    assert got[("nim_pct", "FY26")] == 5.7
    assert got[("cost_to_income_pct", "FY26")] == 57.4
    assert got[("tier1_pct", "FY26")] == 16.9
    assert got[("car_pct", "FY26")] == 18.7
    assert got[("gnpa_pct", "Q1FY27")] == 2.1
    assert got[("nnpa_pct", "FY26")] == 0.7
    assert got[("credit_cost_pct", "FY26")] == 1.0
    assert got[("pcr_pct", "FY26")] == 64.0
    assert got[("pcr_incl_writeoff_pct", "FY26")] == 85.0
    # yield, cost of funds and LCR are real lines but not KPIs this screen ranks on
    assert {m for m, _ in got} == {"nim_pct", "cost_to_income_pct", "tier1_pct", "car_pct",
                                   "gnpa_pct", "nnpa_pct", "credit_cost_pct", "pcr_pct",
                                   "pcr_incl_writeoff_pct"}


def test_a_kpi_keeps_the_basis_it_was_printed_with():
    """Brokers compute NIM on different bases and say so in the label. The label IS the basis, so
    it is kept - throwing it away is what makes four brokers' NIMs look comparable when they are not."""
    text = "Y/E March FY27E FY28E\nNIM on AUM (%) 11.0 10.8\nNIMs (Cal) 10.3 10.1\n"
    rows = parse_estimate_rows([(1, text)])
    assert [(r.metric, r.label) for r in rows if r.period == "FY28E"] == \
        [("nim_pct", "nim on aum (%)"), ("nim_pct", "nims (cal)")]


@pytest.mark.parametrize("label", ["Passenger car (%)", "MFI credit cost", "CASA deposits",
                                   "Credit card spends", "Gross advance yield", "Car volumes"])
def test_lookalike_lines_are_not_kpis(label):
    """A segment's credit cost is not the company's credit cost, and an auto maker's 'car' row is
    not a capital adequacy ratio."""
    assert parse_estimate_rows([(1, f"Y/E March FY27E FY28E\n{label} 1.5 1.6\n")]) == []


def test_credit_cost_in_basis_points_becomes_a_percentage():
    rows = parse_estimate_rows([(1, "Y/E March FY27E FY28E\nCredit cost (bps) 80 75\n")])
    assert [(r.period, r.value) for r in rows] == [("FY27E", 0.8), ("FY28E", 0.75)]


@pytest.mark.parametrize("caption,expected_cr", [
    ("Income statement (INR m)", 17134.6),
    ("FinancialOverview:-(ConsolidatedFiguresRsinMillion)", 17134.6),     # Joindre, squashed
    ("Financials &Valuations (INR b)", 17134600.0),
    ("Exhibit 23: Profit and loss ₹ crore", 171346.0),
])
def test_a_unit_in_the_caption_above_the_header_is_honoured(caption, expected_cr):
    """Trent's revenue was stored as Rs 1,71,346 crore - ten times too large - because 'Rs in
    Million' sat in the caption line, not on the header line."""
    est = parse_estimates_from_text(_pages(f"{caption}\nY/E March FY25 FY26\n"
                                           "Net Sales 1,71,346 2,00,742\n"))
    assert est["FY25"]["revenue_cr"] == expected_cr


def test_money_with_an_unknown_unit_is_dropped_not_assumed():
    """Assuming crore is how a mn table became 10x too large and a bn table 100x too small. A
    ratio or a per-share figure needs no unit, so those rows survive."""
    est = parse_estimates_from_text(_pages("Revenue Model\nY/E March FY25 FY26\n"
                                           "Revenue 28,866 38,994\nEPS -1.1 -0.1\n"))
    assert est == {"FY25": {"eps": -1.1}, "FY26": {"eps": -0.1}}


def test_a_data_row_above_the_header_is_not_mistaken_for_its_caption():
    est = parse_estimates_from_text(_pages("12M Avg Val (INR M) 2403\nY/E March FY26E FY27E\n"
                                           "Revenue 39.0 49.4\n"))
    assert est == {}


def test_rows_carry_page_label_period_and_value():
    rows = parse_estimate_rows([(3, "Y/E March FY26A FY27E\nGNPA (%) 2.0 1.6\n")])
    assert rows == [EstimateRow(page=3, label="gnpa (%)", metric="gnpa_pct", period="FY26A", value=2.0),
                    EstimateRow(page=3, label="gnpa (%)", metric="gnpa_pct", period="FY27E", value=1.6)]


@pytest.mark.parametrize("token,report_date,expected", [
    ("FY26A", "2026-09-04", ("FY26", "actual", "2026-03-31")),
    ("FY27E", "2026-09-04", ("FY27", "estimate", "2027-03-31")),
    # no suffix: actual only once SEBI's 60-day annual filing deadline has passed
    ("FY26", "2026-09-04", ("FY26", "actual", "2026-03-31")),
    ("FY26", "2026-04-20", ("FY26", "estimate", "2026-03-31")),
    # quarters: 45-day deadline; Q1 of FY27 ends 30 Jun 2026
    ("Q1FY27", "2026-09-04", ("Q1FY27", "actual", "2026-06-30")),
    ("Q1FY27", "2026-07-20", ("Q1FY27", "estimate", "2026-06-30")),
    ("Q3FY27", "2026-09-04", ("Q3FY27", "estimate", "2026-12-31")),
    ("Q4FY26", "2026-09-04", ("Q4FY26", "actual", "2026-03-31")),
    ("H1FY27", "2026-12-20", ("H1FY27", "actual", "2026-09-30")),
    ("9MFY26", "2026-09-04", ("9MFY26", "actual", "2025-12-31")),
])
def test_period_info_reads_the_indian_fiscal_calendar(token, report_date, expected):
    assert period_info(token, report_date) == expected


def test_side_by_side_tables_keep_ratios_but_drop_money():
    """Live AU SF Bank page 35: the P&L (INR mn) sits beside a ratio table under one caption, so
    'Total Income 7.4' - a ratio from the RIGHT table - was scaled as INR mn into revenue. Which
    half a money row belongs to cannot be told from the text, so money is dropped; ratios are
    unit-free and safe."""
    text = ("(INR mn, year ending March)\n"
            "FY25A FY26A FY27E FY28E FY25A FY26A FY27E FY28E\n"
            "NIMs (%) 5.8 5.3 5.4 5.6\nTotal Income 7.4 6.9 7.0 7.1\n"
            "Net Profit 21,059 26,413 37,569 48,582\n")
    rows = parse_estimate_rows([(35, text)])
    assert {r.metric for r in rows} == {"nim_pct"}


def test_every_kpi_and_headline_estimate_has_a_display_name():
    from pennystock.data.broker_reports import FORWARD_ESTIMATES, KPI_LABELS, KPI_METRICS
    assert set(KPI_LABELS) == set(KPI_METRICS) | set(FORWARD_ESTIMATES)


# --- sections: a chunk must be a whole argument, not half a sentence ---------------------------
AU_PROSE = """Reflecting this trend, the gross slippage ratio moderated to sub-2% by June'26, from
its peak of 3.5% in Jun'25, while credit cost stabilised at ~1% despite global
uncertainties during H2FY26. Consequently, RoA expanded to 1.8% by Q4FY26 and
remained stable through Q1FY27.
Management has reiterated its full-FY27 RoA guidance at 1.8%, supported by
operating leverage and contained credit cost."""


def test_a_wrapped_line_is_not_mistaken_for_a_heading():
    """Live AU SF Bank chunk: 'uncertainties during H2FY26. Consequently, RoA expanded to 1.8% by
    Q4FY26 and' is short and unpunctuated - exactly like a heading - so the paragraph was cut there
    and retrieval returned '...by Q4FY26 and. remained stable', a sentence with its middle missing."""
    secs = split_sections(AU_PROSE, page_no=10)
    assert len(secs) == 1
    assert "1.8% by Q4FY26 and remained stable through Q1FY27." in secs[0].text


def test_a_heading_after_a_finished_paragraph_still_starts_a_section():
    text = AU_PROSE + "\nCredit cost normalisation to drive RoA\nThe bank expects MFI stress to " \
                      "ease over the next two to four quarters as collections improve further."
    secs = split_sections(text, page_no=10)
    assert [s.heading for s in secs] == ["", "Credit cost normalisation to drive RoA"]


def test_a_section_just_over_one_chunk_folds_its_short_tail_in():
    """`pieces[-2] += pieces.pop()` resolves [-2] BEFORE the pop, so with exactly two pieces it
    indexes past a one-element list. Longer, unbroken sections made this real on AU SF Bank."""
    rep = _report()
    rep.sections = [Section(heading="H", text="z" * 1030, page=1)]
    narrative = [c for c in to_chunks(rep, max_chars=1000) if c["metadata"]["chunk_type"] == "narrative"]
    assert len(narrative) == 1 and narrative[0]["text"].count("z") == 1030


CHART_RESIDUE = """RoE (%) Average RoE (%).
18.0 16.4 15.8 15.4 15.6 16.0 14.2 13.7 14 13.5 13.1 14.0 12 12.0 10.0 8.0 6.0 4.0 2.0 0.0
16.0
14.2 13.7
FY18 FY19 FY20 FY21 FY22 FY23 FY24 FY25 FY26 Q1FY27
1.2% AU Bank. 1.0% 1.0% 0.8% 0.6% 0.6% 0.4% 0.2% 0.0%
Source: Company data, I-Sec research
AU reported annualised net credit cost at 1.3% of total average assets and 1.7% of
average GLP in FY25, higher than its steady state because of unsecured delinquencies."""


def test_chart_axes_and_source_captions_are_not_prose():
    """A chart's axis labels extract as lines of bare numbers and years. Left in, they were the
    first thing retrieval returned for 'why will credit cost fall'. The numbers that matter are in
    broker_metrics; the prose is what this text is for."""
    text = split_sections(CHART_RESIDUE, page_no=10)[0].text
    assert "18.0 16.4" not in text and "FY18 FY19" not in text and "0.8% 0.6%" not in text
    assert "16.0" not in text and "14.2 13.7" not in text          # labels printed one per line
    assert "Source:" not in text
    assert "net credit cost at 1.3% of total average assets and 1.7% of average GLP" in text

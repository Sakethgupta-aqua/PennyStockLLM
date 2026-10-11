"""The fact sheet: one line per fact with its source, every comparison made, gaps first.

Fixture: every pack tool's real output for DIXON, captured 2026-10-11 after Steps 2-3 (year-labelled
financials, the statutory pledge, verified notes, filed-anchored broker figures).
"""
import json
import re
from datetime import date
from pathlib import Path

from pennystock.agents.evidence import PackCall, parse_output
from pennystock.agents.fact_sheet import render_fact_sheet
from pennystock.config import settings

RAW = json.loads((Path(__file__).parent / "fixtures" / "dixon" / "tool_outputs.json").read_text())


def call(name, output=None, ok=True):
    out = RAW[name] if output is None else output
    return PackCall(name, {"symbol": "DIXON"}, out, parse_output(out) if ok else None, ok, 1.0)


def sheet(**over):
    calls = {n: call(n) for n in RAW}
    calls.update(over)
    return render_fact_sheet("DIXON", "Dixon Technologies (India) Limited", calls, today=date(2026, 10, 11))


def test_the_header_and_gaps_come_first_and_the_sheet_fits_its_budget():
    s = sheet()
    assert s.startswith("EVIDENCE PACK - Dixon Technologies (India) Limited (DIXON)")
    assert s.index("### DATA GAPS") < s.index("### PRICE & VALUATION")
    assert len(s) <= settings.evidence_max_chars


def test_valuation_is_worded_against_its_own_history_and_its_peers():
    s = sheet()
    assert "BELOW its historical range" in s
    # from the fields, not the tool's headline '(46.3 vs 27.65)', which the live fact-check misread
    assert "against the sector median of 27.65 - 67.4% ABOVE it" in s


def test_growth_is_labelled_by_year_with_the_filed_figures_beside_it():
    s = sheet()
    assert "FY23 12,173.35" in s and "FY22" not in s and "over FY23->FY26 (3 years)" in s
    assert "profit to shareholders Rs 1,438.64 Cr" in s and "net worth Rs 4,676.67 Cr" in s


def test_forensic_scores_are_stated_with_their_caveats():
    s = sheet()
    assert "Piotroski F-score 6/9 (MIDDLING)" in s and "Altman Z''/EM 5.03 (SAFE)" in s
    assert "Beneish M-score -2.13 (no manipulation signal" in s


def test_off_balance_sheet_items_are_in_crore_with_their_yardsticks():
    s = sheet()
    assert "disputed claims Rs 530.27 Cr" in s and "11.3% of net worth" in s
    assert "BHARTI AIRTEL LTD [group_or_partner" in s and "promoter / KMP entities Rs 0.00 Cr" in s


def test_owners_show_the_statutory_pledge():
    expected = "Promoter pledge: NONE - no promoter shares pledged (statutory shareholding filing, Jun 2026)"
    assert expected in sheet()


def test_brokers_never_mix_bases():
    s = sheet()
    assert "Net profit cr FY26 (reported): filed 1,438.64" in s and "ADJUSTED" in s
    assert "EPS FY26 (reported; each broker's basis taken from its profit figure)" in s


def test_technicals_say_which_side_of_each_average_the_price_is():
    s = sheet()
    assert "ABOVE its 200-day average Rs 12,097" in s and "BELOW its 50-day Rs 13,844" in s


def test_alternatives_come_from_the_screen_without_the_company_itself():
    s = sheet()
    block = s.split("### ALTERNATIVES")[1]
    assert "Oracle Financial Services Software Limited (OFSS)" in block and "(DIXON)" not in block
    # Yahoo's sector is broad (DIXON's Technology holds IT services): the sheet must say they are not peers
    assert "the Technology sector" in block and "own industry, Consumer Electronics" in block


def test_a_failed_source_is_listed_under_data_gaps():
    s = sheet(get_forensics=call("get_forensics", "Error: get_forensics failed (TimeoutError)", ok=False))
    gaps = s.split("### DATA GAPS")[1].split("###")[0]
    assert "trust checks (Piotroski, Altman, Beneish): UNKNOWN" in gaps
    assert "### CAN THE NUMBERS BE TRUSTED" not in s


def test_the_sheet_formats_tool_numbers_and_invents_none():
    s, blob = sheet(), " ".join(RAW.values())
    found = re.findall(r"-?\d[\d,]*\.?\d*", blob)
    tool_numbers = {float(x.replace(",", "")) for x in found if x.strip("-,.")}
    for x in re.findall(r"(?<![\w.])-?\d[\d,]*\.\d+", s):          # every decimal figure in the sheet
        v = float(x.replace(",", ""))
        assert any(abs(v - t) <= max(0.01, abs(t) * 0.0005) for t in tool_numbers), x


def test_a_figure_or_source_the_tool_did_not_give_is_not_printed_as_none_or_zero():
    notes = parse_output(RAW["get_annual_report_notes"])
    notes.pop("yardsticks")
    notes["contingent_liabilities"]["largest_items"][0]["previous_cr"] = None
    s = sheet(get_annual_report_notes=call("get_annual_report_notes", str(notes)))
    block = s.split("### OFF THE BALANCE SHEET")[1].split("###")[0]
    assert "[None]" not in block and "previous year: no figure printed" in block


def test_the_owners_line_names_the_pledge_quarter_and_a_missing_promoter_group():
    sh = parse_output(RAW["get_shareholding"])
    sh.update(promoter_pct=0.0, pledge_pct_of_promoter_holding=0.0, pledge_source="nse_shp_xbrl",
              as_of_quarter="Sep 2026", pledge_quarter="Jun 2026")
    s = sheet(get_shareholding=call("get_shareholding", str(sh)))
    assert "Promoter pledge: NONE - the company has no promoter group (statutory shareholding filing, " \
           "Jun 2026)" in s


def test_altman_is_shown_with_its_zone_boundaries_when_the_tool_gives_them():
    fx = parse_output(RAW["get_forensics"])
    fx["altman_emerging_markets"].update(safe_above=2.6, distress_below=1.1)
    s = sheet(get_forensics=call("get_forensics", str(fx)))
    assert "Altman Z''/EM 5.03 (SAFE; safe above 2.6, distress below 1.1)" in s


def test_the_total_of_the_contingent_liabilities_is_given_with_its_share_of_net_worth():
    notes = parse_output(RAW["get_annual_report_notes"])
    notes["contingent_liabilities"].update(all_contingents_cr={"current": 1516.17, "previous": 1339.8},
                                           all_contingents_pct_of_net_worth=32.4)
    s = sheet(get_annual_report_notes=call("get_annual_report_notes", str(notes)))
    assert "All contingent liabilities together: Rs 1,516.17 Cr (32.4% of net worth)" in s


def test_the_earnings_yield_is_shown_when_the_tool_gives_it():
    f = parse_output(RAW["get_stock_fundamentals"])
    f["earnings_yield_pct"] = 2.68
    s = sheet(get_stock_fundamentals=call("get_stock_fundamentals", str(f)))
    assert "earnings yield 2.68% (1 / trailing P/E)" in s

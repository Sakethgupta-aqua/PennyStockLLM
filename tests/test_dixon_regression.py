"""The Dixon question of 2026-10-10/11, offline: the errors the live runs made must not come back.

Each changed tool's output is rebuilt here with the real code on recorded inputs (Yahoo financials, the
statutory XBRL filings, the annual report's note pages, the broker estimates in market.db); the unchanged
tools keep their recorded outputs (tests/fixtures/dixon/tool_outputs.json, 2026-10-11). Network and models
are stubbed.
"""
import json
from datetime import date
from pathlib import Path

from pennystock.agents.evidence import PackCall, parse_output
from pennystock.agents.fact_sheet import render_fact_sheet
from pennystock.data import broker_metrics as bm
from pennystock.data import contingent_liabilities as cl
from pennystock.data import integrated_filings as itf
from pennystock.data.fundamentals import summarize_financials
from pennystock.data.ownership import Ownership, Quarter, summarize_ownership
from pennystock.data.ownership_providers import parse_shp_pledge
from pennystock.data.providers import Financials
from pennystock.data.related_parties import summarize_rpt

FIX = Path(__file__).parent / "fixtures"
RAW = json.loads((FIX / "dixon" / "tool_outputs.json").read_text())


def _call(name, data):
    return PackCall(name, {"symbol": "DIXON"}, str(data), parse_output(str(data)), True, 1.0)


def _contingent_liabilities() -> dict:
    """The note's rows through the production checks: verify, reconcile, printed guarantees left out."""
    rows = json.loads((FIX / "notes" / "dixon_note46_rows.json").read_text())
    text = "\n".join((FIX / "notes" / f"dixon_p{p}.txt").read_text() for p in (474, 475))
    kept, dropped = cl.verify(rows, text)
    warnings = cl.reconcile(kept) + cl.guarantee_gaps(kept, text)
    return {"fiscal_year": "FY26", "page": 474, "basis": "consolidated", "unit": "lakh",
            "status": cl.status_of(kept, dropped, "lakh", warnings), "rows": kept,
            "totals": cl.totals(kept, "lakh"), "warnings": warnings}


def build_sheet(monkeypatch) -> str:
    fin = summarize_financials(Financials(**json.loads((FIX / "dixon" / "financials.json").read_text())))
    pledge = parse_shp_pledge((FIX / "xbrl" / "dixon_shp_jun2026.xml").read_text())
    own = summarize_ownership(Ownership("DIXON", [Quarter("Jun 2026", promoter_pct=28.55,
                                                          pledge_pct=pledge["pledge_pct"],
                                                          pledge_pct_of_total=pledge["pledge_pct_of_total"],
                                                          pledge_source="nse_shp_xbrl")]))
    disclosure = itf.RptDisclosure("DIXON", "H2 FY26", "2025-10-01", "2026-03-31",
                                   itf.rpt_rows((FIX / "xbrl" / "dixon_rpt_h2fy26.xml").read_text()),
                                   "12-May-2026 23:38:14", "u", 21182.1,
                                   "FY26 revenue minus the April-September half")
    recorded = parse_output(RAW["get_annual_report_notes"])
    notes = {"symbol": "DIXON",
             "contingent_liabilities": cl.summarize_cl(_contingent_liabilities(), 4676.67, False),
             "related_party_transactions": summarize_rpt(disclosure), "yardsticks": recorded["yardsticks"]}
    obs = [bm.Observation(**o) for o in json.loads((FIX / "dixon" / "broker_obs.json").read_text())]
    monkeypatch.setattr(bm, "load_observations", lambda symbol=None, metrics=None, since=None: obs)
    filed = {("net_profit_cr", "FY26"): 1438.64, ("revenue_cr", "FY26"): 48872.8}
    consensus = {**parse_output(RAW["get_broker_consensus"]),
                 "forward_estimates": bm.anchored_estimates("DIXON", filed)}
    calls = {n: _call(n, parse_output(o)) for n, o in RAW.items()}
    calls.update(get_financials=_call("get_financials", fin), get_shareholding=_call("get_shareholding", own),
                 get_annual_report_notes=_call("get_annual_report_notes", notes),
                 get_broker_consensus=_call("get_broker_consensus", consensus))
    return render_fact_sheet("DIXON", "Dixon Technologies (India) Limited", calls, today=date(2026, 10, 11))


def test_contingent_liabilities_are_in_crore_and_against_net_worth(monkeypatch):
    # the live report said "around Rs 50 Cr" / "negligible": the model did lakh -> crore maths on raw text
    s = build_sheet(monkeypatch)
    assert "disputed claims Rs 530.27 Cr" in s and "11.3% of net worth" in s


def test_a_guarantee_stated_below_the_table_is_counted_and_not_given_a_zero_previous_year(monkeypatch):
    # note (ii) on p.475: guarantees of 56,793 lakh to two suppliers on behalf of a subsidiary, this year only
    s = build_sheet(monkeypatch)
    assert "Guarantees given for others: Rs 568.93 Cr (12.2% of net worth)" in s
    assert "Rs 567.93 Cr (previous year: no figure printed)" in s


def test_years_are_labelled_and_growth_has_its_real_span(monkeypatch):
    # the live report called the oldest of four years FY22 and its 3-year growth a 4-year CAGR
    s = build_sheet(monkeypatch)
    assert "FY23 12,173.35" in s and "FY22" not in s and "over FY23->FY26 (3 years)" in s


def test_pledge_is_a_statutory_none_not_unknown(monkeypatch):
    # the live report called the pledge unverified: NSE's pledge API was dead and nothing read the filing
    s = build_sheet(monkeypatch)
    assert "NONE - no promoter shares pledged (statutory shareholding filing, Jun 2026)" in s


def test_related_parties_are_classified_and_brokers_never_mix_bases(monkeypatch):
    # the live report summed subsidiaries with outsiders, and quoted adjusted broker profit as reported
    s = build_sheet(monkeypatch)
    assert "BHARTI AIRTEL LTD [group_or_partner" in s
    assert "Net profit cr FY26 (reported): filed 1,438.64" in s and "ADJUSTED" in s

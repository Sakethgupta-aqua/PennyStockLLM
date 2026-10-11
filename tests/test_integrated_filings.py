"""NSE integrated filings: audited figures and the half-yearly related-party disclosure (real Dixon XBRL)."""
from datetime import date
from pathlib import Path

from pennystock.data import integrated_filings as itf

FIX = Path(__file__).parent / "fixtures" / "xbrl"


def test_year_quarter_and_balance_sheet_figures_come_from_undimensioned_contexts():
    # the fixture also carries a synthetic segment row reusing RevenueFromOperations; it must be ignored
    fin = itf.financial_facts((FIX / "dixon_if_fy26_consolidated.xml").read_text())
    assert fin["revenue_cr"]["FY"] == {"end": "2026-03-31", "value": 48872.8}
    assert fin["revenue_cr"]["Q"]["value"] == 10510.51
    assert fin["profit_to_owners_cr"]["FY"]["value"] == 1438.64
    assert fin["equity_to_owners_cr"]["I"] == {"end": "2026-03-31", "value": 4676.67}
    assert fin["basic_eps"]["FY"]["value"] == 271.59 and fin["exceptional_cr"]["FY"]["value"] == 0.0
    assert fin["auditor"] == "S.N. DHAWAN & CO LLP"


def test_a_six_month_figure_is_read_as_the_half_year():
    fin = itf.financial_facts((FIX / "dixon_if_h1fy26_consolidated.xml").read_text())
    assert fin["revenue_cr"]["H"] == {"end": "2025-09-30", "value": 27690.7}
    assert fin["revenue_cr"]["Q"]["value"] == 14855.04


def test_related_party_rows_are_read_in_crore():
    rows = itf.rpt_rows((FIX / "dixon_rpt_h2fy26.xml").read_text())
    airtel = next(r for r in rows if "AIRTEL" in r.counterparty.upper() and r.amount_cr > 1)
    assert airtel.amount_cr == 1224.67 and airtel.relationship == "Entity having significant influence"
    assert airtel.entity.startswith("Dixon Electro Appliances") and len(rows) == 12


def test_quarter_ends_name_their_half_year():
    assert itf.quarter_end("31-MAR-2026") == date(2026, 3, 31)
    assert itf.half_label(date(2026, 3, 31)) == ("H2 FY26", "2025-10-01", "2026-03-31")
    assert itf.half_label(date(2025, 9, 30)) == ("H1 FY26", "2025-04-01", "2025-09-30")


LISTING = {"data": [
    {"qe_Date": "31-MAR-2026", "consolidated": "Consolidated", "type": itf.FINANCIALS,
     "broadcast_Date": "12-May-2026 23:38:17", "xbrl": "u/fy26c"},
    {"qe_Date": "31-MAR-2026", "consolidated": "Standalone", "type": itf.FINANCIALS,
     "broadcast_Date": "12-May-2026 23:38:14", "xbrl": "u/fy26s"},
    {"qe_Date": "30-SEP-2025", "consolidated": "Consolidated", "type": itf.FINANCIALS,
     "broadcast_Date": "17-Oct-2025 21:15:27", "xbrl": "u/h1c"},
    {"qe_Date": "31-MAR-2026", "consolidated": None, "type": "Integrated Filing- Governance",
     "xbrl": "u/gov"}]}
FILES = {"u/fy26c": "dixon_if_fy26_consolidated.xml", "u/fy26s": "dixon_rpt_h2fy26.xml",
         "u/h1c": "dixon_if_h1fy26_consolidated.xml"}


class FakeFeed:
    def __init__(self, listing=LISTING):
        self.listing, self.gets = listing, []

    def api(self, path):
        return self.listing

    def get(self, url):
        self.gets.append(url)
        return (FIX / FILES[url]).read_bytes() if url in FILES else None


def store(tmp_path, feed):
    return itf.IntegratedFilings(db_path=tmp_path / "m.db", feed_factory=lambda symbol: feed)


def test_the_filed_year_comes_from_the_march_consolidated_filing(tmp_path):
    fy = store(tmp_path, FakeFeed()).filed_year("DIXON")
    assert (fy.fiscal_year, fy.basis, fy.revenue_cr, fy.profit_to_owners_cr, fy.equity_to_owners_cr) == \
           ("FY26", "consolidated", 48872.8, 1438.64, 4676.67)
    assert fy.broadcast_at.startswith("12-May-2026")


def test_the_disclosure_is_the_half_year_with_its_own_revenue(tmp_path):
    d = store(tmp_path, FakeFeed()).rpt_disclosure("DIXON")
    assert (d.half, d.period_start, d.period_end) == ("H2 FY26", "2025-10-01", "2026-03-31")
    assert d.revenue_cr == 21182.1 and "minus" in d.revenue_note        # 48,872.80 - 27,690.70
    assert any("AIRTEL" in r.counterparty.upper() for r in d.rows)


def test_a_parsed_file_is_never_downloaded_twice(tmp_path):
    feed = FakeFeed()
    s = store(tmp_path, feed)
    s.filed_year("DIXON"), s.filed_year("DIXON")
    assert feed.gets.count("u/fy26c") == 1


def test_no_rows_and_no_feed_return_none(tmp_path):
    s = store(tmp_path, FakeFeed(listing=None))
    assert s.filed_year("BDL") is None and s.rpt_disclosure("BDL") is None


def test_a_status_read_uses_the_cache_of_any_age_and_never_the_network(tmp_path):
    # `ingest_integrated_filings.py --status` re-fetched every stale listing from NSE with no delay (2026-10-11)
    store(tmp_path, FakeFeed()).filed_year("DIXON")                       # warm the cache
    import sqlite3
    with sqlite3.connect(tmp_path / "m.db") as con:                       # a month old: past the weekly refresh
        con.execute("UPDATE integrated_listings SET fetched_at = '2026-09-01T00:00:00'")

    def no_network(symbol):
        raise AssertionError("a status read must not fetch")
    offline = itf.IntegratedFilings(db_path=tmp_path / "m.db", feed_factory=no_network, offline=True)
    assert offline.filed_year("DIXON").revenue_cr == 48872.8
    assert offline.filed_year("BDL") is None and offline.rpt_disclosure("BDL") is None

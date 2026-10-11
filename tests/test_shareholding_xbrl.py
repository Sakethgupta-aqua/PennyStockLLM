"""Promoter pledge from each quarter's shareholding filing (XBRL): NSE's pledge API returns nothing."""
from pathlib import Path

from pennystock.data.ownership import Ownership, Quarter, summarize_ownership
from pennystock.data.ownership_providers import NseOwnershipProvider, OwnershipStore, parse_shp_pledge

FIX = Path(__file__).parent / "fixtures" / "xbrl"


def test_no_pledge_is_a_statutory_zero():
    p = parse_shp_pledge((FIX / "dixon_shp_jun2026.xml").read_text())
    assert p["pledge_pct"] == 0.0 and p["pledged_shares"] == 0.0 and p["promoter_shares"] == 17438673.0
    assert p["encumbrance_note"] is None


def test_a_pledge_is_a_share_of_the_promoters_own_holding():
    p = parse_shp_pledge((FIX / "jaynecoind_shp_jun2026.xml").read_text())
    assert p["pledge_pct"] == 99.87 and p["pledge_pct_of_total"] == 55.08       # 534,804,690 / 535,519,366


def test_a_pledged_flag_without_a_count_is_unknown_not_zero():
    xml = (FIX / "jaynecoind_shp_jun2026.xml").read_text()
    xml = "\n".join(l for l in xml.splitlines() if "NumberOfSharesEncumberedUnderPledged" not in l)
    assert parse_shp_pledge(xml) is None


class _Resp:
    def __init__(self, payload=None, text=""):
        self._payload, self.text = payload, text

    def json(self):
        return self._payload


class _Session:
    def __init__(self):
        self.urls = []

    def get(self, url, timeout=None):
        self.urls.append(url)
        if "share-holdings-master" in url:
            return _Resp([{"date": "30-JUN-2026", "pr_and_prgrp": "28.55", "public_val": "71.45",
                           "xbrl": "https://x/SHP_jun.xml"},
                          {"date": "31-MAR-2026", "pr_and_prgrp": "28.68", "public_val": "71.32",
                           "xbrl": None}])
        if url.endswith("SHP_jun.xml"):
            return _Resp(text=(FIX / "dixon_shp_jun2026.xml").read_text())
        raise AssertionError(f"unexpected {url}")


def test_the_provider_reads_pledge_from_the_filing_and_never_calls_the_dead_api(tmp_path, monkeypatch):
    session = _Session()
    prov = NseOwnershipProvider(store=OwnershipStore(tmp_path / "m.db"))
    monkeypatch.setattr(prov, "_session", lambda symbol: session)
    own = prov.fetch("DIXON")
    jun = next(q for q in own.quarters if q.quarter == "Jun 2026")
    assert (jun.pledge_pct, jun.pledge_source, jun.promoter_pct) == (0.0, "nse_shp_xbrl", 28.55)
    assert not any("pledgedata" in u for u in session.urls)


def test_a_quarter_already_read_from_its_filing_is_not_downloaded_again(tmp_path, monkeypatch):
    st = OwnershipStore(tmp_path / "m.db")
    st.save(Ownership("DIXON", [Quarter("Jun 2026", pledge_pct=0.0, pledge_source="nse_shp_xbrl")]))
    session = _Session()
    prov = NseOwnershipProvider(store=st)
    monkeypatch.setattr(prov, "_session", lambda symbol: session)
    prov.fetch("DIXON")
    assert not any(u.endswith("SHP_jun.xml") for u in session.urls)


def test_a_manual_pledge_is_never_overwritten(tmp_path):
    st = OwnershipStore(tmp_path / "m.db")
    st.set_pledge("ABC", "Jun 2026", 12.5)
    st.save(Ownership("ABC", [Quarter("Jun 2026", promoter_pct=40.0, pledge_pct=0.0,
                                      pledge_source="nse_shp_xbrl")]))
    q = st.load("ABC").quarters[0]
    assert (q.pledge_pct, q.pledge_source, q.promoter_pct) == (12.5, "manual", 40.0)


def test_a_statutory_zero_is_reported_as_no_pledge_not_unknown():
    own = Ownership("DIXON", [Quarter("Jun 2026", promoter_pct=28.55, pledge_pct=0.0, pledge_pct_of_total=0.0,
                                      pledge_source="nse_shp_xbrl")])
    s = summarize_ownership(own)
    assert s["pledge_band"] == "none (ideal)"
    assert any(f.startswith("NO PROMOTER PLEDGE") for f in s["risk_flags"])
    assert not any("UNKNOWN" in f for f in s["risk_flags"])


def test_pledge_filings_after_the_quarter_ride_along(monkeypatch):
    # the SHP shows the quarter's level; a pledge created since then is in the event store's SAST filings
    import ast
    import asyncio

    from pennystock.tools import market_data as md
    monkeypatch.setattr(md, "_resolve_ticker", lambda s, e: ("ABC.NS", "Abc Ltd"))
    monkeypatch.setattr(md, "get_ownership", lambda sym: Ownership("ABC", [
        Quarter("Jun 2026", promoter_pct=40.0, pledge_pct=0.0, pledge_source="nse_shp_xbrl")]))
    monkeypatch.setattr(md, "recent_summary", lambda sym: {"pledges": [
        {"date": "2026-08-14", "type": "pledge_created"}, {"date": "2026-05-02", "type": "pledge_released"}]})
    out = ast.literal_eval(asyncio.run(md.get_shareholding.ainvoke({"symbol": "ABC"})))
    assert out["pledge_filings_since_quarter"] == [{"date": "2026-08-14", "type": "pledge_created"}]


# --- a company with no promoter group (banks, Paytm, Swiggy...) has no promoter pledge, not an unknown one
# 11 of the 119 companies read as 'pledge UNKNOWN' on 2026-10-11, which the analyst treats as an open risk
def test_a_company_with_no_promoter_group_has_no_promoter_pledge():
    # YESBANK Jun 2026: the filing has no promoter shareholding; 'promoter' contexts that remain are public
    # categories (bodies corporate where a government is promoter, trusts) and must not count
    p = parse_shp_pledge((FIX / "yesbank_shp_jun2026.xml").read_text())
    assert p["no_promoter_group"] is True and p["promoter_shares"] == 0.0 and p["pledge_pct"] == 0.0
    assert parse_shp_pledge((FIX / "dixon_shp_jun2026.xml").read_text())["no_promoter_group"] is False


def test_an_error_page_served_instead_of_the_filing_is_unknown():
    # TIINDIA Jun 2026: NSE answered the XBRL link with an HTML page
    assert parse_shp_pledge("<!DOCTYPE html><html><body>Service unavailable</body></html>") is None


class _NoPromoterSession(_Session):
    def __init__(self, listed_promoter):
        super().__init__()
        self.listed = listed_promoter

    def get(self, url, timeout=None):
        self.urls.append(url)
        if "share-holdings-master" in url:
            return _Resp([{"date": "30-JUN-2026", "pr_and_prgrp": self.listed, "public_val": "100",
                           "xbrl": "https://x/SHP_jun.xml"}])
        return _Resp(text=(FIX / "yesbank_shp_jun2026.xml").read_text())


def test_no_promoter_group_is_taken_only_when_nse_lists_zero_promoter(tmp_path, monkeypatch):
    for listed, expected in (("0", (0.0, "nse_shp_xbrl")), ("28.55", (None, None))):
        prov = NseOwnershipProvider(store=OwnershipStore(tmp_path / f"m{listed}.db"))
        monkeypatch.setattr(prov, "_session", lambda symbol, s=_NoPromoterSession(listed): s)
        jun = prov.fetch("YESBANK").quarters[-1]
        assert (jun.pledge_pct, jun.pledge_source) == expected, listed


def test_a_company_without_promoters_is_said_to_have_none_not_an_unknown_pledge():
    own = Ownership("YESBANK", [Quarter("Jun 2026", promoter_pct=0.0, pledge_pct=0.0, pledge_pct_of_total=0.0,
                                        pledge_source="nse_shp_xbrl")])
    flags = summarize_ownership(own)["risk_flags"]
    assert any(f.startswith("NO PROMOTER GROUP") for f in flags)
    assert not any("UNKNOWN" in f or f.startswith("NO PROMOTER PLEDGE") for f in flags)


def test_a_newer_quarter_without_a_reading_falls_back_to_the_last_statutory_one():
    # EXIDEIND: a secondary source already lists Sep 2026; the statutory reading is for Jun 2026
    own = Ownership("EXIDEIND", [Quarter("Jun 2026", promoter_pct=45.99, pledge_pct=0.0,
                                         pledge_pct_of_total=0.0, pledge_source="nse_shp_xbrl"),
                                 Quarter("Sep 2026", promoter_pct=45.99)])
    s = summarize_ownership(own)
    assert (s["pledge_pct_of_promoter_holding"], s["pledge_quarter"], s["pledge_source"]) == \
           (0.0, "Jun 2026", "nse_shp_xbrl")
    assert any(f.startswith("PLEDGE AS OF Jun 2026") for f in s["risk_flags"])
    assert not any(f.startswith("PLEDGE UNKNOWN") for f in s["risk_flags"])


def test_pledge_filings_ride_along_from_the_statutory_readings_quarter(monkeypatch):
    # with a newer unread quarter listed, filings since the READING's quarter (Jun) must still show
    import ast
    import asyncio

    from pennystock.tools import market_data as md
    monkeypatch.setattr(md, "_resolve_ticker", lambda s, e: ("ABC.NS", "Abc Ltd"))
    monkeypatch.setattr(md, "get_ownership", lambda sym: Ownership("ABC", [
        Quarter("Jun 2026", promoter_pct=40.0, pledge_pct=0.0, pledge_source="nse_shp_xbrl"),
        Quarter("Sep 2026", promoter_pct=40.0)]))
    monkeypatch.setattr(md, "recent_summary", lambda sym: {"pledges": [
        {"date": "2026-08-14", "type": "pledge_created"}]})
    out = ast.literal_eval(asyncio.run(md.get_shareholding.ainvoke({"symbol": "ABC"})))
    assert out["pledge_filings_since_quarter"] == [{"date": "2026-08-14", "type": "pledge_created"}]


def test_a_symbol_with_an_ampersand_is_url_encoded(tmp_path, monkeypatch):
    # 'symbol=M&MFIN' reads as symbol=M plus a stray parameter: GVT&D and M&MFIN timed out on 2026-10-11
    session = _Session()
    prov = NseOwnershipProvider(store=OwnershipStore(tmp_path / "m.db"))
    monkeypatch.setattr(prov, "_session", lambda symbol: session)
    prov.fetch("M&MFIN")
    master = next(u for u in session.urls if "share-holdings-master" in u)
    assert master.endswith("symbol=M%26MFIN")

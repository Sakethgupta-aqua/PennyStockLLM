from pennystock.data.events import Announcement, Events, classify, summarize_events


def test_routine_filings_are_suppressed():
    # real example that used to be flagged as fraud
    assert classify("Certificate under SEBI (Depositories and Participants) Regulations, 2018") is None
    assert classify("Analysts/Institutional Investor Meet/Con. Call Updates") is None
    assert classify("Intimation of trading window closure") is None


def test_material_events_are_caught():
    assert classify("Statutory auditor has tendered resignation") == "auditor resignation / removal"
    assert classify("Default in payment of interest on NCDs") == "default / insolvency"
    assert classify("Forensic audit ordered by lenders") == "fraud / investigation / enforcement"
    assert classify("Credit rating downgrade by CRISIL") == "credit rating action"
    assert classify("Creation of pledge over promoter shares") == "pledge / encumbrance"


def test_summary_flags_the_severe_categories():
    ev = Events(symbol="X", announcements=[
        Announcement("01-Jan-2026", "x", "auditor has tendered resignation",
                     "auditor resignation / removal")])
    out = summarize_events(ev)
    assert out["material_announcements"] == 1
    assert any("auditor resignation" in f for f in out["event_flags"])


def test_no_data_is_reported_as_unchecked_not_clean():
    out = summarize_events(Events(symbol="X"))
    assert any("UNCHECKED" in f for f in out["event_flags"])


def test_issuing_non_convertible_debentures_is_borrowing_not_dilution():
    """NBFCs allot NCDs every few weeks (Aditya Birla Capital: 10 in two years). 'Non-convertible'
    contains 'convertible', and reading it as a fund raise flooded the material screen."""
    for text in ("Aditya Birla Capital Limited has informed the Exchange regarding allotment of its Non-Convertible Debentures.",
                 "allotment of Non- Convertible Debentures",
                 "Allotment of non convertible debentures on private placement basis"):
        assert classify(text) is None, text
    assert classify("Allotment of fully convertible warrants to promoters") == "fund raise / dilution"
    assert classify("Issue of Compulsorily Convertible Debentures") == "fund raise / dilution"


def test_the_live_provider_no_longer_reads_the_market_wide_rating_list(monkeypatch):
    # NSE's corporate-credit-rating answers a market-wide list of the last few days whatever the
    # symbol, and the old insider feed stopped in April 2026: both now come from event_signals
    from pennystock.data import events
    paths = []
    monkeypatch.setattr(events, "nse_session", lambda s: object())
    monkeypatch.setattr(events, "nse_api", lambda s, p: paths.append(p) or [])
    events.NseEventsProvider().fetch("ABC")
    assert paths and all("credit-rating" not in p and "corporates-pit" not in p for p in paths)


def test_a_symbol_with_an_ampersand_is_url_encoded_in_the_announcements_request(monkeypatch):
    # 'symbol=M&MFIN' reads as symbol=M plus a stray parameter (the session's referer was already encoded)
    from pennystock.data import events
    paths = []
    monkeypatch.setattr(events, "nse_session", lambda s: object())
    monkeypatch.setattr(events, "nse_api", lambda s, p: paths.append(p) or [])
    events.NseEventsProvider().fetch("M&MFIN")
    assert paths == ["corporate-announcements?index=equities&symbol=M%26MFIN"]

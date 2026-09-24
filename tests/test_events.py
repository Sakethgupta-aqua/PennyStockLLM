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

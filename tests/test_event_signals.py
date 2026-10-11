"""Event signals: NSE disclosures typed, stored, graded and used as screener vetoes.

All offline. Rows are shaped like the real NSE feeds measured on 2026-10-05 (see the spec, section 3);
every false alarm found on 6,014 real filings is a negative test.
"""
from __future__ import annotations

from pennystock.data import event_signals as es


# --- Task 1: typing rules -------------------------------------------------------------------------

def pit(name="Venkatesh Rajagopal", cat="Promoters", mode="Market Purchase", value="5499003",
        date="01-Oct-2021 15:36", did="395579", sym="INDTERRAIN", tx="Buy"):
    return {"symbol": sym, "acqName": name, "personCategory": cat, "acqMode": mode, "tdpTransactionType": tx,
            "secAcq": "141000", "secVal": value, "befAcqSharesPer": "11.98", "afterAcqSharesPer": "12.32",
            "acqfromDt": "29-Sep-2021", "acqtoDt": "29-Sep-2021", "intimDt": "30-Sep-2021", "date": date,
            "did": did, "xbrl": "https://nsearchives.nseindia.com/corporate/xbrl/IT_1.xml"}

def test_a_promoter_buying_in_the_market_is_a_bullish_signal():
    [e] = es.insider_events([pit()])
    assert (e.event_type, e.direction, e.conviction, e.severity) == ("insider_buy", "bullish", 3, "signal")
    assert (e.symbol, e.signal_at, e.event_date) == ("INDTERRAIN", "2021-10-01T15:36", "2021-09-29")
    assert e.details["value_rs"] == 5499003.0 and e.details["person"] == "Venkatesh Rajagopal"

def test_small_trades_employees_relatives_and_non_market_modes_are_not_signals():
    rows = [pit(value="200000"), pit(cat="Employees/Designated Employees"), pit(cat="Immediate relative"),
            pit(mode="ESOP"), pit(mode="Off Market"), pit(mode="Inter-se-Transfer")]
    assert es.insider_events(rows) == []

def test_one_persons_split_trades_on_a_day_are_one_event():
    rows = [pit(value="600000", did="1"), pit(value="700000", did="2")]
    [e] = es.insider_events(rows)
    assert e.details["value_rs"] == 1300000.0

def test_sales_and_pledges():
    rows = [pit(mode="Market Sale", tx="Sell", value="10000000", did="3"),
            pit(cat="Promoter Group", mode="Pledge Creation", tx="Pledge", did="4"),
            pit(cat="Promoter Group", mode="Invocation of pledge", tx="Pledge Invoke", did="5"),
            pit(cat="Promoter Group", mode="Revokation of Pledge", tx="Pledge Revoke", did="6")]
    got = {e.event_type: (e.direction, e.conviction, e.severity) for e in es.insider_events(rows)}
    assert got == {"insider_sell": ("bearish", 2, "signal"), "pledge_created": ("bearish", 2, "signal"),
                   "pledge_invoked": ("bearish", 4, "signal"), "pledge_released": (None, None, "info")}

def test_an_unreadable_broadcast_time_skips_the_row():
    assert es.insider_events([pit(date="-")]) == []

def test_a_second_insider_buying_within_30_days_makes_a_cluster():
    a = pit(name="A", date="01-Mar-2025 10:00", did="10")
    b = pit(name="B", cat="Director", date="20-Mar-2025 10:00", did="11")
    c = pit(name="C", date="30-Apr-2025 10:00", did="12")            # 41 days after B: a new window
    got = {e.details["person"]: e.conviction for e in es.apply_clusters(es.insider_events([a, b, c]))}
    assert got == {"A": 3, "B": 4, "C": 3}

XBRL = """<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance" xmlns:in-bse-pit="x">
<in-bse-pit:Symbol contextRef="OneD">EBGNG</in-bse-pit:Symbol>
<in-bse-pit:CategoryOfPerson contextRef="Disclosure1">Promoter</in-bse-pit:CategoryOfPerson>
<in-bse-pit:NameOfThePerson contextRef="Disclosure1">Vidhi S Khandelwal</in-bse-pit:NameOfThePerson>
<in-bse-pit:SecuritiesAcquiredOrDisposedValueOfSecurity contextRef="Disclosure1" unitRef="INR">2308048806</in-bse-pit:SecuritiesAcquiredOrDisposedValueOfSecurity>
<in-bse-pit:SecuritiesAcquiredOrDisposedTransactionType contextRef="Disclosure1">Sell</in-bse-pit:SecuritiesAcquiredOrDisposedTransactionType>
<in-bse-pit:ModeOfAcquisitionOrDisposal contextRef="Disclosure1">Market Sale</in-bse-pit:ModeOfAcquisitionOrDisposal>
<in-bse-pit:DateOfAllotmentAdviceOrAcquisitionOfSharesOrSaleOfSharesSpecifyFromDate contextRef="Disclosure1">2026-09-30</in-bse-pit:DateOfAllotmentAdviceOrAcquisitionOfSharesOrSaleOfSharesSpecifyFromDate>
<in-bse-pit:CategoryOfPerson contextRef="Disclosure2">Director</in-bse-pit:CategoryOfPerson>
<in-bse-pit:NameOfThePerson contextRef="Disclosure2">R Shah</in-bse-pit:NameOfThePerson>
<in-bse-pit:SecuritiesAcquiredOrDisposedValueOfSecurity contextRef="Disclosure2" unitRef="INR">1500000</in-bse-pit:SecuritiesAcquiredOrDisposedValueOfSecurity>
<in-bse-pit:SecuritiesAcquiredOrDisposedTransactionType contextRef="Disclosure2">Buy</in-bse-pit:SecuritiesAcquiredOrDisposedTransactionType>
<in-bse-pit:ModeOfAcquisitionOrDisposal contextRef="Disclosure2">Market Purchase</in-bse-pit:ModeOfAcquisitionOrDisposal>
<in-bse-pit:DateOfAllotmentAdviceOrAcquisitionOfSharesOrSaleOfSharesSpecifyFromDate contextRef="Disclosure2">2026-09-29</in-bse-pit:DateOfAllotmentAdviceOrAcquisitionOfSharesOrSaleOfSharesSpecifyFromDate>
</xbrli:xbrl>"""
INDEX = {"appId": "3835", "broadcastDateTime": "30-Sep-2026 23:44:11", "symbol": "EBGNG",
         "xmlFileName": "https://nsearchives.nseindia.com/corporate/xbrl/IT_25012.xml", "typeOfSubmission": "Original"}

def test_a_new_format_disclosure_reads_like_the_old_feed():
    rows = es.pit_xbrl_rows(XBRL, INDEX)
    got = {(e.event_type, e.details["person"], e.signal_at) for e in es.insider_events(rows)}
    assert got == {("insider_sell", "Vidhi S Khandelwal", "2026-09-30T23:44"),
                   ("insider_buy", "R Shah", "2026-09-30T23:44")}

def test_a_revised_disclosure_of_the_same_trade_is_the_same_event():
    first = es.insider_events(es.pit_xbrl_rows(XBRL, INDEX))
    again = es.insider_events(es.pit_xbrl_rows(XBRL, {**INDEX, "appId": "3901", "typeOfSubmission": "Revised"}))
    assert {e.event_key for e in first} == {e.event_key for e in again}

def rating(action="Downgrade", details="Downgraded from CARE A+; Stable to CARE A; Stable", agency="CARE Ratings Limited",
           when="30-Sep-2026 22:33:29", inst="Non-convertible debentures", sym="SBFC"):
    return {"creditAgencyName": agency, "symbol": sym, "isin": "INE423Y01016", "classOfAction": "Review",
            "ratingAssigned": inst, "creditRating": "CARE A", "outlook": "Stable", "dateOfCurrentCredit": "30-Sep-2026",
            "currentAction": action, "brieFDetails": details, "detailsOfRatingLink": "https://x/pr/1", "broadcastDateTime": when}

def test_rating_actions():
    rows = [rating(), rating(inst="Long-term bank facilities"),               # two instruments, one event
            rating(action="Upgrade", details="Upgraded", agency="ICRA Limited"),
            rating(action="Re-affirm", details="Reaffirmed", agency="CRISIL Ratings Limited"),
            rating(details="IVR B /Negative; ISSUER NOT COOPERATING* (Rating Downgraded)", agency="Infomerics")]
    got = sorted((e.event_type, e.details["agency"], e.severity) for e in es.rating_events(rows))
    assert got == [("rating_downgrade", "CARE Ratings Limited", "signal"), ("rating_inc", "Infomerics", "signal"),
                   ("rating_other", "CRISIL Ratings Limited", "info"), ("rating_upgrade", "ICRA Limited", "signal")]

def ann(desc, text="", seq="101", when="2026-09-28 18:30:12", sym="ABC"):
    return {"symbol": sym, "desc": desc, "attchmntText": text, "seq_id": seq, "sort_date": when,
            "an_dt": "28-Sep-2026 18:30:12", "attchmntFile": "https://nsearchives.nseindia.com/corporate/a.pdf"}

def test_severe_announcements_are_caught():
    cases = {"auditor_exit": ann("Resignation of Statutory Auditor", "Resignation of M/s ABC & Co, Statutory Auditors of the Company"),
             "insolvency": ann("Corporate Insolvency Resolution Process", "admission of application under Section 7 of the IBC by NCLT"),
             "fraud": ann("General Updates", "appointment of forensic auditor to investigate irregularities"),
             "enforcement": ann("Updates", "Receipt of Show Cause Notice from SEBI")}
    for want, row in cases.items():
        e = es.announcement_event(row)
        assert e is not None and e.event_type == want, want
    assert es.announcement_event(cases["auditor_exit"]).severity == "veto"
    assert es.announcement_event(cases["enforcement"]).severity == "info"

def test_the_false_alarms_measured_on_real_filings_are_not_severe():
    sub = es.announcement_event(ann("General Updates", "Resignation of Statutory Auditor of material subsidiary"))
    assert sub.event_type == "auditor_exit_subsidiary" and sub.severity == "info"
    for row in (ann("Scheme of Arrangement", "order passed by the NCLT approving the Scheme of Amalgamation"),
                ann("Shareholders meeting", "Notice of NCLT/Court Convened General Meeting"),
                ann("Updates", "Borrowal fraud of Rs. 319.15 crore in NPA account of M/s Future Lifestyle"),
                ann("General Updates", "Creation of pledge on the shares of Bharat Serums and Vaccines Limited"),
                ann("Updates", "Affirmation that the Director is not debarred by virtue of any SEBI order"),
                ann("Change in Auditors", "Change in Auditors of the company.")):
        assert es.announcement_event(row) is None, row["attchmntText"]


# --- Task 2: storage and ingestion ----------------------------------------------------------------
import pytest  # noqa: E402

@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(es.settings, "market_db", tmp_path / "t.db")
    monkeypatch.setattr(es.settings, "filings_delay_s", 0)
    return tmp_path

def test_saving_twice_stores_once_and_retype_follows_new_rules(db, monkeypatch):
    events = es.insider_events([pit()])
    assert es.save_events(events) == 1 and es.save_events(events) == 0
    monkeypatch.setitem(es.TYPES, "insider_buy", ("signal", "bullish", 2))
    monkeypatch.setattr(es, "RULES_VERSION", es.RULES_VERSION + 1)
    assert es.retype() == 1 and es.load_events("INDTERRAIN")[0].conviction == 2

def test_a_month_is_done_only_when_nse_answered(db):
    calls = []
    def api(path):
        calls.append(path)
        return None                                  # an error page / timeout
    assert es.ingest_month("pit", 2025, 3, api) is None
    assert es.ingest_month("pit", 2025, 3, lambda p: {"data": [pit(date="05-Mar-2025 10:00")]}) == 1
    assert es.ingest_month("pit", 2025, 3, api) == 0 and len(calls) == 1     # done: not fetched again

def test_new_format_details_are_fetched_only_for_the_screening_universe(db):
    index = {"data": [INDEX, {**INDEX, "appId": "9", "symbol": "ZZZ", "xmlFileName": "https://x/z.xml"}]}
    fetched = []
    get = lambda url: fetched.append(url) or XBRL.encode()
    assert es.ingest_month("pit_gg", 2026, 9, lambda p: index, get=get, universe={"EBGNG"}) == 2
    assert fetched == [INDEX["xmlFileName"]]

def test_announcement_calls_encode_the_symbol_and_keep_two_years(db):
    seen = []
    rows = [ann("Resignation of Statutory Auditor", "Statutory Auditors of the Company resigned", seq="7", sym="M&MFIN"),
            ann("Resignation of Statutory Auditor", "Statutory Auditors of the Company resigned", seq="8",
                when="2021-01-01 10:00:00", sym="M&MFIN")]
    assert es.ingest_announcements("M&MFIN", lambda p: seen.append(p) or rows, since="2024-10-01") == 1
    assert "symbol=M%26MFIN" in seen[0]


# --- Task 3: ledger calls and track records -------------------------------------------------------
import pandas as pd  # noqa: E402

from pennystock.data.ledger import Outcome, load_predictions, save_outcomes  # noqa: E402

def test_a_signal_becomes_a_call_dated_the_session_after_it_was_broadcast():
    closes = pd.Series([100.0, 101.0, 102.0], index=pd.to_datetime(["2025-03-04", "2025-03-05", "2025-03-06"]))
    [e] = es.insider_events([pit(date="05-Mar-2025 18:10")])
    p = es.to_prediction([e], closes)
    assert (p.source, p.direction, p.conviction, p.as_of_date) == ("event:insider_buy", "bullish", 3, "2025-03-06")
    assert (p.entry_price, p.entry_price_asof) == (102.0, "2025-03-06")

def test_info_events_are_never_calls():
    [e] = es.insider_events([pit(cat="Promoter Group", mode="Revokation of Pledge", tx="Pledge Revoke")])
    assert es.to_prediction([e]) is None

def test_one_call_per_company_type_and_day_and_recording_is_idempotent(db, monkeypatch):
    monkeypatch.setattr(es, "fetch_closes", lambda *a, **k: None)
    es.save_events(es.apply_clusters(es.insider_events([
        pit(name="A", date="05-Mar-2025 10:00", did="1"), pit(name="B", cat="Director", date="05-Mar-2025 11:00", did="2")])))
    assert es.record_predictions() == 1
    assert es.record_predictions() == 0
    [p] = load_predictions(source="event:insider_buy")
    assert p.conviction == 4 and p.payload["events"] == 2


def test_a_track_record_counts_decided_calls_and_the_companies_behind_them(db, monkeypatch):
    monkeypatch.setattr(es, "fetch_closes", lambda *a, **k: None)
    es.save_events(es.insider_events([pit(sym="AAA", date="05-Mar-2025 10:00", did="1"),
                                      pit(sym="BBB", date="06-Mar-2025 10:00", did="2")]))
    assert es.record_predictions() == 2
    ids = [p.prediction_id for p in load_predictions(source="event:insider_buy")]
    save_outcomes([Outcome(prediction_id=ids[0], horizon_days=20, label="hit", abnormal_return_pct=4.0),
                   Outcome(prediction_id=ids[1], horizon_days=20, label="miss", abnormal_return_pct=-3.0)])
    t = es.track_record("insider_buy")
    assert (t["decided_sample"], t["hit_rate"], t["companies"]) == (2, 0.5, 2)


# --- Task 4: the screener veto and the EVENTS panel -----------------------------------------------
import sys  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def test_a_severe_event_vetoes_for_365_days_and_no_longer(db):
    es.save_events([es.announcement_event(ann("Resignation of Statutory Auditor",
                    "Statutory Auditors of the Company resigned", when="2025-10-05 10:00:00", sym="ABC"))])
    assert "ABC" in es.veto_map(today=date(2026, 10, 4))
    assert "ABC" not in es.veto_map(today=date(2026, 10, 6))


def test_information_events_never_veto(db):
    es.save_events([es.announcement_event(ann("Updates", "Receipt of Show Cause Notice from SEBI", sym="ABC"))])
    assert es.veto_map(today=date(2026, 10, 5)) == {}


def test_the_events_panel_lists_what_happened_and_says_when_nothing_did(db, capsys):
    from screen import print_events_panel
    print_events_panel("NOPE")
    assert "no stored events" in capsys.readouterr().out
    es.save_events(es.insider_events([pit(date="05-Sep-2026 10:00")]))
    print_events_panel("INDTERRAIN")
    out = capsys.readouterr().out
    assert "EVENTS" in out and "insider_buy" in out and "Venkatesh Rajagopal" in out


# --- Task 5: the analyst's view -------------------------------------------------------------------
def test_the_tool_summary_comes_from_stored_dated_events(db):
    es.save_events(es.insider_events([pit(date="05-Sep-2026 10:00")]) +
                   es.rating_events([rating(sym="INDTERRAIN", when="01-Sep-2026 10:00:00")]))
    s = es.recent_summary("INDTERRAIN", today=date(2026, 10, 5))
    assert s["insider"]["buys"] == 1 and s["ratings"][0]["type"] == "rating_downgrade"
    assert es.recent_summary("NOPE", today=date(2026, 10, 5)) is None


def test_the_tool_hands_over_stored_events_or_says_unchecked(monkeypatch):
    import asyncio

    from pennystock.data.events import Events
    from pennystock.tools import market_data as md
    monkeypatch.setattr(md, "_resolve_ticker", lambda s, e="NSE": ("ABC.NS", "Abc Ltd"))
    monkeypatch.setattr(md.NseEventsProvider, "fetch", lambda self, s: Events(symbol=s))
    monkeypatch.setattr(md, "recent_summary", lambda s: None)
    assert "UNCHECKED" in asyncio.run(md.get_corporate_events.ainvoke({"symbol": "ABC"}))
    monkeypatch.setattr(md, "recent_summary", lambda s: {"insider": {"buys": 2}})
    assert "'buys': 2" in asyncio.run(md.get_corporate_events.ainvoke({"symbol": "ABC"}))


# --- Task 6: false alarms found in the full 2-year, 1,486-company run -----------------------------
def kind(desc, text, sym="ABC", seq="9", when="2026-09-28 18:30:12"):
    e = es.announcement_event(ann(desc, text, seq=seq, when=when, sym=sym))
    return e.event_type if e else None


def test_a_subsidiary_auditor_named_before_the_word_subsidiary_is_still_a_subsidiary():
    for text in ("Resignation of Statutory Auditors of Nodwin Gaming Private Limited and Absolute Sports Private "
                 "Limited, Material Subsidiaries of the Company.",
                 "Resignation of the Statutory Auditor in Unlisted Material Subsidiary of the Company",
                 "Resignation by Statutory Auditor ofAmara Raja Advanced Cell Technologies Private Limited, "
                 "wholly owned Material Subsidiaryof the Company",
                 "Resignation of Ms Ernst & Young LLP UK as Statutory Auditors of TVS Logistics Investment UK Limited, "
                 "RICO Logistics Limited, UK and TVS Supply Chain Solutions Limited., UK, Material Subsidiaries"):
        assert kind("Resignation", text) == "auditor_exit_subsidiary", text[:60]


def test_the_companys_own_auditor_still_counts_even_when_subsidiaries_are_mentioned():
    assert kind("Resignation", "Resignation of M/s X & Co, Statutory Auditors of the Company and its subsidiaries") == "auditor_exit"
    assert kind("Outcome of Board Meeting", "appointment of M/s S P Chopra & Co. as the Statutory Auditors of the Company to "
                "fill up the casual vacancy caused by the resignation of M/s Lodha & Co. LLP") == "auditor_exit"


def test_a_list_of_unrelated_changes_and_a_term_ending_are_not_auditor_exits():
    assert kind("Change in Management", "a) Appointment of Statutory Auditors b) Appointment of Secretarial Auditors "
                "c) Resignation of Company Secretary") is None
    assert kind("Change in Auditors", "cessation of M/s X as Statutory Auditors on completion of their term") is None


def test_a_bare_auditor_filing_beside_a_subsidiary_filing_that_week_is_the_subsidiary_story():
    rows = [ann("Resignation", "S.R. Batliboi & Co. LLP have tendered their resignation as the Statutory Auditors of IAC "
                "International Automotive India Private Limited, a material subsidiary of the Company",
                seq="1", when="2025-08-12 10:00:00", sym="LUMAXTECH"),
            ann("Resignation of Statutory Auditor", "LUMAX AUTO TECHNOLOGIES LIMITED has informed the Exchange about "
                "Resignation of Statutory Auditor", seq="2", when="2025-08-13 10:00:00", sym="LUMAXTECH")]
    got = es.reconcile([es.announcement_event(r) for r in rows])
    assert [e.event_type for e in got] == ["auditor_exit_subsidiary", "auditor_exit_subsidiary"]


def test_fraud_against_the_company_news_queries_closures_and_product_launches_are_not_company_fraud():
    for desc, text in (("Press Release", "Airtel launches Fraud Detection Solution, a First in India"),
                       ("News Verification", "clarification with respect to news item captioned Hackers dupe Dr Reddy's"),
                       ("General Updates", "intimation of closure of Forensic Audit"),
                       ("Updates", "Clarification regarding newspaper reports regarding fraud committed by employees"),
                       ("Updates", "update regarding the matter of fraud by sales employee of the company"),
                       ("General Updates", "a subsidiary of the company has suffered a Financial Loss arising from a Fraud Incident")):
        assert kind(desc, text) != "fraud", text[:50]


def test_a_bare_fraud_default_arrest_label_and_a_tax_search_are_shown_not_vetoed():
    assert kind("Fraud/Default/Arrest", "Union Bank of India has informed the Exchange about Fraud/Default/Arrest") == "enforcement"
    assert kind("General Updates", "search and seizure operation was conducted by the GST Department") == "enforcement"
    assert kind("General Updates", "Search and Seizure conducted by the Income Tax Department") == "enforcement"


def test_genuine_fraud_signals_still_veto():
    for desc, text in (("Initiation of Forensic Audit", "Gensol Engineering Limited has informed the Exchange about "
                        "Initiation of Forensic Audit"),
                       ("Action(s) initiated or orders passed", "the Enforcement Directorate has conducted Search and seizure"),
                       ("General Updates", "notice received from Serious Fraud Investigation Office (SFIO) for investigation"),
                       ("Fraud/Default/Arrest", "update on Fraud intimated on February 06, 2026")):
        assert kind(desc, text) == "fraud", text[:50]


def test_buying_or_claiming_against_an_insolvent_company_is_not_insolvency():
    for text in ("LOI Received From Resolution Professional Of Vidarbha Industries Power Limited",
                 "Resolution Plan submitted by the Consortium of NTPC for Sinnar Thermal Power Limited, a company "
                 "undergoing Corporate Insolvency Resolution Process",
                 "Filing of claim against Attharv Sai FlexiPack Private Limited with Interim Resolution Professional",
                 "Identification of Company s subsidiary as Successful Resolution Applicant for Bhilai Jaypee Cement",
                 "Closure of Corporate Insolvency Resolution Process (CIRP) and discharge of the Interim Resolution Professional"):
        assert kind("General Updates", text) != "insolvency", text[:50]
    assert kind("Corporate Insolvency Resolution Process", "minutes of the 02nd COC meeting of the Company") == "insolvency"
    assert kind("General Updates", "BIL VYAPAR LIMITED has informed the Exchange about Appointment Of Resolution Professional") == "insolvency"
    assert kind("Defaults on Payment of Interest/Principal", "default in payment of Interest/principal amount of "
                "0.01% Redeemable Preference Shares") == "insolvency"


def test_a_company_filing_update_after_update_gets_one_call_a_month(db, monkeypatch):
    monkeypatch.setattr(es, "fetch_closes", lambda *a, **k: None)
    es.save_events([es.announcement_event(ann("Corporate Insolvency Resolution Process", f"COC meeting {i}",
                    seq=str(100 + i), when=f"2026-0{m}-{d:02d} 10:00:00", sym="CIRPCO"))
                    for i, (m, d) in enumerate([(3, 1), (3, 11), (3, 25), (4, 20)])])
    assert es.record_predictions() == 2                  # 1 Mar, then 20 Apr (more than 30 days later)


def test_a_cluster_forming_inside_the_month_still_gets_its_stronger_call(db, monkeypatch):
    monkeypatch.setattr(es, "fetch_closes", lambda *a, **k: None)
    es.save_events(es.insider_events([pit(name="A", date="01-Mar-2025 10:00", did="1"),
                                      pit(name="B", cat="Director", date="10-Mar-2025 10:00", did="2")]))
    assert es.record_predictions() == 2
    assert sorted(p.conviction for p in load_predictions(source="event:insider_buy")) == [3, 4]


# --- Task 6 (2): what the real data says about INC ratings, stayed insolvencies and bare auditor filings ---
def rated(grade, details, action="Other", sym="XYZ", agency="CRISIL Ratings Limited"):
    return {**rating(action=action, details=details, agency=agency, sym=sym), "creditRating": grade}


def test_issuer_not_cooperating_is_a_graded_signal_and_only_a_default_grade_vetoes():
    got = {r["creditRating"]: (e.event_type, e.severity) for r in (
        rated("Crisil B(Issuer Not Cooperating)", "Issuer not cooperating and Rating Revised", sym="A"),
        rated("Crisil A4+/Issuer Not Cooperating (Withdrawn)", "Rating Migrated and Withdrawn", sym="B"),
        rated("Crisil D(Issuer Not Cooperating)", "Rating continues at the same level", action="Re-affirm", sym="C"),
        rated("CARE D", "Downgraded", action="Downgrade", sym="D", agency="CARE Ratings Limited"))
        for e in es.rating_events([r])}
    assert got == {"Crisil B(Issuer Not Cooperating)": ("rating_inc", "signal"),
                   "Crisil A4+/Issuer Not Cooperating (Withdrawn)": ("rating_other", "info"),
                   "Crisil D(Issuer Not Cooperating)": ("rating_default", "veto"),
                   "CARE D": ("rating_default", "veto")}


def test_a_stayed_or_closed_insolvency_lifts_and_a_petition_merely_filed_does_not_veto():
    assert kind("Corporate Insolvency Resolution Process",
                "grant of stay by Hon'ble NCLAT against the impugned order passed by Hon'ble NCLT") == "insolvency_lifted"
    assert kind("Corporate Insolvency Resolution Process", "filing of application under Section 7 of IBC, 2016 for the "
                "initiation of CIRP before NCLT Allahabad bench. The matter is yet to come up before the Hon'ble NCLT.") == "enforcement"


def test_a_later_stay_lifts_the_veto_until_insolvency_returns(db):
    cirp = lambda seq, when, text: es.announcement_event(ann("Corporate Insolvency Resolution Process", text, seq=seq,
                                                             when=when, sym="EMB"))
    es.save_events([cirp("1", "2026-01-10 10:00:00", "copy of the order passed by NCLT admitting the petition"),
                    cirp("2", "2026-01-20 10:00:00", "grant of stay by Hon'ble NCLAT against the impugned order")])
    assert "EMB" not in es.veto_map(today=date(2026, 3, 1))
    es.save_events([cirp("3", "2026-02-10 10:00:00", "Public Announcement on Corporate Insolvency Resolution Process")])
    assert "EMB" in es.veto_map(today=date(2026, 3, 1))


def test_a_bare_auditor_filing_is_settled_by_reading_its_letter(db):
    def bare(seq, sym):
        row = ann("Resignation of Statutory Auditor", f"{sym} LIMITED has informed the Exchange about Resignation of "
                  "Statutory Auditor", seq=seq, sym=sym)
        return es.announcement_event({**row, "attchmntFile": f"https://nsearchives.nseindia.com/corporate/{sym}.pdf"})
    es.save_events([bare("1", "SUBCO"), bare("2", "OWNCO"), bare("3", "LOSTCO")])
    letters = {"SUBCO": "resignation of M/s X as Statutory Auditors of ABC Pvt. Ltd., a material subsidiary of the Company",
               "OWNCO": "resignation of M/s Y as Statutory Auditors of the Company with immediate effect"}
    get = lambda url: None if "LOSTCO" in url else url.encode()       # the "PDF" carries its company's name
    extract = lambda body: next(v for k, v in letters.items() if k in body.decode())
    tally = es.resolve_bare_filings(get=get, extract=extract)
    assert (tally["resolved"], tally["failed"]) == (2, 1)
    got = {e.symbol: e.event_type for e in es.load_events(types={"auditor_exit", "auditor_exit_subsidiary"})}
    assert got == {"SUBCO": "auditor_exit_subsidiary", "OWNCO": "auditor_exit", "LOSTCO": "auditor_exit"}


def test_an_auditor_letter_inside_a_zip_is_read(monkeypatch):
    # NSE's resignation form arrives as NUVOCO_ROSA_..._KMP_Doc.zip with the letters as PDFs inside
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("SE_Auditor_resignation_1.pdf", b"letter one")
        z.writestr("readme.txt", b"not a letter")
    monkeypatch.setattr(es, "_one_pdf", lambda body: body.decode())
    assert es._pdf_text(buf.getvalue()).strip() == "letter one"


def test_retyping_leaves_insider_clusters_alone(db):
    es.save_events(es.insider_events([pit(name="A", date="01-Mar-2025 10:00", did="1"),
                                      pit(name="B", cat="Director", date="02-Mar-2025 10:00", did="2")]))
    es.retype()
    assert es.retype() == 0


def test_retyping_keeps_a_reconciled_bare_filing_where_it_is(db):
    rows = [ann("Resignation", "S.R. Batliboi & Co. LLP have tendered their resignation as the Statutory Auditors of IAC "
                "International Automotive India Private Limited, a material subsidiary of the Company",
                seq="1", when="2025-08-12 10:00:00", sym="LUMAXTECH"),
            ann("Resignation of Statutory Auditor", "LUMAX AUTO TECHNOLOGIES LIMITED has informed the Exchange about "
                "Resignation of Statutory Auditor", seq="2", when="2025-08-13 10:00:00", sym="LUMAXTECH")]
    es.save_events(es.reconcile([es.announcement_event(r) for r in rows]))
    es.retype()
    assert es.retype() == 0
    assert {e.event_type for e in es.load_events("LUMAXTECH")} == {"auditor_exit_subsidiary"}


def test_a_subsidiary_named_first_and_its_auditor_after_is_still_a_subsidiary():
    letter = ("..., a Step down Subsidiary of the Company, that the ATPL has considered and approved the following matters "
              "inter alia: 1. Resignation of M/s. S.R. Batliboi & Co. LLP, Chartered Accountants (FRN: 301003E/E300005) "
              "from the position of the Statutory Auditors of ATPL vide their resignation letter")
    assert kind("Resignation of Statutory Auditor", letter) == "auditor_exit_subsidiary"


def test_a_letter_naming_the_companys_own_auditor_is_never_reconciled_away():
    sub = es.announcement_event(ann("Resignation", "resignation as the Statutory Auditors of X Pvt Ltd, a material "
                                    "subsidiary of the Company", seq="1", when="2025-08-12 10:00:00", sym="ARV"))
    bare = es.announcement_event(ann("Resignation of Statutory Auditor", "ARV LIMITED has informed the Exchange about "
                                     "Resignation of Statutory Auditor", seq="2", when="2025-08-13 10:00:00", sym="ARV"))
    bare.details["letter"] = "resignation of M/s Y as Statutory Auditors of the Company"
    assert [e.event_type for e in es.reconcile([sub, bare])] == ["auditor_exit_subsidiary", "auditor_exit"]


def test_two_filings_of_one_severe_event_give_one_veto_reason(db):
    rows = [ann("Resignation of Statutory Auditor", "Statutory Auditors of the Company resigned", seq=s,
                when="2026-07-13 22:17:00", sym="NUV") for s in ("1", "2")]
    es.save_events([es.announcement_event(r) for r in rows])
    assert len(es.veto_map(today=date(2026, 10, 5))["NUV"]) == 1

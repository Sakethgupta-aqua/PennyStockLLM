# Event Signals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store insider trades, promoter pledges, credit-rating actions and severe announcements as dated
`event_signals`, grade each signal type in the ledger, exclude severe cases from the screener, and give the
analyst correct, dated events.

**Architecture:** `pennystock/data/event_signals.py` owns typing (pure functions over NSE rows), storage,
ingestion (injectable `api`/`get` callables), ledger conversion, track records and the veto map.
`scripts/ingest_events.py` is the CLI. The screener takes an `events` veto map beside `kpis`;
`get_corporate_events` reads the store for insider/pledge/rating events and keeps live announcements.

**Tech Stack:** Python 3, SQLite STRICT, pandas (ledger dating), LangChain `@tool`, pytest.

**Spec:** `docs/superpowers/specs/2026-10-05-event-signals-design.md`

## Global Constraints

- Sources: NSE only - `corporates-pit` (market-wide by month, to April 2026), `corporates-pit-gg` + XBRL (from
  May 2026), `credit-rating-sdd-reg30` (market-wide by month, from ~September 2025), `corporate-announcements`
  per company (2 years). ~1 request/s (`settings.filings_delay_s`); no Trendlyne quota.
- `signal_at` = exchange broadcast time. Ledger as-of = `outcomes.actionable_date(signal day, closes, lag=1)`.
- Insider buy/sell: Promoters, Promoter Group, Director, Key Managerial Personnel; Market Purchase / Market Sale;
  one event per (symbol, person, broadcast day, mode), value summed, kept when ≥ ₹10 lakh (`events_min_value_rs`).
- Cluster: insider_buy conviction 4 when ≥ 2 different persons bought in the 30 days ending on its signal day.
- Severe (veto, 12 months = 365 days, lenders included): `auditor_exit`, `insolvency`, `fraud`, `rating_inc`.
- Ledger: source `"event:<type>"`, one call per (symbol, type, signal day), strongest conviction, never rewritten.
- Every false positive measured in spec §3 is a negative test. Tests offline. No commits unless asked.

## Review Focus

- NSE date formats differ per feed ("30-Sep-2026 23:44:11", "17-Jan-2019 12:53", "2026-10-01 18:30:12"): a row
  whose time cannot be read is skipped and counted, never a crash and never dated "today". (Task 1)
- A month whose fetch fails (error page, timeout) must not be marked done: only a real answer - even an empty
  list - completes a month. (Task 2)
- `corporates-pit-gg` revisions (`typeOfSubmission` "Revised", `prevAppId`) must not double-count a trade:
  the event key is person + day + mode, so a revision of the same trade is the same event. (Tasks 1, 2)
- Symbols with `&` (M&MFIN, GVT&D) in per-company announcement calls are URL-encoded. (Task 2)
- Recording thousands of calls: a symbol with no price feed is dated the day after its event and never stops
  the run; closes are fetched once per symbol. (Task 3)

---

### Task 1: Typing rules (pure functions)

**Files:** Create `pennystock/data/event_signals.py`; Test `tests/test_event_signals.py`

**Interfaces — Produces:** `Event` (dataclass: `event_key, symbol, event_type, severity, direction, conviction,
signal_at, event_date, source, source_url, details`), `TYPES` (type → (severity, direction, conviction)),
`insider_events(rows) -> list[Event]` (old-feed-shaped rows), `pit_xbrl_rows(xml: str, index_row: dict) ->
list[dict]` (new feed → old-feed-shaped rows), `rating_events(rows) -> list[Event]`,
`announcement_event(row: dict) -> Event | None`, `apply_clusters(events) -> list[Event]`, `RULES_VERSION`.

- [ ] **Step 1: failing tests** (fixtures are real-shaped; every §3 false positive is a negative)

```python
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
    assert got == [("rating_downgrade", "CARE Ratings Limited", "signal"), ("rating_inc", "Infomerics", "veto"),
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
```

- [ ] **Step 2: run, expect ImportError.** **Step 3: implement** (date parsing for the three formats → `YYYY-MM-DDTHH:MM`;
  event keys: insider `pit:<SYMBOL>:<person>:<day>:<mode>`, rating `rating:<SYMBOL>:<agency>:<day>:<type>`,
  announcement `ann:<seq_id>`; strict regexes with the negatives above). **Step 4: run; full suite.**

### Task 2: Storage and ingestion

**Files:** Modify `pennystock/data/event_signals.py`, `pennystock/config.py`; Create `scripts/ingest_events.py`;
Test `tests/test_event_signals.py`

**Interfaces — Produces:** `save_events(events) -> int`, `load_events(symbol=None, since=None, types=None) -> list[Event]`,
`retype() -> int`, `months(since: date, until: date) -> list[tuple[int, int]]`,
`ingest_month(source, year, month, api, get=None, universe=None) -> int | None` (None = fetch failed),
`ingest_announcements(symbol, api, since) -> int | None`, `ingest(since, until, api, get, universe, symbols=None) -> dict`,
`status() -> dict`. Settings: `events_min_value_rs=1_000_000`, `events_cluster_days=30`, `events_veto_days=365`,
`events_since="2024-10-01"`, `events_stale_days=7`.

- [ ] **Step 1: failing tests**

```python
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
```

- [ ] **Step 2: run, expect failures.** **Step 3: implement** (STRICT `event_signals` per spec §5 plus
  `event_ingest_state(source, period, rows, fetched_at)`; month windows `from_date=01-MM-YYYY&to_date=<last>-MM-YYYY`;
  the current month is never marked done; CLI with `--status`, `--symbol`, `--since`, `--retype`, `--no-ledger`).
  **Step 4: run; full suite.**

### Task 3: Ledger calls and track records

**Files:** Modify `pennystock/data/event_signals.py`; Test `tests/test_event_signals.py`

**Interfaces — Produces:** `to_prediction(group: list[Event], closes=None, lag=1, sector=None, industry=None) -> Prediction | None`,
`record_predictions(lag=1, symbols=None) -> int`, `track_record(event_type, horizon_days=20) -> dict` (`graded`, `hits`,
`hit_rate`, `companies`).

- [ ] **Step 1: failing tests**

```python
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
```

- [ ] **Step 2: run, expect failures.** **Step 3: implement** (natural key (symbol, type, signal day) over
  `load_predictions(include_superseded=True)` for `event:` sources; closes fetched once per symbol; `track_record`
  reads outcomes at the horizon: hit = label "hit"). **Step 4: run; full suite.**

### Task 4: Screener veto and the EVENTS panel

**Files:** Modify `pennystock/data/screener.py`, `scripts/screen.py`, `pennystock/data/event_signals.py`;
Test `tests/test_event_signals.py`, `tests/test_screener.py`

**Interfaces — Produces:** `veto_map(today=None) -> dict[str, list[tuple[str, str]]]`, `screen(..., events=None)`,
`print_events_panel(symbol)` in screen.py.

- [ ] **Step 1: failing tests**

```python
def test_a_severe_event_vetoes_for_365_days_and_no_longer(db):
    es.save_events([es.announcement_event(ann("Resignation of Statutory Auditor",
                    "Statutory Auditors of the Company resigned", when="2025-10-05 10:00:00", sym="ABC"))])
    assert "ABC" in es.veto_map(today=date(2026, 10, 4))
    assert "ABC" not in es.veto_map(today=date(2026, 10, 6))

# tests/test_screener.py
def test_an_event_veto_removes_a_stock_even_a_lender():
    rows = [metrics_row("BANKX", sector="Financial Services"), metrics_row("GOODCO")]
    f = screen(rows, 0, 0, events={"BANKX": [("severe corporate event", "auditor_exit 2026-09-01")]})
    assert f.veto_reasons == {"severe corporate event": 1} and [r.metrics.symbol for r in f.results] == ["GOODCO"]

def test_the_events_panel_lists_what_happened_and_says_when_nothing_did(db, capsys):
    from screen import print_events_panel
    print_events_panel("NOPE")
    assert "no stored events" in capsys.readouterr().out
    es.save_events(es.insider_events([pit(date="05-Sep-2026 10:00")]))
    print_events_panel("INDTERRAIN")
    out = capsys.readouterr().out
    assert "EVENTS" in out and "insider_buy" in out and "Venkatesh Rajagopal" in out
```

- [ ] **Step 2: run, expect failures.** **Step 3: implement** (veto rule name "severe corporate event";
  `screen.py` main and `--explain` pass `events=event_signals.veto_map()`; the panel prints the last 12 months
  grouped by type with dates, people/agencies, values and each type's track record). **Step 4: run; full suite.**

### Task 5: The analyst tool, prompt, coverage, refresh registry

**Files:** Modify `pennystock/data/events.py`, `pennystock/tools/market_data.py`, `pennystock/data/event_signals.py`,
`pennystock/agents/prompts.py`, `pennystock/evals/assertions.py`, `pennystock/data/maintenance.py`;
Test `tests/test_event_signals.py`, `tests/test_events.py`, `tests/test_assertions.py`, `tests/test_maintenance.py`

**Interfaces — Produces:** `recent_summary(symbol, days=365, today=None) -> dict | None`; `NseEventsProvider.fetch` reads
announcements only; `Dataset("events", ...)`.

- [ ] **Step 1: failing tests**

```python
def test_the_tool_summary_comes_from_stored_dated_events(db):
    es.save_events(es.insider_events([pit(date="05-Sep-2026 10:00")]) +
                   es.rating_events([rating(sym="INDTERRAIN", when="01-Sep-2026 10:00:00")]))
    s = es.recent_summary("INDTERRAIN", today=date(2026, 10, 5))
    assert s["insider"]["buys"] == 1 and s["ratings"][0]["type"] == "rating_downgrade"
    assert es.recent_summary("NOPE", today=date(2026, 10, 5)) is None

# tests/test_events.py
def test_the_live_provider_no_longer_reads_the_market_wide_rating_list(monkeypatch):
    paths = []
    monkeypatch.setattr(events, "nse_session", lambda s: object())
    monkeypatch.setattr(events, "nse_api", lambda s, p: paths.append(p) or [])
    events.NseEventsProvider().fetch("ABC")
    assert paths and all("credit-rating" not in p and "corporates-pit" not in p for p in paths)

# tests/test_assertions.py
def test_a_pledge_invocation_question_needs_the_events_tool():
    r = check_tool_coverage(trace(query="Was any promoter pledge invoked at Indian Terrain?",
                                  tool_calls=[{"name": "get_shareholding", "result": "{}"}]))
    assert not r["passed"] and "get_corporate_events" in r["detail"]
```
  plus `test_maintenance`: the dataset list includes "events".

- [ ] **Step 2: run, expect failures.** **Step 3: implement** (tool: live material announcements + `stored_events`
  from `recent_summary` with track records; when the store has nothing: "insider, pledge and rating events
  UNCHECKED - run scripts/ingest_events.py"; prompt line for tool 7 updated; coverage keywords "invoked",
  "invocation", "insider buying", "insider selling", "rating action"; maintenance dataset refresh = incremental
  ingest of the last `events_stale_days` + 30 days, then `record_predictions`). **Step 4: run; full suite.**

### Task 6: Real run, docs

- [ ] Full ingest in the background (`ingest_events.py`, ~1.5-2 h), `--status`, record predictions,
  `screen.py --explain` on a company with insider buying and one with a rating action, the funnel's veto count,
  one tool call, CLAUDE.md commands, CI lint gate, full suite.

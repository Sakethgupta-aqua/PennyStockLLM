from pennystock.evals.assertions import (
    check_citation_provenance, check_context_fits, check_data_warning_surfaced, check_entity_consistency,
    check_no_false_missing, check_numeric_provenance, check_report_complete, check_source_diversity,
    check_task_completion, check_tool_coverage, run_all,
)


def trace(**kw):
    base = {"query": "", "report": "", "tool_calls": [], "retrieved_chunks": [],
            "resolved_symbol": None, "judge": {}}
    base.update(kw)
    return base


def test_entity_consistency_catches_two_companies():
    t = trace(tool_calls=[
        {"name": "get_indian_stock_data", "result": "{'symbol': 'TMPV.NS', 'price': 347.1}"},
        {"name": "get_stock_fundamentals", "result": "{'symbol': 'TMCV.NS', 'pe_trailing': 54.5}"},
    ])
    r = check_entity_consistency(t)
    assert not r["passed"] and "DIFFERENT" in r["detail"]


def test_entity_consistency_passes_when_same():
    t = trace(resolved_symbol="TMPV", tool_calls=[
        {"name": "get_indian_stock_data", "result": "{'symbol': 'TMPV.NS'}"},
        {"name": "get_price_history", "result": "{'symbol': 'TMPV.NS'}"},
    ])
    assert check_entity_consistency(t)["passed"]


def test_citation_provenance_flags_fabricated_quote():
    t = trace(report='Graham said "a stock trading above book value with crushing debt is not cheap at all".',
              retrieved_chunks=[{"source": "b.pdf",
                                 "text": "The margin of safety is always dependent on the price paid."}])
    assert not check_citation_provenance(t)["passed"]


def test_citation_provenance_accepts_real_quote():
    t = trace(report='He wrote "the margin of safety is always dependent on the price paid" here.',
              retrieved_chunks=[{"source": "b.pdf",
                                 "text": "Graham: the margin of safety is always dependent on the price paid, he says."}])
    assert check_citation_provenance(t)["passed"]


def test_numeric_provenance_allows_rounding():
    t = trace(report="P/E is 54.35x and P/B 12.10x.",
              tool_calls=[{"name": "get_stock_fundamentals",
                           "result": "{'pe_trailing': 54.34993, 'price_to_book': 12.101504}"}])
    assert check_numeric_provenance(t)["passed"]


def test_numeric_provenance_flags_invented_figure():
    t = trace(report="Book value is 999.99 per share.",
              tool_calls=[{"name": "get_stock_fundamentals", "result": "{'book_value': 222.807}"}])
    assert not check_numeric_provenance(t)["passed"]


def test_tool_coverage_flags_skipped_researcher():
    t = trace(query="What would Graham say about the behavioral traps here?",
              tool_calls=[{"name": "get_indian_stock_data", "result": "{}"}])
    r = check_tool_coverage(t)
    assert not r["passed"] and "penny_stock_researcher" in r["detail"]


def test_data_warning_must_be_surfaced():
    tc = [{"name": "get_price_history",
           "result": "{'pct_from_peak': -51.0, 'data_warning': {'date': '2025-10-14'}}"}]
    assert not check_data_warning_surfaced(trace(tool_calls=tc, report="All clear, strong buy."))["passed"]
    assert check_data_warning_surfaced(
        trace(tool_calls=tc, report="Note: a corporate action may distort this."))["passed"]


def test_task_completion_and_source_diversity():
    assert not check_task_completion(trace(report="too short"))["passed"]
    assert check_task_completion(trace(report="x" * 500))["passed"]
    one_book = [{"source": "a.pdf", "text": "t"}, {"source": "a.pdf", "text": "t2"}]
    assert not check_source_diversity(trace(retrieved_chunks=one_book))["passed"]


def test_run_all_returns_every_check():
    assert len(run_all(trace())) == 12   # bump when a new check is added


def test_pledge_unknown_must_be_declared():
    from pennystock.evals.assertions import check_pledge_disclosed
    tc = [{"name": "get_shareholding", "result": "{'promoter_pct': 11.7, 'pledge_pct': None}"}]
    silent = trace(tool_calls=tc, report="Promoters hold 11.7%. Strong governance. " * 20)
    assert not check_pledge_disclosed(silent)["passed"]
    honest = trace(tool_calls=tc, report="Promoter pledge is UNKNOWN - unverified from free sources. " * 20)
    assert check_pledge_disclosed(honest)["passed"]


def test_shareholding_expected_for_governance_questions():
    t = trace(query="what is the promoter pledging and shareholding pattern?",
              tool_calls=[{"name": "get_indian_stock_data", "result": "{}"}])
    r = check_tool_coverage(t)
    assert not r["passed"] and "get_shareholding" in r["detail"]


def test_tool_coverage_does_not_match_substrings():
    # "mistakes" must NOT trigger the "stake" rule for get_shareholding
    t = trace(query="what are the classic mistakes people make here?", tool_calls=[])
    assert "get_shareholding" not in check_tool_coverage(t)["detail"]
    # but a real mention still does
    t2 = trace(query="what is the promoter stake?", tool_calls=[])
    assert "get_shareholding" in check_tool_coverage(t2)["detail"]


# --- the broker tools (Layers 1-3) ------------------------------------------------------------
from pennystock.evals.assertions import check_no_unknown_tools  # noqa: E402


def test_the_broker_tools_are_known_tools():
    t = trace(tool_calls=[{"name": "get_broker_consensus", "result": "{}"},
                          {"name": "search_broker_research", "result": "[source: a.pdf] x"}])
    assert check_no_unknown_tools(t)["passed"]


def test_a_target_price_question_needs_the_consensus_tool():
    t = trace(query="What target price do brokers have on AU Small Finance Bank?",
              tool_calls=[{"name": "get_stock_fundamentals", "result": "{}"}])
    r = check_tool_coverage(t)
    assert not r["passed"] and "get_broker_consensus" in r["detail"]


def test_a_research_report_question_needs_the_broker_search():
    t = trace(query="Summarise the broker reports on Trent - what is the bull case they make?",
              tool_calls=[{"name": "get_broker_consensus", "result": "{}"}])
    r = check_tool_coverage(t)
    assert not r["passed"] and "search_broker_research" in r["detail"]


def test_broker_passages_do_not_count_as_extra_books():
    """Source diversity guards the BOOK retrieval. Two broker notes plus one book is still one book."""
    t = trace(retrieved_chunks=[
        {"source": "graham.pdf", "text": "a", "tool": "penny_stock_researcher"},
        {"source": "icici.pdf", "text": "b", "tool": "search_broker_research"},
        {"source": "axis.pdf", "text": "c", "tool": "search_broker_research"}])
    assert not check_source_diversity(t)["passed"]


def test_broker_passages_alone_leave_book_diversity_not_applicable():
    t = trace(retrieved_chunks=[{"source": "icici.pdf", "text": "b", "tool": "search_broker_research"}])
    assert check_source_diversity(t)["applicable"] is False


def test_a_quote_from_a_broker_passage_is_traceable():
    t = trace(report='ICICI writes "credit cost is poised to decline in FY27E owing to its improving portfolio quality".',
              retrieved_chunks=[{"source": "icici.pdf", "tool": "search_broker_research",
                                 "text": "Overall, we believe that credit cost is poised to decline in FY27E, "
                                         "owing to its improving portfolio quality."}])
    assert check_citation_provenance(t)["passed"]


def test_evaluation_knows_every_tool_the_graph_can_call():
    """assertions.py mirrors the graph's tool sets so evaluation never constructs model clients.
    A mirror that drifts would flag a real tool as hallucinated, or stop pinning its entity."""
    from pennystock.agents import nodes
    from pennystock.evals import assertions as a
    assert set(nodes.TOOL_REGISTRY) <= a._KNOWN_TOOLS
    assert a._STOCK_TOOLS == nodes.STOCK_TOOLS


# --- quote detection must pair quotation marks, not scan between them --------------------------
REAL_REPORT_SHAPE = ('both brokers maintaining "Buy" ratings.\n\n---\n\n### **Broker Arguments**\n'
                     'AU now focuses on "super-prime" customers (CIBIL >780) for credit cards.')


def test_the_gap_between_two_short_quoted_words_is_not_a_quote():
    """Live 2026-10-02 report: the regex matched from the CLOSING quote of "Buy" to the OPENING quote
    of "super-prime" and failed provenance on text the report never presented as a quotation."""
    t = trace(report=REAL_REPORT_SHAPE,
              retrieved_chunks=[{"source": "b.pdf", "text": "unrelated passage", "tool": "search_broker_research"}])
    assert check_citation_provenance(t)["applicable"] is False


def test_a_fabricated_quote_beside_short_quoted_words_is_still_caught():
    t = trace(report='Rated "Buy"; ICICI wrote "the bank will double its loan book every year without any '
                     'rise in credit cost" in "Q1".',
              retrieved_chunks=[{"source": "b.pdf", "text": "credit cost is poised to decline in FY27E",
                                 "tool": "search_broker_research"}])
    r = check_citation_provenance(t)
    assert r["applicable"] and not r["passed"]


# --- Trendlyne's third-party scores (Stock Report reader) --------------------------------------

def test_the_trendlyne_tool_is_a_known_tool():
    assert check_no_unknown_tools(trace(tool_calls=[{"name": "get_trendlyne_view", "result": "{}"}]))["passed"]


def test_a_dvm_score_question_needs_the_trendlyne_tool():
    t = trace(query="What is the DVM score of AU Small Finance Bank?",
              tool_calls=[{"name": "get_stock_fundamentals", "result": "{}"}])
    r = check_tool_coverage(t)
    assert not r["passed"] and "get_trendlyne_view" in r["detail"]


def test_a_pledge_invocation_question_needs_the_events_tool():
    r = check_tool_coverage(trace(query="Was any promoter pledge invoked at Indian Terrain?",
                                  tool_calls=[{"name": "get_shareholding", "result": "{}"}]))
    assert not r["passed"] and "get_corporate_events" in r["detail"]


def test_a_disclosure_question_needs_the_filings_search():
    r = check_tool_coverage(trace(query="What did AU Small Finance Bank disclose about the merger?", tool_calls=[]))
    assert not r["passed"] and "search_filings" in r["detail"]


def test_context_fits_fails_when_a_model_call_filled_its_window():
    # the Dixon run's final draft: a 16,169-token prompt left 214 tokens of room; the report stopped mid-sentence
    flagged = {"role": "analyst", "input_tokens": 16169, "output_tokens": 214, "num_ctx": 16384,
               "context_full": True, "prompt_near_limit": True}
    r = check_context_fits(trace(llm_calls=[flagged]))
    assert not r["passed"] and "analyst" in r["detail"]
    fits = {**flagged, "input_tokens": 9000, "output_tokens": 2000, "context_full": False,
            "prompt_near_limit": False}
    assert check_context_fits(trace(llm_calls=[fits]))["passed"]
    assert not check_context_fits(trace())["applicable"]          # traces written before the token log


# --- no false "missing", no cut-off report (the DIXON report of 2026-10-10 did both) ---------------------
def test_saying_data_is_missing_when_the_tool_returned_it_fails():
    t = trace(report="No Piotroski F-Score or Altman Z-Score was returned in the provided data.",
              tool_calls=[{"name": "get_forensics", "result": "{'piotroski': {'f_score': 6}}"}])
    r = check_no_false_missing(t)
    assert not r["passed"] and "get_forensics" in r["detail"]


def test_saying_it_is_missing_when_it_really_is_passes():
    t = trace(report="The pledge could not be verified.",
              tool_calls=[{"name": "get_shareholding", "result": "{'pledge_pct_of_promoter_holding': None}"}])
    assert check_no_false_missing(t)["passed"]


def test_false_missing_is_caught_against_the_real_tool_outputs():
    # the check's patterns are matched against what the tools really print (DIXON, captured 2026-10-11)
    import json
    from pathlib import Path
    raw = json.loads((Path(__file__).parent / "fixtures" / "dixon" / "tool_outputs.json").read_text())
    calls = [{"name": n, "result": out} for n, out in raw.items()]
    for claim in ("Related-party transactions were not disclosed in the data.",
                  "The contingent liabilities could not be quantified.",
                  "Promoter pledging data is unavailable.",
                  "The stock's own historical P/E band is missing."):
        assert not check_no_false_missing(trace(report=claim, tool_calls=calls))["passed"], claim
    assert check_no_false_missing(trace(report="Revenue grew 19%.", tool_calls=calls))["passed"]


def test_a_report_cut_mid_sentence_fails():
    cut = ("## Verdict\n*   **System View:** The system views the stock as a mid-range performer "
           "(Durability: High")
    assert not check_report_complete(trace(report=cut))["passed"]
    assert check_report_complete(trace(report="## Verdict\nA growth stock, not a value stock."))["passed"]
    stopped = trace(report="Fine.", llm_calls=[{"role": "analyst", "done_reason": "length"}])
    assert not check_report_complete(stopped)["passed"]


def test_a_bullet_ending_in_a_number_is_complete_but_a_dangling_word_is_not():
    bullet = "## Technicals\n*   **Price vs 200-day average:** above it, at Rs 12,097"
    assert check_report_complete(trace(report=bullet))["passed"]
    dangling = "The risks are concentrated in two customers, and the order book depends on"
    assert not check_report_complete(trace(report=dangling))["passed"]


def test_a_quote_taken_from_a_tool_result_counts_as_traced():
    # the live DIXON run (2026-10-11) quoted the earnings call verbatim from get_management_commentary; only
    # book and broker passages were searched, so the check failed a real quote
    t = trace(report='Management said "we expect the JV to commence operations and start reflecting in our '
                     'revenues from Q3" this year.',
              retrieved_chunks=[{"source": "b.pdf",
                                 "text": "The margin of safety is always dependent on price."}],
              tool_calls=[{"name": "get_management_commentary",
                           "result": "{'guidance_latest': ['We expect the JV to commence operations and "
                                     "start reflecting in our revenues from Q3 of the current fiscal.']}"}])
    assert check_citation_provenance(t)["passed"]


def test_an_elided_quote_is_checked_piece_by_piece():
    # live DIXON run: "higher P/E ratios... shrink the margin of safety" elides "and interest rates" from the
    # book's sentence; every 8-word window crossed the '...' and the real quote failed
    book = [{"source": "ii.epub",
             "text": "Conversely, higher P/E ratios and interest rates shrink the margin of safety."}]
    real = trace(report='Graham: "higher P/E ratios... shrink the margin of safety".', retrieved_chunks=book)
    made_up = trace(report='Graham: "higher P/E ratios... always destroy the value of every '
                           'shareholder".', retrieved_chunks=book)
    assert check_citation_provenance(real)["passed"] and not check_citation_provenance(made_up)["passed"]

from pennystock.evals.assertions import (
    check_citation_provenance, check_data_warning_surfaced, check_entity_consistency,
    check_numeric_provenance, check_source_diversity, check_task_completion,
    check_tool_coverage, run_all,
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
    assert len(run_all(trace())) == 9   # bump when a new check is added


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

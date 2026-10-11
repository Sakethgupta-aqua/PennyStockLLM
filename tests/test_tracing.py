"""Trace building for the broker research tool: a quote taken from a broker passage must be
checkable against that passage, and must not count as a second BOOK for diversity."""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from pennystock.tracing import build_trace


def _state(tool, result):
    return {"messages": [HumanMessage(content="q"),
                         AIMessage(content="", tool_calls=[{"name": tool, "args": {}, "id": "t1"}]),
                         ToolMessage(content=result, tool_call_id="t1"),
                         AIMessage(content="final report")]}


def test_broker_passages_become_retrieved_chunks_tagged_with_their_tool():
    t = build_trace("q", _state("search_broker_research", "[source: a.pdf] credit cost to normalise"))
    assert t["retrieved_chunks"] == [{"source": "a.pdf", "text": "credit cost to normalise",
                                      "tool": "search_broker_research"}]


def test_book_passages_are_tagged_too():
    t = build_trace("q", _state("penny_stock_researcher", "[source: graham.pdf] margin of safety"))
    assert t["retrieved_chunks"][0]["tool"] == "penny_stock_researcher"


def test_a_result_set_aside_to_fit_the_window_keeps_its_full_text_in_the_trace():
    # the model saw a one-line note; the trace must still hold what the tool returned (numeric checks use it)
    msgs = [AIMessage(content="", tool_calls=[{"name": "get_financials", "args": {}, "id": "c1"}]),
            ToolMessage(content="[get_financials result left out to fit the model's window]",
                        tool_call_id="c1", artifact="{'revenue_cr': [48537.39]}",
                        additional_kwargs={"trimmed": True})]
    call = build_trace("q", {"messages": msgs})["tool_calls"][0]
    assert call["result"] == "{'revenue_cr': [48537.39]}" and call["trimmed"] is True


def test_the_trace_records_the_pack_calls_and_the_fact_sheet():
    state = {"messages": [AIMessage(content="REPORT")],
             "evidence": {"fact_sheet": "EVIDENCE PACK - x", "built_at": "t", "call_keys": [],
                          "calls": [{"name": "get_forensics", "args": {"symbol": "ABC"}, "ok": True,
                                     "seconds": 2.0, "result": "{'piotroski': 6}"}]}}
    t = build_trace("q", state)
    assert t["tool_calls"][0] == {"seq": 1, "name": "get_forensics", "args": {"symbol": "ABC"},
                                  "result": "{'piotroski': 6}", "ok": True, "trimmed": False,
                                  "source": "evidence_pack", "seconds": 2.0}
    assert t["evidence"]["fact_sheet"] == "EVIDENCE PACK - x"

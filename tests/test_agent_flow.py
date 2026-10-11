"""The agent graph's plumbing: routing, what each model is shown, and what a run records.

Everything here was written in response to a real run. The Dixon question of 2026-10-10 called 23 tools, and
the report then said the forensics, shareholding and five-year financials were "not returned" - the tools had
returned them, but the conversation outgrew the analyst's 16,384-token window and Ollama silently dropped the
oldest messages. An 8B router with an 8,192-token window sent the work back to the analyst nine times (11
drafts piling up in the history), and the fact-checker read only the first 8,000 characters of tool output.
The models here are stand-ins; the graph, nodes and routing are the real ones.
"""
import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from pennystock.agents import nodes
from pennystock.agents.graph import build_graph
from pennystock.data.resolver import resolve_from_query as _real_resolve


class FakeLLM:
    """A stand-in for one ChatOllama role: scripted replies in order (the last repeats); payloads kept."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.payloads = []

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, payload):
        self.payloads.append(payload)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, dict):                     # a tool-calling turn, or a reply with token counts
            usage = reply.get("usage")
            return AIMessage(content=reply.get("content", ""), tool_calls=reply.get("tool_calls", []),
                             usage_metadata={"input_tokens": usage[0], "output_tokens": usage[1],
                                             "total_tokens": sum(usage)} if usage else None,
                             response_metadata={"done_reason": reply.get("done_reason", "stop")})
        return AIMessage(content=reply)


class FakeTool:
    """A stand-in tool: fixed output, every call's arguments kept."""

    def __init__(self, output):
        self.output, self.calls = output, []

    async def ainvoke(self, args):
        self.calls.append(args)
        return self.output


def tool_call(name, call_id="c1", **args):
    return {"tool_calls": [{"name": name, "args": args, "id": call_id}]}


class NoSupervisor:
    def with_structured_output(self, *args, **kwargs):
        raise AssertionError("the 8B supervisor must not be called: the pipeline is a fixed sequence")


def run_graph(monkeypatch, analyst, grounding="GROUNDED", judge="Clear and grounded. Score: 5", resolve=None,
              question="Analyze ABC"):
    """Run the real graph with stand-in models. `grounding` / `judge` take a reply or a FakeLLM; a company is
    resolved only when `resolve` is given or the test pinned one itself (no company: no evidence pack)."""
    if resolve is not None:
        monkeypatch.setattr(nodes, "resolve_from_query", resolve)
    elif nodes.resolve_from_query is _real_resolve:
        monkeypatch.setattr(nodes, "resolve_from_query", lambda q: None)
    monkeypatch.setattr(nodes, "llm_data_agent", FakeLLM("Live price: Rs 100"))
    monkeypatch.setattr(nodes, "llm_analyst_agent", analyst)
    as_llm = lambda x: x if isinstance(x, FakeLLM) else FakeLLM(x)      # noqa: E731
    monkeypatch.setattr(nodes, "llm_grounding", as_llm(grounding))
    monkeypatch.setattr(nodes, "llm_judge_agent", as_llm(judge))
    monkeypatch.setattr(nodes, "llm_supervisor", NoSupervisor(), raising=False)

    async def go():
        path, final = [], None
        async for mode, chunk in build_graph().astream({"messages": [HumanMessage(content=question)]},
                                                       stream_mode=["updates", "values"]):
            if mode == "updates":
                path.extend(chunk)
            else:
                final = chunk
        return path, final

    return asyncio.run(go())


# --- routing: a fixed sequence, no LLM router ----------------------------------------------------
def test_a_run_goes_data_then_analyst_then_fact_check_then_judge_without_a_router(monkeypatch):
    path, final = run_graph(monkeypatch, FakeLLM("REPORT: ABC looks fine."))
    assert path == ["resolve_entity", "MarketDataAgent", "evidence_pack", "FinancialAnalystAgent",
                    "grounding_check", "EvaluatorAgent"]
    assert "Passed" in final["messages"][-1].content


def test_the_graph_has_no_supervisor_node():
    assert "supervisor_node" not in build_graph().nodes


def test_a_fact_check_that_has_used_its_retries_goes_straight_to_the_judge(monkeypatch):
    monkeypatch.setattr(nodes, "llm_grounding", FakeLLM("UNSUPPORTED: everything"))
    state = {"messages": [HumanMessage(content="q"), ToolMessage(content="price 100", tool_call_id="1"),
                          AIMessage(content="REPORT")],
             "grounding_retries": nodes.settings.max_grounding_retries}
    assert asyncio.run(nodes.grounding_check_node(state)).goto == "EvaluatorAgent"


# --- history: only the latest draft stays ---------------------------------------------------------
def test_a_rewrite_sees_its_previous_draft_but_only_the_latest_draft_stays_in_the_history(monkeypatch):
    # 11 drafts of ~1,700 tokens each piled up in the Dixon run and pushed the tool results out of the window
    monkeypatch.setitem(nodes.TOOL_REGISTRY, "get_price_history",
                        (FakeTool("last close 100"), "FinancialAnalystAgent"))
    analyst = FakeLLM(tool_call("get_price_history", symbol="ABC"), "DRAFT 1", "DRAFT 2")
    path, final = run_graph(monkeypatch, analyst, grounding="UNSUPPORTED: the margin figure")
    assert path.count("FinancialAnalystAgent") == 3                # tools, draft 1, then the bounced rewrite
    assert any(m.content == "DRAFT 1" for m in analyst.payloads[2])  # the rewrite can see what it is revising
    drafts = [m.content for m in final["messages"]
              if isinstance(m, AIMessage) and m.content.startswith("DRAFT")]
    assert drafts == ["DRAFT 2"]


# --- tools: an identical call is answered with a pointer, not a second copy ---------------------
def test_repeating_a_tool_call_points_to_the_earlier_result_instead_of_fetching_it_again(monkeypatch):
    # calls 17-22 of the Dixon run re-sent 4,497 tokens the conversation already held; the args differed only
    # by a default the model sometimes spells out (exchange="NSE")
    from langchain_core.tools import tool

    fetched = []

    @tool
    async def get_price_history(symbol: str, exchange: str = "NSE", period: str = "1y") -> str:
        """Price history."""
        fetched.append(symbol)
        return "last close 100"

    monkeypatch.setitem(nodes.TOOL_REGISTRY, "get_price_history",
                        (get_price_history, "FinancialAnalystAgent"))
    analyst = FakeLLM(tool_call("get_price_history", "c1", symbol="ABC", exchange="NSE"),
                      tool_call("get_price_history", "c2", symbol="ABC"), "REPORT")
    _, final = run_graph(monkeypatch, analyst)
    assert fetched == ["ABC"]
    repeat = next(m for m in final["messages"] if isinstance(m, ToolMessage) and m.tool_call_id == "c2")
    assert "earlier get_price_history call" in repeat.content and "last close 100" not in repeat.content


# --- the fact-check reads what the writer read -----------------------------------------------------
def _grounding_prompt(monkeypatch, tool_msgs):
    checker = FakeLLM("GROUNDED")
    monkeypatch.setattr(nodes, "llm_grounding", checker)
    state = {"messages": [HumanMessage(content="q"), *tool_msgs, AIMessage(content="REPORT")],
             "grounding_retries": 0}
    asyncio.run(nodes.grounding_check_node(state))
    return checker.payloads[0][0].content


def test_the_fact_check_sees_late_tool_results_and_skips_repeat_pointers(monkeypatch):
    # it read only the first 8,000 characters (calls 1-8 in the Dixon run: the half the writer had lost)
    msgs = [ToolMessage(content="EARLY piotroski 6/9", tool_call_id="1")]
    msgs += [ToolMessage(content="filler " * 400, tool_call_id=f"f{i}") for i in range(4)]   # ~11,200 chars
    msgs += [ToolMessage(content="LATE target 16,300", tool_call_id="2"),
             ToolMessage(content="Same result as the earlier get_x call", tool_call_id="3",
                         additional_kwargs={"call_key": "k", "repeat": True})]
    prompt = _grounding_prompt(monkeypatch, msgs)
    assert "EARLY piotroski 6/9" in prompt and "LATE target 16,300" in prompt
    assert "Same result as the earlier" not in prompt


def test_evidence_over_the_budget_drops_the_oldest_results_first(monkeypatch):
    monkeypatch.setattr(nodes.settings, "max_grounding_chars", 60)
    prompt = _grounding_prompt(monkeypatch, [ToolMessage(content="OLD " * 20, tool_call_id="1"),
                                             ToolMessage(content="NEWEST result", tool_call_id="2")])
    assert "NEWEST result" in prompt and "OLD" not in prompt


# --- every model call records its tokens against its window ---------------------------------------
def test_every_model_call_is_recorded_and_one_that_filled_its_window_is_flagged(monkeypatch):
    # Ollama trims an over-long prompt silently and traces kept no token counts, so the overflow looked like
    # bad extraction. 11 of 13 analyst calls in the Dixon run ended at the ceiling: prompt + answer = window.
    from pennystock.tracing import build_trace

    window = nodes.settings.large_num_ctx
    analyst = FakeLLM({"content": "REPORT", "usage": (window - 1536, 1536), "done_reason": "length"})
    _, final = run_graph(monkeypatch, analyst)
    calls = final["llm_calls"]
    assert [c["role"] for c in calls] == ["data", "analyst", "judge"]
    assert (calls[1]["input_tokens"], calls[1]["output_tokens"], calls[1]["num_ctx"]) == (window - 1536, 1536,
                                                                                         window)
    assert calls[1]["context_full"] and not calls[0]["context_full"]
    assert build_trace("q", final)["context_overflow"] is True


# --- the analyst's prompt is fitted to its window on purpose, never silently by Ollama -----------
def test_an_over_budget_prompt_sets_aside_the_oldest_tool_results_and_keeps_the_question(monkeypatch):
    # Ollama would drop whole messages from the start - the user's question first - with no error
    s = nodes.settings
    monkeypatch.setattr(s, "large_num_ctx", 20_000)
    monkeypatch.setattr(s, "answer_reserve_tokens", 4_000)
    monkeypatch.setattr(s, "tool_schema_tokens", 0)
    monkeypatch.setattr(s, "chars_per_token_estimate", 4.0)        # budget: 16,000 tokens = 64,000 chars
    analyst = FakeLLM("REPORT")
    monkeypatch.setattr(nodes, "llm_analyst_agent", analyst)
    state = {"messages": [
        HumanMessage(content="Is ABC a buy?", id="q"),
        AIMessage(content="", tool_calls=[{"name": "get_financials", "args": {}, "id": "c1"},
                                          {"name": "get_price_history", "args": {}, "id": "c2"}], id="a1"),
        ToolMessage(content="OLD " * 15_000, tool_call_id="c1", name="get_financials", id="old"),  # 60K chars
        ToolMessage(content="NEW last close 100", tool_call_id="c2", name="get_price_history", id="new")]}
    update = asyncio.run(nodes.financial_analyst_agent(state)).update
    sent = " ".join(str(m.content) for m in analyst.payloads[0])
    assert "Is ABC a buy?" in sent and "NEW last close 100" in sent and "OLD OLD" not in sent
    assert "get_financials result left out to fit" in sent
    swapped = next(m for m in update["messages"] if getattr(m, "id", None) == "old")   # replaced in history
    assert swapped.additional_kwargs.get("trimmed") is True
    assert update["llm_calls"][0]["trimmed_results"] == 1


# --- the evidence pack: the data tools run in code, the analyst gets one fact sheet ---------------------
# The Dixon run of 2026-10-10 spent 23 tool calls and ~766 s gathering what code can gather in parallel; the
# fact sheet carries every figure with its source and every comparison already made (agents/evidence.py).
def _pack(monkeypatch, sheet="EVIDENCE PACK - Abc (ABC) ... Piotroski F-score 6/9",
          ok_tools=("get_forensics",)):
    from pennystock.agents.evidence import EvidencePack, PackCall
    calls = [PackCall(n, {"symbol": "ABC"}, "{'company': 'Abc'}", {"company": "Abc"}, True, 1.0)
             for n in ok_tools]

    async def build(symbol, question):
        return EvidencePack(symbol, "Abc", sheet, calls, "2026-10-11T10:00:00")
    monkeypatch.setattr(nodes, "build_pack", build)


def _abc(q):
    return {"symbol": "ABC", "company_name": "Abc"}


def test_a_resolved_company_gets_its_evidence_pack_before_the_analyst(monkeypatch):
    _pack(monkeypatch)
    analyst = FakeLLM("REPORT")
    path, final = run_graph(monkeypatch, analyst, resolve=_abc)
    assert path[:3] == ["resolve_entity", "MarketDataAgent", "evidence_pack"]
    sent = analyst.payloads[0]
    assert any(getattr(m, "name", None) == nodes.EVIDENCE and "Piotroski" in m.content for m in sent)
    assert final["evidence"]["call_keys"] and final["evidence"]["calls"][0]["name"] == "get_forensics"


def test_re_calling_a_pack_tool_points_to_the_pack(monkeypatch):
    _pack(monkeypatch)
    forensics = FakeTool("{'piotroski': 6}")
    monkeypatch.setitem(nodes.TOOL_REGISTRY, "get_forensics", (forensics, "FinancialAnalystAgent"))
    analyst = FakeLLM(tool_call("get_forensics", symbol="ABC"), "REPORT")
    _, final = run_graph(monkeypatch, analyst, resolve=_abc)
    reply = next(m for m in final["messages"] if isinstance(m, ToolMessage) and m.tool_call_id == "c1")
    assert forensics.calls == [] and "Already in the EVIDENCE PACK" in reply.content


def test_the_fact_check_and_the_judge_read_the_fact_sheet(monkeypatch):
    _pack(monkeypatch, sheet="EVIDENCE PACK - sheet text 4,676.67")
    monkeypatch.setitem(nodes.TOOL_REGISTRY, "get_price_history",
                        (FakeTool("last close 100"), "FinancialAnalystAgent"))
    checker, judge = FakeLLM("GROUNDED"), FakeLLM("Faithful. Score: 5")
    analyst = FakeLLM(tool_call("get_price_history", symbol="ABC"), "REPORT")
    run_graph(monkeypatch, analyst, grounding=checker, judge=judge, resolve=_abc)
    assert "sheet text 4,676.67" in checker.payloads[0][0].content
    assert "sheet text 4,676.67" in judge.payloads[0][0].content


def test_a_failed_pack_leaves_the_analyst_its_tools(monkeypatch):
    async def broken(symbol, question):
        raise TimeoutError("every source timed out")
    monkeypatch.setattr(nodes, "build_pack", broken)
    analyst = FakeLLM("REPORT")
    path, final = run_graph(monkeypatch, analyst, resolve=_abc)
    assert "FinancialAnalystAgent" in path and not final.get("evidence")
    assert not any(getattr(m, "name", None) == nodes.EVIDENCE for m in analyst.payloads[0])


# --- the fact-check's verdict: a reply that found nothing passes even when it explains itself ------------
def test_a_verdict_that_finds_nothing_and_ends_grounded_passes(monkeypatch):
    # the live DIXON run (2026-10-11): this reply was read as a failure and forced a full rewrite
    verdict = ("No unsupported claims found. All specific figures, ratios, and attributions in the report "
               "are either directly present in the evidence or are direct derivations.\n\nGROUNDED")
    monkeypatch.setattr(nodes, "llm_grounding", FakeLLM(verdict))
    state = {"messages": [HumanMessage(content="q"), ToolMessage(content="price 100", tool_call_id="1"),
                          AIMessage(content="REPORT")], "grounding_retries": 0}
    assert asyncio.run(nodes.grounding_check_node(state)).goto == "EvaluatorAgent"


# --- the text the question needs: filings, broker reasoning, the books --------------------------------
# The live DIXON run (2026-10-11) answered from the evidence pack alone: no filings search, no broker report,
# no book - and wrote 'brokers cite the strong order book...' with no broker passage behind it.
TEXT_Q = "Analyze ABC: what has it disclosed in its filings, and does it pass Graham's margin of safety?"


def test_the_pack_message_says_which_text_tools_the_question_needs(monkeypatch):
    _pack(monkeypatch)
    analyst = FakeLLM("REPORT")
    run_graph(monkeypatch, analyst, resolve=_abc, question=TEXT_Q)
    sheet = next(m for m in analyst.payloads[0] if getattr(m, "name", None) == nodes.EVIDENCE).content
    assert "TEXT TO FETCH" in sheet and "search_filings" in sheet and "penny_stock_researcher" in sheet
    assert "search_broker_research" not in sheet.split("TEXT TO FETCH")[1]


def test_a_first_draft_that_skipped_the_text_tools_is_sent_back_once(monkeypatch):
    _pack(monkeypatch)
    analyst = FakeLLM("DRAFT 0", "DRAFT 1")              # never calls a tool
    path, final = run_graph(monkeypatch, analyst, resolve=_abc, question=TEXT_Q)
    nudge = str(analyst.payloads[1][-1].content)
    assert "search_filings" in nudge and "penny_stock_researcher" in nudge
    assert path.count("FinancialAnalystAgent") == 2      # one nudge, then the draft is accepted
    texts = [m.content for m in final["messages"] if isinstance(m, AIMessage)]
    assert "DRAFT 0" not in texts and "DRAFT 1" in texts


# --- a runaway fact-check never reaches the analyst --------------------------------------------------
def _audit(monkeypatch, reply):
    checker = FakeLLM(reply)
    monkeypatch.setattr(nodes, "llm_grounding", checker)
    state = {"messages": [HumanMessage(content="q"), ToolMessage(content="price 100", tool_call_id="1"),
                          AIMessage(content="REPORT")], "grounding_retries": 0}
    return asyncio.run(nodes.grounding_check_node(state))


def test_a_verdict_cut_at_its_length_limit_is_not_used(monkeypatch):
    runaway = {"content": "Let's re-read carefully. " * 4000, "done_reason": "length"}
    out = _audit(monkeypatch, runaway)
    assert out.goto == "EvaluatorAgent" and not (out.update or {}).get("messages")


def test_a_long_verdict_is_clipped_before_it_goes_back_to_the_analyst(monkeypatch):
    out = _audit(monkeypatch, "UNSUPPORTED: the margin figure\n" + "detail " * 3000)
    feedback = out.update["messages"][0].content
    assert out.goto == "FinancialAnalystAgent" and "the margin figure" in feedback
    assert len(feedback) < nodes.settings.grounding_feedback_chars + 1000


# --- the same passages twice are held once -------------------------------------------------------------
def test_a_result_identical_to_one_already_held_becomes_a_pointer(monkeypatch):
    # penny_stock_researcher, three queries, the same 10,180 characters three times (live DIXON run)
    books = FakeTool("[source: graham.pdf] margin of safety " * 50)
    monkeypatch.setitem(nodes.TOOL_REGISTRY, "penny_stock_researcher", (books, "FinancialAnalystAgent"))
    analyst = FakeLLM(tool_call("penny_stock_researcher", "c1", query="Graham margin of safety"),
                      tool_call("penny_stock_researcher", "c2", query="Graham criteria"), "REPORT")
    _, final = run_graph(monkeypatch, analyst)
    second = next(m for m in final["messages"] if isinstance(m, ToolMessage) and m.tool_call_id == "c2")
    assert "Same text as the earlier penny_stock_researcher result" in second.content
    assert second.additional_kwargs.get("repeat") is True


def test_a_long_judge_feedback_is_clipped_before_it_goes_back_to_the_analyst(monkeypatch):
    monkeypatch.setattr(nodes, "llm_judge_agent", FakeLLM("Weak report. " + "reason " * 3000 + "Score: 2"))
    state = {"messages": [HumanMessage(content="q"), AIMessage(content="REPORT")], "revisions": 0}
    out = asyncio.run(nodes.evaluator_node(state))
    feedback = out.update["messages"][0].content
    assert out.goto == "FinancialAnalystAgent" and len(feedback) < nodes.settings.grounding_feedback_chars + 1000


def test_the_closing_instruction_asks_for_the_missing_text_first(monkeypatch):
    # live runs 2-3: the model ignored the pack's TEXT TO FETCH section and wrote a draft the gate then threw
    # away (~2,600 tokens); the last message it reads is where the instruction belongs
    _pack(monkeypatch)
    analyst = FakeLLM("REPORT")
    run_graph(monkeypatch, analyst, resolve=_abc, question=TEXT_Q)
    closing = str(analyst.payloads[0][-1].content)
    assert closing.startswith("Before writing the report, call") and "search_filings" in closing


def test_a_verdict_of_not_grounded_is_a_failure(monkeypatch):
    # 'GROUNDED' anywhere in the first 60 characters used to pass - including 'NOT GROUNDED' and 'UNGROUNDED'
    for verdict in ("NOT GROUNDED: the margin figure is not in the evidence.",
                    "UNGROUNDED - the target price of Rs 18,000 appears nowhere."):
        assert _audit(monkeypatch, verdict).goto == "FinancialAnalystAgent", verdict
    for verdict in ("GROUNDED", "**GROUNDED**", "Verdict: GROUNDED"):
        assert _audit(monkeypatch, verdict).goto == "EvaluatorAgent", verdict

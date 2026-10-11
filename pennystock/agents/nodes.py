"""The LangGraph agent nodes (entity resolution, data, evidence pack, analyst, fact-check, evaluator, tools).

The pipeline is a fixed sequence - resolve -> data agent -> evidence pack -> analyst (<-> tools) -> fact-check
-> judge - so routing is code, not a model. An 8B LLM router used to sit between the stages; with an
8,192-token window it could not follow a 16-tool run and sent the analyst back to rewrite nine times
(2026-10-10)."""
import json
import logging
import re
from contextlib import suppress
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph import END
from langgraph.types import Command


from pennystock.agents.prompts import (
    ANALYST_PROMPT,
    DATA_AGENT_PROMPT,
    GROUNDING_CHECK_PROMPT,
    JUDGE_REFERENCE_ANSWER,
    JUDGE_RUBRIC,
)

from pennystock.agents.evidence import build_pack
from pennystock.evals.assertions import expected_tools
from pennystock.data.resolver import resolve_from_query
from pennystock.agents.state import AgentState
from pennystock.config import settings
from pennystock.llms import analyst_llm, data_agent_llm, grounding_llm, judge_llm
from pennystock.tools import (
    get_annual_report_notes,
    get_broker_consensus,
    get_trendlyne_view,
    search_filings,
    screen_stocks,
    search_broker_research,
    analyst_tools,
    data_tools,
    get_corporate_actions,
    get_corporate_events,
    get_financials,
    get_forensics,
    get_management_commentary,
    get_indian_stock_data,
    get_price_history,
    get_shareholding,
    get_stock_fundamentals,
    get_valuation_context,
    penny_stock_researcher,
)

logger = logging.getLogger(__name__)

# tags the analyst's report drafts (ChatOllama never sends a message's name to the model)
DRAFT = "analyst_draft"
EVIDENCE = "evidence_pack"        # the name of the message that carries the fact sheet
# the tools that return TEXT - the evidence pack holds numbers only. The live DIXON run (2026-10-11) answered
# from the pack alone and wrote "brokers cite the strong order book" with no broker passage behind it.
TEXT_TOOLS = {"search_filings": "what the company disclosed in its filings, presentations and calls",
              "search_broker_research": "the brokers' reasons, in their own words",
              "penny_stock_researcher": "the framework or principle asked about (e.g. Graham)"}
NUDGE = "BEFORE WRITING THE REPORT"

llm_data_agent = data_agent_llm()
llm_analyst_agent = analyst_llm()
llm_judge_agent = judge_llm()
llm_grounding = grounding_llm()

def _usage(role: str, response, num_ctx: int) -> dict:
    """The tokens one model call used against its window. Ollama trims an over-long prompt WITHOUT an
    error, so a call whose prompt + answer reached the window, or whose prompt came close, is flagged."""
    usage = getattr(response, "usage_metadata", None) or {}
    prompt, answer = usage.get("input_tokens"), usage.get("output_tokens")
    full = prompt is not None and answer is not None and prompt + answer >= num_ctx - 1
    near = prompt is not None and prompt >= settings.context_warn_share * num_ctx
    if full or near:
        logger.error(f"{role} call {'filled' if full else 'nearly filled'} its {num_ctx:,}-token window "
                     f"(prompt {prompt:,}, answer {answer or 0:,}): Ollama may have dropped the oldest "
                     "messages or cut the answer short.")
    return {"role": role, "input_tokens": prompt, "output_tokens": answer, "num_ctx": num_ctx,
            "done_reason": (getattr(response, "response_metadata", None) or {}).get("done_reason"),
            "context_full": full, "prompt_near_limit": near}


async def resolve_entity_node(state: AgentState) -> Command:
    """Resolve the ONE stock named in the user's query — deterministically, ONCE, before any
    agent runs. Pins it in state so every downstream tool analyzes the SAME company, no matter
    what string an LLM later types. This is the single source of truth for 'which stock'."""
    query = str(state["messages"][0].content)
    match = resolve_from_query(query)                     # prefers the LONGEST name -> TMPV, not TMCV
    if match:
        logger.info(f"Entity pinned: {match['symbol']} ({match['company_name']}).")
        return Command(goto="MarketDataAgent", update={
            "resolved_symbol": match["symbol"], "resolved_name": match["company_name"]})
    logger.info("No confident entity in the query; tools will resolve per-call (legacy fallback).")
    return Command(goto="MarketDataAgent")

async def market_data_agent(state: AgentState) -> Command[Literal["tools", "evidence_pack"]]:
    system_prompt = SystemMessage(content=DATA_AGENT_PROMPT)
    response = await llm_data_agent.bind_tools(data_tools).ainvoke([system_prompt] + state["messages"])
    calls = [_usage("data", response, settings.num_ctx)]
    if hasattr(response, "tool_calls") and response.tool_calls:
        return Command(goto="tools", update={"messages": [response], "llm_calls": calls})
    return Command(goto="evidence_pack", update={"messages": [response], "llm_calls": calls})


async def evidence_pack_node(state: AgentState) -> Command[Literal["FinancialAnalystAgent"]]:
    """Gather the structured data in code and hand the analyst one fact sheet (pennystock/agents/evidence.py).
    Discovery questions (no company) keep the tool-calling path, and so does a pack that fails."""
    symbol = state.get("resolved_symbol")
    if not symbol:
        return Command(goto="FinancialAnalystAgent")
    try:
        pack = await build_pack(symbol, str(state["messages"][0].content))
    except Exception as e:                            # the analyst still has every tool
        logger.warning(f"Evidence pack failed for {symbol} ({type(e).__name__}: {e}); the analyst will call "
                       "tools.")
        return Command(goto="FinancialAnalystAgent")
    keys = [_call_key(c.name, TOOL_REGISTRY[c.name][0], c.args) for c in pack.calls
            if c.ok and c.name in TOOL_REGISTRY]
    evidence = {"fact_sheet": pack.fact_sheet, "call_keys": keys, "built_at": pack.built_at,
                "calls": [{"name": c.name, "args": c.args, "ok": c.ok, "seconds": c.seconds,
                           "result": c.output} for c in pack.calls]}
    needed = [t for t in expected_tools(str(state["messages"][0].content)) if t in TEXT_TOOLS]
    note = ("\n\n### TEXT TO FETCH BEFORE WRITING\nThis pack holds numbers only; the question also asks for "
            "text - call these first:\n" + "\n".join(f"- {t}: {TEXT_TOOLS[t]}" for t in needed)
            if needed else "")
    return Command(goto="FinancialAnalystAgent",
                   update={"messages": [HumanMessage(content=pack.fact_sheet + note, name=EVIDENCE)],
                           "evidence": evidence})


def _missing_text_tools(state: AgentState) -> list[str]:
    used = {tc["name"] for m in state["messages"] for tc in (getattr(m, "tool_calls", None) or [])}
    used |= {c["name"] for c in (state.get("evidence") or {}).get("calls") or []}
    return [t for t in expected_tools(str(state["messages"][0].content)) if t in TEXT_TOOLS and t not in used]


def _estimated_tokens(messages: list) -> int:
    chars = sum(len(str(m.content)) + len(json.dumps(getattr(m, "tool_calls", None) or [], default=str))
                for m in messages)
    return int(chars / settings.chars_per_token_estimate) + settings.tool_schema_tokens


def _fit_window(messages: list, extra: list) -> tuple[list, list]:
    """Fit the analyst's prompt to its window on purpose: set aside the oldest tool results first, each
    replaced by a one-line note (the full result stays in `artifact`, which the model never sees, for the
    trace), until the estimate leaves room for the answer. Returns (messages to send, replacements)."""
    budget = settings.large_num_ctx - settings.answer_reserve_tokens
    msgs, swaps = list(messages), []
    for i, m in enumerate(msgs):
        if _estimated_tokens(msgs + extra) <= budget:
            break
        if isinstance(m, ToolMessage) and not m.additional_kwargs.get("trimmed"):
            name = m.name or "tool"
            msgs[i] = ToolMessage(
                content=f"[{name} result left out to fit the model's window ({len(str(m.content)):,} "
                        f"characters). Call {name} again if you need it.]",
                tool_call_id=m.tool_call_id, name=m.name, id=m.id, artifact=m.content,
                additional_kwargs={**m.additional_kwargs, "trimmed": True})
            swaps.append(msgs[i])
    if swaps:
        logger.warning(f"Analyst prompt over budget: set aside the {len(swaps)} oldest tool result(s) to fit "
                       f"the {settings.large_num_ctx:,}-token window.")
    if _estimated_tokens(msgs + extra) > budget:
        logger.error("Analyst prompt is over budget even with every tool result set aside.")
    return msgs, swaps


async def financial_analyst_agent(state: AgentState) -> Command[
        Literal["tools", "grounding_check", "FinancialAnalystAgent", "__end__"]]:
    # LangGraph 1.1 allows 10,007 steps by default: this only fires under an explicit, lower recursion_limit
    if state.get("remaining_steps", settings.recursion_steps_default) <= settings.recursion_floor:
        logger.warning("Recursion limit approaching. Forcefully terminating graph.")
        return Command(goto=END)
    system_prompt = SystemMessage(content=ANALYST_PROMPT)
    last = str(state["messages"][-1].content)
    is_feedback = ("Your previous report failed quality checks" in last or "GROUNDING CHECK FAILED" in last
                   or last.startswith(NUDGE))
    to_fetch = [] if is_feedback or state.get("text_nudged") else _missing_text_tools(state)
    if is_feedback:
        extra = []
    elif to_fetch:     # the last message read is where the model decides; a section in the pack was ignored
        extra = [HumanMessage(content="Before writing the report, call " + "; ".join(
            f"{t} ({TEXT_TOOLS[t]})" for t in to_fetch) + " - the question asks for text the data above does "
            "not hold.")]
    else:
        extra = [HumanMessage(content="The tool execution is complete. Please review the data above and "
                                      "proceed with your analysis or further tool calls.")]
    messages, swaps = _fit_window(state["messages"], [system_prompt] + extra)
    payload = [system_prompt] + messages + extra

    response = await llm_analyst_agent.bind_tools(analyst_tools).ainvoke(payload)
    calls = [{**_usage("analyst", response, settings.large_num_ctx), "trimmed_results": len(swaps)}]
    if hasattr(response, "tool_calls") and response.tool_calls:
        return Command(goto="tools", update={"messages": swaps + [response], "llm_calls": calls})
    # once per run: a draft written without the text the question asks for goes back (it is not kept)
    missing = [] if state.get("text_nudged") else _missing_text_tools(state)
    if missing:
        logger.warning(f"First draft skipped text tools the question needs: {missing}; sending it back once.")
        nudge = HumanMessage(content=(
            f"{NUDGE}: the question asks for text that the evidence pack does not hold. Call "
            + "; ".join(f"{t} ({TEXT_TOOLS[t]})" for t in missing)
            + " - then write the COMPLETE report, quoting what they return with its source."))
        return Command(goto="FinancialAnalystAgent",
                       update={"messages": swaps + [nudge], "text_nudged": True, "llm_calls": calls})
    # a new draft replaces the old ones: a rewrite has just read its previous draft, and every older draft
    # left in the history is ~1,700 tokens that would push tool results out of the window
    response.name = DRAFT
    stale = [RemoveMessage(id=m.id) for m in state["messages"]
             if isinstance(m, AIMessage) and m.name == DRAFT]
    return Command(goto="grounding_check",
                   update={"messages": swaps + stale + [response], "llm_calls": calls})


def _evidence(messages: list) -> str:
    """The fact sheet, then every tool result the writer holds, oldest first. Over budget, the OLDEST results
    go first - the same end the writer's own view is trimmed from - so the auditor never judges against data
    the writer could not see. The fact sheet is never trimmed from the writer's view, so never from this."""
    sheet = next((str(m.content) for m in messages if isinstance(m, HumanMessage) and m.name == EVIDENCE), "")
    budget = max(settings.max_grounding_chars - len(sheet), 0)
    parts = [str(m.content) for m in messages
             if isinstance(m, ToolMessage) and not m.additional_kwargs.get("repeat")]
    kept, used = [], 0
    for part in reversed(parts):
        if used + len(part) > budget:
            if not kept and not sheet:                    # never audit against nothing
                kept.append(part[: settings.max_grounding_chars])
            logger.warning(f"Fact-check evidence over budget: left out the {len(parts) - len(kept)} oldest "
                           f"of {len(parts)} tool results.")
            break
        kept.append(part)
        used += len(part) + 2
    return "\n\n".join(([sheet] if sheet else []) + list(reversed(kept)))


def _grounded(verdict: str) -> bool:
    """The auditor is asked for one word. A reply that explains itself and ends with that word, or says it
    found no unsupported claims, passes too: the live DIXON run (2026-10-11) got 'No unsupported claims
    found. ... GROUNDED', read it as a failure, and rewrote the whole report."""
    v = verdict.strip()
    if re.search(r"\b(not\s+grounded|ungrounded)\b", v, re.I):
        return False
    lines = [ln.strip(" .*`#") for ln in v.splitlines() if ln.strip()]
    return (len(v) < 8 or re.match(r"[\W_]*(verdict\s*:\s*)?[\W_]*grounded\b", v, re.I) is not None
            or (bool(lines) and lines[-1].upper() == "GROUNDED")
            or re.match(r"no\s+unsupported\s+claims", v, re.I) is not None)


async def grounding_check_node(
        state: AgentState) -> Command[Literal["FinancialAnalystAgent", "EvaluatorAgent"]]:
    """CRAG grounding gate: reject report claims not backed by the retrieved passages or tool data.

    On a flag, bounce back to the analyst once (capped by max_grounding_retries) to ground or remove
    the claim; otherwise proceed to the evaluator. Fails open (never blocks)."""
    retries = state.get("grounding_retries", 0)
    if retries >= settings.max_grounding_retries:
        logger.info("Grounding retries exhausted; proceeding to evaluation.")
        return Command(goto="EvaluatorAgent")

    report = str(state["messages"][-1].content)
    evidence = _evidence(state["messages"])
    if not evidence.strip():                                   # nothing retrieved -> nothing to audit against
        return Command(goto="EvaluatorAgent")
    try:
        prompt = GROUNDING_CHECK_PROMPT.format(evidence=evidence, report=report)
        checked = await llm_grounding.ainvoke([HumanMessage(content=prompt)])
        verdict = str(checked.content)
    except Exception as e:
        logger.warning(f"Grounding check errored ({e}); proceeding.")
        return Command(goto="EvaluatorAgent")

    calls = [_usage("grounding", checked, settings.large_num_ctx)]
    # a verdict cut at its length limit is an essay, not a verdict: the live DIXON run (2026-10-11) sent
    # 51,211 tokens of it back to the analyst and every later prompt sat at the window's ceiling
    if (getattr(checked, "response_metadata", None) or {}).get("done_reason") == "length":
        logger.warning("Grounding verdict was cut at its length limit; not used.")
        return Command(goto="EvaluatorAgent", update={"llm_calls": calls})
    if _grounded(verdict):
        logger.info("Grounding check passed.")
        return Command(goto="EvaluatorAgent", update={"llm_calls": calls})

    logger.warning(f"Grounding check flagged unsupported claims (grounding revision {retries + 1}).")
    feedback = HumanMessage(content=(
        "GROUNDING CHECK FAILED. Your report states claims NOT supported by the retrieved passages or "
        f"the tool data:\n{verdict[: settings.grounding_feedback_chars]}\n\nRevise: for each, either call a "
        "tool / penny_stock_researcher to "
        "support it, or remove it. Never state a quote, framework, or figure that isn't in the evidence."
        # Measured 2026-10-02 (A/B, 3 runs each): without the next sentence the analyst removed both
        # flagged claims 1/3 times; with it, 3/3. A thinking-on run once answered with a list of
        # "verified" claims instead of a report - and the reader only ever sees this last message.
        " Then write the COMPLETE revised report from the beginning, in the same structure - the "
        "reader sees ONLY your last message, so never reply with a list of checks or verifications."))
    return Command(goto="FinancialAnalystAgent",
                   update={"messages": [feedback], "grounding_retries": retries + 1, "llm_calls": calls})

def _extract_score(eval_text: str, default: int = 1) -> int:
    """Pull the judge's 1-5 score out of Prometheus-2's free text, robustly.

    Prometheus echoes the template line '[an integer number between 1 and 5]', whose '1'
    and '5' fool a naive 'last digit' parser. We strip that phrase first, then prefer a
    number that follows the word 'score' (the actual verdict), and only fall back to a
    bare digit if needed."""
    text = re.sub(r"between\s+1\s+and\s+5", " ", eval_text, flags=re.IGNORECASE)   # kill the boilerplate
    after_score = re.findall(r"score\D{0,20}?([1-5])", text, flags=re.IGNORECASE)  # "...score of 4"
    if after_score:
        return int(after_score[-1])                # the LAST 'score of X' is the final verdict
    digits = re.findall(r"\b([1-5])\b", text)       # fallback: last standalone 1-5
    return int(digits[-1]) if digits else default


async def evaluator_node(state: AgentState) -> Command[Literal["FinancialAnalystAgent", "__end__"]]:
    """Prometheus-2 Judgement Node. Evaluates the Analyst's report for faithfulness and relevance."""
    logger.info("Evaluating Analyst Report using Prometheus-2...")
    report_content = state["messages"][-1].content
    user_question = state["messages"][0].content

    instruction = (
        "Evaluate the financial report generated by the Analyst. The report was written to answer "
        "the following user request, using live market data and vector-database context:\n\n"
        f"USER REQUEST: {user_question}"
    )
    sheet = (state.get("evidence") or {}).get("fact_sheet")
    if sheet:
        instruction += ("\n\nThe analyst was given this EVIDENCE PACK of verified facts; a faithful report "
                        "agrees with it:\n" + sheet[: settings.judge_evidence_chars])
    prompt = f"""###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general. Refer to the given context when writing the feedback and making an assessment.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "(write a feedback for criteria) (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{report_content}

###Reference Answer (Score 5):
{JUDGE_REFERENCE_ANSWER}

###Score Rubrics:
{JUDGE_RUBRIC}

###Feedback:
"""
    response = await llm_judge_agent.ainvoke([HumanMessage(content=prompt)])
    calls = [_usage("judge", response, settings.num_ctx)]
    eval_text = response.content
    logger.info(f"Prometheus Evaluation Output: {eval_text}")

    score = _extract_score(eval_text)
    revisions = state.get("revisions", 0)
    logger.info(f"Assigned Quality Score: {score} (revision {revisions}/{settings.max_revisions})")

    if score >= settings.pass_score:
        logger.info("Report passed evaluation. Ending workflow.")
        final_msg = AIMessage(content=f"LLM Judge Evaluation Passed (Score:{score})\n\nFeedback: {eval_text}")
        return Command(goto=END, update={"messages": [final_msg], "llm_calls": calls})

    if revisions >= settings.max_revisions:                      # loop cap: stop chasing an unwinnable score
        logger.warning(f"Max revisions ({settings.max_revisions}) reached. Accepting best-effort report (Score:{score}).")
        best_msg = AIMessage(content=(
            f"LLM Judge Evaluation: accepted as best-effort after {revisions} revision(s) (Score:{score}).\n\n"
            f"Feedback: {eval_text}"))
        return Command(goto=END, update={"messages": [best_msg], "llm_calls": calls})

    logger.warning(f"Report failed evaluation (Score:{score}). Routing back for revision {revisions + 1}/{settings.max_revisions}.")
    feedback_msg = HumanMessage(content=(
        "Your previous report failed quality checks. Here is the evaluator's feedback:\n"
        f"{str(eval_text)[: settings.grounding_feedback_chars]}\n\n"
        "Please revise to rely strictly on the provided live data and retrieved context, and explicitly state "
        "any figures that are unavailable rather than inventing them."))
    return Command(goto="FinancialAnalystAgent",
                   update={"messages": [feedback_msg], "revisions": revisions + 1, "llm_calls": calls})

STOCK_TOOLS = {"get_indian_stock_data", "get_stock_fundamentals", "get_price_history",
               "get_financials", "get_corporate_actions", "get_shareholding",
               "get_forensics", "get_valuation_context", "get_corporate_events",
               "get_management_commentary", "get_annual_report_notes",
               "get_broker_consensus", "search_broker_research", "get_trendlyne_view",
               "search_filings"}

# --- tool dispatch: name -> (tool, agent to return to) ---
TOOL_REGISTRY = {
    "get_indian_stock_data": (get_indian_stock_data, "MarketDataAgent"),
    "get_stock_fundamentals": (get_stock_fundamentals, "FinancialAnalystAgent"),
    "get_price_history": (get_price_history, "FinancialAnalystAgent"),
    "get_financials": (get_financials, "FinancialAnalystAgent"),
    "get_corporate_actions": (get_corporate_actions, "FinancialAnalystAgent"),
    "get_shareholding": (get_shareholding, "FinancialAnalystAgent"),
    "get_forensics": (get_forensics, "FinancialAnalystAgent"),
    "get_valuation_context": (get_valuation_context, "FinancialAnalystAgent"),
    "get_corporate_events": (get_corporate_events, "FinancialAnalystAgent"),
    "get_management_commentary": (get_management_commentary, "FinancialAnalystAgent"),
    "get_annual_report_notes": (get_annual_report_notes, "FinancialAnalystAgent"),
    "penny_stock_researcher": (penny_stock_researcher, "FinancialAnalystAgent"),
    "get_broker_consensus": (get_broker_consensus, "FinancialAnalystAgent"),
    "search_broker_research": (search_broker_research, "FinancialAnalystAgent"),
    "get_trendlyne_view": (get_trendlyne_view, "FinancialAnalystAgent"),
    "search_filings": (search_filings, "FinancialAnalystAgent"),
    # screen_stocks is universe-level, so it is deliberately NOT in STOCK_TOOLS -
    # injecting a resolved symbol into it would silently narrow a discovery query
    "screen_stocks": (screen_stocks, "FinancialAnalystAgent"),
}


def _sanitize_stock_args(args: dict) -> dict:
    """Defensive: unwrap nested/list values some models emit, then normalize casing."""
    exchange = args.get("exchange", "NSE")
    symbol = args.get("symbol", "")
    if isinstance(exchange, dict):
        exchange = exchange.get("value", "NSE")
    if isinstance(symbol, dict):
        symbol = symbol.get("value", "")
    elif isinstance(symbol, list):
        symbol = symbol[-1]
    return {"symbol": str(symbol).upper(), "exchange": str(exchange).upper()}


def _call_key(name: str, tool, args: dict) -> str:
    """One key per distinct call, defaults filled in: exchange="NSE" spelled out or left off match."""
    schema = getattr(tool, "args_schema", None)
    if hasattr(schema, "model_validate"):
        with suppress(TypeError, ValueError):            # bad arguments: the tool itself will say so
            args = schema.model_validate(args).model_dump()
    return f"{name}:{json.dumps(args, sort_keys=True, default=str)}"


async def execute_tools(state: AgentState) -> Command:
    """Execute each requested tool and route back to the agent that owns it."""
    last_msg = state["messages"][-1]
    tool_msgs = []
    return_target = "FinancialAnalystAgent"
    # results already in the conversation, by call key (a trimmed one no longer counts as held)
    held = {m.additional_kwargs.get("call_key") for m in state["messages"]
            if isinstance(m, ToolMessage) and not m.additional_kwargs.get("trimmed")}
    # results the evidence pack already put in the fact sheet
    pack_keys = set((state.get("evidence") or {}).get("call_keys") or [])
    # texts already held, by content: three different book queries returned the same 10,180 characters
    held_text = {str(m.content): m.name for m in state["messages"] if isinstance(m, ToolMessage)
                 and not m.additional_kwargs.get("trimmed") and not m.additional_kwargs.get("repeat")}

    for tc in last_msg.tool_calls:
        name, args, key, repeat = tc["name"], tc["args"], None, False
        if name not in TOOL_REGISTRY:
            logger.warning(f"LLM hallucinated an unknown tool: {name}")
            result = f"System Error: The tool '{name}' does not exist. Please use the correct provided tool."
            return_target = "FinancialAnalystAgent"
        else:
            tool, return_target = TOOL_REGISTRY[name]
            if name in STOCK_TOOLS and state.get("resolved_symbol"):
                args = {**args, "symbol": state["resolved_symbol"]}   # force ONE company for every tool
            if name == "get_indian_stock_data":
                args = _sanitize_stock_args(args)
            key = _call_key(name, tool, args)
            repeat = key in held or key in pack_keys
            if key in pack_keys and key not in held:
                result = f"Already in the EVIDENCE PACK above ({name}) - use those figures as given."
                logger.info(f"Tool skipped (in the evidence pack): {name}")
            elif repeat:
                # the same data is already in the conversation; a second copy only pushes other results out
                result = (f"Same result as the earlier {name} call in this conversation - it has not "
                          "changed. Use that result.")
                logger.info(f"Tool skipped (already held): {name}")
            else:
                result = await tool.ainvoke(args)
                held.add(key)
                logger.info(f"Tool executed: {name}")
                text = str(result)
                if len(text) > 200 and text in held_text:
                    result = (f"Same text as the earlier {held_text[text]} result in this conversation - "
                              "use that result.")
                    repeat = True
                else:
                    held_text[text] = name

        meta = {"call_key": key, "repeat": True} if repeat else ({"call_key": key} if key else {})
        tool_msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"], name=name,
                                     additional_kwargs=meta))

    return Command(goto=return_target, update={"messages": tool_msgs})

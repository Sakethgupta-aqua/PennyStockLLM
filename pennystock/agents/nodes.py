"""The LangGraph agent nodes (supervisor, data, analyst, evaluator, tool executor)."""
import logging
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END
from langgraph.types import Command


from pennystock.agents.prompts import (
    ANALYST_PROMPT,
    DATA_AGENT_PROMPT,
    GROUNDING_CHECK_PROMPT,
    JUDGE_REFERENCE_ANSWER,
    JUDGE_RUBRIC,
    SUPERVISOR_PROMPT,
)

from pennystock.data.resolver import resolve_from_query
from pennystock.agents.state import AgentState, Router
from pennystock.config import settings
from pennystock.llms import analyst_llm, data_agent_llm, grounding_llm, judge_llm, supervisor_llm
from pennystock.tools import (
    get_annual_report_notes,
    screen_stocks,
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

llm_data_agent = data_agent_llm()
llm_analyst_agent = analyst_llm()
llm_judge_agent = judge_llm()
llm_supervisor = supervisor_llm()
llm_grounding = grounding_llm()

async def resolve_entity_node(state: AgentState) -> Command:
    """Resolve the ONE stock named in the user's query — deterministically, ONCE, before any
    agent runs. Pins it in state so every downstream tool analyzes the SAME company, no matter
    what string an LLM later types. This is the single source of truth for 'which stock'."""
    query = str(state["messages"][0].content)
    match = resolve_from_query(query)                     # prefers the LONGEST name -> TMPV, not TMCV
    if match:
        logger.info(f"Entity pinned: {match['symbol']} ({match['company_name']}).")
        return Command(goto="supervisor_node", update={
            "resolved_symbol": match["symbol"], "resolved_name": match["company_name"]})
    logger.info("No confident entity in the query; tools will resolve per-call (legacy fallback).")
    return Command(goto="supervisor_node")

async def supervisor_node(state: AgentState) -> Command:
    if state.get("remaining_steps", settings.recursion_steps_default) <= settings.recursion_floor:
        logger.warning("Recursion limit approaching. Forcefully terminating graph.")
        return Command(goto=END)

    sanitized = []
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage) and len(str(msg.content)) > settings.max_tool_chars:
            sanitized.append(ToolMessage(
                content=str(msg.content)[: settings.max_tool_chars] + "...", tool_call_id=msg.tool_call_id))
        else:
            sanitized.append(msg)

    router = llm_supervisor.with_structured_output(Router, method="json_schema")
    try:
        decision = await router.ainvoke([SystemMessage(content=SUPERVISOR_PROMPT)] + sanitized)
    except Exception as e:
        logger.error(f"Supervisor failed to output valid structured JSON: {e}. Defaulting to FINISH.")
        return Command(goto=END)

    logger.info(f"Supervisor routing to: {decision.next_agent}")
    if decision.next_agent == "FINISH":
        return Command(goto=END)
    return Command(goto=decision.next_agent)


async def market_data_agent(state: AgentState) -> Command[Literal["tools", "supervisor_node"]]:
    system_prompt = SystemMessage(content=DATA_AGENT_PROMPT)
    response = await llm_data_agent.bind_tools(data_tools).ainvoke([system_prompt] + state["messages"])
    if hasattr(response, "tool_calls") and response.tool_calls:
        return Command(goto="tools", update={"messages": [response]})
    return Command(goto="supervisor_node", update={"messages": [response]})


async def financial_analyst_agent(state: AgentState) -> Command[Literal["tools", "grounding_check"]]:
    system_prompt = SystemMessage(content=ANALYST_PROMPT)
    last = str(state["messages"][-1].content)
    is_feedback = "Your previous report failed quality checks" in last or "GROUNDING CHECK FAILED" in last
    if is_feedback:
        payload = [system_prompt] + state["messages"]
    else:
        transitional = HumanMessage(
            content="The tool execution is complete. Please review the data above and proceed with your analysis or further tool calls.")
        payload = [system_prompt] + state["messages"] + [transitional]

    response = await llm_analyst_agent.bind_tools(analyst_tools).ainvoke(payload)
    if hasattr(response, "tool_calls") and response.tool_calls:
        return Command(goto="tools", update={"messages": [response]})
    return Command(goto="grounding_check", update={"messages": [response]})


async def grounding_check_node(state: AgentState) -> Command[Literal["FinancialAnalystAgent", "supervisor_node"]]:
    """CRAG grounding gate: reject report claims not backed by the retrieved passages or tool data.

    On a flag, bounce back to the analyst once (capped by max_grounding_retries) to ground or remove
    the claim; otherwise proceed to the evaluator via the supervisor. Fails open (never blocks)."""
    retries = state.get("grounding_retries", 0)
    if retries >= settings.max_grounding_retries:
        logger.info("Grounding retries exhausted; proceeding to evaluation.")
        return Command(goto="supervisor_node")

    report = str(state["messages"][-1].content)
    evidence = "\n\n".join(
        str(m.content) for m in state["messages"] if isinstance(m, ToolMessage))[: settings.max_grounding_chars]
    if not evidence.strip():                                   # nothing retrieved -> nothing to audit against
        return Command(goto="supervisor_node")
    try:
        prompt = GROUNDING_CHECK_PROMPT.format(evidence=evidence, report=report)
        verdict = str((await llm_grounding.ainvoke([HumanMessage(content=prompt)])).content)
    except Exception as e:
        logger.warning(f"Grounding check errored ({e}); proceeding.")
        return Command(goto="supervisor_node")

    if "GROUNDED" in verdict.upper()[:60] or len(verdict.strip()) < 8:
        logger.info("Grounding check passed.")
        return Command(goto="supervisor_node")

    logger.warning(f"Grounding check flagged unsupported claims (grounding revision {retries + 1}).")
    feedback = HumanMessage(content=(
        "GROUNDING CHECK FAILED. Your report states claims NOT supported by the retrieved passages or "
        f"the tool data:\n{verdict}\n\nRevise: for each, either call a tool / penny_stock_researcher to "
        "support it, or remove it. Never state a quote, framework, or figure that isn't in the evidence."))
    return Command(goto="FinancialAnalystAgent",
                   update={"messages": [feedback], "grounding_retries": retries + 1})

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
    eval_text = response.content
    logger.info(f"Prometheus Evaluation Output: {eval_text}")

    score = _extract_score(eval_text)
    revisions = state.get("revisions", 0)
    logger.info(f"Assigned Quality Score: {score} (revision {revisions}/{settings.max_revisions})")

    if score >= settings.pass_score:
        logger.info("Report passed evaluation. Ending workflow.")
        final_msg = AIMessage(content=f"LLM Judge Evaluation Passed (Score:{score})\n\nFeedback: {eval_text}")
        return Command(goto=END, update={"messages": [final_msg]})

    if revisions >= settings.max_revisions:                      # loop cap: stop chasing an unwinnable score
        logger.warning(f"Max revisions ({settings.max_revisions}) reached. Accepting best-effort report (Score:{score}).")
        best_msg = AIMessage(content=(
            f"LLM Judge Evaluation: accepted as best-effort after {revisions} revision(s) (Score:{score}).\n\n"
            f"Feedback: {eval_text}"))
        return Command(goto=END, update={"messages": [best_msg]})

    logger.warning(f"Report failed evaluation (Score:{score}). Routing back for revision {revisions + 1}/{settings.max_revisions}.")
    feedback_msg = HumanMessage(content=(
        f"Your previous report failed quality checks. Here is the evaluator's feedback:\n{eval_text}\n\n"
        "Please revise to rely strictly on the provided live data and retrieved context, and explicitly state "
        "any figures that are unavailable rather than inventing them."))
    return Command(goto="FinancialAnalystAgent", update={"messages": [feedback_msg], "revisions": revisions + 1})

STOCK_TOOLS = {"get_indian_stock_data", "get_stock_fundamentals", "get_price_history",
               "get_financials", "get_corporate_actions", "get_shareholding",
               "get_forensics", "get_valuation_context", "get_corporate_events",
               "get_management_commentary", "get_annual_report_notes"}

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


async def execute_tools(state: AgentState) -> Command:
    """Execute each requested tool and route back to the agent that owns it."""
    last_msg = state["messages"][-1]
    tool_msgs = []
    return_target = "supervisor_node"

    for tc in last_msg.tool_calls:
        name, args = tc["name"], tc["args"]
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
            result = await tool.ainvoke(args)

        tool_msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        logger.info(f"Tool executed: {name}")

    return Command(goto=return_target, update={"messages": tool_msgs})

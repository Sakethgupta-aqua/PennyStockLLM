"""Assemble the LangGraph pipeline from the nodes."""
from langgraph.graph import START, StateGraph

from pennystock.agents.nodes import (
    evaluator_node,
    evidence_pack_node,
    execute_tools,
    financial_analyst_agent,
    grounding_check_node,
    market_data_agent,
    resolve_entity_node,
)
from pennystock.agents.state import AgentState


def build_graph():
    """Build and compile the multi-agent graph. Compilation is cheap — no DB/model I/O."""
    builder = StateGraph(AgentState)
    builder.add_node("resolve_entity", resolve_entity_node)
    builder.add_node("MarketDataAgent", market_data_agent)
    builder.add_node("evidence_pack", evidence_pack_node)
    builder.add_node("FinancialAnalystAgent", financial_analyst_agent)
    builder.add_node("grounding_check", grounding_check_node)
    builder.add_node("EvaluatorAgent", evaluator_node)
    builder.add_node("tools", execute_tools)
    builder.add_edge(START, "resolve_entity")
    return builder.compile()

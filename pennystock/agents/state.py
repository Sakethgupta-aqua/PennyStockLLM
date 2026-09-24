"""Graph state and the supervisor's routing schema."""
from typing import Literal

from langgraph.graph import MessagesState
from langgraph.managed.is_last_step import RemainingSteps
from pydantic import BaseModel, Field


class AgentState(MessagesState):
    remaining_steps: RemainingSteps
    revisions: int          # times the evaluator has bounced the report back (loop cap)
    grounding_retries: int  # times the CRAG grounding gate has bounced the report back
    resolved_symbol: str | None
    resolved_name: str | None


class Router(BaseModel):
    next_agent: Literal["MarketDataAgent", "FinancialAnalystAgent", "EvaluatorAgent", "FINISH"] = Field(
        description="The next agent to route to based on the current context."
    )

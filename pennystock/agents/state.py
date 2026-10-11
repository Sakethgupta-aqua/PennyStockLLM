"""Graph state."""
import operator
from typing import Annotated

from langgraph.graph import MessagesState
from langgraph.managed.is_last_step import RemainingSteps


class AgentState(MessagesState):
    remaining_steps: RemainingSteps
    revisions: int          # times the evaluator has bounced the report back (loop cap)
    grounding_retries: int  # times the CRAG grounding gate has bounced the report back
    resolved_symbol: str | None
    resolved_name: str | None
    llm_calls: Annotated[list[dict], operator.add]  # every model call's tokens against its window (traced)
    evidence: dict | None   # the evidence pack: fact sheet, call keys, built_at, the calls behind it (traced)
    text_nudged: bool       # the first draft was sent back once for text tools the question needs


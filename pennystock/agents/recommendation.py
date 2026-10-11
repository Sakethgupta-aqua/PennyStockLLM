"""Turning the analyst's prose report into a gradeable call.

The analyst writes for a human: paragraphs, caveats, judgement. Nobody can grade a
paragraph. This module reads the finished report and fills in a small form - action,
conviction, horizon, the price that would prove it wrong - so the ledger has something
with an answer.

WHY A SEPARATE STEP, NOT A CHANGE TO THE ANALYST. Forcing the analyst itself to emit
structured output would change the job it does and risk degrading the report a person
actually reads. Extraction afterwards leaves the report untouched and costs one cheap
call.

THE VALIDATION SANDWICH. Industry practice for production structured output is three
layers, and this module implements all three:

  1. SCHEMA, at generation time. `with_structured_output(..., method="json_schema")`
     constrains decoding so the model cannot emit a field that is not in the schema.
     Already proven three times in this codebase (nodes.py:76, query_expand.py:29,
     build_eval_set.py:118).
  2. SEMANTICS, at the application boundary. The layer a schema cannot do for you:
     `action="buy"` with an invalidation price ABOVE the current price is valid JSON and
     financial nonsense - you cannot be "wrong if it falls below" a level the stock is
     already under. Pydantic sees no entry price, so this check lives here.
  3. FAILURE HANDLING that separates STRUCTURAL from SEMANTIC problems. A structural
     failure (unparseable output) is worth one retry. A semantic problem is not - the
     model was clear, it was just wrong, so the offending FIELD is dropped and the rest
     of the call is kept.

WHAT IT NEVER DOES: guess. Two failed attempts produce no recommendation and a recorded
reason. Inventing `buy` to fill the row would quietly stuff the ledger with calls nobody
made, and every hit rate computed afterwards would be fiction.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 2                 # one retry. Healthy average retry count is 0.5-1.5.
MIN_REPORT_CHARS = 200           # shorter than this is not a report worth extracting from
# An invalidation level further than this from the entry price is not a stop, it is a
# mis-parsed number - a paise/rupee slip, or a figure lifted from the wrong table.
MAX_INVALIDATION_DISTANCE = 0.60

ACTION_TO_DIRECTION = {"buy": "bullish", "avoid": "bearish", "watch": "neutral"}


class Recommendation(BaseModel):
    """The gradeable core of a report. Deliberately small: every field must be either
    checkable against the market later, or needed to interpret one that is."""
    action: Literal["buy", "avoid", "watch"] = Field(
        description="buy = expect outperformance, avoid = expect underperformance, "
                    "watch = no directional call")
    conviction: int = Field(ge=1, le=5, description="1 = tentative, 5 = high conviction")
    horizon: Literal["1w", "1m", "3m"] = Field(description="how long the call is meant to hold")
    thesis: str = Field(description="one or two sentences, the core reason")
    catalysts: list[str] = Field(default_factory=list, description="what would prove it right")
    risks: list[str] = Field(default_factory=list, description="what would prove it wrong")
    invalidation_price: float | None = Field(
        default=None, description="the close that would prove this call wrong; null if none")


@dataclass
class ExtractionResult:
    """Outcome of one extraction, including the reasons - never a bare None."""
    ok: bool
    recommendation: Recommendation | None = None
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
    attempts: int = 0


EXTRACTION_PROMPT = """You are reading a finished equity research report and recording the
call it makes, so that it can be checked against the market later.

Extract ONLY what the report actually says. Do not add a view of your own.

  action              buy   = the report expects this to outperform
                      avoid = the report expects it to underperform
                      watch = the report reaches no directional conclusion
                      If the report hedges or reaches no conclusion, use "watch".
                      Do NOT read a cautious buy as "avoid", or a hedged view as a call.
  conviction          1-5, from how strongly the report states its case.
  horizon             1w, 1m or 3m - whichever the report's own reasoning implies.
  thesis              One or two sentences, in the report's own terms.
  catalysts / risks   Short phrases, only ones the report names.
  invalidation_price  The price level the report says would prove it wrong. For a BUY this
                      must be BELOW the current price of {price}; for an AVOID, ABOVE it.
                      If the report names no such level, return null. Do not invent one.

CURRENT PRICE: {price}

REPORT:
{report}
"""


def validate_semantics(rec: Recommendation,
                       entry_price: float | None) -> tuple[Recommendation, list[str]]:
    """Check what the schema cannot: does this call make sense against the actual price?

    A bad field is DROPPED, not the whole call. The direction and conviction are still
    worth grading, and discarding a usable call over one unusable number loses more
    information than it protects.
    """
    problems: list[str] = []
    if entry_price is None or not entry_price or rec.invalidation_price is None:
        return rec, problems                       # nothing to check against

    level, price = rec.invalidation_price, entry_price
    distance = abs(level - price) / price

    if distance > MAX_INVALIDATION_DISTANCE:
        problems.append(
            f"invalidation {level} is {distance:.0%} away from {price} - too far to be a "
            f"stop level; reads as a units or parsing error")
    elif rec.action == "buy" and level >= price:
        problems.append(
            f"a buy cannot be invalidated at {level}, which is at or above the current "
            f"{price}: the call would be wrong the moment it was made. Must be below.")
    elif rec.action == "avoid" and level <= price:
        problems.append(
            f"an avoid cannot be invalidated at {level}, at or below the current {price}. "
            f"Must be above.")

    if problems:
        rec = rec.model_copy(update={"invalidation_price": None})
    return rec, problems


def to_prediction_fields(rec: Recommendation) -> dict:
    """Map a Recommendation onto the ledger's shared vocabulary.

    The ledger grades `bullish`/`bearish`/`neutral` so that an agent's call and a broker's
    call are scored by exactly the same code. Everything the ledger does not grade on
    travels in the payload.
    """
    return {
        "direction": ACTION_TO_DIRECTION[rec.action],
        "conviction": rec.conviction,
        "horizon": rec.horizon,
        "invalidation_price": rec.invalidation_price,
        "payload": {"action": rec.action, "thesis": rec.thesis,
                    "catalysts": rec.catalysts, "risks": rec.risks},
    }


async def extract_recommendation(report: str, entry_price: float | None = None,
                                 llm=None) -> ExtractionResult:
    """Read a report, return a gradeable call - or an honest reason there is none."""
    if not report or len(report.strip()) < MIN_REPORT_CHARS:
        return ExtractionResult(ok=False, reason="report too short to extract a call from")

    if llm is None:                                # imported lazily: no model client at import time
        from pennystock.llms import analyst_llm
        llm = analyst_llm()

    prompt = EXTRACTION_PROMPT.format(report=report[:12000],
                                      price=entry_price if entry_price else "unknown")
    structured = llm.with_structured_output(Recommendation, method="json_schema")

    last_error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            rec = await structured.ainvoke(prompt)
        except Exception as e:                     # STRUCTURAL failure - worth one retry
            last_error = f"{type(e).__name__}: {e}"
            logger.warning(f"recommendation extraction attempt {attempt} failed: {last_error}")
            continue

        # SEMANTIC problems are NOT retried: the model was clear, it was simply wrong, and
        # asking again usually returns the same answer at the same cost.
        rec, problems = validate_semantics(rec, entry_price)
        if problems:
            logger.info(f"recommendation semantics corrected: {'; '.join(problems)}")
        return ExtractionResult(ok=True, recommendation=rec, warnings=problems, attempts=attempt)

    return ExtractionResult(ok=False, attempts=MAX_ATTEMPTS,
                            reason=f"extraction failed after {MAX_ATTEMPTS} attempts: {last_error}")

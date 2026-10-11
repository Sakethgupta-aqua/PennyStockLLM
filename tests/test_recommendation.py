"""Turning the analyst's prose report into a gradeable call.

This is the riskiest step in the ledger: an LLM reads a report and fills in a form. The
industry pattern for that is the VALIDATION SANDWICH - the schema is enforced at
generation time, semantics are checked at the application boundary, and structural
failures are handled differently from semantic ones.

The semantic layer is the one a schema cannot do for you. `action="buy"` with an
invalidation price ABOVE the current price is perfectly valid JSON and financial
nonsense: you cannot be "wrong if it falls below" a level the stock is already under.
"""
import asyncio

import pytest

from pennystock.agents.recommendation import (Recommendation, extract_recommendation,
                                              to_prediction_fields, validate_semantics)


# Long enough to clear MIN_REPORT_CHARS. A two-word "report" is not a report, and the
# guard that rejects one is behaviour worth keeping, not a fixture to work around.
REPORT = ("Hindalco screens well on the forensic checks. " * 12)


def rec(**kw) -> Recommendation:
    base = dict(action="buy", conviction=4, horizon="1m", thesis="Cheap on peer multiples.",
                catalysts=["capacity ramp"], risks=["aluminium price"], invalidation_price=590.0)
    return Recommendation(**(base | kw))


class FakeLLM:
    """Stands in for the analyst model. Each call returns the next queued answer."""
    def __init__(self, *answers):
        self.answers, self.calls = list(answers), 0

    def with_structured_output(self, *a, **k):
        return self

    async def ainvoke(self, *a, **k):
        self.calls += 1
        answer = self.answers[min(self.calls - 1, len(self.answers) - 1)]
        if isinstance(answer, Exception):
            raise answer
        return answer


# --- the schema itself ------------------------------------------------------------------
def test_a_well_formed_recommendation_is_accepted():
    r = rec()
    assert (r.action, r.conviction, r.horizon) == ("buy", 4, "1m")


@pytest.mark.parametrize("bad", [0, 6, -1, 99])
def test_conviction_outside_one_to_five_is_rejected(bad):
    """A 7-out-of-5 conviction would silently dominate any confidence-weighted score."""
    with pytest.raises(Exception):
        rec(conviction=bad)


def test_an_invented_action_is_rejected_by_the_schema():
    with pytest.raises(Exception):
        rec(action="strong buy")


def test_catalysts_and_risks_default_to_empty_rather_than_none():
    r = Recommendation(action="watch", conviction=2, horizon="1m", thesis="x")
    assert r.catalysts == [] and r.risks == []


# --- the semantic layer: what a schema CANNOT catch ---------------------------------------
def test_a_buy_whose_invalidation_sits_above_the_price_is_rejected():
    """"Buy at 640, I'm wrong below 700" is valid JSON and nonsense - the stock is already
    below 700, so the call is invalidated the moment it is made."""
    r, problems = validate_semantics(rec(action="buy", invalidation_price=700.0), entry_price=640.0)
    assert r.invalidation_price is None
    assert any("below" in p for p in problems)


def test_an_avoid_whose_invalidation_sits_below_the_price_is_rejected():
    r, problems = validate_semantics(rec(action="avoid", invalidation_price=500.0), entry_price=640.0)
    assert r.invalidation_price is None
    assert problems


def test_a_sensible_invalidation_price_is_kept():
    r, problems = validate_semantics(rec(action="buy", invalidation_price=590.0), entry_price=640.0)
    assert r.invalidation_price == 590.0
    assert problems == []


def test_a_bad_invalidation_price_is_dropped_not_the_whole_call():
    """The direction and conviction are still worth grading. Throwing the call away over
    one unusable field would lose more information than it protects."""
    r, problems = validate_semantics(rec(action="buy", invalidation_price=700.0), entry_price=640.0)
    assert r.action == "buy" and r.conviction == 4
    assert problems


def test_an_absurd_invalidation_price_is_rejected_as_a_units_error():
    """A level 90% away is not a stop, it is a mis-parsed number - a paise/rupee slip or a
    figure lifted from the wrong table."""
    r, problems = validate_semantics(rec(action="buy", invalidation_price=6.4), entry_price=640.0)
    assert r.invalidation_price is None
    assert any("far" in p for p in problems)


def test_semantics_pass_when_no_entry_price_is_known():
    """Without a price there is nothing to validate against; the call still stands."""
    r, problems = validate_semantics(rec(), entry_price=None)
    assert r.invalidation_price == 590.0 and problems == []


def test_a_watch_call_needs_no_invalidation_price():
    r, problems = validate_semantics(rec(action="watch", invalidation_price=None), entry_price=640.0)
    assert problems == []


# --- mapping into the ledger ---------------------------------------------------------------
def test_actions_map_onto_the_ledgers_shared_vocabulary():
    """The ledger grades `bullish/bearish/neutral` so an agent call and a broker call can
    be scored by the same code."""
    assert to_prediction_fields(rec(action="buy"))["direction"] == "bullish"
    assert to_prediction_fields(rec(action="avoid"))["direction"] == "bearish"
    assert to_prediction_fields(rec(action="watch"))["direction"] == "neutral"


def test_the_thesis_and_lists_travel_into_the_payload():
    f = to_prediction_fields(rec())
    assert f["payload"]["thesis"].startswith("Cheap")
    assert f["payload"]["catalysts"] == ["capacity ramp"]


# --- extraction end to end -------------------------------------------------------------------
def test_a_clean_extraction_succeeds_on_the_first_try():
    out = asyncio.run(extract_recommendation(REPORT, entry_price=640.0,
                                       llm=FakeLLM(rec())))
    assert out.ok and out.attempts == 1
    assert out.recommendation.action == "buy"


def test_a_structural_failure_is_retried_once():
    """Structural failures (the model emitted unparseable output) are worth one retry.
    Measured practice puts a healthy average retry count between 0.5 and 1.5."""
    out = asyncio.run(extract_recommendation(REPORT, entry_price=640.0,
                                       llm=FakeLLM(ValueError("not valid json"), rec())))
    assert out.ok and out.attempts == 2


def test_repeated_failure_never_guesses_a_direction():
    """A run that produced no gradeable call is itself a fact. Inventing 'buy' to fill the
    row would quietly stuff the ledger with calls nobody made."""
    out = asyncio.run(extract_recommendation(REPORT, entry_price=640.0,
                                       llm=FakeLLM(ValueError("x"), ValueError("y"))))
    assert not out.ok
    assert out.recommendation is None
    assert "extraction failed" in out.reason.lower()


def test_an_empty_report_is_not_sent_to_the_model_at_all():
    llm = FakeLLM(rec())
    out = asyncio.run(extract_recommendation("", entry_price=640.0, llm=llm))
    assert not out.ok and llm.calls == 0


def test_semantic_warnings_are_carried_out_not_swallowed():
    out = asyncio.run(extract_recommendation(REPORT, entry_price=640.0,
                                       llm=FakeLLM(rec(invalidation_price=700.0))))
    assert out.ok
    assert out.warnings
    assert out.recommendation.invalidation_price is None

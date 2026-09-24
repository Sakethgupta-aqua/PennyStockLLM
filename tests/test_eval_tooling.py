import sys
from pathlib import Path

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_eval_set import is_self_contained          # noqa: E402
from eval_rag import rank_of_source                   # noqa: E402


def test_rejects_questions_that_reference_the_source():
    for bad in ["According to the passage, what is a margin of safety?",
                "What does the author say about promoter pledging in Indian markets?",
                "As mentioned above, what is the coffee can approach about?"]:
        ok, why = is_self_contained(bad)
        assert not ok, bad


def test_rejects_dangling_pointers():
    """The real failure that motivated this gate."""
    ok, why = is_self_contained("What was the value of the portfolio at the start of the period?")
    assert not ok and "vague reference" in why


def test_rejects_non_questions_and_stubs():
    assert not is_self_contained("The margin of safety is important.")[0]
    assert not is_self_contained("What is ROCE?")[0]        # too short to be findable


def test_keeps_genuinely_self_contained_questions():
    for good in ["What minimum return on capital employed does the coffee-can approach require?",
                 "How does promoter pledging create forced-selling risk in Indian small-cap stocks?",
                 "Why does Benjamin Graham argue that a margin of safety depends on the price paid?"]:
        ok, why = is_self_contained(good)
        assert ok, f"{good} -> {why}"


def _doc(text, score):
    return Document(page_content=text, metadata={"rerank_score": score})


def test_rank_uses_relevance_order_not_display_order():
    """advanced_retrieve REORDERS for the LLM; the rank must be recovered from rerank_score."""
    source = "the margin of safety is always dependent on the price paid"
    docs = [_doc("irrelevant filler", 0.1), _doc(source, 0.9), _doc("more filler", 0.5)]
    assert rank_of_source(source, docs) == 1        # highest score = rank 1, despite being 2nd


def test_rank_is_none_when_source_absent():
    assert rank_of_source("a passage never retrieved", [_doc("something else", 0.4)]) is None


def test_rank_tolerates_whitespace_differences():
    src = "margin   of \n safety depends on the price paid and nothing else at all whatsoever"
    assert rank_of_source(src, [_doc(" ".join(src.split()), 0.8)]) == 1


def test_decomposition_always_keeps_the_original_query():
    """Measured regression: a specific question was generalised away and never retrieved.

    The decomposer turned "What confirms a bearish marubozu?" into ['value investing',
    'behavioral finance', 'technical analysis'] - the distinctive term was gone, so the right
    chunk could not be found. Keeping the original query means decomposition can only ADD.
    """
    import asyncio
    from unittest.mock import patch

    from pennystock.retrieval.query_expand import decompose_query

    class FakeResult:
        queries = ["value investing", "behavioral finance", "technical analysis"]

    class FakeLLM:
        def with_structured_output(self, *a, **k):
            return self

        async def ainvoke(self, *a, **k):
            return FakeResult()

    question = "What confirms a bearish marubozu?"
    with patch("pennystock.llms.decomposer_llm", return_value=FakeLLM()):
        out = asyncio.run(decompose_query(question))
    assert out[0] == question                 # the original survives decomposition
    assert "value investing" in out           # sub-queries are still added


def test_decomposition_falls_back_to_the_original_on_failure():
    import asyncio
    from unittest.mock import patch

    from pennystock.retrieval.query_expand import decompose_query

    with patch("pennystock.llms.decomposer_llm", side_effect=RuntimeError("ollama down")):
        out = asyncio.run(decompose_query("margin of safety"))
    assert out == ["margin of safety"]         # degrades, never breaks retrieval

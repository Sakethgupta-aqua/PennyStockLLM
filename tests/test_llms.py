"""Model factories: each role gets the model, context and thinking mode the config gives it.

Thinking is the one setting with a measured, large cost. Qwen3.5 reasons in hidden tokens unless told
not to: on this M4 Max (2026-10-02) the same short answer took 1,381 generated tokens / 24.7 s with
thinking and 157 tokens / 2.1 s without. The flag must reach the roles that run a thinking model, and
must NOT be sent to roles that do not - a judge or router model given a flag it does not support is a
needless way to fail.
"""
from pennystock import llms


def test_the_analyst_and_grounding_roles_get_the_configured_thinking_mode(monkeypatch):
    monkeypatch.setattr(llms.settings, "analyst_reasoning", False)
    monkeypatch.setattr(llms.settings, "grounding_reasoning", True)
    llms.grounding_llm.cache_clear()          # cached: importing the graph builds it at default settings
    assert llms.analyst_llm().reasoning is False
    assert llms.grounding_llm().reasoning is True


def test_roles_without_a_thinking_model_never_send_the_flag():
    for factory in (llms.judge_llm, llms.data_agent_llm):
        assert factory().reasoning is None


def test_the_analyst_and_grounding_roles_get_the_large_window_and_the_rest_keep_the_default():
    # A 16,384 window overflowed on a 16-tool run (2026-10-10): Ollama silently dropped the oldest messages,
    # i.e. the first tool results. The 35B is hybrid, so 64K costs ~1 GB more and still runs 100% on the GPU.
    llms.grounding_llm.cache_clear()
    assert llms.settings.large_num_ctx == 65536
    assert llms.analyst_llm().num_ctx == llms.settings.large_num_ctx
    assert llms.grounding_llm().num_ctx == llms.settings.large_num_ctx
    for factory in (llms.judge_llm, llms.data_agent_llm):
        assert factory().num_ctx == llms.settings.num_ctx


def test_the_extraction_role_runs_the_analyst_model_without_thinking():
    llm = llms.extraction_llm()
    assert llm.model == llms.settings.analyst_model and llm.reasoning is False
    assert llm.num_ctx == llms.settings.extraction_num_ctx and llm.temperature == 0.0


def teardown_module():
    llms.grounding_llm.cache_clear()           # do not leak the patched instance into later tests


def test_the_fact_check_cannot_write_without_end():
    # the live DIXON run (2026-10-11): the auditor wrote 51,211 tokens arguing with itself and the whole text
    # went back to the analyst as feedback - every later prompt sat at the 65K ceiling (1,687 s)
    llms.grounding_llm.cache_clear()
    assert llms.grounding_llm().num_predict == llms.settings.grounding_max_tokens <= 2048


def test_the_judge_cannot_write_without_end_either():
    # its feedback goes back to the analyst too when a report fails
    assert llms.judge_llm().num_predict == llms.settings.judge_max_tokens <= 2048

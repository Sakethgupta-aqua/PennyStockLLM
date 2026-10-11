"""The evidence pack: the structured tools run in code, in parallel, before the analyst writes a word."""
import asyncio

from pennystock.agents import evidence as ev


class Tool:
    def __init__(self, name, output, fail=False):
        self.name, self.output, self.fail, self.calls = name, output, fail, []

    async def ainvoke(self, args):
        self.calls.append(args)
        if self.fail:
            raise TimeoutError("NSE did not answer")
        return self.output


def _tools(monkeypatch, fail=()):
    tools = [Tool(t.name, str({"company": "Dixon Technologies", "sector": "Technology"}), t.name in fail)
             for t in ev.PACK_TOOLS]
    monkeypatch.setattr(ev, "PACK_TOOLS", tuple(tools))
    monkeypatch.setattr(ev, "filed_year", lambda symbol: None)
    return tools


def test_every_structured_tool_runs_for_the_resolved_company(monkeypatch):
    tools = _tools(monkeypatch)
    calls = asyncio.run(ev.gather("DIXON", "Is Dixon a buy?"))
    assert all(t.calls == [{"symbol": "DIXON"}] for t in tools)
    assert {c.name for c in calls} >= {t.name for t in tools}
    assert all(c.ok for c in calls if c.name != "nse_integrated_filing")


def test_a_failing_tool_becomes_a_failed_call_not_a_failed_pack(monkeypatch):
    _tools(monkeypatch, fail=("get_forensics",))
    calls = {c.name: c for c in asyncio.run(ev.gather("DIXON", "q"))}
    assert not calls["get_forensics"].ok and "TimeoutError" in calls["get_forensics"].output
    assert calls["get_financials"].ok


def test_alternatives_are_screened_in_the_companys_sector_only_when_asked(monkeypatch):
    _tools(monkeypatch)
    screen = Tool("screen_stocks", str({"candidates": []}))
    monkeypatch.setattr(ev, "screen_stocks", screen)
    asyncio.run(ev.gather("DIXON", "Is Dixon a buy?"))
    assert screen.calls == []
    asyncio.run(ev.gather("DIXON", "if not, which stocks would you shortlist instead?"))
    assert screen.calls == [{"sector": "Technology", "top_n": ev.settings.evidence_alternatives}]


def test_tool_output_is_parsed_back_into_a_dict():
    assert ev.parse_output("{'a': 1, 'b': [('2021-03-18', 5.0)]}") == {"a": 1, "b": [("2021-03-18", 5.0)]}
    assert ev.parse_output("Error fetching x") is None

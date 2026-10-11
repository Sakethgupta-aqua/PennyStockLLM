"""The analyst's Trendlyne tool: a third-party view, attributed, and UNKNOWN when there is none."""
import asyncio

from pennystock.tools import trendlyne as tool_mod


def test_no_report_on_file_is_unknown_never_neutral(monkeypatch):
    monkeypatch.setattr(tool_mod, "_resolve_ticker", lambda s, e: ("XYZ.NS", "Xyz Ltd"))
    monkeypatch.setattr(tool_mod, "view", lambda s: None)
    out = asyncio.run(tool_mod.get_trendlyne_view.ainvoke({"symbol": "XYZ"}))
    assert "UNKNOWN" in out and "neutral" not in out.lower()


def test_the_view_is_attributed_and_carries_how_to_read(monkeypatch):
    monkeypatch.setattr(tool_mod, "_resolve_ticker", lambda s, e: ("AUBANK.NS", "AU Small Finance Bank"))
    monkeypatch.setattr(tool_mod, "view", lambda s: {"as_of": "2026-10-01T15:31", "durability": 45})
    out = asyncio.run(tool_mod.get_trendlyne_view.ainvoke({"symbol": "AUBANK"}))
    assert "Trendlyne" in out and "how_to_read" in out and "45" in out


def test_the_pe_zone_is_said_in_words(monkeypatch):
    # 'Strong Buy zone, 12%' was read as 'P/E in the top 12% of its history' (DIXON, 2026-10-10)
    monkeypatch.setattr(tool_mod, "_resolve_ticker", lambda s, e: ("DIXON.NS", "Dixon Technologies"))
    monkeypatch.setattr(tool_mod, "view", lambda s: {"as_of": "2026-10-01T15:31", "pe_zone": "Strong Buy",
                                                     "pe_zone_pct": 12.0})
    out = asyncio.run(tool_mod.get_trendlyne_view.ainvoke({"symbol": "DIXON"}))
    assert "higher than only 12% of its own history" in out and "cheap against its past" in out


def test_pe_zone_meaning_reads_both_ends():
    from pennystock.data.trendlyne_scores import pe_zone_meaning
    assert "expensive against its past" in pe_zone_meaning("Sell", 85.0)
    assert pe_zone_meaning(None, None) is None

"""Agent tools + the per-agent tool assignments."""
from pennystock.tools.market_data import (
    get_annual_report_notes,
    get_corporate_actions,
    get_corporate_events,
    get_forensics,
    get_management_commentary,
    get_financials,
    get_indian_stock_data,
    get_price_history,
    get_shareholding,
    get_stock_fundamentals,
    get_valuation_context,
)
from pennystock.tools.broker_research import get_broker_consensus, search_broker_research
from pennystock.tools.trendlyne import get_trendlyne_view
from pennystock.tools.filings import search_filings
from pennystock.tools.researcher import penny_stock_researcher
from pennystock.tools.screening import screen_stocks

# Keep tools separate so each agent only sees the tools it should use.
data_tools = [get_indian_stock_data]                                     # fine-tuned penny-llama
analyst_tools = [penny_stock_researcher, get_stock_fundamentals, get_price_history,
                 get_financials, get_corporate_actions, get_shareholding,
                 get_forensics,
    get_management_commentary, get_valuation_context, get_corporate_events,
                 get_annual_report_notes, screen_stocks,
                 get_broker_consensus, search_broker_research, get_trendlyne_view, search_filings]

__all__ = [
    "get_indian_stock_data", "get_stock_fundamentals", "get_price_history",
    "get_financials", "get_corporate_actions", "get_shareholding", "get_forensics",
    "get_valuation_context", "get_corporate_events", "get_management_commentary",
    "get_annual_report_notes",
    "penny_stock_researcher", "screen_stocks",
    "get_broker_consensus", "search_broker_research", "get_trendlyne_view", "search_filings",
    "data_tools", "analyst_tools",
]

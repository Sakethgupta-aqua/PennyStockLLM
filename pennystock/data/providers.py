"""Structured market-data providers — the LIVE half of the data spine.

The reference architecture's central rule: BOOK knowledge (frameworks) and LIVE data
(prices, fundamentals) are separate sources. This module owns the LIVE side. The
MarketDataProvider interface is deliberately thin so the yfinance backend can later be
swapped for a broker/NSE feed without touching the tools or agents that depend on it.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# tiny TTL cache so snapshot+fundamentals for the same stock share ONE .info network call
_TTL_SECONDS = 60
_cache: dict[tuple, tuple[float, Any]] = {}

CRORE = 1e7


def _cached(key: tuple, producer):
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < _TTL_SECONDS:
        return hit[1]
    val = producer()
    _cache[key] = (now, val)
    return val


@dataclass
class Snapshot:
    symbol: str | None = None
    company: str | None = None
    price: float | None = None
    previous_close: float | None = None
    day_high: float | None = None
    day_low: float | None = None
    volume: int | None = None
    market_cap: int | None = None


@dataclass
class Fundamentals:
    symbol: str | None = None
    company: str | None = None
    sector: str | None = None
    industry: str | None = None
    market_cap: int | None = None
    pe_trailing: float | None = None
    pe_forward: float | None = None
    eps_trailing: float | None = None
    book_value: float | None = None
    price_to_book: float | None = None
    debt_to_equity_pct: float | None = None      # yfinance gives this as a PERCENT (e.g. 3.776 == 3.78%)
    return_on_equity: float | None = None
    profit_margin: float | None = None
    dividend_yield: float | None = None
    beta: float | None = None
    fifty_two_week_high: float | None = None
    fifty_two_week_low: float | None = None


@dataclass
class Financials:
    """Multi-year statement series, oldest -> newest, all money in Rs CRORE (EPS/shares excepted).

    The first block is what the summary tools use. The second block exists so
    pennystock.data.forensics can compute Piotroski / Altman / Beneish without a second fetch -
    those scores need balance-sheet and cash-flow lines the summary never shows.
    """
    fiscal_years: list = field(default_factory=list)      # e.g. [2022, 2023, 2024, 2025, 2026] ascending
    revenue_cr: list = field(default_factory=list)
    net_income_cr: list = field(default_factory=list)
    eps: list = field(default_factory=list)
    operating_income_cr: list = field(default_factory=list)
    total_debt_cr: list = field(default_factory=list)
    equity_cr: list = field(default_factory=list)
    free_cash_flow_cr: list = field(default_factory=list)

    # --- extra lines required by the forensic scores ---
    total_assets_cr: list = field(default_factory=list)
    current_assets_cr: list = field(default_factory=list)
    current_liabilities_cr: list = field(default_factory=list)
    total_liabilities_cr: list = field(default_factory=list)
    retained_earnings_cr: list = field(default_factory=list)
    long_term_debt_cr: list = field(default_factory=list)
    gross_profit_cr: list = field(default_factory=list)
    cost_of_revenue_cr: list = field(default_factory=list)
    operating_cash_flow_cr: list = field(default_factory=list)
    receivables_cr: list = field(default_factory=list)
    inventory_cr: list = field(default_factory=list)
    payables_cr: list = field(default_factory=list)
    net_ppe_cr: list = field(default_factory=list)
    depreciation_cr: list = field(default_factory=list)
    sga_cr: list = field(default_factory=list)
    interest_expense_cr: list = field(default_factory=list)
    ebitda_cr: list = field(default_factory=list)
    tax_provision_cr: list = field(default_factory=list)
    pretax_income_cr: list = field(default_factory=list)
    shares_outstanding: list = field(default_factory=list)   # count, NOT crore


@dataclass
class CorporateActions:
    splits: list = field(default_factory=list)      # [(date, ratio), ...]
    dividends: list = field(default_factory=list)   # [(date, amount), ...]


class MarketDataProvider(ABC):
    """Swappable interface for the live-data side of the spine."""

    @abstractmethod
    def get_snapshot(self, ticker: str) -> Snapshot: ...

    @abstractmethod
    def get_fundamentals(self, ticker: str) -> Fundamentals: ...

    @abstractmethod
    def get_price_history(self, ticker: str, period: str, interval: str): ...

    @abstractmethod
    def get_financials(self, ticker: str, years: int) -> "Financials": ...

    @abstractmethod
    def get_corporate_actions(self, ticker: str) -> "CorporateActions": ...


class YFinanceProvider(MarketDataProvider):
    """yfinance backend. `ticker` must already be fully-qualified (e.g. 'TMPV.NS');
    name->ticker resolution happens upstream in the tools layer."""

    def _info(self, ticker: str) -> dict[str, Any]:
        import yfinance as yf
        return _cached(("info", ticker), lambda: yf.Ticker(ticker).info or {})

    def get_snapshot(self, ticker: str) -> Snapshot:
        i = self._info(ticker)
        return Snapshot(
            symbol=i.get("symbol"),
            company=i.get("longName"),
            price=i.get("currentPrice", i.get("regularMarketPrice")),
            previous_close=i.get("previousClose"),
            day_high=i.get("dayHigh"),
            day_low=i.get("dayLow"),
            volume=i.get("volume"),
            market_cap=i.get("marketCap"),
        )

    def get_fundamentals(self, ticker: str) -> Fundamentals:
        i = self._info(ticker)
        return Fundamentals(
            symbol=i.get("symbol"),
            company=i.get("longName"),
            sector=i.get("sector"),
            industry=i.get("industry"),
            market_cap=i.get("marketCap"),
            pe_trailing=i.get("trailingPE"),
            pe_forward=i.get("forwardPE"),
            eps_trailing=i.get("trailingEps"),
            book_value=i.get("bookValue"),
            price_to_book=i.get("priceToBook"),
            debt_to_equity_pct=i.get("debtToEquity"),      # already a percentage number; no math needed
            return_on_equity=i.get("returnOnEquity"),
            profit_margin=i.get("profitMargins"),
            dividend_yield=i.get("dividendYield"),
            beta=i.get("beta"),
            fifty_two_week_high=i.get("fiftyTwoWeekHigh"),
            fifty_two_week_low=i.get("fiftyTwoWeekLow"),
        )

    def get_price_history(self, ticker: str, period: str = "1y", interval: str = "1d"):
        import yfinance as yf
        return _cached(
            ("hist", ticker, period, interval),
            lambda: yf.Ticker(ticker).history(period=period, interval=interval),
        )

    def get_financials(self, ticker: str, years: int = 5) -> Financials:
        return _cached(("fin", ticker, years), lambda: self._build_financials(ticker, years))

    def _build_financials(self, ticker: str, years: int) -> Financials:
        import pandas as pd
        import yfinance as yf
        t = yf.Ticker(ticker)
        inc, bal, cf = t.income_stmt, t.balance_sheet, t.cashflow
        if not isinstance(inc, pd.DataFrame) or inc.empty:
            return Financials()                                   # nothing available -> all-empty
        yrs = sorted({c.year for c in inc.columns})[-years:]      # ascending, last N fiscal years

        def series(df, labels, scale=CRORE):
            out = [None] * len(yrs)
            if not isinstance(df, pd.DataFrame) or df.empty:
                return out
            row = next((df.loc[l] for l in labels if l in df.index), None)
            if row is None:
                return out
            by_year = {c.year: row[c] for c in df.columns}        # match by YEAR (robust to date drift)
            for i, y in enumerate(yrs):
                v = by_year.get(y)
                out[i] = round(float(v) / scale, 2) if v is not None and pd.notna(v) else None
            return out

        return Financials(
            fiscal_years=yrs,
            revenue_cr=series(inc, ["Total Revenue", "Operating Revenue"]),
            net_income_cr=series(inc, ["Net Income", "Net Income Common Stockholders"]),
            eps=series(inc, ["Diluted EPS", "Basic EPS"], scale=1),
            operating_income_cr=series(inc, ["Operating Income", "EBIT"]),
            total_debt_cr=series(bal, ["Total Debt", "Long Term Debt"]),
            equity_cr=series(bal, ["Stockholders Equity", "Total Equity Gross Minority Interest"]),
            free_cash_flow_cr=series(cf, ["Free Cash Flow"]),
            # --- forensic inputs ---
            total_assets_cr=series(bal, ["Total Assets"]),
            current_assets_cr=series(bal, ["Current Assets", "Total Current Assets"]),
            current_liabilities_cr=series(bal, ["Current Liabilities", "Total Current Liabilities"]),
            total_liabilities_cr=series(bal, ["Total Liabilities Net Minority Interest", "Total Liabilities"]),
            retained_earnings_cr=series(bal, ["Retained Earnings"]),
            long_term_debt_cr=series(bal, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"]),
            gross_profit_cr=series(inc, ["Gross Profit"]),
            cost_of_revenue_cr=series(inc, ["Cost Of Revenue", "Reconciled Cost Of Revenue"]),
            operating_cash_flow_cr=series(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]),
            receivables_cr=series(bal, ["Accounts Receivable", "Receivables"]),
            inventory_cr=series(bal, ["Inventory"]),
            payables_cr=series(bal, ["Accounts Payable", "Payables"]),
            net_ppe_cr=series(bal, ["Net PPE", "Net Property Plant And Equipment"]),
            depreciation_cr=series(cf, ["Depreciation And Amortization", "Depreciation Amortization Depletion"]),
            sga_cr=series(inc, ["Selling General And Administration", "Selling General And Administrative"]),
            interest_expense_cr=series(inc, ["Interest Expense", "Interest Expense Non Operating"]),
            ebitda_cr=series(inc, ["EBITDA", "Normalized EBITDA"]),
            tax_provision_cr=series(inc, ["Tax Provision", "Income Tax Expense"]),
            pretax_income_cr=series(inc, ["Pretax Income", "Income Before Tax"]),
            shares_outstanding=series(bal, ["Ordinary Shares Number", "Share Issued"], scale=1),
        )

    def get_corporate_actions(self, ticker: str) -> CorporateActions:
        return _cached(("actions", ticker), lambda: self._build_actions(ticker))

    def _build_actions(self, ticker: str) -> CorporateActions:
        import yfinance as yf
        t = yf.Ticker(ticker)
        sp, dv = t.splits, t.dividends
        splits = [(str(d.date()), round(float(r), 3)) for d, r in sp.items()][-5:] if len(sp) else []
        divs = [(str(d.date()), round(float(v), 2)) for d, v in dv.items()][-6:] if len(dv) else []
        return CorporateActions(splits=splits, dividends=divs)


@lru_cache(maxsize=1)
def get_provider() -> MarketDataProvider:
    """Return the configured provider (cached). Swap backends here later."""
    from pennystock.config import settings
    if settings.market_provider == "yfinance":
        return YFinanceProvider()
    raise ValueError(f"Unknown market_provider: {settings.market_provider}")

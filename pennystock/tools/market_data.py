"""LIVE-DATA tools (the structured half of the data spine).

Tools, one per intent (the doc's 'route by intent' rule):
  get_indian_stock_data  -> live price snapshot          (MarketDataAgent)
  get_stock_fundamentals -> snapshot valuation & health  (FinancialAnalystAgent)
  get_price_history      -> computed technical indicators (FinancialAnalystAgent)
  get_financials         -> 5-year statement trends      (FinancialAnalystAgent)
  get_corporate_actions  -> splits & dividends           (FinancialAnalystAgent)
  get_shareholding       -> promoter/FII/DII + PLEDGE    (FinancialAnalystAgent)
  get_forensics          -> Piotroski/Altman/Beneish     (FinancialAnalystAgent)
  get_valuation_context  -> peer median + own P/E band   (FinancialAnalystAgent)
  get_corporate_events   -> announcements/insider/rating (FinancialAnalystAgent)
  get_management_commentary -> earnings-call language     (FinancialAnalystAgent)

Each resolves a name/ticker via the resolver, then delegates to a swappable provider.
Blocking network calls run in a thread so the LangGraph event loop stays responsive.
"""
import asyncio
import logging
from datetime import datetime
from dataclasses import asdict

from langchain_core.tools import tool

from pennystock.config import settings
from pennystock.data.fundamentals import snapshot_warnings, summarize_financials
from pennystock.data.indicators import detect_structural_break, summarize
from pennystock.data.annual_report import AnnualReportNotes, fetch_notes, notes_from_pages, summarize_notes
from pennystock.data.contingent_liabilities import latest_report, summarize_cl
from pennystock.data.contingent_liabilities import stored as stored_cl
from pennystock.data.events import NseEventsProvider, summarize_events
from pennystock.data.filing_text import pages_of
from pennystock.data.integrated_filings import filed_year, rpt_disclosure
from pennystock.data.event_signals import recent_summary
from pennystock.data.forensics import summarize_forensics
from pennystock.data.transcripts import fetch_transcripts, summarize_transcripts
from pennystock.data.ownership import summarize_ownership
from pennystock.data.ownership_providers import get_ownership
from pennystock.data.peers import compare_to_peers
from pennystock.data.providers import get_provider
from pennystock.data.valuation import valuation_bands
from pennystock.data.related_parties import summarize_rpt
from pennystock.data.resolver import resolve_symbol
from pennystock.data.sectors import profile_for

logger = logging.getLogger(__name__)


def _quarter_end(label: str | None) -> str | None:
    """'Jun 2026' -> '2026-06-30': the last day of the quarter a shareholding filing covers."""
    import calendar
    try:
        month = datetime.strptime(str(label), "%b %Y")
    except ValueError:
        return None
    return f"{month.year}-{month.month:02d}-{calendar.monthrange(month.year, month.month)[1]:02d}"


def _resolve_ticker(symbol: str, exchange: str) -> tuple[str, str | None]:
    """Resolve a name/ticker to a fully-qualified yfinance ticker + the official name."""
    match = resolve_symbol(symbol)
    if match:
        logger.info(f"Resolved '{symbol}' -> {match['symbol']} "
                    f"({match['company_name']}, {match['how']} {match['score']})")
        symbol, name = match["symbol"], match["company_name"]
    else:
        name = None
        logger.warning(f"Could not confidently resolve '{symbol}' to a listed NSE symbol.")
    suffix = ".NS" if exchange.upper() == "NSE" else ".BO"
    return f"{symbol.upper()}{suffix}", name


@tool
async def get_indian_stock_data(symbol: str, exchange: str = "NSE") -> str:
    """Fetches the live PRICE SNAPSHOT for an Indian stock (NSE/BSE).

    `symbol` accepts EITHER an exact ticker (e.g. SUZLON) OR the company name
    (e.g. "Kaynes Technology") - names are resolved automatically."""
    try:
        ticker, name = _resolve_ticker(symbol, exchange)
        snap = await asyncio.to_thread(get_provider().get_snapshot, ticker)
        if snap.price is None:
            return (f"Error: Could not find live data for '{ticker}'. "
                    f"If you guessed the ticker, retry with the full company name instead.")
        data = asdict(snap)
        data["company"] = data.get("company") or name        # visibility guardrail
        return str(data)
    except Exception as e:
        logger.warning(f"snapshot fetch failed for {symbol}: {e}")
        return f"Error fetching data for {symbol}."


@tool
async def get_stock_fundamentals(symbol: str, exchange: str = "NSE") -> str:
    """Fetches valuation & financial-health FUNDAMENTALS: P/E (trailing & forward), EPS,
    book value, price-to-book, debt-to-equity, ROE, profit margin, dividend yield, 52-week
    range, beta, sector/industry.

    Use for VALUE-INVESTING / Graham questions (margin of safety, is it cheap?). Any field
    returned as None is genuinely unavailable - say so, do NOT invent it. `symbol` may be a
    company name or a ticker."""
    try:
        ticker, name = _resolve_ticker(symbol, exchange)
        fund = await asyncio.to_thread(get_provider().get_fundamentals, ticker)
        data = asdict(fund)
        data["company"] = data.get("company") or name
        # Graham's margin of safety starts from the earnings yield; the live DIXON report worked out 1/37
        pe = data.get("pe_trailing")
        data["earnings_yield_pct"] = round(100 / pe, 2) if isinstance(pe, (int, float)) and pe > 0 else None
        warnings = snapshot_warnings(fund)
        if warnings:
            data["data_warnings"] = warnings
        return str(data)
    except Exception as e:
        logger.warning(f"fundamentals fetch failed for {symbol}: {e}")
        return f"Error fetching fundamentals for {symbol}."


@tool
async def get_price_history(symbol: str, exchange: str = "NSE", period: str | None = None) -> str:
    """Fetches historical prices and returns COMPUTED technical indicators: 50/200-day moving
    averages, RSI(14), trend, support & resistance, period high/low, % from high, avg volume.

    Use for TECHNICAL-ANALYSIS questions (levels to watch, overbought?). Indicators are
    computed in code, not estimated. `symbol` may be a name or ticker; `period` like
    '6mo', '1y', '5y'."""
    try:
        ticker, _ = _resolve_ticker(symbol, exchange)
        df = await asyncio.to_thread(
            get_provider().get_price_history,
            ticker, period or settings.history_period, settings.history_interval)
        result = summarize(df, settings.sr_lookback, settings.rsi_window, settings.history_recent_days)
        recent = df.tail(settings.history_recent_days) if df is not None and not df.empty else df
        brk = detect_structural_break(recent, settings.structural_break_pct)   # recent window only
        if brk:
            result["data_warning"] = brk
        return str(result)
    except Exception as e:
        logger.warning(f"history fetch failed for {symbol}: {e}")
        return f"Error fetching price history for {symbol}."


@tool
async def get_financials(symbol: str, exchange: str = "NSE") -> str:
    """Fetches 5 YEARS of financial statements and returns a COMPUTED trend summary: revenue +
    CAGR, net income + how many years were profitable, EPS, total-debt trend, latest
    debt-to-equity, free cash flow. Use to judge earnings CONSISTENCY, growth, leverage over
    time and cyclicality (Graham's earnings-stability test). `symbol` may be a name or ticker."""
    try:
        ticker, name = _resolve_ticker(symbol, exchange)
        fin = await asyncio.to_thread(get_provider().get_financials, ticker, settings.financials_years)
        summary = summarize_financials(fin)
        summary["company"] = name
        return str(summary)
    except Exception as e:
        logger.warning(f"financials fetch failed for {symbol}: {e}")
        return f"Error fetching financials for {symbol}."


@tool
async def get_corporate_actions(symbol: str, exchange: str = "NSE") -> str:
    """Fetches recent CORPORATE ACTIONS - stock splits and dividends. NOTE: this source does
    NOT record demergers/spin-offs; for those, a 'data_warning' from get_price_history (a large
    single-day gap) is the better signal. `symbol` may be a name or ticker."""
    try:
        ticker, name = _resolve_ticker(symbol, exchange)
        ca = await asyncio.to_thread(get_provider().get_corporate_actions, ticker)
        return str({"company": name, "recent_splits": ca.splits, "recent_dividends": ca.dividends})
    except Exception as e:
        logger.warning(f"corporate actions fetch failed for {symbol}: {e}")
        return f"Error fetching corporate actions for {symbol}."


@tool
async def get_shareholding(symbol: str, exchange: str = "NSE") -> str:
    """Fetches the OWNERSHIP picture: promoter holding % and its trend, FII/DII (institutional)
    holding and trend, public holding, shareholder count, PROMOTER PLEDGE % with a risk band, and
    explicit governance risk_flags. Includes up to 8 quarters of history.

    Use for governance / red-flag questions, especially on small and micro-caps: promoter
    pledging, promoters selling down, institutions exiting.

    `pledge_pct_of_promoter_holding` is the market-convention figure the risk band uses;
    `pledge_pct_of_total_shares` is NSE's headline number and is much smaller - quote the FORMER.
    If pledge is None it is genuinely UNKNOWN (not filed / source unavailable) - say so explicitly
    and treat it as unverified; NEVER assume it is zero."""
    try:
        ticker, name = _resolve_ticker(symbol, exchange)
        base = ticker.rsplit(".", 1)[0]                     # screener uses the bare NSE symbol
        own = await asyncio.to_thread(get_ownership, base)
        summary = summarize_ownership(own, settings.ownership_lookback_quarters)
        summary["company"] = name
        try:                                   # pledges created / invoked since the quarter the filing covers
            events = await asyncio.to_thread(recent_summary, base)
            # since the statutory reading's quarter - a newer quarter can be listed before its filing is read
            q_end = _quarter_end(summary.get("pledge_quarter") or summary.get("as_of_quarter"))
            summary["pledge_filings_since_quarter"] = [p for p in (events or {}).get("pledges", [])
                                                       if q_end and p.get("date", "") > q_end]
        except Exception as e:
            logger.warning(f"pledge events unavailable for {symbol}: {e}")
        return str(summary)
    except Exception as e:
        logger.warning(f"shareholding fetch failed for {symbol}: {e}")
        return f"Error fetching shareholding for {symbol}."


@tool
async def get_forensics(symbol: str, exchange: str = "NSE") -> str:
    """FORENSIC ACCOUNTING screen - "can I trust these numbers, and can this company survive?".

    Returns the Piotroski F-Score (0-9, fundamental improvement; 8-9 strong, <=3 weak), the Altman
    Z-Score (bankruptcy distress; >2.99 safe, <1.81 distress), the Beneish M-Score (earnings
    manipulation; above -1.78 is a flag), cumulative cash conversion (operating cash flow vs
    reported profit - profit that never becomes cash is paper), receivable/inventory/payable days
    with year-on-year change, and an explicit red_flags list.

    Run this BEFORE judging valuation: if the accounts cannot be trusted or the balance sheet is
    breaking, no multiple is cheap enough. Report the caveats the tool returns - Altman is
    unreliable for banks/NBFCs and Beneish is calibrated on US filers."""
    try:
        ticker, name = _resolve_ticker(symbol, exchange)
        prov = get_provider()
        fin = await asyncio.to_thread(prov.get_financials, ticker, settings.financials_years)
        snap = await asyncio.to_thread(prov.get_snapshot, ticker)
        mcap_cr = (snap.market_cap / 1e7) if snap.market_cap else None
        out = summarize_forensics(fin, mcap_cr)
        out["company"] = name
        return str(out)
    except Exception as e:
        logger.warning(f"forensics failed for {symbol}: {e}")
        return f"Error computing forensics for {symbol}."


@tool
async def get_valuation_context(symbol: str, exchange: str = "NSE") -> str:
    """VALUATION IN CONTEXT - answers "expensive versus WHAT?".

    Two references a bare multiple can never provide:
      peers       - median P/E, P/B, ROE, debt and margin for the company's own industry, plus
                    where this stock ranks (percentile, higher = better)
      own history - its P/E band over the last few fiscal years vs today's multiple

    ALWAYS call this before calling a stock cheap or expensive. A P/E of 33 is high for a metals
    company and low for an FMCG one; a P/E of 13 can still be that stock's own 4-year peak."""
    try:
        ticker, name = _resolve_ticker(symbol, exchange)
        base = ticker.rsplit(".", 1)[0]
        prov = get_provider()
        peers = await asyncio.to_thread(compare_to_peers, base)
        fin = await asyncio.to_thread(prov.get_financials, ticker, settings.financials_years)
        hist = await asyncio.to_thread(prov.get_price_history, ticker,
                                       settings.history_period, settings.history_interval)
        fund = await asyncio.to_thread(prov.get_fundamentals, ticker)
        bands = valuation_bands(fin, hist, fund.pe_trailing, fund.price_to_book, fund.book_value)
        return str({"company": name, "vs_peers": peers, "vs_own_history": bands})
    except Exception as e:
        logger.warning(f"valuation context failed for {symbol}: {e}")
        return f"Error building valuation context for {symbol}."


@tool
async def get_corporate_events(symbol: str, exchange: str = "NSE") -> str:
    """CORPORATE EVENTS & NEWS FLOW from NSE's statutory filings - what moves a stock before the
    numbers show it.

    Returns MATERIAL announcements (routine compliance filings are screened out): auditor
    resignations, defaults/insolvency, fraud or enforcement action, senior-management exits,
    litigation, fund raises, M&A. Plus `stored_events`, the last 12 months of DATED disclosures:
    insider buying/selling by promoters, directors and KMP, promoter pledges created/invoked, credit-
    rating upgrades/downgrades/Issuer Not Cooperating, severe events (which keep a stock out of the
    screen), and each event type's track record in this system's ledger.

    An auditor resigning mid-term is never routine and is among the highest-signal warnings in
    Indian markets. If no announcement data comes back, say event risk is UNCHECKED, not clean."""
    try:
        ticker, name = _resolve_ticker(symbol, exchange)
        base = ticker.rsplit(".", 1)[0]
        ev = await asyncio.to_thread(NseEventsProvider().fetch, base)
        out = summarize_events(ev)
        out["company"] = name
        stored = await asyncio.to_thread(recent_summary, base)
        out["stored_events"] = stored or ("UNCHECKED - no stored insider, pledge or rating events for "
                                          "this company (scripts/ingest_events.py). Do not read this as clean.")
        return str(out)
    except Exception as e:
        logger.warning(f"corporate events failed for {symbol}: {e}")
        return f"Error fetching corporate events for {symbol}."


@tool
async def get_management_commentary(symbol: str, exchange: str = "NSE") -> str:
    """EARNINGS-CALL LANGUAGE ANALYSIS - what management SAYS, and what they avoid saying.

    Downloads the last few quarterly earnings-call transcripts from NSE and returns:
      tone_latest / tone_previous - Loughran-McDonald financial sentiment on management's own
        words, including `hedging_ratio` (weak-modal share). "We WILL commission the plant"
        becoming "we MAY commission it" is a downgrade nobody announces.
      evasion - `explicit_non_answers` is the reliable number (management literally declined to
        answer); `possible_deflections` is a noisy heuristic to skim, NOT to score.
      guidance_latest vs guidance_previous - forward-looking sentences from each call. COMPARE
        THEM: a target stated last quarter and absent this quarter is the highest-signal item in
        any transcript. Quote management's own words when you make the point.
      new_risk_language - negative/uncertainty words used this quarter but not last.

    Use for management-quality, guidance and credibility questions. If no transcript is found, say
    management commentary is UNCHECKED - not that it is clean."""
    try:
        ticker, name = _resolve_ticker(symbol, exchange)
        base = ticker.rsplit(".", 1)[0]
        tx = await asyncio.to_thread(fetch_transcripts, base, settings.transcript_max_quarters)
        out = await asyncio.to_thread(summarize_transcripts, tx)
        out["company"] = name
        return str(out)
    except Exception as e:
        logger.warning(f"management commentary failed for {symbol}: {e}")
        return f"Error fetching management commentary for {symbol}."


def _is_lender(ticker: str) -> bool:
    f = get_provider().get_fundamentals(ticker)
    return profile_for(f.sector, f.industry).name == "financials"


def _excerpt_notes(base: str) -> AnnualReportNotes:
    """The two notes as text: from the report already in filings.db, else NSE's PDF (the old download
    path)."""
    report = latest_report(base)
    if report:
        texts = [p["text"] for p in pages_of(report["content_sha"])]
        return notes_from_pages(base, texts, None, None, report.get("fiscal_period"))
    return fetch_notes(base)


async def _yardsticks(ticker: str, fy) -> tuple[float | None, float | None, str]:
    """Net worth and revenue to judge the notes by: the company's own NSE filing, else Yahoo's latest year."""
    if fy and fy.equity_to_owners_cr is not None:
        return (fy.equity_to_owners_cr, fy.revenue_cr,
                f"NSE integrated filing {fy.fiscal_year} ({(fy.broadcast_at or '')[:11]})")
    fin = await asyncio.to_thread(get_provider().get_financials, ticker, settings.financials_years)

    def latest(xs):                          # series run oldest -> newest
        return next((v for v in reversed(xs or []) if v is not None), None)
    return latest(fin.equity_cr), latest(fin.revenue_cr), "Yahoo statements"


@tool
async def get_annual_report_notes(symbol: str, exchange: str = "NSE") -> str:
    """OFF-BALANCE-SHEET & RELATED-PARTY CHECK, from the company's own filings.

      contingent_liabilities - the annual report's note, read into numbers and VERIFIED against the page
        (machine_verified True): disputed claims, guarantees given for others (the red-flag class) and
        trade-finance contingents, in Rs crore, judged against NET WORTH. A guarantee to a group company or a
        large disputed tax demand is real risk that sits off the balance sheet. When machine_verified is False
        you get the note's text and candidate figures, which are HINTS only.
      related_party_transactions - the company's half-yearly related-party disclosure to NSE (SEBI Reg 23(9),
        XBRL): totals by kind of related party, the largest counterparties outside the subsidiaries and any
        loans or guarantees to them, judged against the same half-year's REVENUE (~10% is the long-standing
        threshold) - and, just as important, WHO the counterparties are.

    If nothing is returned, report off-balance-sheet exposure and related-party dealings as UNCHECKED -
    never as clean. `symbol` may be a company name or a ticker."""
    try:
        ticker, name = _resolve_ticker(symbol, exchange)
        base = ticker.rsplit(".", 1)[0]
        fy = await asyncio.to_thread(filed_year, base)
        net_worth, revenue, source = await _yardsticks(ticker, fy)
        cl = await asyncio.to_thread(stored_cl, base)
        rpt = await asyncio.to_thread(rpt_disclosure, base)
        verified = bool(cl and cl.get("status") == "verified")
        notes = None if (verified and rpt) else await asyncio.to_thread(_excerpt_notes, base)
        excerpt = summarize_notes(notes, net_worth, revenue) if notes else {}
        if verified:
            cl_view = summarize_cl(cl, net_worth, await asyncio.to_thread(_is_lender, ticker))
        else:
            why = (f"the note was read but not verified ({cl['status']})" if cl else
                   "the note has not been machine-read yet (scripts/extract_contingent_liabilities.py)")
            cl_view = {**(excerpt.get("contingent_liabilities") or {}), "machine_verified": False, "why": why}
        rpt_view = summarize_rpt(rpt) if rpt else {
            **(excerpt.get("related_party_transactions") or {}), "machine_verified": False,
            "why": "no related-party disclosure found in the company's NSE integrated filings"}
        out = {"symbol": base, "company": name, "contingent_liabilities": cl_view,
               "related_party_transactions": rpt_view,
               "yardsticks": {"net_worth_cr": net_worth, "revenue_cr": revenue, "source": source}}
        if not (verified and rpt) and excerpt.get("caveat"):
            out["caveat"] = excerpt["caveat"]
        return str(out)
    except Exception as e:
        logger.warning(f"annual report notes failed for {symbol}: {e}")
        return f"Error fetching annual report notes for {symbol}."

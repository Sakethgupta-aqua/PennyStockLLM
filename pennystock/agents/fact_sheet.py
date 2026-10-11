"""The fact sheet: one line per fact, each with its source, every comparison already made.

Pure: takes the evidence pack's parsed tool outputs and returns text. It formats the numbers the tools
returned and computes no new ones (rounding is to two decimals at most), so every figure in the sheet traces
to a tool - the numeric-provenance check relies on it. Units are converted, years labelled and verdicts worded
upstream, in the tools, before the analyst reads a word; a source that failed or came back empty is listed
under DATA GAPS.
"""
from __future__ import annotations

from datetime import date

from pennystock.config import settings

_TOPICS = {
    "get_stock_fundamentals": "price, P/E and market value", "get_price_history": "price and technicals",
    "get_financials": "five-year financials", "get_forensics": "trust checks (Piotroski, Altman, Beneish)",
    "get_valuation_context": "valuation against peers and own history",
    "get_shareholding": "shareholding and pledge",
    "get_corporate_events": "insider trades, ratings, severe events",
    "get_corporate_actions": "dividends and splits",
    "get_annual_report_notes": "contingent liabilities and related parties",
    "get_broker_consensus": "broker views", "get_trendlyne_view": "Trendlyne's scores",
    "get_management_commentary": "earnings-call guidance",
    "nse_integrated_filing": "filed annual figures (NSE integrated filing)",
    "screen_stocks": "alternatives screen"}


def _d(calls: dict, name: str) -> dict:
    c = calls.get(name)
    return c.data if c is not None and c.ok and isinstance(c.data, dict) else {}


def _n(v, places: int = 2) -> str:
    if v is None:
        return "UNKNOWN"
    if isinstance(v, (int, float)) and float(v).is_integer():
        return f"{v:,.0f}"
    return f"{v:,.{places}f}"


def _cr(v) -> str:
    return "UNKNOWN" if v is None else f"Rs {v:,.2f} Cr"


def _rs(v) -> str:
    return "UNKNOWN" if v is None else f"Rs {_n(v)}"


def _years(d: dict | None) -> str:
    return " | ".join(f"{fy} {v:,.2f}" for fy, v in (d or {}).items()) or "UNKNOWN"


def _clip(text, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


# --- sections: each returns its lines and may add to the gaps -------------------------------------
def _price_valuation(calls: dict, gaps: list) -> list[str]:
    f, ph = _d(calls, "get_stock_fundamentals"), _d(calls, "get_price_history")
    v = _d(calls, "get_valuation_context")
    out = []
    if ph.get("last_close") is not None:
        line = f"- Price {_rs(ph['last_close'])} (last close)"
        if f.get("fifty_two_week_low") is not None and f.get("fifty_two_week_high") is not None:
            line += f"; 52-week range {_rs(f['fifty_two_week_low'])} to {_rs(f['fifty_two_week_high'])}"
        if ph.get("pct_from_peak") is not None:
            line += f"; {ph['pct_from_peak']}% from its 5-year peak {_rs(ph.get('peak_5y'))}"
        out.append(line + ". [Yahoo]")
    if f:
        ey = (f"; earnings yield {_n(f['earnings_yield_pct'])}% (1 / trailing P/E)"
              if f.get("earnings_yield_pct") is not None else "")
        out.append(f"- P/E {_n(f.get('pe_trailing'))} trailing, {_n(f.get('pe_forward'))} forward{ey}; P/B "
                   f"{_n(f.get('price_to_book'))}; dividend yield {_n(f.get('dividend_yield'))}%; "
                   "debt/equity "
                   f"{_n(f.get('debt_to_equity_pct'))}% (a percentage of equity, not a multiple). [Yahoo]")
    own, peers = v.get("vs_own_history") or {}, v.get("vs_peers") or {}
    if own.get("verdict"):
        out.append(f"- Against its own history: {own['verdict']} [get_valuation_context]")
    pe = (peers.get("metrics") or {}).get("P/E") or {}
    if all(pe.get(k) is not None for k in ("value", "peer_median", "vs_median_pct")):
        # from the fields: the tool's headline '(46.3 vs 27.65)' left the order to the reader, and the live
        # fact-check (2026-10-11) argued over it for 51,211 tokens
        pct = pe["vs_median_pct"]
        line = (f"- Against peers ({peers.get('compared_against')}): P/E {_n(pe['value'])} in the peer table "
                f"against the sector median of {_n(pe['peer_median'])} - {_n(abs(pct), 1)}% "
                f"{'ABOVE' if pct > 0 else 'BELOW'} it.")
    elif peers.get("headline"):
        line = f"- Against peers: {peers['headline']} Compared against {peers.get('compared_against')}."
    else:
        line = None
    if line:
        pe_peer = ((peers.get("metrics") or {}).get("P/E") or {}).get("value")
        pe_now = f.get("pe_trailing")
        if pe_peer and pe_now and abs(pe_peer - pe_now) > 0.1 * pe_now:
            line += (f" (the peer table uses the screening cache's P/E {_n(pe_peer)}; Yahoo's trailing P/E "
                     "is "
                     f"{_n(pe_now)})")
        out.append(line)
    return out


def _growth(calls: dict, gaps: list) -> list[str]:
    fin, fy = _d(calls, "get_financials"), _d(calls, "nse_integrated_filing")
    out = []
    if fin.get("revenue_cr"):
        g = fin.get("revenue_cagr")
        span = f"; growth {g['pct']}% a year over {g['from']}->{g['to']} ({g['years']} years)" if g else ""
        out.append(f"- Revenue (Rs Cr, Yahoo): {_years(fin['revenue_cr'])}{span}.")
        out.append(f"- Net profit (Rs Cr, Yahoo): {_years(fin.get('net_income_cr'))}; profitable "
                   f"{fin.get('profitable_years')} years; trend {fin.get('net_income_trend')}.")
        if fin.get("eps"):
            out.append(f"- EPS (Rs, Yahoo): {_years(fin.get('eps'))}.")
        out.append(f"- {fin.get('latest_year')}: ROE {_n(fin.get('roe_pct_latest'))}% | ROCE "
                   f"{_n(fin.get('roce_pct_latest'))}% | debt/equity "
                   f"{_n(fin.get('debt_to_equity_pct_latest'))}% | "
                   f"free cash flow (Rs Cr) {_years(fin.get('free_cash_flow_cr'))}.")
    if fy:
        out.append(f"- Filed {fy.get('fiscal_year')} ({fy.get('basis')}, NSE integrated filing "
                   f"{str(fy.get('broadcast_at') or '')[:11]}): revenue from operations "
                   f"{_cr(fy.get('revenue_cr'))}, "
                   f"profit to shareholders {_cr(fy.get('profit_to_owners_cr'))}, net worth "
                   f"{_cr(fy.get('equity_to_owners_cr'))}, basic EPS {_rs(fy.get('basic_eps'))}, "
                   "exceptional items "
                   f"{_cr(fy.get('exceptional_cr'))}; auditor {fy.get('auditor') or 'UNKNOWN'}. These are "
                   "the "
                   "company's own audited figures; Yahoo's can differ slightly in definition.")
    return out


def _trust(calls: dict, gaps: list) -> list[str]:
    fo = _d(calls, "get_forensics")
    if not fo:
        return []
    p, a, b = fo.get("piotroski") or {}, fo.get("altman_emerging_markets") or {}, fo.get("beneish") or {}
    out = []
    if p:
        failed = ", ".join(t.replace("_", " ") for t in p.get("failed_tests") or [])
        out.append(f"- Piotroski F-score {p.get('f_score')}/{p.get('out_of')} ({p.get('verdict')})"
                   + (f"; failed: {failed}" if failed else "") + ". [get_forensics]")
    if a:
        # this variant's own boundaries: the live DIXON report supplied the 1968 model's 2.99 from memory
        bounds = (f"; safe above {a['safe_above']:g}, distress below {a['distress_below']:g}"
                  if a.get("safe_above") is not None and a.get("distress_below") is not None else "")
        out.append(f"- Altman Z''/EM {a.get('z_em_score')} ({a.get('zone')}{bounds}) - {a.get('note')}")
    if b:
        out.append(f"- Beneish M-score {b.get('m_score')} ({b.get('verdict')}; {b.get('note')})")
    ic, de = fo.get("interest_coverage") or {}, fo.get("debt_to_ebitda") or {}
    cc, wc = fo.get("cash_conversion") or {}, fo.get("working_capital") or {}
    out.append(f"- Interest cover {ic.get('interest_coverage')}x ({ic.get('verdict')}); debt/EBITDA "
               f"{de.get('debt_to_ebitda')}x ({de.get('verdict')}); cash from operations / profit "
               f"{cc.get('cfo_to_pat')} over {cc.get('years')} years ({cc.get('verdict')}); cash conversion "
               "cycle "
               f"{wc.get('cash_conversion_cycle_days')} days.")
    out += [f"- Red flag: {x}" for x in fo.get("red_flags") or []
            if not str(x).lower().startswith("no forensic")]
    return out


def _off_balance_sheet(calls: dict, gaps: list) -> list[str]:
    ar = _d(calls, "get_annual_report_notes")
    if not ar:
        return []
    out = []
    cl, rp, yard = ar.get("contingent_liabilities") or {}, ar.get("related_party_transactions") or {}, \
        ar.get("yardsticks") or {}
    if cl.get("machine_verified"):
        t = cl.get("totals_cr") or {}
        claims, others = t.get("disputed_claims") or {}, t.get("guarantees_for_others") or {}
        trade, other = t.get("trade_finance") or {}, t.get("other") or {}
        line = (f"- Contingent liabilities ({cl.get('source')}; unit as printed: "
                f"{cl.get('unit_as_printed')}; every "
                f"figure verified against the page): disputed claims {_cr(claims.get('current'))} (previous "
                "year "
                f"{_cr(claims.get('previous'))}) = {cl.get('disputed_claims_pct_of_net_worth')}% of net "
                "worth "
                f"{_cr(cl.get('net_worth_cr'))}"
                + (f" [{yard['source']}]." if yard.get("source") else "."))
        out.append(line)
        if "guarantees_for_others_pct_of_net_worth" in cl:
            out.append(f"  Guarantees given for others: {_cr(others.get('current'))} "
                       f"({cl.get('guarantees_for_others_pct_of_net_worth')}% of net worth). Trade-finance "
                       f"contingents (normal operating items): {_cr(trade.get('current'))}. Other: "
                       f"{_cr(other.get('current'))}.")
            every = cl.get("all_contingents_cr") or {}
            if every.get("current") is not None:
                out.append(f"  All contingent liabilities together: {_cr(every['current'])} "
                           f"({cl.get('all_contingents_pct_of_net_worth')}% of net worth) - the sum of the "
                           "lines above; the per-line shares are the ones to judge.")
        else:
            out.append(f"  {cl.get('how_to_judge')}")
        for item in (cl.get("largest_items") or [])[:4]:
            out.append(f"  - {_clip(item.get('label'), 80)} [{item.get('category')}]: "
                       f"{_cr(item.get('current_cr'))} "
                       + (f"(previous {_cr(item['previous_cr'])})" if item.get("previous_cr") is not None
                          else "(previous year: no figure printed)"))
        out += [f"  - Check: {w}" for w in cl.get("warnings") or []]
    elif cl:
        out.append(f"- Contingent liabilities: NOT machine-verified ({cl.get('why')}). Indicative figures "
                   "only: "
                   f"{cl.get('figures_mentioned')} - quote them as indicative, never as totals.")
        gaps.append("- contingent liabilities: not machine-verified - quote only as indicative")
    if rp.get("source", "").startswith("NSE"):
        out.append(f"- Related-party transactions, {rp.get('period')} ({rp.get('source')}, filed "
                   f"{rp.get('filed')}; "
                   f"amounts as disclosed, can include taxes): outside the subsidiaries "
                   f"{_cr(rp.get('outside_subsidiaries_cr'))} = "
                   f"{rp.get('outside_subsidiaries_pct_of_revenue')}% of "
                   f"the half-year's revenue {_cr(rp.get('half_year_revenue_cr'))}; with promoter / KMP "
                   "entities "
                   f"{_cr(rp.get('promoter_kmp_entities_cr'))} "
                   f"({rp.get('promoter_kmp_entities_pct_of_revenue')}%).")
        for x in (rp.get("largest_outside_subsidiaries") or [])[:5]:
            out.append(f"  - {_clip(x.get('entity'), 40)} -> {_clip(x.get('counterparty'), 40)} "
                       f"[{x.get('class')}: "
                       f"{_clip(x.get('relationship'), 60)}]: {_clip(x.get('type'), 40)} "
                       f"{_cr(x.get('amount_cr'))}")
        for x in (rp.get("loans_guarantees_outside_subsidiaries") or [])[:3]:
            out.append(f"  - Loan / guarantee outside the subsidiaries: {_clip(x.get('counterparty'), 40)} "
                       f"[{x.get('class')}]: {_clip(x.get('type'), 40)} {_cr(x.get('amount_cr'))}")
        out.append(f"  {rp.get('how_to_judge')}")
    elif rp:
        out.append(f"- Related-party transactions: no NSE disclosure found ({rp.get('why')}); the "
                   "annual-report "
                   "excerpt is indicative only.")
        gaps.append("- related-party transactions: no structured disclosure - quote only as indicative")
    return out


def _owners(calls: dict, gaps: list) -> list[str]:
    sh, ev = _d(calls, "get_shareholding"), _d(calls, "get_corporate_events").get("stored_events") or {}
    out = []
    if sh.get("promoter_pct") is not None:
        change = next((v for k, v in sh.items() if k.startswith("promoter_change_")), None)
        out.append(f"- Promoter holding {sh['promoter_pct']}% ({sh.get('as_of_quarter')}), change over four "
                   f"quarters {change} points; FII {sh.get('fii_pct')}%, DII {sh.get('dii_pct')}%. "
                   "[get_shareholding]")
        p, src = sh.get("pledge_pct_of_promoter_holding"), sh.get("pledge_source")
        quarter = sh.get("pledge_quarter") or sh.get("as_of_quarter")
        if p is None:
            out.append("- Promoter pledge: UNKNOWN - the statutory shareholding filing could not be read.")
            gaps.append("- promoter pledge: UNKNOWN (statutory filing unreadable)")
        elif p == 0 and src == "nse_shp_xbrl" and sh.get("promoter_pct") == 0:
            out.append("- Promoter pledge: NONE - the company has no promoter group (statutory shareholding "
                       f"filing, {quarter}).")
        elif p == 0 and src == "nse_shp_xbrl":
            out.append("- Promoter pledge: NONE - no promoter shares pledged (statutory shareholding "
                       "filing, "
                       f"{quarter}).")
        else:
            out.append(f"- Promoter pledge: {p}% of the promoters' own holding "
                       f"({sh.get('pledge_pct_of_total_shares')}"
                       f"% of all shares; band {sh.get('pledge_band')}) per {src}, {quarter}.")
        if sh.get("encumbrance_note"):
            out.append(f"- Encumbrance: {sh['encumbrance_note']}.")
        since = sh.get("pledge_filings_since_quarter") or []
        listed = "; ".join(f"{x.get('date')} {x.get('type')}" for x in since[:4])
        out.append(f"- Pledge filings since that quarter: {len(since)}" + (f": {listed}." if since else "."))
    ins = ev.get("insider") or {}
    if ev:
        out.append(f"- Insider trades in the last {ev.get('window_days')} days (Rs 10 lakh or more): "
                   f"{ins.get('buys', 0)} buys, {ins.get('sells', 0)} sells. [NSE disclosures]")
        ratings = "; ".join(f"{r.get('agency')} {r.get('rating')}"
                            + (f" ({r.get('outlook')})" if r.get("outlook") else "") + f" {r.get('date')}"
                            for r in (ev.get("ratings") or [])[:3])
        out.append(f"- Credit ratings: {ratings or 'none on record in the window'}.")
        severe = ev.get("severe") or []
        named = "; ".join(f"{x.get('date')} {x.get('type')}" for x in severe[:3]) if severe else "none"
        out.append(f"- Severe events (own auditor quitting, default, fraud, insolvency): {named}.")
    return out


def _street(calls: dict, gaps: list) -> list[str]:
    bc = _d(calls, "get_broker_consensus")
    if not bc:
        return []
    if str(bc.get("broker_coverage", "")).startswith("UNKNOWN"):
        gaps.append("- broker views: UNKNOWN - no broker report on file")
        return ["- Broker coverage: UNKNOWN - no broker report or call is on file."]
    cons, out = bc.get("consensus") or {}, []
    calls_txt = "; ".join(f"{c.get('broker')} {c.get('rating')} {_rs(c.get('target'))} ({c.get('date')})"
                          for c in cons.get("calls") or [])
    out.append(f"- {cons.get('brokers')} broker(s) on file - attributed opinions, not facts: {calls_txt}.")
    tgt = cons.get("target") or {}
    if tgt:
        out.append(f"- Median target {_rs(tgt.get('median'))} = {tgt.get('implied_upside_pct')}% from "
                   "today's price; "
                   f"range {_rs(tgt.get('low'))} to {_rs(tgt.get('high'))}; dispersion "
                   f"{tgt.get('dispersion_pct')}%; "
                   f"flags: {', '.join(cons.get('flags') or []) or 'none'}.")
    for e in bc.get("forward_estimates") or []:
        if e.get("kind") == "reported":
            if e.get("filed") is None and "brokers_median" in e:
                out.append(f"- {e.get('kpi')} {e.get('period')} (reported, as each broker prints it): median "
                           f"{_n(e.get('brokers_median'))} ({', '.join(e.get('brokers') or [])}).")
                continue
            line = (f"- {e.get('kpi')} {e.get('period')} (reported): filed {_n(e.get('filed'))}"
                    if e.get("filed") is not None else
                    f"- {e.get('kpi')} {e.get('period')} (reported; each broker's basis taken from its "
                    "profit "
                    "figure)")
            on = e.get("brokers_on_filed_basis") or []
            if on:
                line += "; on the filed basis: " + ", ".join(f"{b.get('broker')} {_n(b.get('value'))}"
                                                             for b in on)
            adj = e.get("brokers_adjusted")
            if adj:
                line += (f"; ADJUSTED basis ({', '.join(adj.get('brokers') or [])}): median "
                         f"{_n(adj.get('median'))} "
                         f"- compare their forecasts with this, not with the filed figure")
            out.append(line + ".")
        else:
            line = (f"- {e.get('kpi')} {e.get('period')} forecast ({e.get('basis')}): median "
                    f"{_n(e.get('median'))} "
                    f"(range {_n(e.get('low'))} to {_n(e.get('high'))})")
            others = e.get("other_basis") or []
            if others:
                line += "; on another basis: " + ", ".join(
                    f"{o.get('broker')} ({o.get('basis')}) {_n(o.get('value'))}" for o in others)
            out.append(line + ".")
    return out


def _trendlyne(calls: dict, gaps: list) -> list[str]:
    tv = _d(calls, "get_trendlyne_view").get("trendlyne_view")
    if isinstance(tv, str):
        gaps.append("- Trendlyne's scores: UNKNOWN - no Stock Report on file")
        return [f"- Trendlyne: {_clip(tv, 140)}"]
    if not isinstance(tv, dict):
        return []
    out = [f"- Trendlyne (a vendor's opinion, report {str(tv.get('as_of'))[:10]}): Durability "
           f"{tv.get('durability')} "
           f"({tv.get('durability_label')}), Valuation {tv.get('valuation')} ({tv.get('valuation_label')}), "
           "Momentum "
           f"{tv.get('momentum')} ({tv.get('momentum_label')}) - '{tv.get('classification')}'; checklist "
           f"{tv.get('checklist_passed')} passed, {tv.get('checklist_failed')} failed.",
           f"- Trendlyne's analyst aggregate: {tv.get('analysts')} analysts, consensus "
           f"{tv.get('consensus')}, average "
           f"target {_rs(tv.get('target_price'))} ({tv.get('upside_pct')}% upside); "
           f"{tv.get('n_strong_buy')} strong "
           f"buy, {tv.get('n_buy')} buy, {tv.get('n_hold')} hold, {tv.get('n_sell')} sell, "
           f"{tv.get('n_strong_sell')} "
           "strong sell."]
    if tv.get("pe_zone_meaning"):
        out.append(f"- P/E zone: {tv['pe_zone_meaning']}.")
    return out


def _technicals(calls: dict, gaps: list) -> list[str]:
    ph = _d(calls, "get_price_history")
    last, s200, s50 = ph.get("last_close"), ph.get("sma_200"), ph.get("sma_50")
    if None in (last, s200, s50):
        return []
    rsi = ph.get("rsi_14")
    mood = "overbought" if rsi and rsi >= 70 else "oversold" if rsi and rsi <= 30 else "neutral"
    return [f"- Price {_rs(last)} is {'ABOVE' if last > s200 else 'BELOW'} its 200-day average "
            f"{_rs(round(s200))} and "
            f"{'ABOVE' if last > s50 else 'BELOW'} its 50-day {_rs(round(s50))}; RSI(14) {rsi} ({mood}); "
            "trend "
            f"{ph.get('trend')}; support {_rs(ph.get('support'))}, resistance {_rs(ph.get('resistance'))}. "
            "[get_price_history]"]


def _corporate_actions(calls: dict, gaps: list) -> list[str]:
    ca = _d(calls, "get_corporate_actions")
    if not ca:
        return []
    divs = " | ".join(f"{str(d)[:7]} Rs {_n(a)}" for d, a in ca.get("recent_dividends") or [])
    splits = ", ".join(f"{_n(r)}-for-1 on {d}" for d, r in ca.get("recent_splits") or [])
    return [f"- Dividends per share (Yahoo, ex-date): {divs or 'none in the data'}.",
            "- Splits / bonus issues (Yahoo records a bonus as a split ratio): "
            f"{splits or 'none in the data'}."]


def _management(calls: dict, gaps: list) -> list[str]:
    mc = _d(calls, "get_management_commentary")
    if not mc.get("latest_call"):
        return []
    lc, tl, tp, ev = mc["latest_call"], mc.get("tone_latest") or {}, mc.get("tone_previous") or {}, \
        mc.get("evasion") or {}
    out = [f"- Latest earnings call {lc.get('date')} ({lc.get('questions')} analyst questions): net tone "
           f"{tl.get('net_tone')} against {tp.get('net_tone')} on the call before; hedging "
           f"{tl.get('hedging_ratio')} "
           f"against {tp.get('hedging_ratio')}; explicit non-answers {ev.get('explicit_non_answers')} of "
           f"{ev.get('questions_analysed')}. [get_management_commentary]"]
    out += [f"  - this call: \"{_clip(s, 220)}\"" for s in (mc.get("guidance_latest") or [])[:6]]
    out += [f"  - the call before: \"{_clip(s, 220)}\"" for s in (mc.get("guidance_previous") or [])[:4]]
    return out


def _alternatives(calls: dict, gaps: list) -> list[str]:
    if "screen_stocks" not in calls:
        return []
    sc = _d(calls, "screen_stocks")
    subject = next((c.data.get("symbol") for c in calls.values() if c.ok and isinstance(c.data, dict)
                    and c.data.get("symbol")), None)
    cands = [x for x in sc.get("candidates") or [] if x.get("symbol") != subject]
    if not cands:
        gaps.append("- alternatives: the screen returned none "
                    f"({_clip(sc.get('screen') or sc.get('note') or '', 80)})")
        return []
    fun = sc.get("funnel") or {}
    sector, industry = cands[0].get("sector"), _d(calls, "get_stock_fundamentals").get("industry")
    scope = (f"the {sector} sector (Yahoo's sector, broader than the company's own industry, {industry}: "
             "these "
             f"are sector names, not like-for-like peers)" if industry else f"the {sector} sector")
    out = [f"- Screen of {scope}; cached universe, scores are percentile ranks among survivors, not absolute "
           f"quality: {fun.get('universe')} names -> {fun.get('tradeable')} tradeable -> "
           f"{fun.get('survived_veto_gates')} passed the veto gates. [screen_stocks]"]
    for i, x in enumerate(cands[: settings.evidence_alternatives], 1):
        out.append(f"  {i}. {x.get('company')} ({x.get('symbol')}): score {x.get('score_out_of_100')}, ROCE "
                   f"{_n(x.get('roce_pct'))}%, P/E {_n(x.get('pe'))}, debt/EBITDA "
                   f"{_n(x.get('debt_to_ebitda'))}, red "
                   f"flags {x.get('red_flag_count')}")
    return out


_SECTIONS = (("PRICE & VALUATION", _price_valuation), ("GROWTH", _growth),
             ("CAN THE NUMBERS BE TRUSTED", _trust), ("OFF THE BALANCE SHEET", _off_balance_sheet),
             ("OWNERS", _owners), ("STREET VIEW", _street), ("TRENDLYNE", _trendlyne),
             ("TECHNICALS", _technicals),
             ("DIVIDENDS & CORPORATE ACTIONS", _corporate_actions), ("MANAGEMENT", _management),
             ("ALTERNATIVES", _alternatives))


def render_fact_sheet(symbol: str, company: str | None, calls: dict, today: date) -> str:
    gaps = [f"- {_TOPICS.get(n, n)}: UNKNOWN - {_clip(c.output, 120)}" for n, c in calls.items() if not c.ok]
    body: list[str] = []
    for title, section in _SECTIONS:
        lines = section(calls, gaps)
        if lines:
            body += [f"### {title}", *lines]
    head = [f"EVIDENCE PACK - {company or symbol} ({symbol}), computed by code on {today:%Y-%m-%d} from the "
            "tools in brackets. Figures are exact as stated (money in Rs crore unless marked; percentages "
            "already in %). Use them as given: do not recompute, re-label years or convert units. Every "
            "UNKNOWN below is unknown - say so; never fill it from memory.",
            "### DATA GAPS", *(gaps or ["- none: every source answered"])]
    text = "\n".join(head + body)
    if len(text) > settings.evidence_max_chars:
        text = (text[: settings.evidence_max_chars - 40].rsplit("\n", 1)[0]
                + "\n[... cut to fit the evidence budget]")
    return text

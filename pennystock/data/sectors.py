"""Sector rulebooks: what each kind of business is actually judged on.

The oldest discipline in equity research is comparing like with like, and a single-rulebook screen
breaks it in three separate ways. Measured across a live 500-name universe:

    metric          Technology   Utilities   Financials     spread
    ROCE                 21.1%        6.8%        11.1%       3.1x
    Debt/Equity           9.9%      113.1%       243.9%        25x
    Debt/EBITDA           0.29        4.31         5.55        19x
    P/E (median)          27.8        28.1         22.4    Energy 10.1 vs Healthcare 46.4

  1. RANKING. Percentile-ranking ROCE across all sectors at once does not measure quality, it
     measures asset intensity. An IT firm is asset-light and structurally earns 20%+; a regulated
     utility earns single digits on a rate base and that is what the regulator INTENDS. Ranked
     together, the best utility in the country loses to a mediocre software company every time.

  2. GATES. A Debt/EBITDA veto at 4.5x is sensible for an FMCG company and nonsense for a lender,
     a utility or a developer, whose leverage is the business model. The median financial-sector
     name sits at 5.55x - the standard rulebook disqualifies an entire sector by construction.

  3. VALUATION. Banks are valued on book, cyclicals on replacement cost or EV/EBITDA, asset-light
     compounders on earnings. Ranking everyone on P/E rewards whichever sector happens to sit in a
     structurally low-multiple part of the market - usually a commodity at peak earnings, which is
     the oldest value trap there is.

Each profile also carries BLIND_SPOTS: the metrics that sector is genuinely judged on which this
screen does not have. A screen that cannot say what it is failing to measure invites the reader to
assume it measured everything.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SectorProfile:
    name: str
    sectors: tuple[str, ...] = ()          # yfinance sector labels
    industries: tuple[str, ...] = ()       # substrings matched against the industry label
    skip_metrics: frozenset[str] = frozenset()
    skip_vetoes: frozenset[str] = frozenset()
    veto_overrides: dict[str, float] = field(default_factory=dict)
    valuation_metrics: tuple[tuple[str, bool], ...] = (("pe", False), ("pb", False))
    weights: dict[str, float] = field(default_factory=dict)
    blind_spots: tuple[str, ...] = ()
    note: str = ""
    # Sector KPIs read from broker reports that may join a pillar - (metric, higher_is_better).
    # They only ever vote once enough companies in the sector carry them; see screener.py.
    kpi_pillars: dict[str, tuple[tuple[str, bool], ...]] = field(default_factory=dict)


# Metrics that are meaningless for lenders. Deposits are not Debt/EBITDA-style debt, interest is a
# lender's cost of goods rather than a burden to cover, growing a loan book is an operating cash
# OUTFLOW, and neither Altman nor Beneish was estimated on financial filers.
_FIN_SKIP = frozenset({"debt_to_ebitda", "interest_coverage", "altman_zem",
                       "cash_conversion", "beneish_m", "debt_to_equity_pct"})

PROFILES: tuple[SectorProfile, ...] = (
    SectorProfile(
        name="financials",
        sectors=("financial services", "financials", "financial"),
        industries=("bank", "insur", "credit services", "capital markets", "asset management",
                    "financial", "mortgage", "brokerage"),
        skip_metrics=_FIN_SKIP,
        skip_vetoes=frozenset({"interest cover", "Altman EM distress zone", "cash conversion",
                               "debt/EBITDA", "Beneish manipulation flag"}),
        # a bank is bought and sold on book value; P/E is secondary
        valuation_metrics=(("pb", False), ("pe", False)),
        weights={"quality": 0.30, "safety": 0.20, "governance": 0.25, "valuation": 0.15,
                 "momentum": 0.10},
        blind_spots=("net interest margin (NIM)", "GNPA/NNPA and slippage", "provision coverage",
                     "capital adequacy (CRAR)", "CASA ratio", "credit cost", "cost-to-income"),
        note="Judge a lender on asset quality and capital, not on leverage ratios built for "
             "manufacturers. None of those inputs is in this screen.",
        # every manufacturer safety ratio is skipped for a lender, which left 20% of its score a
        # constant 0.50; asset quality and capital are what a lender's safety actually means
        kpi_pillars={"safety": (("gnpa_pct", False), ("nnpa_pct", False), ("car_pct", True),
                                ("pcr_pct", True), ("credit_cost_pct", False)),
                     "quality": (("nim_pct", True), ("roa_pct", True), ("cost_to_income_pct", False))},
    ),
    SectorProfile(
        name="utilities_power",
        sectors=("utilities",),
        industries=("utilities", "power", "electric"),
        # regulated returns on a long-gestation asset base: high leverage is the design, not a flaw
        veto_overrides={"debt_to_ebitda": 7.0, "interest_coverage": 1.2},
        blind_spots=("plant load factor (PLF)", "regulated RoE and the tariff order",
                     "PPA coverage and merchant exposure", "receivables from discoms",
                     "capacity under construction"),
        note="Leverage is structural against contracted cash flows. What matters is PLF, the "
             "tariff order and whether the discoms actually pay.",
    ),
    SectorProfile(
        name="real_estate",
        sectors=("real estate",),
        industries=("real estate", "realty",),
        # debt funds inventory and cash flow is lumpy across the project cycle
        veto_overrides={"debt_to_ebitda": 6.0, "cash_conversion": 0.0},
        blind_spots=("pre-sales / bookings", "collections versus pre-sales", "unsold inventory",
                     "net debt / equity against the board's own ceiling", "land bank and NAV",
                     "project approvals and RERA status"),
        note="Pre-sales are bookings, not cash. Collections are what fund the next project, and "
             "the gap between the two is where developers get into trouble.",
    ),
    SectorProfile(
        name="materials_cyclical",
        sectors=("basic materials",),
        industries=("steel", "cement", "aluminum", "copper", "mining", "building materials",
                    "coking coal", "thermal coal", "paper", "chemicals"),
        blind_spots=("EBITDA per tonne", "EV per tonne against replacement cost (~$120-140/t for "
                     "cement)", "capacity utilisation", "realisation per tonne", "cost of "
                     "production", "position in the commodity cycle"),
        note="A low P/E on peak-cycle earnings is a warning, not a bargain - the E falls before "
             "the P does. Judge these on cost position and where the cycle sits.",
    ),
    SectorProfile(
        name="technology",
        sectors=("technology",),
        industries=("information technology services", "software", "computer"),
        blind_spots=("deal TCV and book-to-bill", "constant-currency revenue growth",
                     "attrition (a two-year high of ~13.8% at TCS)", "utilisation (~85% norm)",
                     "revenue per employee (~USD 50k across FY20-24)", "onsite/offshore mix"),
        note="Asset-light, so ROCE is structurally high and means less here. Growth is bought "
             "with deal wins and defended against attrition, neither of which this screen sees.",
    ),
    SectorProfile(
        name="healthcare_pharma",
        sectors=("healthcare",),
        industries=("drug manufacturers", "pharmaceutical", "biotechnology", "medical"),
        blind_spots=("USFDA inspection outcomes (Form 483s, warning letters, import alerts)",
                     "ANDA/DMF filings and approvals", "R&D as % of sales",
                     "US generics versus domestic formulations mix", "price erosion in US generics"),
        note="A single USFDA warning letter can remove a plant's earnings overnight. That risk is "
             "regulatory, not financial, and nothing in this screen detects it.",
    ),
    SectorProfile(
        name="consumer_staples",
        sectors=("consumer defensive",),
        industries=("packaged foods", "household", "personal products", "beverages", "tobacco"),
        # negative working capital is a STRENGTH here (the trade funds the business)
        veto_overrides={"cash_conversion": 0.4},
        blind_spots=("volume growth versus value growth", "advertising spend as % of sales",
                     "distribution reach (outlets covered)", "rural versus urban mix",
                     "gross margin and the raw-material cycle"),
        note="Value growth without VOLUME growth is just price inflation, and it reverses. These "
             "names carry structurally high multiples; cheapness is rarely the reason to own one.",
    ),
    SectorProfile(
        name="consumer_cyclical",
        sectors=("consumer cyclical",),
        industries=("auto", "retail", "apparel", "restaurant", "leisure", "travel", "hotel"),
        blind_spots=("volume growth and realisation/ASP", "same-store sales growth (SSSG)",
                     "capacity utilisation", "discounting and channel inventory",
                     "revenue per square foot"),
        note="Demand is cyclical and operating leverage cuts both ways. Volumes and realisation "
             "explain these businesses; the annual ratios arrive far too late.",
    ),
    SectorProfile(
        name="industrials_infra",
        sectors=("industrials",),
        industries=("engineering & construction", "aerospace", "defense", "machinery",
                    "electrical equipment", "infrastructure", "conglomerates", "railroads"),
        veto_overrides={"debt_to_ebitda": 5.5},
        blind_spots=("order book and book-to-bill ratio", "order inflow versus execution",
                     "working-capital days and retention money", "execution cycle length",
                     "client concentration and government receivables"),
        note="The order book is the leading indicator and the P&L is the lagging one. A screen "
             "without the order book is reading last year's execution, not next year's.",
    ),
    SectorProfile(
        name="energy",
        sectors=("energy",),
        industries=("oil & gas", "refining", "petroleum"),
        blind_spots=("gross refining margin (GRM)", "marketing margin", "refinery utilisation",
                     "crude price and inventory gains/losses", "subsidy sharing"),
        note="Reported profit swings on crude and inventory revaluation. A single year's ratios "
             "say almost nothing about the underlying business.",
    ),
    SectorProfile(
        name="communication",
        sectors=("communication services",),
        industries=("telecom", "media", "entertainment", "publishing", "broadcasting"),
        veto_overrides={"debt_to_ebitda": 5.5},
        blind_spots=("ARPU and subscriber additions", "churn", "spectrum liabilities and AGR dues",
                     "capex intensity", "data usage per subscriber"),
        note="Telecom is a capex treadmill with regulatory liabilities sitting off the balance "
             "sheet. ARPU is the number that decides it.",
    ),
)

DEFAULT_PROFILE = SectorProfile(
    name="general",
    blind_spots=("sector-specific operating metrics",),
    note="No sector rulebook matched, so the general one was applied. Check what this kind of "
         "business is actually judged on before trusting the rank.",
)


# Which broker-report KPIs (broker_reports.KPI_ROWS) fill which declared blind spot. Keyed by the
# blind spot's exact text; tests/test_sectors.py fails if a rulebook's wording drifts from it.
BLIND_SPOT_KPIS: dict[str, tuple[str, ...]] = {
    "net interest margin (NIM)": ("nim_pct",),
    "GNPA/NNPA and slippage": ("gnpa_pct", "nnpa_pct"),
    "provision coverage": ("pcr_pct", "pcr_incl_writeoff_pct"),
    "capital adequacy (CRAR)": ("car_pct", "tier1_pct"),
    "CASA ratio": ("casa_pct",),
    "credit cost": ("credit_cost_pct",),
    "cost-to-income": ("cost_to_income_pct",),
    "plant load factor (PLF)": ("plf_pct",),
    "pre-sales / bookings": ("presales_cr",),
    "EBITDA per tonne": ("ebitda_per_tonne",),
    "capacity utilisation": ("capacity_util_pct",),
    "realisation per tonne": ("realisation_per_tonne",),
    "deal TCV and book-to-bill": ("book_to_bill",),
    "attrition (a two-year high of ~13.8% at TCS)": ("attrition_pct",),
    "volume growth versus value growth": ("volume_growth_pct",),
    "volume growth and realisation/ASP": ("volume_growth_pct",),
    "same-store sales growth (SSSG)": ("sssg_pct",),
    "order book and book-to-bill ratio": ("order_book_cr", "book_to_bill"),
    "order inflow versus execution": ("order_inflow_cr",),
    "gross refining margin (GRM)": ("grm_usd_bbl",),
    "ARPU and subscriber additions": ("arpu",),
}


def blind_spot_coverage(profile: SectorProfile, available: set[str] | frozenset[str]
                        ) -> tuple[list[str], list[str]]:
    """(blind spots broker data now covers, blind spots still unmeasured), in rulebook order.
    A blind spot is covered only when at least one of its KPIs actually has data."""
    covered = [b for b in profile.blind_spots if set(BLIND_SPOT_KPIS.get(b, ())) & set(available)]
    return covered, [b for b in profile.blind_spots if b not in covered]


def profile_for(sector: str | None, industry: str | None) -> SectorProfile:
    """The rulebook for one company. INDUSTRY is matched first because it is more specific: a life
    insurer mislabelled under Consumer Cyclical is still a financial, and the industry label is
    what gives it away."""
    ind = (industry or "").strip().lower()
    if ind:
        for p in PROFILES:
            if any(h in ind for h in p.industries):
                return p
    sec = (sector or "").strip().lower()
    if sec:
        for p in PROFILES:
            if sec in p.sectors:
                return p
    return DEFAULT_PROFILE


def applies_to_profile(profile: SectorProfile, metric: str) -> bool:
    """Is this metric meaningful for this kind of business at all?"""
    return metric not in profile.skip_metrics

"""
Query Analyzer: turns a natural-language question into a structured
QueryAnalysis (intents, tickers, user-supplied prices).

Rule-based and transparent: every detected intent lists the exact terms that
triggered it (QueryAnalysis.matched_terms). No LLM is used here yet.
"""

import re
from typing import Dict, List, Optional, Tuple

from core.config_loader import load_config
from core.logger import get_logger
from core.task_schema import Intent, QueryAnalysis

log = get_logger(__name__)

# Regex fragments matched with word boundaries against the lower-cased query.
INTENT_PATTERNS: Dict[Intent, List[str]] = {
    Intent.INVESTMENT_DECISION: [
        r"buy", r"buying", r"sell", r"selling", r"hold", r"holdings?", r"invest(?:ing|ment)?",
        r"should i", r"add to my position", r"trim", r"entry point", r"take profits?",
        r"good (?:stock|investment)", r"portfolio",
    ],
    Intent.FINANCIAL: [
        r"financial(?:ly)?", r"financials", r"health(?:y)?", r"revenue", r"sales", r"profit\w*",
        r"margins?", r"earnings", r"income", r"cash ?flow", r"balance sheet", r"debt",
        r"roe", r"roa", r"growth", r"cagr", r"eps",
    ],
    Intent.VALUATION: [
        r"valuation", r"valued", r"overvalued", r"undervalued", r"fair value", r"intrinsic",
        r"dcf", r"p/?e", r"pe ratio", r"price target", r"worth", r"expensive", r"cheap",
        r"current price", r"multiples?", r"ev/ebit",
    ],
    Intent.MARKET: [
        r"market", r"markets", r"industry", r"sector", r"market share", r"market position",
        r"stock price", r"share price", r"trading", r"traded",
    ],
    Intent.COMPETITOR: [
        r"compare", r"compared", r"comparison", r"competitors?", r"competition", r"competitive",
        r"vs\.?", r"versus", r"rivals?", r"peers?",
    ],
    Intent.NEWS: [
        r"news", r"latest", r"recent(?:ly)?", r"happened", r"announce\w*", r"events?",
        r"8-k", r"press release", r"earnings report", r"earnings call", r"guidance",
        r"headlines?", r"update",
    ],
    Intent.RISK: [
        r"risks?", r"risky", r"threats?", r"uncertaint\w*", r"exposure", r"vulnerab\w*",
        r"concerns?", r"downside", r"due diligence", r"safe",
    ],
    Intent.RESEARCH: [
        r"business", r"segments?", r"products?", r"strategy", r"overview", r"operations",
        r"what does", r"describe", r"explain", r"tell me about", r"business model",
    ],
    Intent.VERIFICATION: [
        r"verify", r"verified", r"check", r"accurate", r"true", r"reliable", r"evidence",
        r"double[- ]check", r"validate", r"fact[- ]check",
    ],
}

# Tie-break order when several intents match: the most specific/decisive first.
PRIMARY_INTENT_PRIORITY = [
    Intent.INVESTMENT_DECISION,
    Intent.COMPETITOR,
    Intent.NEWS,
    Intent.VALUATION,
    Intent.RISK,
    Intent.FINANCIAL,
    Intent.MARKET,
    Intent.RESEARCH,
    Intent.VERIFICATION,
]

# Phrases that place a $ amount into a named slot. Checked against the text
# of the clause containing the amount.
PRICE_SLOTS: List[Tuple[str, str]] = [
    ("cost_basis", r"average|avg|purchase price|bought|cost basis|paid|entry"),
    ("previous_price", r"yesterday|previous|prior|last week|was"),
    ("target_price", r"target"),
    ("current_price", r"today|now|current|currently|trading at|trades at|is at"),
]
DOLLAR_PATTERN = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")


def _compile(fragment: str) -> re.Pattern:
    return re.compile(rf"(?<![a-z0-9]){fragment}(?![a-z0-9])")


_COMPILED = {intent: [(p, _compile(p)) for p in pats] for intent, pats in INTENT_PATTERNS.items()}


def detect_intents(query: str) -> Dict[Intent, List[str]]:
    """Return {intent: [matched terms]} for every intent with at least one match."""
    text = query.lower()
    found: Dict[Intent, List[str]] = {}
    for intent, patterns in _COMPILED.items():
        terms = [m.group(0) for _, rx in patterns for m in [rx.search(text)] if m]
        if terms:
            found[intent] = terms
    return found


def choose_primary_intent(intents: Dict[Intent, List[str]]) -> Intent:
    for intent in PRIMARY_INTENT_PRIORITY:
        if intent in intents:
            return intent
    return Intent.GENERAL


def company_aliases(config: dict) -> Dict[str, str]:
    """alias (lower-case) -> ticker, for the target company and competitors."""
    aliases: Dict[str, str] = {}
    entries = [config["company"]] + list(config.get("competitors", []))
    for entry in entries:
        ticker = entry["ticker"]
        for alias in (ticker, entry.get("name"), entry.get("short_name")):
            if alias:
                aliases[alias.lower()] = ticker
        first_word = (entry.get("name") or "").split(" ")[0]
        if len(first_word) > 3:
            aliases[first_word.lower()] = ticker  # "Broadcom", "Intel", "NVIDIA"
    return aliases


def extract_tickers(query: str, config: dict) -> List[str]:
    """Known tickers mentioned in the query, in order of first appearance."""
    text = query.lower()
    hits = []
    for alias, ticker in company_aliases(config).items():
        m = _compile(re.escape(alias)).search(text)
        if m:
            hits.append((m.start(), ticker))
    ordered: List[str] = []
    for _, ticker in sorted(hits):
        if ticker not in ordered:
            ordered.append(ticker)
    return ordered


def extract_user_prices(query: str) -> Dict[str, float]:
    """
    Pull $ amounts out of the query and label them by the words in the same
    clause, e.g. "$500 today" -> current_price, "average purchase price is
    $120" -> cost_basis. Unlabelled amounts are ignored rather than guessed.
    """
    prices: Dict[str, float] = {}
    for clause in re.split(r"(?<!\d),|,(?!\d)|;|\band\b|\bbut\b|[.?!](?:\s|$)", query):
        for m in DOLLAR_PATTERN.finditer(clause):
            context = clause.lower()
            for slot, pattern in PRICE_SLOTS:
                if slot not in prices and re.search(pattern, context):
                    prices[slot] = float(m.group(1).replace(",", ""))
                    break
    return prices


def analyze_query(query: str, config: Optional[dict] = None) -> QueryAnalysis:
    """Main entry: build a QueryAnalysis for the question."""
    config = config or load_config()
    query = (query or "").strip()
    intents = detect_intents(query)
    primary = choose_primary_intent(intents)

    company_ticker = config["company"]["ticker"]
    competitor_set = {c["ticker"] for c in config.get("competitors", [])}
    mentioned = extract_tickers(query, config)

    if company_ticker in mentioned or not mentioned:
        target = company_ticker
    else:
        target = mentioned[0]
    competitors = [t for t in mentioned if t != target and (t in competitor_set or t == company_ticker)]

    # Naming a second known company is itself a comparison signal.
    if competitors and Intent.COMPETITOR not in intents:
        intents[Intent.COMPETITOR] = [f"mentions {', '.join(competitors)}"]
        primary = choose_primary_intent(intents)

    ordered = [i for i in PRIMARY_INTENT_PRIORITY if i in intents]
    analysis = QueryAnalysis(
        query=query,
        primary_intent=primary,
        intents=ordered,
        target_ticker=target,
        mentioned_tickers=mentioned,
        competitor_tickers=competitors,
        user_prices=extract_user_prices(query),
        matched_terms={i.value: intents[i] for i in ordered},
    )
    log.info(
        "[ANALYZER] primary_intent=%s intents=%s target=%s competitors=%s user_prices=%s",
        primary.value, [i.value for i in ordered], target, competitors, analysis.user_prices,
    )
    return analysis

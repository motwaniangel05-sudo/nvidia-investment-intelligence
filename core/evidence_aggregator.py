"""
Evidence Aggregator: standard AgentResults -> one compact, structured context
for the local Qwen synthesis step (small model, ~4096-token window).

Rules:
- Only successful/partial agents contribute evidence; failed agents are listed
  under agent_status and missing_evidence, never silently dropped.
- Numerical evidence (calculations, valuations, market data) and conflicts are
  never trimmed. Only text evidence (news sentences, filing passages,
  uncertainties) is ranked and capped to fit the token budget, and every cut
  is counted in meta.omitted.
- Nothing is invented: values that were not computed are listed as such.
- Conflicts are detected mechanically and both sides are kept, unresolved.
- Every item keeps two separate signals: `quality` (source tier, below) and
  `confidence` (the agent's own score). They are never merged.

Source quality tiers:
    HIGH   SEC filing / official financial statement data (XBRL, 10-K, 10-Q)
    MEDIUM market data, company event filings (8-K)
    LOW    model output (DCF), project configuration, interpretations
"""

import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from core.agent_result import SCHEMA_VERSION, AgentResult
from core.logger import get_logger

log = get_logger(__name__)

DEFAULT_TOKEN_BUDGET = 2800  # leaves ~1300 tokens for instructions + answer in a 4096 window
CHARS_PER_TOKEN = 4          # rough estimate for English/JSON text
DEFAULT_LIMITS = {"news": 12, "facts": 5, "uncertainties": 10}
MIN_LIMITS = {"news": 5, "facts": 2, "uncertainties": 3}
SNIPPET_CHARS = 240
MIN_SNIPPET_CHARS = 120
TEXT_CHARS = 300
# Trimming order when over budget: fewer news items, shorter snippets, then the floors.
SHRINK_STEPS = (("news", 8), ("snippet", 160), ("facts", MIN_LIMITS["facts"]),
                ("uncertainties", MIN_LIMITS["uncertainties"]), ("news", MIN_LIMITS["news"]),
                ("snippet", MIN_SNIPPET_CHARS))

QUALITY_SCALE = {
    "HIGH": "SEC filing / official financial statement data",
    "MEDIUM": "market data or company event filing (8-K)",
    "LOW": "model output, project configuration or interpretation",
}
QUALITY_WEIGHT = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}
SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
LOW_RELIABILITY_LABELS = {"INTERPRETATION", "CLAIM"}
VERIFICATION_AGENT = "RedTeamAgent"
CONTENT_FREE_AGENTS = {"SynthesisAgent"}  # restates other agents; status only
STRUCTURED_AGENTS = {"FinancialAgent", "MarketAgent", "ValuationAgent", "RiskAgent"}
RECENCY_WORDS = {"latest", "recent", "recently", "now", "today", "current", "currently", "new", "last"}
STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "with", "that", "this", "what", "how", "why",
    "has", "have", "had", "its", "it's", "from", "into", "about", "after", "before", "should",
    "would", "could", "does", "did", "can", "will", "our", "your", "you", "they", "their", "any",
    "all", "but", "not", "who", "which", "when", "there", "been", "also", "more", "than",
}
NUMERIC_TEXT = re.compile(r"\$\s?\d|\d+(?:\.\d+)?\s?%|\d+(?:\.\d+)?\s(?:million|billion|thousand)", re.I)
DATE_IN_TEXT = re.compile(r"\d{4}-\d{2}-\d{2}")

# Numbers equal after rounding to one decimal are not a conflict.
ABS_TOLERANCE = 0.051
REL_TOLERANCE = 0.005


# --- helpers ----------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def source_quality(source: Optional[str]) -> str:
    """Map a source/form string to a quality tier. Unknown sources are LOW."""
    s = (source or "").replace("_", " ").lower()
    if "dcf" in s or "config" in s:
        return "LOW"
    if "market data" in s or "8-k" in s:
        return "MEDIUM"
    if "xbrl" in s or "10-k" in s or "10-q" in s:
        return "HIGH"
    return "LOW"


def _source_id(source: Optional[str]) -> Optional[str]:
    """'SEC XBRL company facts (NVDA_roe_2026)' or '8-K:NVDA_8-K_...' -> the id."""
    if not source:
        return None
    m = re.search(r"\(([^()]+)\)\s*$", source)
    if m:
        return m.group(1)
    return source.split(":", 1)[1] if ":" in source else source


def _form(source: Optional[str]) -> Optional[str]:
    if source and ":" in source and "(" not in source:
        return source.split(":", 1)[0]
    return re.sub(r"\s*\([^()]*\)\s*$", "", source).strip() if source else None


def _snippet(text: Optional[str], limit: int) -> str:
    text = re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _norm(text: Optional[str]) -> str:
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


def _stem(word: str) -> str:
    for suffix in ("ings", "ing", "ies", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


QUERY_EXPANSIONS = {
    "earnings": ("results", "quarter", "quarterly", "revenue", "guidance"),
    "acquisition": ("acquire", "merger", "transaction"),
    "lawsuit": ("litigation", "legal"),
    "dividend": ("repurchase", "buyback"),
}


def _terms(text: str) -> set:
    return {_stem(w) for w in re.findall(r"[a-z0-9][a-z0-9\-/]+", (text or "").lower())
            if w not in STOPWORDS and len(w) > 2}


def _compact(d: Dict[str, Any]) -> Dict[str, Any]:
    """Drop None / empty values so the context stays small (missing != zero)."""
    return {k: v for k, v in d.items() if v not in (None, "", [], {})}


def _ticker_and_metric(metric: str, source_id: Optional[str]) -> Tuple[Optional[str], str]:
    """'AMD Revenue CAGR' -> ('AMD', 'Revenue CAGR') only when AMD is in the source id."""
    m = re.match(r"^([A-Z]{1,5}) (.+)$", metric)
    if m and m.group(1) in (source_id or "").split("_"):
        return m.group(1), m.group(2)
    m = re.match(r"^([A-Z]{1,5})_", source_id or "")
    return (m.group(1) if m else None), metric


def _same_value(a: float, b: float) -> bool:
    return abs(a - b) <= max(ABS_TOLERANCE, REL_TOLERANCE * max(abs(a), abs(b)))


EXPANSIONS = {_stem(k): {_stem(w) for w in v} for k, v in QUERY_EXPANSIONS.items()}


class Relevance:
    """IDF-weighted term overlap between the question and candidate texts.
    Terms present in most candidates (e.g. the company name) carry ~no weight."""

    def __init__(self, query: str, texts: List[str]):
        self.weights = {t: 1.0 for t in _terms(query)}
        for t in list(self.weights):
            for extra in EXPANSIONS.get(t, ()):
                self.weights.setdefault(extra, 0.5)  # related words count half
        self.n = len(texts)
        self.df: Dict[str, int] = {}
        for text in texts:
            for t in _terms(text) & self.weights.keys():
                self.df[t] = self.df.get(t, 0) + 1

    def _weight(self, term: str) -> float:
        return self.weights[term] * math.log((self.n + 1) / (self.df.get(term, 0) + 1))

    def score(self, text: str) -> float:
        total = sum(self._weight(t) for t in self.weights)
        if not total:
            return 0.0
        return round(sum(self._weight(t) for t in self.weights.keys() & _terms(text)) / total, 3)


# --- output -------------------------------------------------------------------------

@dataclass
class EvidenceContext:
    query: str
    intent: Optional[str] = None
    company_ticker: Optional[str] = None
    user_inputs: Dict[str, Any] = field(default_factory=dict)
    facts: List[dict] = field(default_factory=list)
    financial_metrics: List[dict] = field(default_factory=list)
    market_findings: List[dict] = field(default_factory=list)
    competitor_findings: List[dict] = field(default_factory=list)
    valuation: List[dict] = field(default_factory=list)
    risks: List[dict] = field(default_factory=list)
    news_events: List[dict] = field(default_factory=list)
    interpretations: List[dict] = field(default_factory=list)
    assumptions: Dict[str, Any] = field(default_factory=dict)
    uncertainties: List[str] = field(default_factory=list)
    conflicts: List[dict] = field(default_factory=list)
    verification_flags: List[dict] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    sources: List[dict] = field(default_factory=list)
    agent_status: List[dict] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: Optional[int] = None) -> str:
        """Compact JSON by default -- this is what goes into the Qwen prompt."""
        separators = None if indent else (",", ":")
        return json.dumps(self.to_dict(), indent=indent, separators=separators, allow_nan=False)

    @property
    def token_estimate(self) -> int:
        return estimate_tokens(self.to_json())


# --- aggregator ---------------------------------------------------------------------

class EvidenceAggregator:
    def __init__(self, query: str, results: Iterable[AgentResult], intent: Optional[str] = None,
                 company_ticker: Optional[str] = None, user_prices: Optional[Dict[str, float]] = None,
                 token_budget: int = DEFAULT_TOKEN_BUDGET, limits: Optional[Dict[str, int]] = None):
        self.query = query
        self.results = list(results)
        self.intent = intent
        self.company_ticker = company_ticker
        self.user_prices = dict(user_prices or {})
        self.token_budget = token_budget
        self.limits = {**DEFAULT_LIMITS, **(limits or {})}
        self.sources: Dict[str, dict] = {}
        self.confidence_by_source: Dict[str, float] = {}
        self.numeric_entries: List[dict] = []

    # sources ------------------------------------------------------------------
    def _register(self, source_id: Optional[str], quality: str, form: Optional[str]) -> Optional[str]:
        if source_id and source_id not in self.sources:
            self.sources[source_id] = {"form": form or "unknown", "quality": quality}
        return source_id

    # sections -----------------------------------------------------------------
    def _metric_groups(self, agent: str, calcs) -> List[dict]:
        """Group a metric's yearly values into one item; every value is kept."""
        groups: Dict[tuple, dict] = {}
        for c in calcs:
            sid = _source_id(c.source)
            ticker, metric = _ticker_and_metric(c.metric, sid)
            quality = source_quality(c.source)
            self.numeric_entries.append({"agent": agent, "ticker": ticker, "metric": metric, "unit": c.unit,
                                         "period": c.period, "value": c.value, "source_id": sid})
            g = groups.setdefault((ticker, metric, c.unit), {
                "metric": metric, "ticker": ticker, "unit": c.unit, "formula": c.formula,
                "quality": quality, "values": [], "ids": [], "confidences": set(), "form": _form(c.source)})
            if [c.period, c.value] not in g["values"]:
                g["values"].append([c.period, c.value])
                g["ids"].append(sid)
            if sid in self.confidence_by_source:
                g["confidences"].add(self.confidence_by_source[sid])

        items = []
        for g in groups.values():
            pairs = sorted(zip(g["values"], g["ids"]), key=lambda p: str(p[0][0]))
            values, ids = [p[0] for p in pairs], [p[1] for p in pairs]
            dates = [DATE_IN_TEXT.findall(str(v[0])) for v in values]
            first = dates[0][-1] if dates and dates[0] else ""
            prefix = ids[0][: -len(first)] if first and ids[0] and ids[0].endswith(first) else None
            periods = ["n/a" if v[0] is None else str(v[0]) for v in values]
            period_prefix = os.path.commonprefix(periods) if len(periods) > 1 else ""
            period_prefix = period_prefix[: period_prefix.rfind(" ") + 1]
            by_period: Dict[str, Any] = {}
            for period, (_, value) in zip(periods, values):
                key = period[len(period_prefix):]
                by_period[key] = [*(by_period[key] if isinstance(by_period[key], list) else [by_period[key]]), value] \
                    if key in by_period else value  # two values for one period: keep both
            item = _compact({"metric": g["metric"], "ticker": g["ticker"], "unit": g["unit"],
                             "formula": g["formula"], "period_prefix": period_prefix or None,
                             "values": by_period, "quality": g["quality"],
                             "confidence": min(g["confidences"]) if g["confidences"] else None,
                             "agent": agent})
            if len(ids) > 1 and prefix and all(d and i == prefix + d[-1] for i, d in zip(ids, dates)):
                item["source_id_pattern"] = prefix + "{period_end_date}"
                self._register(item["source_id_pattern"], g["quality"], g["form"])
            else:
                item["source_ids"] = [self._register(i, g["quality"], g["form"]) for i in ids if i]
            items.append(item)
        return items

    def _competitor_items(self, result: AgentResult) -> List[dict]:
        items, seen = [], {}
        for c in result.calculations:
            sid = _source_id(c.source)
            ticker, metric = _ticker_and_metric(c.metric, sid)
            self.numeric_entries.append({"agent": result.agent_name, "ticker": ticker, "metric": metric,
                                         "unit": c.unit, "period": c.period, "value": c.value, "source_id": sid})
            key = (c.metric, c.period, c.value, c.unit)
            quality = source_quality(c.source)
            self._register(sid, quality, _form(c.source))
            if key in seen:
                seen[key]["source_ids"].append(sid)
                continue
            seen[key] = _compact({"metric": c.metric, "value": c.value, "unit": c.unit, "period": c.period,
                                  "quality": quality, "confidence": self.confidence_by_source.get(sid),
                                  "agent": result.agent_name})
            seen[key]["source_ids"] = [sid]
            items.append(seen[key])
        for claim in result.claims:
            if isinstance(claim.value, str) and _source_id(claim.source or "").endswith("_inclusion_rationale"):
                sid = self._register(_source_id(claim.source), source_quality(claim.source), _form(claim.source))
                items.append(_compact({"competitor": claim.value, "claim": _snippet(claim.claim, 110),
                                       "quality": source_quality(claim.source), "confidence": claim.confidence,
                                       "source_id": sid, "agent": result.agent_name}))
        return items

    def _valuation_items(self, result: AgentResult) -> Tuple[List[dict], List[dict], set, dict]:
        """Valuation records, observed market data, ids of calculations folded in, shared DCF assumptions."""
        valuations, market, attached = [], [], set()
        for v in result.valuations:
            components = {}
            for c in result.calculations:
                suffix = f" ({v.scenario})"
                if v.scenario and c.metric.endswith(suffix) and c.value != v.enterprise_value:
                    components[c.metric[: -len(suffix)]] = c.value
                    attached.add(id(c))
            sid = _source_id(v.source)
            quality = source_quality(v.source)
            self._register(sid, quality, _form(v.source))
            not_computed = [k for k in ("enterprise_value", "equity_value", "implied_share_price", "sensitivity")
                            if getattr(v, k) in (None, {}) and not (k == "enterprise_value" and v.multiple is not None)]
            valuations.append(_compact({
                "method": v.method, "scenario": v.scenario, "enterprise_value": v.enterprise_value,
                "equity_value": v.equity_value, "implied_share_price": v.implied_share_price,
                "multiple": v.multiple, "components": components, "assumptions": dict(v.assumptions),
                "sensitivity": v.sensitivity,
                "not_computed": not_computed, "quality": quality,
                "confidence": self.confidence_by_source.get(sid), "source_id": sid, "agent": result.agent_name}))
            a = v.assumptions or {}
            if v.method == "P/E" and a.get("price") is not None:
                market.append(_compact({"metric": "Share price (close)", "value": a["price"], "unit": "USD",
                                        "date": a.get("price_date"), "quality": "MEDIUM",
                                        "source_id": sid, "agent": result.agent_name}))
            if v.method == "EV/EBIT" and a.get("market_cap") is not None:
                market.append(_compact({"metric": "Market capitalization", "value": a["market_cap"], "unit": "USD",
                                        "quality": "MEDIUM", "source_id": sid, "agent": result.agent_name}))
        dcf = [v for v in valuations if v["method"] == "DCF"]
        shared = {k: val for k, val in dcf[0].get("assumptions", {}).items()
                  if all(d.get("assumptions", {}).get(k, object()) == val for d in dcf)} if len(dcf) > 1 else {}
        for d in dcf:
            for k in shared:
                d["assumptions"].pop(k)
            if not d.get("assumptions"):
                d.pop("assumptions", None)
        if len(dcf) > 1 and all(d.get("not_computed") == dcf[0].get("not_computed") for d in dcf) \
                and dcf[0].get("not_computed"):
            shared["not_computed"] = dcf[0]["not_computed"]
            for d in dcf:
                d.pop("not_computed")
        return valuations, market, attached, shared

    def _risk_items(self, result: AgentResult, relevance: Relevance, snippet: int) -> List[dict]:
        risk_claims = [c for c in result.claims if c.value is not None]
        items = []
        for i, r in enumerate(result.risks):
            claim = risk_claims[i] if i < len(risk_claims) else None
            sid = self._register(_source_id(claim.source), source_quality(claim.source),
                                 _form(claim.source)) if claim else None
            items.append(_compact({
                "risk": r.risk, "severity": r.severity, "probability": r.probability, "impact": r.impact,
                "not_assessed": [k for k in ("probability", "impact") if getattr(r, k) is None],
                "evidence": [_snippet(e, snippet) for e in r.evidence],
                "contradicting_evidence": [_snippet(e, snippet) for e in r.contradicting_evidence],
                "quality": source_quality(claim.source) if claim else "LOW",
                "confidence": claim.confidence if claim else None, "date": claim.date if claim else None,
                "source_id": sid, "agent": result.agent_name,
                "_rank": (SEVERITY_ORDER.get((r.severity or "").upper(), 3), -relevance.score(r.risk + " " + " ".join(r.evidence)))}))
        return sorted(items, key=lambda i: i["_rank"])

    def _news_candidates(self, result: AgentResult) -> List[dict]:
        """Deduplicate identical sentences, keeping dates and occurrence counts."""
        by_text: Dict[str, dict] = {}
        for c in result.claims:
            key = _norm(c.evidence)
            if not key:
                continue
            sid = _source_id(c.source)
            cand = by_text.get(key)
            if cand is None:
                by_text[key] = {"text": c.evidence, "label": c.value, "dates": [c.date] if c.date else [],
                                "source_id": sid, "form": _form(c.source), "confidence": c.confidence,
                                "agent": result.agent_name}
                continue
            if c.date:
                cand["dates"].append(c.date)
                if c.date >= max(cand["dates"]):
                    cand["source_id"] = sid  # cite the most recent occurrence
            if (c.confidence or 0) > (cand["confidence"] or 0):
                cand["confidence"] = c.confidence
        return list(by_text.values())

    def _rank_news(self, candidates: List[dict]) -> List[dict]:
        relevance = Relevance(self.query, [c["text"] for c in candidates])
        all_dates = sorted({max(c["dates"]) for c in candidates if c["dates"]})
        date_rank = {d: (i + 1) / len(all_dates) for i, d in enumerate(all_dates)}
        recency_weight = 0.2 if _terms(self.query) & {_stem(w) for w in RECENCY_WORDS} else 0.05
        for c in candidates:
            c["relevance"] = relevance.score(c["text"])
            latest = max(c["dates"]) if c["dates"] else None
            recency = recency_weight * date_rank.get(latest, 0) * (1.0 if c["relevance"] > 0 else 0.25)
            c["_score"] = (0.6 * c["relevance"] + recency
                           + 0.1 * (c["confidence"] or 0) + (0.1 if NUMERIC_TEXT.search(c["text"]) else 0)
                           - (0.1 if c["label"] in LOW_RELIABILITY_LABELS else 0))
        return sorted(candidates, key=lambda c: (-c["_score"], -(c["confidence"] or 0), c["text"]))

    def _news_item(self, c: dict, snippet: int) -> dict:
        quality = "LOW" if c["label"] in LOW_RELIABILITY_LABELS else source_quality(c["form"])
        self._register(c["source_id"], source_quality(c["form"]), c["form"])
        dates = sorted(set(c["dates"]))
        return _compact({"text": _snippet(c["text"], snippet), "label": c["label"],
                         "date": dates[-1] if dates else None,
                         "first_date": dates[0] if len(dates) > 1 else None,
                         "occurrences": len(c["dates"]) if len(c["dates"]) > 1 else None,
                         "quality": quality, "confidence": c["confidence"], "relevance": c["relevance"],
                         "source_id": c["source_id"], "agent": c["agent"]})

    def _fact_candidates(self, result: AgentResult, structured: bool) -> List[dict]:
        out = []
        for c in result.claims:
            if structured and c.value is not None:
                continue  # mirrored by calculations / valuations / risks / competitor items
            generic = c.claim.startswith("Relevant evidence found regarding")
            out.append({"claim": None if generic else c.claim, "evidence": c.evidence, "date": c.date,
                        "source": c.source, "confidence": c.confidence, "agents": [result.agent_name]})
        return out

    # conflicts ---------------------------------------------------------------
    def _numeric_conflicts(self) -> List[dict]:
        groups: Dict[tuple, List[dict]] = {}
        for e in self.numeric_entries:
            if e["value"] is None:
                continue
            period = "full available history" if "cagr" in e["metric"].lower() else e["period"]
            groups.setdefault((e["ticker"], e["metric"].lower(), e["unit"], period), []).append(e)
        conflicts = []
        for (ticker, _, unit, period), entries in groups.items():
            distinct: List[dict] = []
            for e in entries:
                if not any(_same_value(e["value"], d["value"]) for d in distinct):
                    distinct.append(e)
            if len(distinct) > 1:
                conflicts.append({
                    "type": "numeric_mismatch",
                    "subject": f"{ticker + ' ' if ticker else ''}{entries[0]['metric']} ({period})",
                    "positions": [_compact({"agent": d["agent"], "value": d["value"], "unit": unit,
                                            "period": d["period"], "source_id": d["source_id"]}) for d in distinct],
                    "note": "Values disagree; both kept, not resolved.",
                })
        return conflicts

    def _risk_conflicts(self, risks: List[dict]) -> List[dict]:
        by_name: Dict[str, List[dict]] = {}
        for r in risks:
            by_name.setdefault(r["risk"].lower(), []).append(r)
        conflicts = []
        for items in by_name.values():
            if len({(i.get("severity") or "").upper() for i in items}) > 1:
                conflicts.append({"type": "severity_mismatch", "subject": f"{items[0]['risk']} risk severity",
                                  "positions": [_compact({"agent": i["agent"], "value": i.get("severity"),
                                                          "source_id": i.get("source_id")}) for i in items],
                                  "note": "Agents disagree on severity; both kept, not resolved."})
        return conflicts

    def _user_price_checks(self, market: List[dict], conflicts: List[dict], missing: List[str]) -> None:
        user_price = self.user_prices.get("current_price")
        if user_price is None:
            return
        stored = next((m for m in market if m["metric"] == "Share price (close)"), None)
        if stored is None:
            missing.append("No stored share price available to compare with the user-supplied current price.")
        elif not _same_value(user_price, stored["value"]):
            conflicts.append({
                "type": "user_price_mismatch",
                "subject": f"{self.company_ticker or ''} current share price".strip(),
                "positions": [{"source": "user", "value": user_price, "unit": "USD"},
                              _compact({"agent": stored["agent"], "value": stored["value"], "unit": "USD",
                                        "date": stored.get("date"), "source_id": stored.get("source_id")})],
                "note": "User-supplied price differs from the stored price; the stored price is the data, the user price is shown for comparison only.",
            })

    # main ----------------------------------------------------------------------
    def build(self) -> EvidenceContext:
        ok = [r for r in self.results if r.status != "failed"]
        agent_status, missing = [], []
        for r in self.results:
            has_evidence = r.claims or r.calculations or r.valuations or r.risks
            agent_status.append(_compact({"agent": r.agent_name, "status": r.status,
                                          "summary": None if has_evidence and r.status == "success"
                                          else _snippet(r.summary, 100),
                                          "errors": r.errors}))
            has_evidence = r.claims or r.calculations or r.valuations or r.risks
            if r.status == "failed":
                missing.append(f"{r.agent_name} failed: {_snippet('; '.join(r.errors) or 'unknown error', 200)}")
            elif not has_evidence and r.agent_name not in CONTENT_FREE_AGENTS | {VERIFICATION_AGENT}:
                missing.append(f"{r.agent_name} returned no evidence ({r.status}).")

        for r in ok:
            for c in r.claims:
                sid = _source_id(c.source)
                if sid and c.confidence is not None:
                    self.confidence_by_source[sid] = c.confidence

        financial, competitor, valuation, market, risks_src = [], [], [], [], []
        news_cands, fact_cands, flags, assumptions, uncertainties = [], [], [], {}, []
        for r in ok:
            if r.agent_name in CONTENT_FREE_AGENTS:
                continue
            if r.agent_name == VERIFICATION_AGENT:
                flags += _group_flags(r.findings)
                continue
            vals, mk, attached, shared_dcf = self._valuation_items(r)
            valuation += vals
            market += mk
            if r.agent_name == "MarketAgent":
                competitor += self._competitor_items(r)
            elif r.calculations:
                financial += self._metric_groups(r.agent_name, _not_in_valuations(r, attached))
            if r.risks:
                risks_src.append(r)
            if r.agent_name == "NewsAgent":
                news_cands += self._news_candidates(r)
            else:
                fact_cands += self._fact_candidates(r, r.agent_name in STRUCTURED_AGENTS)
            agent_assumptions = {**r.assumptions, **({"dcf_shared": shared_dcf} if shared_dcf else {})}
            if agent_assumptions:
                assumptions[r.agent_name] = agent_assumptions
            for u in r.uncertainties:
                text = f"[{r.agent_name}] {_snippet(u, TEXT_CHARS)}"
                if text not in uncertainties:
                    uncertainties.append(text)

        facts = self._dedupe_facts(fact_cands)
        ranked_news = self._rank_news(news_cands) if news_cands else []
        fact_rel = Relevance(self.query, [f["evidence"] or "" for f in facts])
        for f in facts:
            f["relevance"] = fact_rel.score((f["claim"] or "") + " " + (f["evidence"] or ""))
            f["quality"] = source_quality(f["source"])
        facts.sort(key=lambda f: -(0.6 * f["relevance"] + 0.25 * QUALITY_WEIGHT[f["quality"]] + 0.15 * (f["confidence"] or 0)))

        conflicts = self._numeric_conflicts()
        self._user_price_checks(market, conflicts, missing)

        state = {**self.limits, "snippet": SNIPPET_CHARS}
        steps = list(SHRINK_STEPS)
        while True:
            ctx = self._render(facts, financial, market, competitor, valuation, risks_src, ranked_news,
                               assumptions, uncertainties, conflicts, flags, missing, agent_status,
                               state, state["snippet"])
            if ctx.token_estimate <= self.token_budget or not _shrink(state, steps):
                break

        ctx.meta["token_estimate"] = ctx.token_estimate
        ctx.meta["budget_exceeded"] = ctx.meta["token_estimate"] > self.token_budget
        log.info("[AGGREGATOR] %d agents, ~%d tokens (budget %d), conflicts=%d, omitted=%s",
                 len(self.results), ctx.meta["token_estimate"], self.token_budget,
                 len(ctx.conflicts), ctx.meta["omitted"])
        return ctx

    def _dedupe_facts(self, candidates: List[dict]) -> List[dict]:
        by_key: Dict[str, dict] = {}
        for c in candidates:
            key = _norm(c["evidence"]) or _norm(c["claim"])
            if key in by_key:
                existing = by_key[key]
                existing["agents"] += [a for a in c["agents"] if a not in existing["agents"]]
                if (c["confidence"] or 0) > (existing["confidence"] or 0):
                    existing["confidence"] = c["confidence"]
            else:
                by_key[key] = dict(c)
        return list(by_key.values())

    def _render(self, facts, financial, market, competitor, valuation, risks_src, ranked_news,
                assumptions, uncertainties, conflicts, flags, missing, agent_status, limits, snippet) -> EvidenceContext:
        news = [self._news_item(c, snippet) for c in ranked_news[: limits["news"]]]
        risk_items = []
        for r in risks_src:
            relevance = Relevance(self.query, [x.risk + " " + " ".join(x.evidence) for x in r.risks])
            risk_items += self._risk_items(r, relevance, snippet)
        for item in risk_items:
            item.pop("_rank", None)
        fact_items = []
        for f in facts[: limits["facts"]]:
            sid = self._register(_source_id(f["source"]), f["quality"], _form(f["source"]))
            fact_items.append(_compact({"claim": f["claim"], "evidence": _snippet(f["evidence"], snippet),
                                        "date": f["date"], "quality": f["quality"], "confidence": f["confidence"],
                                        "relevance": f["relevance"], "source_id": sid,
                                        "agents": f["agents"] if len(f["agents"]) > 1 else None,
                                        "agent": f["agents"][0] if len(f["agents"]) == 1 else None}))

        ctx = EvidenceContext(
            query=self.query, intent=self.intent, company_ticker=self.company_ticker,
            user_inputs=({**self.user_prices, "note": "User-supplied; not verified; never used as data."}
                         if self.user_prices else {}),
            facts=fact_items, financial_metrics=financial, market_findings=market,
            competitor_findings=competitor, valuation=valuation, risks=risk_items,
            news_events=[n for n in news if n.get("label") not in LOW_RELIABILITY_LABELS],
            interpretations=[n for n in news if n.get("label") in LOW_RELIABILITY_LABELS],
            assumptions=assumptions,
            uncertainties=[_snippet(u, snippet + 60) for u in uncertainties[: limits["uncertainties"]]],
            conflicts=conflicts + self._risk_conflicts(risk_items), verification_flags=flags,
            missing_evidence=missing, agent_status=agent_status,
        )
        groups: Dict[tuple, dict] = {}
        for sid in _referenced_ids(ctx):
            src = self.sources[sid]
            groups.setdefault((src["form"], src["quality"]), {**src, "count": 0})["count"] += 1
        ctx.sources = list(groups.values())
        section_agents = {}
        for name in ("facts", "financial_metrics", "market_findings", "competitor_findings", "valuation",
                     "risks", "news_events", "interpretations"):
            items = [dict(i) for i in getattr(ctx, name)]  # shared across renders; never mutate
            setattr(ctx, name, items)
            agents = {i.get("agent") for i in items}
            if items and len(agents) == 1 and None not in agents:
                section_agents[name] = agents.pop()  # stated once instead of on every item
                for i in items:
                    i.pop("agent")
        ctx.meta = {
            "schema_version": SCHEMA_VERSION,
            "token_budget": self.token_budget,
            "quality_scale": QUALITY_SCALE,
            "section_agents": section_agents,
            "omitted": {k: v for k, v in {"news_items": max(0, len(ranked_news) - limits["news"]),
                                 "facts": max(0, len(facts) - limits["facts"]),
                                 "uncertainties": max(0, len(uncertainties) - limits["uncertainties"])}.items() if v},
        }
        return ctx


def _shrink(state: Dict[str, int], steps: list) -> bool:
    """Apply the next trimming step (one item, or one snippet-length cut). False when nothing is left."""
    while steps:
        key, floor = steps[0]
        if state[key] > floor:
            state[key] = floor if key == "snippet" else state[key] - 1
            return True
        steps.pop(0)
    return False


def _group_flags(findings: List[dict]) -> List[dict]:
    """Red-Team issues grouped per agent and result; problems that differ only by numbers shown once."""
    groups: Dict[tuple, dict] = {}
    for f in findings:
        g = groups.setdefault((f.get("source_agent"), f.get("verification_result")), {
            "agent": f.get("source_agent"), "result": f.get("verification_result"),
            "count": 0, "claims": [], "problems": [], "_templates": set()})
        g["count"] += 1
        g["claims"].append(_snippet(f.get("claim"), 100))
        template = re.sub(r"\d[\d,.]*", "#", f.get("problem") or "")
        if template not in g["_templates"]:
            g["_templates"].add(template)
            g["problems"].append(_snippet(f.get("problem"), 140))
    for g in groups.values():
        g.pop("_templates")
    return [_compact(g) for g in groups.values()]


def _not_in_valuations(result: AgentResult, attached: set) -> list:
    """Calculations not already carried by the agent's valuation records (avoids duplicates)."""
    carried = {v.enterprise_value for v in result.valuations} | {v.multiple for v in result.valuations}
    return [c for c in result.calculations
            if id(c) not in attached and (c.value is None or c.value not in carried)]


def _referenced_ids(ctx: EvidenceContext) -> List[str]:
    ids: List[str] = []
    sections = (ctx.facts, ctx.financial_metrics, ctx.market_findings, ctx.competitor_findings,
                ctx.valuation, ctx.risks, ctx.news_events, ctx.interpretations)
    for section in sections:
        for item in section:
            for sid in [item.get("source_id"), item.get("source_id_pattern"), *item.get("source_ids", [])]:
                if sid and sid not in ids:
                    ids.append(sid)
    return ids


def aggregate_evidence(query: str, results: Iterable[AgentResult], **kwargs) -> EvidenceContext:
    """Main entry: standard AgentResults -> compact EvidenceContext for Qwen."""
    return EvidenceAggregator(query, results, **kwargs).build()

"""
Adapters that convert each existing agent's output into the standard
core.agent_result.AgentResult, without changing any agent.

    BaseAgent subclasses -> core.schemas.AgentResult -> normalize_result()
    Red-Team             -> List[VerificationResult] -> normalize_verification()
    Synthesis            -> SynthesisReport          -> normalize_synthesis()

Structured values (percentages, multiples, enterprise values, severities) are
parsed from the agents' own claim/evidence text and chunk ids, so they always
match what the agent reported. If parsing ever fails, the result falls back to
the generic form (claims + evidence) and records why under `uncertainties`;
it never raises.
"""

import re
from dataclasses import asdict
from statistics import mean
from typing import Any, Callable, Dict, List, Optional

from core.agent_result import (
    AgentResult, Calculation, Claim, Evidence, Risk, SourceReference, Valuation,
)
from core.logger import get_logger
from core.schemas import AgentResult as LegacyAgentResult
from core.schemas import Finding

log = get_logger(__name__)

FAILURE_PREFIX = "Agent execution failed:"
RED_TEAM_NAME = "RedTeamAgent"
SYNTHESIS_NAME = "SynthesisAgent"
XBRL_SOURCE = "SEC XBRL company facts"

_PCT = re.compile(r"([+-]?\d+(?:\.\d+)?)%")
_MONEY = r"\$([\d,]+(?:\.\d+)?)"


def _num(text: str) -> float:
    return float(text.replace(",", ""))


def _search(pattern: str, text: str) -> Optional[re.Match]:
    return re.search(pattern, text)


def _date(value: Optional[str]) -> Optional[str]:
    return None if value in (None, "", "n/a") else value


def _source(f: Finding) -> str:
    return f"{f.source_form}:{f.source_chunk_id}"


def _claim(f: Finding, value: Any = None, unit: Optional[str] = None) -> Claim:
    return Claim(claim=f.claim, value=value, unit=unit, source=_source(f),
                 date=_date(f.source_filing_date), evidence=f.evidence_text, confidence=f.confidence)


def _base(legacy: LegacyAgentResult, summary: str = "") -> AgentResult:
    """Fields every agent shares: findings, generic claims, evidence, sources, errors."""
    failed = legacy.status == "failed"
    errors = [w for w in legacy.warnings if failed or w.startswith(FAILURE_PREFIX)]
    uncertainties = [w for w in legacy.warnings if w not in errors]

    refs, seen = [], set()
    for f in legacy.findings:
        if f.source_chunk_id not in seen:
            seen.add(f.source_chunk_id)
            refs.append(SourceReference(f.source_chunk_id, f.source_form, _date(f.source_filing_date)))

    confidences = [f.confidence for f in legacy.findings]
    status = legacy.status if legacy.status in ("success", "partial", "failed") else "partial"
    return AgentResult(
        agent_name=legacy.agent_name,
        task=legacy.task,
        status=status,
        summary=summary or (
            f"{legacy.agent_name} failed." if failed
            else f"{legacy.agent_name} produced {len(legacy.findings)} finding(s)."
        ),
        findings=[asdict(f) for f in legacy.findings],
        metrics=dict(legacy.metrics),
        claims=[_claim(f) for f in legacy.findings],
        evidence=[Evidence(f.evidence_text, f.source_chunk_id, f.source_form,
                           _date(f.source_filing_date), f.confidence) for f in legacy.findings],
        uncertainties=uncertainties,
        confidence=round(mean(confidences), 4) if confidences and not failed else None,
        source_references=refs,
        errors=errors,
    )


# --- FinancialAgent ----------------------------------------------------------

FINANCIAL_METRICS = {
    "revenue_cagr": ("Revenue CAGR", "((end / start) ^ (1 / years) - 1) * 100"),
    "revenue_yoy": ("Revenue Growth (YoY)", "(current - prior) / |prior| * 100"),
    "net_margin": ("Net Margin", "Net Income / Revenue * 100"),
    "roe": ("Return on Equity", "Net Income / Stockholders' Equity * 100"),
}


def _financial(legacy: LegacyAgentResult) -> AgentResult:
    out = _base(legacy)
    claims, calcs = [], []
    for f in legacy.findings:
        kind = _search(r"_(revenue_cagr|revenue_yoy|net_margin|roe)(?:_(.+))?$", f.source_chunk_id)
        pct = _PCT.search(f.claim)
        if not (kind and pct):
            claims.append(_claim(f))
            continue
        metric, formula = FINANCIAL_METRICS[kind.group(1)]
        value = float(pct.group(1))
        span = _search(r"from (\S+) to (\S+?)\.?$", f.claim)
        period = f"{span.group(1)} to {span.group(2)}" if span else f"FY ended {kind.group(2)}"
        calcs.append(Calculation(metric, value, "%", period, formula, f"{XBRL_SOURCE} ({f.source_chunk_id})"))
        claims.append(_claim(f, value, "%"))
    out.claims, out.calculations = claims, calcs
    cagr = legacy.metrics.get("revenue_cagr_pct")
    if legacy.status != "failed":
        out.summary = f"{len(calcs)} financial calculation(s) from SEC XBRL data" + (
            f"; revenue CAGR {cagr:.1f}% over {legacy.metrics.get('years_of_revenue_data')} years." if cagr is not None else "."
        )
    return out


# --- MarketAgent -------------------------------------------------------------

def _market(legacy: LegacyAgentResult) -> AgentResult:
    out = _base(legacy)
    claims, calcs, competitors = [], [], []
    for f in legacy.findings:
        cid = f.source_chunk_id
        if cid.endswith("_inclusion_rationale"):
            ticker = cid[: -len("_inclusion_rationale")]
            competitors.append(ticker)
            claims.append(_claim(f, ticker))
            continue
        cagr = _search(r"^(\S+) revenue CAGR \(([-\d.]+)%\) was \w+ than (\S+?)'s \(([-\d.]+)%\)", f.claim)
        ratio = _search(rf"calendar (\d{{4}}), (\S+) revenue \({_MONEY}\) was ([\d.]+)x (\S+?)'s \({_MONEY}\)", f.claim)
        if cagr:
            target, t_val, comp, c_val = cagr.group(1), float(cagr.group(2)), cagr.group(3), float(cagr.group(4))
            for ticker, value in ((target, t_val), (comp, c_val)):
                calcs.append(Calculation(f"{ticker} Revenue CAGR", value, "%", "available fiscal history",
                                         FINANCIAL_METRICS["revenue_cagr"][1], f"{XBRL_SOURCE} ({cid})"))
            claims.append(_claim(f, {target: t_val, comp: c_val}, "%"))
        elif ratio:
            year, target, value, comp = ratio.group(1), ratio.group(2), float(ratio.group(4)), ratio.group(5)
            calcs.append(Calculation(f"Revenue ratio {target}/{comp}", value, "x",
                                     f"fiscal years ending in calendar {year}",
                                     f"{target} revenue / {comp} revenue", f"{XBRL_SOURCE} ({cid})"))
            claims.append(_claim(f, value, "x"))
        else:
            claims.append(_claim(f))
    out.claims, out.calculations = claims, calcs
    if competitors:
        out.assumptions = {
            "competitors": competitors,
            "competitor_source": "config.yaml",
            "fiscal_year_alignment": "grouped by calendar year of fiscal year-end (approximation)",
        }
    if legacy.status != "failed":
        out.summary = f"Compared against {len(competitors)} competitor(s): {', '.join(competitors) or 'none'}."
    return out


# --- ValuationAgent ----------------------------------------------------------

def _dcf(f: Finding, scenario: str, out: AgentResult) -> Valuation:
    ev = _search(rf"Enterprise Value = {_MONEY}", f.claim)
    ev_value = _num(ev.group(1)) if ev else None
    growth = _search(r"Revenue growth rate assumption=(\d+(?:\.\d+)?|None)", f.evidence_text)
    assumptions = {
        "wacc": float(m.group(1)) if (m := _search(r"WACC=(\d+(?:\.\d+)?)", f.evidence_text)) else None,
        "terminal_growth": float(m.group(1)) if (m := _search(r"Terminal Growth=(\d+(?:\.\d+)?)", f.evidence_text)) else None,
        "revenue_growth_rate": float(growth.group(1)) if growth and growth.group(1) != "None" else None,
        "fcf_proxy": "Operating Cash Flow (CapEx unavailable)",
        "flagged_unrealistic": "likely unrealistic" in f.claim,
    }
    for label, pattern in (("PV of explicit-period FCF", r"Sum of PV \(explicit period\) = "),
                           ("PV of terminal value", r"PV of Terminal Value = ")):
        m = _search(pattern + _MONEY, f.evidence_text)
        if m:
            out.calculations.append(Calculation(f"{label} ({scenario})", _num(m.group(1)), "USD", None,
                                                None, f"DCF model ({f.source_chunk_id})"))
    out.calculations.append(Calculation(f"Enterprise value ({scenario})", ev_value, "USD", None,
                                        "PV explicit period + PV terminal value", f"DCF model ({f.source_chunk_id})"))
    return Valuation("DCF", scenario, enterprise_value=ev_value, assumptions=assumptions,
                     source=f"DCF model ({f.source_chunk_id})")


def _valuation(legacy: LegacyAgentResult) -> AgentResult:
    out = _base(legacy)
    claims, valuations = [], []
    for f in legacy.findings:
        cid = f.source_chunk_id
        scenario = _search(r"_dcf_(.+)$", cid)
        if scenario:
            v = _dcf(f, scenario.group(1), out)
            valuations.append(v)
            claims.append(_claim(f, v.enterprise_value, "USD"))
        elif cid.endswith("_pe_ratio") and (pe := _search(rf"P/E ratio: ([\d.]+)x \(price {_MONEY} / EPS \$([-\d.]+)\)", f.claim)):
            multiple = float(pe.group(1))
            eps_period = _search(r"Diluted EPS \(([^)]+)\)", f.evidence_text)
            valuations.append(Valuation("P/E", multiple=multiple, source=f"market data + {XBRL_SOURCE} ({cid})",
                                        assumptions={"price": _num(pe.group(2)), "price_date": _date(f.source_filing_date),
                                                     "eps_diluted": float(pe.group(3)),
                                                     "eps_period": eps_period.group(1) if eps_period else None}))
            out.calculations.append(Calculation("P/E ratio", multiple, "x", f"price {f.source_filing_date}",
                                                "Price / Diluted EPS", f"market data + {XBRL_SOURCE} ({cid})"))
            claims.append(_claim(f, multiple, "x"))
        elif cid.endswith("_ev_ebit") and (ratio := _search(r"EV/EBIT ratio: ([\d.]+)x", f.claim)):
            multiple = float(ratio.group(1))
            parts = {k: _num(m.group(1)) for k, p in (("enterprise_value", rf"Enterprise Value = {_MONEY}"),
                                                      ("market_cap", rf"Market Cap {_MONEY}"),
                                                      ("total_debt", rf"Debt {_MONEY}"),
                                                      ("cash", rf"Cash {_MONEY}"),
                                                      ("ebit", rf"EBIT \([^)]+\) = {_MONEY}"))
                     if (m := _search(p, f.evidence_text))}
            ev = parts.pop("enterprise_value", None)
            valuations.append(Valuation("EV/EBIT", enterprise_value=ev, multiple=multiple,
                                        assumptions={**parts, "ebit_period": _date(f.source_filing_date)},
                                        source=f"market data + {XBRL_SOURCE} ({cid})"))
            out.calculations.append(Calculation("EV/EBIT", multiple, "x", f"EBIT FY ended {f.source_filing_date}",
                                                "(Market Cap + Debt - Cash) / EBIT", f"market data + {XBRL_SOURCE} ({cid})"))
            claims.append(_claim(f, multiple, "x"))
        else:
            claims.append(_claim(f))
    out.claims, out.valuations = claims, valuations
    if any(v.method == "DCF" for v in valuations):
        out.assumptions = {"fcf_proxy": "Operating Cash Flow used as FCF proxy (CapEx unavailable at annual granularity)"}
    if legacy.status != "failed":
        methods = sorted({v.method for v in valuations})
        out.summary = f"{len(valuations)} valuation result(s) using {', '.join(methods) or 'no method'}."
    return out


# --- RiskAgent ---------------------------------------------------------------

def _risk(legacy: LegacyAgentResult) -> AgentResult:
    out = _base(legacy)
    claims, risks = [], []
    for f in legacy.findings:
        m = _search(r"^(.+?) risk: severity=(\w+) \((\d+) distinct fiscal years mentioned\)", f.claim)
        if not m:
            claims.append(_claim(f))
            continue
        risks.append(Risk(risk=m.group(1), severity=m.group(2), evidence=[f.evidence_text]))
        claims.append(_claim(f, m.group(2)))
    out.claims, out.risks = claims, risks
    if risks:
        out.assumptions = {"severity_method": "frequency proxy: distinct fiscal years mentioning the risk topic"}
    if legacy.status != "failed":
        dist = legacy.metrics.get("severity_distribution", {})
        out.summary = f"{len(risks)} risk categor{'y' if len(risks) == 1 else 'ies'} assessed" + (
            "; " + ", ".join(f"{k}={v}" for k, v in dist.items()) + "." if dist else ".")
    return out


# --- NewsAgent / ResearchAgent ----------------------------------------------

def _news(legacy: LegacyAgentResult) -> AgentResult:
    out = _base(legacy)
    claims = []
    for f in legacy.findings:
        m = _search(r"^Classified as (\w+)", f.claim)
        claims.append(_claim(f, m.group(1) if m else None))
    out.claims = claims
    counts = legacy.metrics.get("label_counts", {})
    if legacy.status != "failed":
        out.summary = f"{len(legacy.findings)} 8-K sentence(s) classified" + (
            "; " + ", ".join(f"{k}={v}" for k, v in counts.items()) + "." if counts else ".")
    return out


def _research(legacy: LegacyAgentResult) -> AgentResult:
    out = _base(legacy)
    if legacy.status != "failed":
        out.summary = f"{len(legacy.findings)} filing passage(s) retrieved for: {legacy.task}"
    return out


NORMALIZERS: Dict[str, Callable[[LegacyAgentResult], AgentResult]] = {
    "FinancialAgent": _financial,
    "MarketAgent": _market,
    "ValuationAgent": _valuation,
    "RiskAgent": _risk,
    "NewsAgent": _news,
    "ResearchAgent": _research,
}


def normalize_result(legacy: LegacyAgentResult) -> AgentResult:
    """Convert one BaseAgent output (core.schemas.AgentResult) to the standard contract."""
    normalizer = NORMALIZERS.get(legacy.agent_name, _base)
    try:
        return normalizer(legacy)
    except Exception as e:  # never lose an agent's output because a parser broke
        log.warning("[ADAPTER] %s normalization fell back to generic: %s", legacy.agent_name, e)
        out = _base(legacy)
        out.uncertainties.append(f"Structured parsing unavailable ({e}); claims are unparsed.")
        return out


def normalize_verification(verification: Optional[list], task: str = "verify agent findings",
                           error: str = "") -> AgentResult:
    """Red-Team: List[VerificationResult] -> standard AgentResult."""
    if error:
        return AgentResult(RED_TEAM_NAME, task, "failed", summary="Red-Team verification failed.", errors=[error])
    verification = verification or []
    by_result: Dict[str, int] = {}
    by_agent: Dict[str, int] = {}
    for v in verification:
        by_result[v.verification_result] = by_result.get(v.verification_result, 0) + 1
        by_agent[v.source_agent] = by_agent.get(v.source_agent, 0) + 1
    problems = list(dict.fromkeys(f"[{v.source_agent}] {v.verification_result}: {v.problem}" for v in verification))
    return AgentResult(
        agent_name=RED_TEAM_NAME,
        task=task,
        status="success",
        summary=f"{len(verification)} verification issue(s)" + (f": {by_result}." if by_result else "."),
        findings=[asdict(v) for v in verification],
        metrics={"total_issues": len(verification), "by_result": by_result, "by_agent": by_agent},
        claims=[Claim(claim=v.claim, value=v.verification_result, source=v.source_agent,
                      evidence=v.evidence, confidence=v.confidence) for v in verification],
        uncertainties=problems,
        assumptions={"method": "mechanical checks: confidence threshold, numeric support, disclaimers, outliers"},
    )


SYNTHESIS_STATUS = {"complete": "success", "partial": "partial", "failed": "failed"}


def normalize_synthesis(report) -> AgentResult:
    """Synthesis: SynthesisReport -> standard AgentResult."""
    vs = report.verification_summary or {}
    claims = [
        Claim(claim=h["claim"], source=f"{agent}:{h.get('source_form')}", confidence=h.get("confidence"),
              value=None if not h.get("issues") else {"issues": h["issues"]})
        for agent, summary in report.agents.items() for h in summary.get("headline_findings", [])
    ]
    return AgentResult(
        agent_name=SYNTHESIS_NAME,
        task=report.query,
        status=SYNTHESIS_STATUS.get(report.overall_status, "partial"),
        summary=(f"Overall status {report.overall_status}: {len(report.agents)} agent(s), "
                 f"{vs.get('total_issues', 0)} verification issue(s)."),
        findings=[{"agent": name, **summary} for name, summary in report.agents.items()],
        metrics={"agent_count": len(report.agents), "verification_summary": vs},
        claims=claims,
        uncertainties=list(report.caveats),
        errors=["All agents failed."] if report.overall_status == "failed" else [],
        details={"company_ticker": report.company_ticker, "text": report.text},
    )


def run_standardized(agent_class: type, company_ticker: str, task: str) -> AgentResult:
    """Run an existing BaseAgent subclass and return the standard AgentResult."""
    return normalize_result(agent_class(company_ticker=company_ticker).run(task))


def standardize_response(response: dict) -> List[AgentResult]:
    """
    Convert a core.orchestrator.run_query() dict (agent_results, verification,
    optional synthesis) into standard results, in execution order.
    """
    results = [normalize_result(r) for r in response.get("agent_results", {}).values()]
    if "verification" in response:
        results.append(normalize_verification(response["verification"]))
    if response.get("synthesis") is not None:
        results.append(normalize_synthesis(response["synthesis"]))
    return results

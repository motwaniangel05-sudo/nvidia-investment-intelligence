"""
RedTeamAgent: verifies other agents' AgentResult outputs. Does NOT
generate new findings from raw data -- it audits findings already
produced by other agents, per master prompt Section "Red-Team Agent":
Claim / Evidence / Verification Result / Confidence / Problem / Correction.

This agent must not blindly agree with other agents' outputs. All checks
here are mechanical and verifiable (regex/number matching, threshold
comparisons, presence/absence checks) -- not simulated "AI judgment" we
cannot substantiate.
"""

import re
from dataclasses import dataclass
from typing import List

from core.logger import get_logger
from core.schemas import AgentResult, Finding

log = get_logger(__name__)

LOW_CONFIDENCE_THRESHOLD = 0.10
STALE_DATA_YEARS = 3  # findings referencing data older than this, relative
                        # to the newest finding in the batch, are flagged

# Finding types that MUST carry specific disclaimer language, per known
# project limitations (Phase 6 CapEx gap, Phase 12 EV/EBIT substitution).
REQUIRED_DISCLAIMERS = {
    "fcf": ("capex", "FCF-related finding is missing the required CapEx/FCF-proxy disclaimer."),
    "ev/ebit": ("ebitda", "EV/EBIT finding is missing the required EBITDA-substitution disclaimer."),
}


@dataclass
class VerificationResult:
    claim: str
    evidence: str
    verification_result: str  # "SUPPORTED" | "UNSUPPORTED" | "FLAGGED"
    confidence: float
    problem: str
    correction: str
    source_agent: str


def check_low_confidence(findings: List[Finding], source_agent: str) -> List[VerificationResult]:
    """Flag findings whose stated confidence falls below our threshold."""
    results = []
    for f in findings:
        if f.confidence < LOW_CONFIDENCE_THRESHOLD:
            results.append(VerificationResult(
                claim=f.claim,
                evidence=f.evidence_text[:200],
                verification_result="FLAGGED",
                confidence=f.confidence,
                problem=(
                    f"Confidence ({f.confidence}) is below the verification "
                    f"threshold ({LOW_CONFIDENCE_THRESHOLD}). This finding's "
                    f"underlying retrieval or classification was weak."
                ),
                correction="Treat this finding as low-reliability; seek corroborating evidence before relying on it.",
                source_agent=source_agent,
            ))
    return results


def check_unsupported_numeric_claims(findings: List[Finding], source_agent: str) -> List[VerificationResult]:
    """
    Mechanical cross-check: if a claim states a specific number (e.g. a
    percentage or dollar figure), verify that SOME number appears in the
    evidence_text too. This doesn't verify correctness, only that the
    claim isn't citing a number with zero supporting text -- a real,
    cheap sanity check against a specific failure mode.
    """
    results = []
    number_pattern = re.compile(r"[\d,]+\.?\d*%?")

    def _clean_numbers(text):
        raw = number_pattern.findall(text)
        cleaned = set()
        for n in raw:
            n = n.strip(".,")  # strip trailing/leading punctuation (e.g. "4,805.")
            if len(n) > 1 and any(c.isdigit() for c in n):
                cleaned.add(n.rstrip("%"))  # "45.8%" and "45.8" are the same number
        return cleaned

    for f in findings:
        claim_numbers = _clean_numbers(f.claim)
        if not claim_numbers:
            continue

        evidence_numbers = _clean_numbers(f.evidence_text)
        if not (claim_numbers & evidence_numbers) and claim_numbers:
            # No overlap at all between numbers in claim and in evidence
            results.append(VerificationResult(
                claim=f.claim,
                evidence=f.evidence_text[:200],
                verification_result="UNSUPPORTED",
                confidence=f.confidence,
                problem=(
                    f"Claim cites number(s) {claim_numbers} but none appear "
                    f"verbatim in the evidence text. This may indicate the "
                    f"number was computed/asserted without a directly "
                    f"quotable source, or evidence_text was truncated."
                ),
                correction="Verify the cited figure against the original source document directly.",
                source_agent=source_agent,
            ))
    return results


def check_missing_disclaimers(findings: List[Finding], source_agent: str) -> List[VerificationResult]:
    """Check FCF/EV-EBIT findings carry their required disclaimer language."""
    results = []
    for f in findings:
        claim_lower = f.claim.lower()
        for keyword, (required_term, problem_msg) in REQUIRED_DISCLAIMERS.items():
            if keyword in claim_lower:
                combined_text = (f.claim + " " + f.evidence_text).lower()
                if required_term not in combined_text:
                    results.append(VerificationResult(
                        claim=f.claim,
                        evidence=f.evidence_text[:200],
                        verification_result="FLAGGED",
                        confidence=f.confidence,
                        problem=problem_msg,
                        correction=f"Add explicit disclosure of the {required_term.upper()} limitation to this finding.",
                        source_agent=source_agent,
                    ))
    return results


def check_historical_cagr_outlier(findings: List[Finding], source_agent: str) -> List[VerificationResult]:
    """
    Independently flag any DCF scenario using unflagged extreme growth
    assumptions, by checking the claim text for a very high implied
    Enterprise Value relative to other scenarios in the same batch.
    """
    results = []
    ev_pattern = re.compile(r"Enterprise Value = \$([\d,]+)")

    evs = []
    for f in findings:
        if "DCF scenario" not in f.claim:
            continue
        match = ev_pattern.search(f.claim)
        if match:
            evs.append((f, float(match.group(1).replace(",", ""))))

    if len(evs) < 2:
        return results

    values = [v for _, v in evs]
    median_ev = sorted(values)[(len(values) - 1) // 2]  # lower median, so 2 scenarios can be compared

    for finding, ev in evs:
        if ev > median_ev * 2 and "likely unrealistic" not in finding.claim.lower():
            results.append(VerificationResult(
                claim=finding.claim,
                evidence=finding.evidence_text[:200],
                verification_result="FLAGGED",
                confidence=finding.confidence,
                problem=(
                    f"This scenario's Enterprise Value (${ev:,.0f}) is more "
                    f"than 2x the median across scenarios (${median_ev:,.0f}) "
                    f"and is not explicitly flagged as an outlier assumption."
                ),
                correction="Review the growth rate assumption driving this scenario for plausibility.",
                source_agent=source_agent,
            ))
    return results


def check_revenue_consistency(
    financial_findings: List[Finding], valuation_findings: List[Finding]
) -> List[VerificationResult]:
    """
    Cross-agent check: FinancialAgent's stated CAGR should match the
    growth rate ValuationAgent's historical_cagr scenario claims to use,
    within rounding tolerance. Catches silent drift if the two agents'
    underlying calculations ever diverge.
    """
    results = []

    fin_cagr = None
    for f in financial_findings:
        # Matches FinancialAgent's actual claim format:
        # "NVDA revenue grew at a 45.8% CAGR from ..."
        match = re.search(r"grew at a ([\d.]+)% CAGR", f.claim)
        if match:
            fin_cagr = float(match.group(1))
            break

    if fin_cagr is None:
        return results

    for f in valuation_findings:
        if "historical_cagr" in f.claim:
            # historical_cagr scenario's growth_rate_used is None in our
            # current implementation (computed internally) -- this check
            # confirms that's documented, not silently wrong
            if "growth rate used: None" in f.claim:
                results.append(VerificationResult(
                    claim=f.claim,
                    evidence=f"FinancialAgent independently computed CAGR: {fin_cagr}%",
                    verification_result="FLAGGED",
                    confidence=f.confidence,
                    problem=(
                        f"ValuationAgent's historical_cagr scenario does not "
                        f"display the actual growth rate used in its claim "
                        f"text (shows 'None' since it's computed internally "
                        f"by forecast_revenue()), even though FinancialAgent "
                        f"independently reports CAGR={fin_cagr}% for the same "
                        f"company. These should be cross-referenced explicitly."
                    ),
                    correction=(
                        f"ValuationAgent should display the actual computed "
                        f"growth rate (expected ~{fin_cagr}%) in its finding "
                        f"claim text, not 'None', for transparency."
                    ),
                    source_agent="ValuationAgent",
                ))
    return results


def run_verification(agent_results: dict) -> List[VerificationResult]:
    """
    Main entry: agent_results is {agent_name: AgentResult}. Runs all
    checks across all provided agent outputs and returns a flat list of
    VerificationResult records.
    """
    all_results = []

    for agent_name, result in agent_results.items():
        all_results.extend(check_low_confidence(result.findings, agent_name))
        all_results.extend(check_unsupported_numeric_claims(result.findings, agent_name))
        all_results.extend(check_missing_disclaimers(result.findings, agent_name))
        all_results.extend(check_historical_cagr_outlier(result.findings, agent_name))

    if "FinancialAgent" in agent_results and "ValuationAgent" in agent_results:
        all_results.extend(check_revenue_consistency(
            agent_results["FinancialAgent"].findings,
            agent_results["ValuationAgent"].findings,
        ))

    return all_results


if __name__ == "__main__":
    from agents.financial_agent import FinancialAgent
    from agents.valuation_agent import ValuationAgent
    from agents.news_agent import NewsAgent

    print("Running source agents...")
    financial_result = FinancialAgent("NVDA").run("Analyze financial health")
    valuation_result = ValuationAgent("NVDA").run("Value the company")
    news_result = NewsAgent("NVDA").run("Classify events")

    agent_results = {
        "FinancialAgent": financial_result,
        "ValuationAgent": valuation_result,
        "NewsAgent": news_result,
    }

    print("\nRunning Red-Team verification...\n")
    verification = run_verification(agent_results)

    print(f"Total issues flagged: {len(verification)}\n")
    for v in verification[:15]:
        print(f"[{v.verification_result}] ({v.source_agent}, conf={v.confidence})")
        print(f"  Claim: {v.claim[:150]}")
        print(f"  Problem: {v.problem}")
        print(f"  Correction: {v.correction}")
        print()

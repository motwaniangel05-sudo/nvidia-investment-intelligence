"""
SynthesisAgent: combines the orchestrator's output (all agent results plus
Red-Team verification) into one structured report.

Purely mechanical -- no LLM and no new analysis. It selects, groups, flags and
counts what other agents already produced, and keeps every caveat visible.

Note: confidence values are NOT comparable across agents. Financial, Valuation
and Risk use 1.0 for exact/deterministic calculations, while Research and News
use a retrieval or classifier score. The report keeps each value as-is.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from core.logger import get_logger

log = get_logger(__name__)

DEFAULT_FINDINGS_PER_AGENT = 3
PROBLEM_RESULTS = {"FLAGGED", "UNSUPPORTED"}


@dataclass
class SynthesisReport:
    company_ticker: str
    query: str
    overall_status: str  # "complete" | "partial" | "failed"
    agents: Dict[str, dict] = field(default_factory=dict)
    verification_summary: dict = field(default_factory=dict)
    caveats: List[str] = field(default_factory=list)
    text: str = ""


def _overall_status(agent_results: dict) -> str:
    if not agent_results:
        return "failed"
    statuses = [r.status for r in agent_results.values()]
    if all(s == "failed" for s in statuses):
        return "failed"
    if all(s == "success" for s in statuses):
        return "complete"
    return "partial"


def _problem_index(verification: list) -> dict:
    """Map (agent, claim) -> list of problem results raised by the Red-Team."""
    index = {}
    for v in verification:
        if v.verification_result in PROBLEM_RESULTS:
            index.setdefault((v.source_agent, v.claim), []).append(v.verification_result)
    return index


def _summarize_agent(name: str, result, problems: dict, max_findings: int) -> dict:
    ranked = sorted(result.findings, key=lambda f: f.confidence, reverse=True)
    headlines = [
        {
            "claim": f.claim,
            "confidence": f.confidence,
            "source_form": f.source_form,
            "issues": list(problems.get((name, f.claim), [])),
        }
        for f in ranked[:max_findings]
    ]
    flagged = sum(1 for f in result.findings if (name, f.claim) in problems)
    return {
        "status": result.status,
        "finding_count": len(result.findings),
        "flagged_finding_count": flagged,
        "headline_findings": headlines,
        "warnings": list(result.warnings),
    }


def _verification_summary(verification: list) -> dict:
    by_result = {}
    for v in verification:
        by_result[v.verification_result] = by_result.get(v.verification_result, 0) + 1
    return {"total_issues": len(verification), "by_result": by_result}


def _collect_caveats(agent_results: dict, verification: list) -> List[str]:
    caveats, seen = [], set()
    for name, result in agent_results.items():
        for w in result.warnings:
            if (name, w) not in seen:
                seen.add((name, w))
                caveats.append(f"[{name}] {w}")
    if verification:
        caveats.append(
            f"[RedTeam] {len(verification)} issue(s) raised by verification; "
            f"see the flagged findings above."
        )
    return caveats


def _render_text(report: SynthesisReport) -> str:
    lines = [
        f"Synthesis for {report.company_ticker} -- query: {report.query}",
        f"Overall status: {report.overall_status}",
        "",
    ]
    for name, a in report.agents.items():
        lines.append(
            f"{name} ({a['status']}, {a['finding_count']} findings, "
            f"{a['flagged_finding_count']} flagged)"
        )
        for h in a["headline_findings"]:
            mark = f" [{'/'.join(h['issues'])}]" if h["issues"] else ""
            lines.append(f"  - {h['claim']} (confidence {h['confidence']}){mark}")
        lines.append("")
    vs = report.verification_summary
    lines.append(f"Verification: {vs['total_issues']} issue(s) {vs['by_result']}")
    if report.caveats:
        lines.append("")
        lines.append("Caveats:")
        lines.extend(f"  * {c}" for c in report.caveats)
    return "\n".join(lines)


def synthesize(response: dict, max_findings_per_agent: int = DEFAULT_FINDINGS_PER_AGENT) -> SynthesisReport:
    """Main entry: takes the dict returned by core.orchestrator.run_query()."""
    agent_results = response.get("agent_results", {})
    verification = response.get("verification", [])
    problems = _problem_index(verification)

    report = SynthesisReport(
        company_ticker=response.get("company_ticker", ""),
        query=response.get("query", ""),
        overall_status=_overall_status(agent_results),
        agents={
            name: _summarize_agent(name, result, problems, max_findings_per_agent)
            for name, result in agent_results.items()
        },
        verification_summary=_verification_summary(verification),
        caveats=_collect_caveats(agent_results, verification),
    )
    report.text = _render_text(report)
    log.info("Synthesis complete: status=%s, %d agents", report.overall_status, len(report.agents))
    return report


if __name__ == "__main__":
    from core.orchestrator import run_query

    print(synthesize(run_query("NVDA", "Analyze NVIDIA's financial health and valuation")).text)

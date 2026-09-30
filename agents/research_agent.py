"""
ResearchAgent: answers qualitative research questions using RAG retrieval
over the filing chunks knowledge base. Findings are structured with
evidence and confidence, never generated from the agent's "own knowledge"
(there is none -- everything is retrieved).
"""

from core.logger import get_logger
from core.schemas import AgentResult, Finding
from agents.base_agent import BaseAgent

log = get_logger(__name__)

# Below this retrieval score, evidence is too weak to state as a finding
# with any confidence -- included only as a low-confidence flag, not
# presented as a solid answer. Chosen based on Phase 5 testing, where
# specific queries scored 0.20+ and vague queries scored ~0.08.
MIN_USEFUL_SCORE = 0.05


class ResearchAgent(BaseAgent):
    agent_name = "ResearchAgent"

    def analyze(self, task: str) -> AgentResult:
        """
        task is treated as a natural-language research question, e.g.
        "What are NVIDIA's main business segments?"
        Retrieves top chunks, converts each into a Finding.
        """
        chunks = self.retrieve(task, top_k=5)

        findings = []
        for chunk in chunks:
            if chunk["score"] < MIN_USEFUL_SCORE:
                continue
            findings.append(Finding(
                claim=f"Relevant evidence found regarding: {task}",
                evidence_text=chunk["text"][:500],
                source_chunk_id=chunk["chunk_id"],
                source_form=chunk["form"],
                source_filing_date=chunk["filing_date"],
                confidence=round(chunk["score"], 4),
            ))

        status = "success" if findings else "partial"
        warnings = []
        if not findings:
            warnings.append(
                f"No chunks scored above the minimum usefulness threshold "
                f"({MIN_USEFUL_SCORE}) for this query. Try a more specific question."
            )

        return AgentResult(
            agent_name=self.agent_name,
            task=task,
            findings=findings,
            metrics={"chunks_retrieved": len(chunks), "chunks_used": len(findings)},
            warnings=warnings,
            status=status,
        )


if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) or "What are NVIDIA's main business segments and products?"

    agent = ResearchAgent(company_ticker="NVDA")
    result = agent.run(question)

    print(f"Task: {result.task}")
    print(f"Status: {result.status}")
    print(f"Metrics: {result.metrics}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")
    print()

    for i, f in enumerate(result.findings, 1):
        print(f"[{i}] confidence={f.confidence} | {f.source_form} filed {f.source_filing_date}")
        print(f"    {f.evidence_text[:250]}...")
        print()

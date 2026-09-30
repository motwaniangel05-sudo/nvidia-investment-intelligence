"""
BaseAgent: shared lifecycle every specialized agent inherits.

Per master prompt Section 10: agents never talk to each other directly.
They communicate through AgentResult objects passed via the orchestrator
(built in Phase 14). Each agent follows the same run() lifecycle:
    1. retrieve() -- get evidence from the knowledge base (RAG)
    2. analyze()  -- subclass-specific logic on that evidence
    3. verify()   -- self-check: confidence, completeness, warnings
    4. return a standardized AgentResult
"""

from abc import ABC, abstractmethod
from typing import List

from core.logger import get_logger
from core.schemas import AgentResult, Finding
from rag.retriever import retrieve as rag_retrieve

log = get_logger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all specialized agents. Subclasses must
    implement analyze(). retrieve(), verify(), and run() have sensible
    shared defaults but can be overridden if an agent's needs differ
    (e.g. FinancialAgent in Phase 8 won't need RAG retrieval at all).
    """

    agent_name: str = "BaseAgent"

    def __init__(self, company_ticker: str):
        self.company_ticker = company_ticker

    def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        """
        Default retrieval: delegate to rag.retriever. Subclasses that
        don't need RAG (e.g. purely numeric agents) can skip calling this.
        """
        try:
            return rag_retrieve(query, top_k=top_k)
        except Exception as e:
            log.error("Retrieval failed for query '%s': %s", query, e)
            return []

    @abstractmethod
    def analyze(self, task: str) -> AgentResult:
        """
        Subclass-specific logic: given a task description, produce an
        AgentResult. This is the one method every concrete agent must
        implement -- everything else has a shared default.
        """
        raise NotImplementedError

    def verify(self, result: AgentResult) -> AgentResult:
        """
        Default verification: flag low-confidence or empty results.
        Subclasses may override for more specific checks (e.g. the
        Red-Team agent in Phase 13 does much deeper verification).
        """
        if not result.findings:
            result.warnings.append("No findings were produced for this task.")
            result.status = "partial" if result.status == "success" else result.status

        low_confidence = [f for f in result.findings if f.confidence < 0.05]
        if low_confidence:
            result.warnings.append(
                f"{len(low_confidence)} finding(s) have very low confidence "
                f"(retrieval score < 0.05) and should be treated with caution."
            )

        return result

    def run(self, task: str) -> AgentResult:
        """Main entry: orchestrates analyze() then verify()."""
        log.info("[%s] Running task: %s", self.agent_name, task)
        try:
            result = self.analyze(task)
        except Exception as e:
            log.error("[%s] Analysis failed: %s", self.agent_name, e)
            return AgentResult(
                agent_name=self.agent_name,
                task=task,
                status="failed",
                warnings=[f"Agent execution failed: {e}"],
            )

        result = self.verify(result)
        log.info(
            "[%s] Completed with status=%s, %d findings, %d warnings",
            self.agent_name, result.status, len(result.findings), len(result.warnings),
        )
        return result

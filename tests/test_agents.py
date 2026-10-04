"""Tests for BaseAgent lifecycle and ResearchAgent behavior."""

from unittest.mock import patch

import pytest

from agents.base_agent import BaseAgent
from agents.research_agent import ResearchAgent
from core.schemas import AgentResult, Finding


def test_base_agent_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseAgent("NVDA")


class DummyAgent(BaseAgent):
    agent_name = "DummyAgent"

    def analyze(self, task):
        return AgentResult(agent_name=self.agent_name, task=task, findings=[
            Finding("claim", "evidence", "chunk_1", "10-K", "2024-01-28", 0.5)
        ])


class FailingAgent(BaseAgent):
    agent_name = "FailingAgent"

    def analyze(self, task):
        raise ValueError("something broke")


def test_run_calls_analyze_and_verify_in_order():
    agent = DummyAgent("NVDA")
    result = agent.run("test task")
    assert result.status == "success"
    assert len(result.findings) == 1


def test_run_catches_exceptions_and_returns_failed_status():
    agent = FailingAgent("NVDA")
    result = agent.run("test task")
    assert result.status == "failed"
    assert "something broke" in result.warnings[0]


def test_verify_flags_empty_findings_as_partial():
    agent = DummyAgent("NVDA")
    empty_result = AgentResult(agent_name="DummyAgent", task="x", findings=[])
    verified = agent.verify(empty_result)
    assert verified.status == "partial"
    assert len(verified.warnings) == 1


def test_verify_flags_low_confidence_findings():
    agent = DummyAgent("NVDA")
    low_conf_result = AgentResult(
        agent_name="DummyAgent", task="x",
        findings=[Finding("c", "e", "id1", "10-K", "2024-01-01", 0.01)],
    )
    verified = agent.verify(low_conf_result)
    assert any("low confidence" in w for w in verified.warnings)


def test_research_agent_filters_low_score_chunks():
    fake_chunks = [
        {"chunk_id": "c1", "score": 0.20, "form": "10-K",
         "filing_date": "2024-01-28", "text": "Relevant text about revenue."},
        {"chunk_id": "c2", "score": 0.02, "form": "10-K",
         "filing_date": "2024-01-28", "text": "Barely relevant text."},
    ]
    agent = ResearchAgent("NVDA")
    with patch.object(agent, "retrieve", return_value=fake_chunks):
        result = agent.run("What is NVIDIA's revenue?")

    assert len(result.findings) == 1  # only c1 passes the 0.05 threshold
    assert result.findings[0].source_chunk_id == "c1"
    assert result.status == "success"


def test_research_agent_no_chunks_returns_partial_with_warning():
    agent = ResearchAgent("NVDA")
    with patch.object(agent, "retrieve", return_value=[]):
        result = agent.run("An extremely obscure unanswerable question")

    assert result.status == "partial"
    assert len(result.findings) == 0
    # Two warnings are expected: one from ResearchAgent.analyze() explaining
    # WHY (no chunks met the score threshold), and one from BaseAgent.verify()
    # confirming THAT no findings resulted -- both are legitimate, layered checks.
    assert len(result.warnings) == 2
    assert any("threshold" in w for w in result.warnings)
    assert any("No findings were produced" in w for w in result.warnings)


def test_research_agent_finding_has_correct_metadata():
    fake_chunks = [
        {"chunk_id": "NVDA_10-K_2024-01-28_0005", "score": 0.30, "form": "10-K",
         "filing_date": "2024-01-28", "text": "NVIDIA's data center segment grew significantly."},
    ]
    agent = ResearchAgent("NVDA")
    with patch.object(agent, "retrieve", return_value=fake_chunks):
        result = agent.run("data center segment growth")

    finding = result.findings[0]
    assert finding.source_form == "10-K"
    assert finding.source_filing_date == "2024-01-28"
    assert finding.confidence == 0.30
    assert "data center segment" in finding.evidence_text



# ---- BaseAgent.retrieve, verify edge cases, abstract method ----
from agents import base_agent as ba


def test_default_retrieve_delegates_to_rag_retriever_with_top_k():
    with patch.object(ba, "rag_retrieve", return_value=[{"chunk_id": "c1"}]) as mock_rag:
        out = DummyAgent("NVDA").retrieve("revenue?", top_k=3)

    mock_rag.assert_called_once_with("revenue?", top_k=3)
    assert out == [{"chunk_id": "c1"}]


def test_default_retrieve_uses_top_k_of_5_by_default():
    with patch.object(ba, "rag_retrieve", return_value=[]) as mock_rag:
        DummyAgent("NVDA").retrieve("q")
    mock_rag.assert_called_once_with("q", top_k=5)


def test_default_retrieve_returns_empty_list_when_retriever_raises():
    with patch.object(ba, "rag_retrieve", side_effect=RuntimeError("index down")):
        assert DummyAgent("NVDA").retrieve("q") == []


def test_verify_does_not_flag_confidence_exactly_at_threshold():
    result = AgentResult(
        agent_name="DummyAgent", task="x",
        findings=[Finding("c", "e", "id1", "10-K", "2024-01-01", 0.05)],
    )
    verified = DummyAgent("NVDA").verify(result)
    assert verified.warnings == []
    assert verified.status == "success"


def test_verify_counts_every_low_confidence_finding():
    findings = [
        Finding("c", "e", f"id{i}", "10-K", "2024-01-01", conf)
        for i, conf in enumerate([0.01, 0.02, 0.5])
    ]
    result = AgentResult(agent_name="DummyAgent", task="x", findings=findings)
    verified = DummyAgent("NVDA").verify(result)
    assert any("2 finding(s)" in w for w in verified.warnings)


def test_verify_empty_findings_keeps_failed_status():
    result = AgentResult(agent_name="DummyAgent", task="x", findings=[], status="failed")
    verified = DummyAgent("NVDA").verify(result)
    assert verified.status == "failed"
    assert any("No findings" in w for w in verified.warnings)


def test_run_skips_verify_when_analyze_raises():
    agent = FailingAgent("NVDA")
    with patch.object(agent, "verify") as mock_verify:
        result = agent.run("task")
    mock_verify.assert_not_called()
    assert result.status == "failed"
    assert result.agent_name == "FailingAgent"
    assert result.task == "task"


def test_abstract_analyze_raises_not_implemented_when_called_via_super():
    class CallsSuper(BaseAgent):
        def analyze(self, task):
            return super().analyze(task)

    with pytest.raises(NotImplementedError):
        CallsSuper("NVDA").analyze("t")

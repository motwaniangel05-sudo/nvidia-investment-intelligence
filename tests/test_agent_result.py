"""Tests for the standard agent result contract (core/agent_result.py)."""

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from core.agent_result import (
    SCHEMA_VERSION, AgentResult, Calculation, Claim, Evidence, Risk,
    SourceReference, Valuation, results_from_json, results_to_dict,
    results_to_json, validate_confidence,
)


def full_result(**overrides):
    kwargs = dict(
        agent_name="FinancialAgent",
        task="How healthy is NVIDIA?",
        status="success",
        summary="1 calculation.",
        findings=[{"claim": "c", "evidence_text": "e", "source_chunk_id": "id",
                   "source_form": "XBRL_metrics", "source_filing_date": "2026-01-25", "confidence": 1.0}],
        metrics={"revenue_cagr_pct": 45.8},
        calculations=[Calculation("Revenue Growth", 42.5, "%", "FY2026", "(cur - prior) / |prior| * 100", "XBRL")],
        claims=[Claim("Revenue grew 42.5%", 42.5, "%", "XBRL", "2026-01-25", "Revenue ...", 1.0)],
        evidence=[Evidence("Revenue ...", "id", "XBRL_metrics", "2026-01-25", 1.0)],
        valuations=[Valuation("DCF", "base", 1e12, None, None, None, {"wacc": 0.1}, {})],
        risks=[Risk("Supply chain", "HIGH", None, None, ["evidence"], [])],
        assumptions={"wacc": 0.1},
        uncertainties=["CapEx unavailable"],
        confidence=0.9,
        source_references=[SourceReference("id", "XBRL_metrics", "2026-01-25")],
        timestamp="2026-10-05T00:00:00+00:00",
    )
    kwargs.update(overrides)
    return AgentResult(**kwargs)


# --- valid / optional fields ---------------------------------------------------

def test_valid_agent_result_with_every_field():
    r = full_result()
    d = r.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["calculations"][0] == {
        "metric": "Revenue Growth", "value": 42.5, "unit": "%", "period": "FY2026",
        "formula": "(cur - prior) / |prior| * 100", "source": "XBRL",
    }
    assert set(d["claims"][0]) == {"claim", "value", "unit", "source", "date", "evidence", "confidence"}
    assert set(d["valuations"][0]) >= {"method", "enterprise_value", "equity_value",
                                       "implied_share_price", "assumptions", "sensitivity"}
    assert set(d["risks"][0]) == {"risk", "severity", "probability", "impact",
                                  "evidence", "contradicting_evidence"}


def test_missing_optional_fields_default_to_null_or_empty():
    r = AgentResult("ResearchAgent", "task", "partial")
    d = r.to_dict()
    assert d["summary"] == "" and d["confidence"] is None
    for key in ("findings", "calculations", "claims", "evidence", "valuations", "risks",
                "uncertainties", "source_references", "errors"):
        assert d[key] == []
    assert d["metrics"] == {} and d["assumptions"] == {} and d["details"] == {}
    assert d["timestamp"].endswith("+00:00")
    assert Claim("only a claim").value is None
    assert Valuation("P/E").enterprise_value is None


def test_nested_dicts_are_converted_to_typed_items():
    r = AgentResult("X", "t", "success", claims=[{"claim": "a", "confidence": 0.5}],
                    risks=[{"risk": "r"}])
    assert isinstance(r.claims[0], Claim) and isinstance(r.risks[0], Risk)


# --- validation ------------------------------------------------------------------

@pytest.mark.parametrize("bad", [1.5, -0.1, float("nan"), "high", True, [0.5]])
def test_invalid_confidence_is_rejected_everywhere(bad):
    with pytest.raises(ValueError, match="confidence"):
        AgentResult("X", "t", "success", confidence=bad)
    with pytest.raises(ValueError, match="confidence"):
        Claim("c", confidence=bad)
    with pytest.raises(ValueError, match="confidence"):
        Evidence("e", confidence=bad)
    with pytest.raises(ValueError, match="confidence"):
        AgentResult("X", "t", "success", claims=[{"claim": "c", "confidence": bad}])


def test_confidence_boundaries_and_ints_are_accepted():
    assert validate_confidence(0, "x") == 0.0
    assert validate_confidence(1, "x") == 1.0
    assert validate_confidence(None, "x") is None


def test_invalid_status_and_missing_name_rejected():
    with pytest.raises(ValueError, match="status"):
        AgentResult("X", "t", "done")
    with pytest.raises(ValueError, match="agent_name"):
        AgentResult("", "t", "success")


def test_nested_lists_must_be_lists_of_the_right_type():
    with pytest.raises(ValueError, match="claims must be a list"):
        AgentResult("X", "t", "success", claims="nope")
    with pytest.raises(ValueError, match="Claim items"):
        AgentResult("X", "t", "success", claims=[Risk("r")])


# --- serialization ---------------------------------------------------------------

def test_json_round_trip_is_lossless():
    r = full_result()
    text = r.to_json()
    assert json.loads(text)["agent_name"] == "FinancialAgent"
    back = AgentResult.from_json(text)
    assert back == r
    assert back.to_json() == text


def test_serialization_is_strict_json_safe():
    r = AgentResult("X", "t", "success", metrics={
        "nan": float("nan"), "inf": math.inf, "np_float": np.float64(1.5), "np_int": np.int64(3),
        "when": date(2026, 1, 25), "path": Path("a/b"), "tuple": (1, 2), "nested": {1: {"x"}},
    })
    d = json.loads(r.to_json())["metrics"]
    assert d == {"nan": None, "inf": None, "np_float": 1.5, "np_int": 3, "when": "2026-01-25",
                 "path": "a/b", "tuple": [1, 2], "nested": {"1": ["x"]}}


def test_from_dict_ignores_unknown_keys():
    d = full_result().to_dict()
    d["added_in_a_future_version"] = 1
    assert AgentResult.from_dict(d).agent_name == "FinancialAgent"


def test_indented_json():
    assert "\n  " in full_result().to_json(indent=2)


# --- multiple results / failed result ------------------------------------------

def test_multiple_agent_results_serialize_together():
    results = [full_result(), AgentResult("RiskAgent", "t", "partial"),
               AgentResult("NewsAgent", "t", "failed", errors=["boom"])]
    doc = results_to_dict(results)
    assert doc["agent_count"] == 3 and doc["schema_version"] == SCHEMA_VERSION
    assert [r["agent_name"] for r in doc["results"]] == ["FinancialAgent", "RiskAgent", "NewsAgent"]
    back = results_from_json(results_to_json(results, indent=2))
    assert back == results


def test_failed_agent_result():
    r = AgentResult("RiskAgent", "t", "failed", summary="RiskAgent failed.",
                    errors=["Agent execution failed: vector store missing"])
    d = json.loads(r.to_json())
    assert d["status"] == "failed"
    assert d["errors"] == ["Agent execution failed: vector store missing"]
    assert d["claims"] == [] and d["confidence"] is None

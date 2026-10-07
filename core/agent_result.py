"""
Standard, machine-readable result contract for every agent.

All 8 agents (6 BaseAgent subclasses plus Red-Team and Synthesis) are wrapped
into this one structure by core/result_adapters.py, and serialized to JSON for
the downstream (local Qwen) synthesis layer.

This sits alongside core.schemas.AgentResult, which the agents still return
unchanged; that legacy object is the input to the adapters.

Every field except agent_name/task/status is optional: agents leave fields that
do not apply as None or empty. Confidence values must be in [0, 1].
"""

import json
import math
from dataclasses import asdict, dataclass, field, fields
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

SCHEMA_VERSION = "1.0"
VALID_STATUSES = ("success", "partial", "failed")


def validate_confidence(value: Any, where: str) -> Optional[float]:
    """None is allowed; otherwise a real number in [0, 1]."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: confidence must be a number in [0, 1], got {value!r}")
    value = float(value)
    if math.isnan(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{where}: confidence must be in [0, 1], got {value!r}")
    return value


def _jsonable(value: Any) -> Any:
    """Make a value strict-JSON safe: NaN/inf -> None, numpy scalars -> Python, dates -> ISO."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):  # numpy scalar
        return _jsonable(value.item())
    return str(value)


@dataclass
class Claim:
    claim: str
    value: Any = None
    unit: Optional[str] = None
    source: Optional[str] = None
    date: Optional[str] = None
    evidence: Optional[str] = None
    confidence: Optional[float] = None

    def __post_init__(self):
        self.confidence = validate_confidence(self.confidence, f"Claim {self.claim[:40]!r}")


@dataclass
class Calculation:
    metric: str
    value: Optional[float] = None
    unit: Optional[str] = None
    period: Optional[str] = None
    formula: Optional[str] = None
    source: Optional[str] = None


@dataclass
class Valuation:
    method: str                      # "DCF" | "P/E" | "EV/EBIT" ...
    scenario: Optional[str] = None
    enterprise_value: Optional[float] = None
    equity_value: Optional[float] = None
    implied_share_price: Optional[float] = None
    multiple: Optional[float] = None
    assumptions: Dict[str, Any] = field(default_factory=dict)
    sensitivity: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None


@dataclass
class Risk:
    risk: str
    severity: Optional[str] = None
    probability: Optional[str] = None
    impact: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)


@dataclass
class Evidence:
    text: str
    source_id: Optional[str] = None
    source_form: Optional[str] = None
    date: Optional[str] = None
    confidence: Optional[float] = None

    def __post_init__(self):
        self.confidence = validate_confidence(self.confidence, f"Evidence {self.source_id!r}")


@dataclass
class SourceReference:
    source_id: str
    source_form: Optional[str] = None
    date: Optional[str] = None


_NESTED = {
    "claims": Claim, "calculations": Calculation, "valuations": Valuation,
    "risks": Risk, "evidence": Evidence, "source_references": SourceReference,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class AgentResult:
    agent_name: str
    task: str
    status: str
    summary: str = ""
    findings: List[Dict[str, Any]] = field(default_factory=list)  # original findings, lossless
    metrics: Dict[str, Any] = field(default_factory=dict)
    calculations: List[Calculation] = field(default_factory=list)
    claims: List[Claim] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    valuations: List[Valuation] = field(default_factory=list)
    risks: List[Risk] = field(default_factory=list)
    assumptions: Dict[str, Any] = field(default_factory=dict)
    uncertainties: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    source_references: List[SourceReference] = field(default_factory=list)
    timestamp: str = field(default_factory=_utc_now)
    errors: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)  # agent-specific extras
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        if not self.agent_name:
            raise ValueError("agent_name is required")
        if self.status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {VALID_STATUSES}, got {self.status!r}")
        self.confidence = validate_confidence(self.confidence, f"AgentResult {self.agent_name}")
        for name, cls in _NESTED.items():
            items = getattr(self, name)
            if not isinstance(items, list):
                raise ValueError(f"{name} must be a list")
            setattr(self, name, [cls(**i) if isinstance(i, dict) else i for i in items])
            bad = [i for i in getattr(self, name) if not isinstance(i, cls)]
            if bad:
                raise ValueError(f"{name} must contain {cls.__name__} items, got {type(bad[0]).__name__}")

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentResult":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_json(cls, text: str) -> "AgentResult":
        return cls.from_dict(json.loads(text))


def results_to_dict(results: Iterable[AgentResult]) -> Dict[str, Any]:
    results = list(results)
    return {
        "schema_version": SCHEMA_VERSION,
        "agent_count": len(results),
        "results": [r.to_dict() for r in results],
    }


def results_to_json(results: Iterable[AgentResult], indent: Optional[int] = None) -> str:
    """Serialize several agent results as one JSON document."""
    return json.dumps(results_to_dict(results), indent=indent, allow_nan=False)


def results_from_json(text: str) -> List[AgentResult]:
    return [AgentResult.from_dict(r) for r in json.loads(text)["results"]]

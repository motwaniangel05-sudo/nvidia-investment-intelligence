"""
NewsAgent: classifies sentences from 8-K event filings into
FACT / CLAIM / INTERPRETATION / UNCERTAINTY using the classifier trained
in Phase 10 (models/classification/train_classifier.py).

KNOWN LIMITATION (see Phase 10 training results): the CLAIM class has
only 5 training examples and achieved 0% precision/recall in cross-
validation. Any CLAIM prediction from this model should be treated with
substantial skepticism -- this is disclosed explicitly in every result
this agent produces, not hidden.
"""

import pickle
from pathlib import Path
from typing import List

from core.config_loader import get_path, load_config
from core.logger import get_logger
from core.schemas import AgentResult, Finding
from agents.base_agent import BaseAgent
from models.classification.labeling_tools import (
    is_labelable_sentence,
    split_into_sentences,
)

log = get_logger(__name__)

# Per Phase 10 cross-validation results: CLAIM predictions are backed by
# only 5 training examples (0% precision/recall observed) and should
# never be presented with confidence.
UNRELIABLE_LABELS = {"CLAIM"}


class NewsAgent(BaseAgent):
    agent_name = "NewsAgent"

    def __init__(self, company_ticker: str):
        super().__init__(company_ticker)
        self._classifier = None

    def retrieve(self, query: str, top_k: int = 5):
        """Not used -- NewsAgent classifies pre-chunked 8-K text directly."""
        return []

    def _load_classifier(self) -> dict:
        if self._classifier is None:
            config = load_config()
            path = get_path(config, "models") / "classification" / "event_classifier.pkl"
            if not path.exists():
                raise FileNotFoundError(
                    f"Classifier not found at {path}. "
                    "Run `python -m models.classification.train_classifier` first."
                )
            with open(path, "rb") as f:
                self._classifier = pickle.load(f)
        return self._classifier

    def _classify_sentence(self, sentence: str) -> dict:
        clf = self._load_classifier()
        vec = clf["vectorizer"].transform([sentence])
        pred = clf["model"].predict(vec)[0]
        proba = clf["model"].predict_proba(vec)[0]
        confidence = float(max(proba))
        return {"label": pred, "confidence": confidence}

    def analyze(self, task: str) -> AgentResult:
        """
        task is currently informational only. This agent processes all
        available 8-K chunks for the configured company (a future
        refinement could filter to a specific date range via task text).
        """
        config = load_config()
        chunks_path = get_path(config, "data_processed") / f"chunks_{self.company_ticker}_8K.csv"

        if not chunks_path.exists():
            return AgentResult(
                agent_name=self.agent_name, task=task, status="failed",
                warnings=[f"No 8-K chunks found at {chunks_path}."],
            )

        import pandas as pd
        chunks_df = pd.read_csv(chunks_path)

        findings = []
        warnings = [
            "CLAIM classification is unreliable (0% precision/recall in training "
            "evaluation due to only 5 training examples) -- CLAIM predictions "
            "below are flagged individually but should not be trusted."
        ]
        label_counts = {}

        for _, row in chunks_df.iterrows():
            sentences = split_into_sentences(row["text"])
            for sentence in sentences:
                if not is_labelable_sentence(sentence):
                    continue

                result = self._classify_sentence(sentence)
                label = result["label"]
                confidence = result["confidence"]
                label_counts[label] = label_counts.get(label, 0) + 1

                claim_text = f"Classified as {label}"
                if label in UNRELIABLE_LABELS:
                    claim_text += " (LOW RELIABILITY -- see agent warnings)"

                findings.append(Finding(
                    claim=claim_text,
                    evidence_text=sentence,
                    source_chunk_id=row["chunk_id"],
                    source_form="8-K",
                    source_filing_date=row["filing_date"],
                    confidence=round(confidence, 4),
                ))

        status = "success" if findings else "partial"
        return AgentResult(
            agent_name=self.agent_name,
            task=task,
            findings=findings,
            metrics={"sentences_classified": len(findings), "label_counts": label_counts},
            warnings=warnings,
            status=status,
        )


if __name__ == "__main__":
    agent = NewsAgent(company_ticker="NVDA")
    result = agent.run("Classify NVIDIA 8-K event statements")

    print(f"Status: {result.status}")
    print(f"Metrics: {result.metrics}")
    print(f"Warnings: {result.warnings}")
    print(f"\nTotal findings: {len(result.findings)}\n")

    print("Sample findings (first 10):")
    for f in result.findings[:10]:
        print(f"[{f.claim}] (confidence={f.confidence})")
        print(f"  {f.evidence_text[:150]}")
        print()

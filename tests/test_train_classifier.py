"""Tests for classifier training pipeline mechanics (not re-validating the
real model's documented accuracy -- see Phase 10 notes for honest results)."""

import pickle

import pandas as pd
import pytest

from models.classification.train_classifier import (
    LABELS,
    save_classifier,
    train_and_evaluate,
)


def fake_labeled_df(n_per_class=10):
    """Small synthetic dataset: distinct vocabulary per class so the
    model has something learnable to test the pipeline mechanics with."""
    rows = []
    templates = {
        "FACT": "revenue was {} million dollars in the quarter",
        "CLAIM": "we believe our product will transform the industry",
        "INTERPRETATION": "the increase was primarily due to strong demand",
        "UNCERTAINTY": "results may differ materially due to risks and uncertainty",
    }
    for label, template in templates.items():
        for i in range(n_per_class):
            rows.append({"sentence": template.format(i), "label": label})
    return pd.DataFrame(rows)


def test_train_and_evaluate_produces_model_and_vectorizer():
    df = fake_labeled_df(n_per_class=10)
    results = train_and_evaluate(df, n_folds=5)

    assert results["model"] is not None
    assert results["vectorizer"] is not None
    assert results["n_examples"] == 40


def test_train_and_evaluate_report_covers_all_labels():
    df = fake_labeled_df(n_per_class=10)
    results = train_and_evaluate(df, n_folds=5)

    for label in LABELS:
        assert label in results["report_dict"]


def test_train_and_evaluate_reduces_folds_for_small_class():
    df = fake_labeled_df(n_per_class=10)
    # Shrink one class down to 3 examples
    small_class_df = df[df["label"] == "CLAIM"].head(3)
    other_df = df[df["label"] != "CLAIM"]
    df_imbalanced = pd.concat([other_df, small_class_df], ignore_index=True)

    results = train_and_evaluate(df_imbalanced, n_folds=5)
    assert results["n_folds_used"] == 3  # reduced to match smallest class


def test_train_and_evaluate_raises_with_too_few_examples_in_a_class():
    df = fake_labeled_df(n_per_class=10)
    tiny_class_df = df[df["label"] == "CLAIM"].head(1)
    other_df = df[df["label"] != "CLAIM"]
    df_tiny = pd.concat([other_df, tiny_class_df], ignore_index=True)

    with pytest.raises(ValueError):
        train_and_evaluate(df_tiny, n_folds=5)


def test_save_classifier_round_trips(tmp_path):
    df = fake_labeled_df(n_per_class=10)
    results = train_and_evaluate(df, n_folds=5)

    path = tmp_path / "test_classifier.pkl"
    save_classifier(results["model"], results["vectorizer"], path)

    assert path.exists()
    with open(path, "rb") as f:
        loaded = pickle.load(f)

    assert loaded["labels"] == LABELS
    assert loaded["model"] is not None
    assert loaded["vectorizer"] is not None


def test_trained_model_predicts_a_valid_label():
    df = fake_labeled_df(n_per_class=10)
    results = train_and_evaluate(df, n_folds=5)

    sample_vec = results["vectorizer"].transform(["revenue was 500 million dollars"])
    prediction = results["model"].predict(sample_vec)[0]
    assert prediction in LABELS

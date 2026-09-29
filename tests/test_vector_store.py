"""Tests for TF-IDF vector store: tokenization, TF, IDF, matrix construction."""

import math

import numpy as np
import pytest

from rag.vector_store import (
    build_tfidf_matrix,
    build_vocabulary,
    compute_idf,
    compute_tf,
    tokenize,
)


def test_tokenize_lowercases_and_splits():
    result = tokenize("NVIDIA's Revenue Growth!")
    assert "nvidia" in result
    assert "revenue" in result
    assert "growth" in result


def test_tokenize_removes_stop_words():
    result = tokenize("The revenue and the growth of the company")
    assert "the" not in result
    assert "and" not in result
    assert "of" not in result
    assert "revenue" in result
    assert "growth" in result
    assert "company" in result


def test_tokenize_removes_punctuation_and_numbers():
    result = tokenize("Revenue grew 25% in Q3, 2024.")
    assert "25" not in result
    assert "2024" not in result
    assert "revenue" in result
    assert "grew" in result


def test_tokenize_empty_string_returns_empty_list():
    assert tokenize("") == []
    assert tokenize("the and of") == []  # all stop words


def test_build_vocabulary_assigns_unique_indices():
    tokenized = [["revenue", "growth"], ["revenue", "margin"]]
    vocab = build_vocabulary(tokenized)
    assert set(vocab.keys()) == {"revenue", "growth", "margin"}
    assert len(set(vocab.values())) == 3  # all indices unique


def test_compute_tf_matches_hand_calculation():
    # Chunk: "revenue revenue growth" -> revenue: 2/3, growth: 1/3
    tokens = ["revenue", "revenue", "growth"]
    vocab = {"revenue": 0, "growth": 1, "margin": 2}
    tf = compute_tf(tokens, vocab)

    assert tf[0] == pytest.approx(2 / 3)  # revenue
    assert tf[1] == pytest.approx(1 / 3)  # growth
    assert 2 not in tf  # margin never appears, not in the sparse dict


def test_compute_tf_empty_tokens_returns_empty_dict():
    vocab = {"revenue": 0}
    assert compute_tf([], vocab) == {}


def test_compute_idf_matches_hand_calculation():
    tokenized_chunks = [
        ["revenue", "revenue", "growth"],   # chunk 0
        ["revenue", "margin"],               # chunk 1
        ["supply", "chain", "risk"],         # chunk 2
    ]
    vocab = build_vocabulary(tokenized_chunks)
    idf = compute_idf(tokenized_chunks, vocab)

    # "revenue" appears in 2 of 3 chunks -> log(3 / (1+2)) = log(1) = 0
    assert idf[vocab["revenue"]] == pytest.approx(math.log(3 / 3), abs=1e-9)

    # "supply" appears in 1 of 3 chunks -> log(3 / (1+1)) = log(1.5)
    assert idf[vocab["supply"]] == pytest.approx(math.log(1.5), abs=1e-9)

    # A rarer word should have strictly higher IDF than a common one
    assert idf[vocab["supply"]] > idf[vocab["revenue"]]


def test_build_tfidf_matrix_shape_matches_corpus():
    tokenized_chunks = [
        ["revenue", "revenue", "growth"],
        ["revenue", "margin"],
        ["supply", "chain", "risk"],
    ]
    vocab = build_vocabulary(tokenized_chunks)
    idf = compute_idf(tokenized_chunks, vocab)
    matrix = build_tfidf_matrix(tokenized_chunks, vocab, idf)

    assert matrix.shape == (3, len(vocab))


def test_build_tfidf_matrix_common_word_scores_zero():
    # "revenue" appears in 2/3 chunks -> idf = log(1) = 0
    # so its tfidf score should be exactly 0 everywhere, regardless of TF
    tokenized_chunks = [
        ["revenue", "revenue", "growth"],
        ["revenue", "margin"],
        ["supply", "chain", "risk"],
    ]
    vocab = build_vocabulary(tokenized_chunks)
    idf = compute_idf(tokenized_chunks, vocab)
    matrix = build_tfidf_matrix(tokenized_chunks, vocab, idf)

    revenue_col = vocab["revenue"]
    revenue_scores = matrix[:, revenue_col].toarray().flatten()
    assert np.allclose(revenue_scores, 0.0)


def test_build_tfidf_matrix_rare_word_scores_nonzero():
    tokenized_chunks = [
        ["revenue", "revenue", "growth"],
        ["revenue", "margin"],
        ["supply", "chain", "risk"],
    ]
    vocab = build_vocabulary(tokenized_chunks)
    idf = compute_idf(tokenized_chunks, vocab)
    matrix = build_tfidf_matrix(tokenized_chunks, vocab, idf)

    supply_col = vocab["supply"]
    supply_scores = matrix[:, supply_col].toarray().flatten()
    # Only chunk 2 contains "supply" -- its score should be > 0,
    # and chunks 0/1 (which don't contain it) should be exactly 0
    assert supply_scores[2] > 0
    assert supply_scores[0] == 0
    assert supply_scores[1] == 0

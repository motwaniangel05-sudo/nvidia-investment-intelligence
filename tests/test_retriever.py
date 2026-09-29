"""Tests for TF-IDF retrieval: query vectorization, cosine similarity, ranking."""

import numpy as np
import pytest
import scipy.sparse as sp

from rag.retriever import cosine_similarity, vectorize_query


def test_vectorize_query_ignores_unknown_words():
    vocab = {"revenue": 0, "growth": 1}
    idf = np.array([0.5, 0.8])
    vec = vectorize_query("revenue skyrocketed", vocab, idf)
    assert vec.shape == (1, 2)
    assert vec[0, 0] > 0   # "revenue" known
    # "skyrocketed" unknown -> silently ignored, no crash


def test_vectorize_query_empty_returns_zero_vector():
    vocab = {"revenue": 0}
    idf = np.array([0.5])
    vec = vectorize_query("", vocab, idf)
    assert vec.nnz == 0


def test_vectorize_query_all_unknown_words_returns_zero_vector():
    vocab = {"revenue": 0}
    idf = np.array([0.5])
    vec = vectorize_query("zzzznotinvocab", vocab, idf)
    assert vec.nnz == 0


def test_cosine_similarity_identical_vectors_equals_one():
    matrix = sp.csr_matrix(np.array([[1.0, 2.0, 0.0]]))
    query = sp.csr_matrix(np.array([[1.0, 2.0, 0.0]]))
    sims = cosine_similarity(query, matrix)
    assert sims[0] == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_equals_zero():
    matrix = sp.csr_matrix(np.array([[1.0, 0.0]]))
    query = sp.csr_matrix(np.array([[0.0, 1.0]]))
    sims = cosine_similarity(query, matrix)
    assert sims[0] == pytest.approx(0.0)


def test_cosine_similarity_ranks_more_similar_higher():
    # Chunk 0 close to query, chunk 1 unrelated
    matrix = sp.csr_matrix(np.array([
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]))
    query = sp.csr_matrix(np.array([[1.0, 1.0, 0.0]]))
    sims = cosine_similarity(query, matrix)
    assert sims[0] > sims[1]


def test_cosine_similarity_zero_vector_does_not_crash():
    matrix = sp.csr_matrix(np.array([[0.0, 0.0]]))
    query = sp.csr_matrix(np.array([[1.0, 1.0]]))
    sims = cosine_similarity(query, matrix)
    assert sims[0] == 0.0  # no division-by-zero error

"""
Query the TF-IDF vector store built in vector_store.py.

Pipeline: query text -> tokenize -> vectorize using the SAME vocabulary/IDF
as the corpus -> cosine similarity against every chunk -> top-K results.
"""

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import scipy.sparse as sp

from core.config_loader import get_path, load_config
from core.logger import get_logger
from core.schemas import get_connection
from rag.vector_store import tokenize

log = get_logger(__name__)


class RetrieverError(Exception):
    """Raised when the vector store cannot be loaded or queried."""


def load_vector_store(path: Path = None) -> dict:
    """Load the pickled TF-IDF store (chunk_ids, vocab, idf, matrix)."""
    if path is None:
        config = load_config()
        path = get_path(config, "models") / "embeddings" / "tfidf_store.pkl"

    if not Path(path).exists():
        raise RetrieverError(
            f"Vector store not found at {path}. Run `python -m rag.vector_store` first."
        )

    with open(path, "rb") as f:
        return pickle.load(f)


def vectorize_query(query: str, vocab: Dict[str, int], idf: np.ndarray) -> sp.csr_matrix:
    """
    Turn a query string into a TF-IDF vector using the corpus's existing
    vocabulary and IDF values (NOT recomputed -- the query must live in
    the same vector space as the chunks to be comparable).
    Words in the query not seen in the corpus vocabulary are ignored.
    """
    tokens = tokenize(query)
    if not tokens:
        return sp.csr_matrix((1, len(vocab)))

    counts: Dict[int, int] = {}
    for token in tokens:
        if token in vocab:
            idx = vocab[token]
            counts[idx] = counts.get(idx, 0) + 1

    total = len(tokens)
    cols = list(counts.keys())
    values = [(count / total) * idf[col] for col, count in counts.items()]
    rows = [0] * len(cols)

    return sp.csr_matrix((values, (rows, cols)), shape=(1, len(vocab)))


def cosine_similarity(query_vec: sp.csr_matrix, matrix: sp.csr_matrix) -> np.ndarray:
    """
    Cosine similarity between the query vector and every chunk row.
    cos(theta) = (A . B) / (|A| * |B|)
    Returns a 1D array of similarity scores, one per chunk.
    """
    dot_products = matrix.dot(query_vec.T).toarray().flatten()

    query_norm = np.sqrt(query_vec.multiply(query_vec).sum())
    chunk_norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A.flatten()

    denom = chunk_norms * query_norm
    # Avoid division by zero for empty chunks/queries
    similarities = np.divide(
        dot_products, denom,
        out=np.zeros_like(dot_products), where=denom != 0,
    )
    return similarities


def retrieve(query: str, top_k: int = 5, store: dict = None) -> List[dict]:
    """
    Main entry: search the vector store for the top_k chunks most similar
    to `query`. Returns a list of dicts with chunk_id, score, and text
    (text fetched fresh from the knowledge base, not cached in the pickle,
    so it always reflects the current database).
    """
    store = store or load_vector_store()
    chunk_ids = store["chunk_ids"]
    vocab = store["vocab"]
    idf = store["idf"]
    matrix = store["matrix"]

    query_vec = vectorize_query(query, vocab, idf)
    scores = cosine_similarity(query_vec, matrix)

    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    conn = get_connection()
    try:
        for idx in ranked_indices:
            if scores[idx] <= 0:
                continue
            chunk_id = chunk_ids[idx]
            row = conn.execute(
                "SELECT c.text, c.document_id, d.form, d.filing_date "
                "FROM chunks c JOIN documents d ON c.document_id = d.document_id "
                "WHERE c.chunk_id = ?", (chunk_id,)
            ).fetchone()
            if row is None:
                continue
            results.append({
                "chunk_id": chunk_id,
                "score": float(scores[idx]),
                "form": row["form"],
                "filing_date": row["filing_date"],
                "text": row["text"],
            })
    finally:
        conn.close()

    return results


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "What was NVIDIA's revenue?"
    print(f"Query: {query}\n")

    results = retrieve(query, top_k=5)
    if not results:
        print("No relevant results found.")
    for i, r in enumerate(results, 1):
        print(f"[{i}] score={r['score']:.4f} | {r['form']} filed {r['filing_date']} | {r['chunk_id']}")
        print(f"    {r['text'][:200]}...")
        print()

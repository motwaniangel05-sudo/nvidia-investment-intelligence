"""
TF-IDF vector store, built from scratch with numpy/scipy.

Implements the core RAG retrieval math ourselves (per project constraints:
no external embedding APIs, no pretrained LLMs). This module builds the
TF-IDF matrix; rag/retriever.py (Part 2) uses it to answer queries.

TF-IDF recap:
  TF(word, chunk)  = how often `word` appears in `chunk`, normalized by
                      chunk length (so long chunks don't dominate purely
                      by having more words)
  IDF(word)        = log(total_chunks / chunks_containing_word) -- rare
                      words across the corpus get a higher score
  TF-IDF(word, chunk) = TF(word, chunk) * IDF(word)

Each chunk becomes a vector (one dimension per vocabulary word), stored
as a sparse matrix since the vast majority of entries are zero.
"""

import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp

from core.config_loader import get_path, load_config
from core.logger import get_logger
from core.schemas import get_connection

log = get_logger(__name__)

# A minimal, standard English stop word list. These carry little topical
# meaning (they appear in almost every document) and would otherwise
# receive near-zero IDF anyway -- removing them upfront saves computation.
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "has", "have", "in", "is", "it", "its", "of", "on", "or", "our", "such",
    "that", "the", "their", "this", "to", "was", "were", "will", "with",
    "we", "us", "these", "those", "which", "who", "may", "can", "could",
    "would", "should", "not", "no", "if", "than", "then", "also", "each",
}

TOKEN_PATTERN = re.compile(r"[a-z]{2,}")  # words of 2+ letters, lowercase


def tokenize(text: str) -> List[str]:
    """
    Split text into lowercase word tokens, dropping punctuation, numbers,
    and stop words. Single-letter tokens are dropped too (mostly noise
    from stray characters).
    """
    text = text.lower()
    tokens = TOKEN_PATTERN.findall(text)
    return [t for t in tokens if t not in STOP_WORDS]


def build_vocabulary(tokenized_chunks: List[List[str]]) -> Dict[str, int]:
    """
    Build a {word: column_index} mapping across the whole corpus.
    Every unique word gets one fixed position in the TF-IDF matrix.
    """
    vocab: Dict[str, int] = {}
    for tokens in tokenized_chunks:
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def compute_tf(tokens: List[str], vocab: Dict[str, int]) -> Dict[int, float]:
    """
    Term frequency for one chunk: count of each word divided by total
    word count in that chunk (so a 100-word chunk mentioning 'revenue'
    twice and a 1000-word chunk mentioning it twice are NOT scored the
    same -- the shorter chunk's mention is proportionally more significant).
    Returns {vocab_index: tf_value}, sparse by construction (only words
    that actually appear).
    """
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {
        vocab[word]: count / total
        for word, count in counts.items()
        if word in vocab
    }


def compute_idf(tokenized_chunks: List[List[str]], vocab: Dict[str, int]) -> np.ndarray:
    """
    Inverse document frequency for every word in the vocabulary:
    idf(word) = log(N / (1 + df(word)))
    where N = total chunks, df(word) = number of chunks containing word.
    The "+1" avoids division by zero and slightly dampens extremely rare
    words (standard smoothing, same idea scikit-learn uses by default).
    """
    n_chunks = len(tokenized_chunks)
    doc_freq = np.zeros(len(vocab))

    for tokens in tokenized_chunks:
        unique_words = set(tokens)
        for word in unique_words:
            if word in vocab:
                doc_freq[vocab[word]] += 1

    idf = np.log(n_chunks / (1 + doc_freq))
    return idf


def build_tfidf_matrix(
    tokenized_chunks: List[List[str]],
    vocab: Dict[str, int],
    idf: np.ndarray,
) -> sp.csr_matrix:
    """
    Combine TF and IDF into the full sparse TF-IDF matrix:
    one row per chunk, one column per vocabulary word.
    """
    n_chunks = len(tokenized_chunks)
    n_vocab = len(vocab)

    rows, cols, values = [], [], []
    for row_idx, tokens in enumerate(tokenized_chunks):
        tf = compute_tf(tokens, vocab)
        for col_idx, tf_value in tf.items():
            rows.append(row_idx)
            cols.append(col_idx)
            values.append(tf_value * idf[col_idx])

    matrix = sp.csr_matrix((values, (rows, cols)), shape=(n_chunks, n_vocab))
    return matrix


def load_chunks_from_db() -> Tuple[List[str], List[str]]:
    """Read all chunks from the knowledge base. Returns (chunk_ids, texts)."""
    conn = get_connection()
    try:
        cursor = conn.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id")
        rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        raise ValueError(
            "No chunks found in the knowledge base. Run `python -m core.memory` first."
        )

    chunk_ids = [r["chunk_id"] for r in rows]
    texts = [r["text"] for r in rows]
    return chunk_ids, texts


def build_and_save_vector_store() -> dict:
    """
    Main entry: load chunks, tokenize, build vocabulary/IDF/TF-IDF matrix,
    and persist everything to disk so retriever.py can load it instantly
    without recomputing on every query.
    """
    config = load_config()
    chunk_ids, texts = load_chunks_from_db()

    log.info("Tokenizing %d chunks...", len(texts))
    tokenized_chunks = [tokenize(t) for t in texts]

    log.info("Building vocabulary...")
    vocab = build_vocabulary(tokenized_chunks)
    log.info("Vocabulary size: %d unique words", len(vocab))

    log.info("Computing IDF...")
    idf = compute_idf(tokenized_chunks, vocab)

    log.info("Building TF-IDF matrix...")
    matrix = build_tfidf_matrix(tokenized_chunks, vocab, idf)

    out_dir = get_path(config, "models") / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "tfidf_store.pkl"

    with open(out_file, "wb") as f:
        pickle.dump({
            "chunk_ids": chunk_ids,
            "vocab": vocab,
            "idf": idf,
            "matrix": matrix,
        }, f)

    log.info("Saved TF-IDF store to %s (%d chunks x %d vocab)",
              out_file, matrix.shape[0], matrix.shape[1])

    return {
        "n_chunks": matrix.shape[0],
        "vocab_size": matrix.shape[1],
        "matrix_nnz": matrix.nnz,  # number of non-zero entries
        "output_path": str(out_file),
    }


if __name__ == "__main__":
    result = build_and_save_vector_store()
    print(f"Chunks indexed: {result['n_chunks']}")
    print(f"Vocabulary size: {result['vocab_size']}")
    print(f"Non-zero entries: {result['matrix_nnz']} "
          f"({100 * result['matrix_nnz'] / (result['n_chunks'] * result['vocab_size']):.2f}% dense)")
    print(f"Saved to: {result['output_path']}")

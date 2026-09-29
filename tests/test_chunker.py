"""Tests for chunking: paragraph merging, splitting, overlap, metadata."""

from pathlib import Path

from rag.chunker import (
    Chunk,
    chunk_document,
    merge_and_split,
    split_into_paragraphs,
    split_long_paragraph,
)


def make_words(n: int, prefix: str = "word") -> str:
    """Helper: build a string of n space-separated words."""
    return " ".join(f"{prefix}{i}" for i in range(n))


def make_sentence(n_words: int) -> str:
    """Helper: build a single 'sentence' of n words ending in a period."""
    return make_words(n_words) + "."


def test_split_into_paragraphs_basic():
    text = "Para one.\n\nPara two.\n\nPara three."
    result = split_into_paragraphs(text)
    assert result == ["Para one.", "Para two.", "Para three."]


def test_split_into_paragraphs_ignores_empty_blocks():
    text = "Para one.\n\n\n\nPara two."
    result = split_into_paragraphs(text)
    assert result == ["Para one.", "Para two."]


def test_split_long_paragraph_respects_sentence_boundaries():
    # 3 sentences of 200 words each = 600 words total; max 300 per piece
    para = " ".join([make_sentence(200) for _ in range(3)])
    pieces = split_long_paragraph(para, max_words=300)
    assert len(pieces) >= 2
    for piece in pieces:
        assert len(piece.split()) <= 300 + 5  # small tolerance


def test_split_long_paragraph_hard_splits_a_single_giant_sentence():
    # One "sentence" (no periods) of 500 words, max 100 per piece
    giant = make_words(500)
    pieces = split_long_paragraph(giant, max_words=100)
    assert len(pieces) == 5
    for piece in pieces:
        assert len(piece.split()) == 100


def test_merge_and_split_merges_small_paragraphs_toward_target():
    # 10 paragraphs of 50 words each = 500 words; target 800
    # Should all merge into a single chunk since 500 < 800
    paragraphs = [make_sentence(50) for _ in range(10)]
    chunks = merge_and_split(paragraphs, target_words=800, overlap_words=100)
    assert len(chunks) == 1
    assert len(chunks[0].split()) >= 490  # roughly all 500 words present


def test_merge_and_split_creates_multiple_chunks_when_exceeding_target():
    # 5 paragraphs of 300 words = 1500 words; target 800 -> at least 2 chunks
    paragraphs = [make_sentence(300) for _ in range(5)]
    chunks = merge_and_split(paragraphs, target_words=800, overlap_words=100)
    assert len(chunks) >= 2


def test_merge_and_split_produces_overlap_between_chunks():
    paragraphs = [make_sentence(300) for _ in range(5)]
    overlap_words = 100
    chunks = merge_and_split(paragraphs, target_words=800, overlap_words=overlap_words)

    # The exact overlap_words-sized tail of chunk 0 should equal the exact
    # overlap_words-sized head of chunk 1, since chunk 1 is seeded with
    # chunk 0's tail before any new paragraph content is appended.
    tail_of_first = chunks[0].split()[-overlap_words:]
    head_of_second = chunks[1].split()[:overlap_words]
    assert tail_of_first == head_of_second


def test_merge_and_split_tiny_leftover_merges_into_previous_chunk():
    # 3 paragraphs of 300 words (fills a chunk near target) + 1 tiny paragraph
    paragraphs = [make_sentence(300) for _ in range(3)] + [make_sentence(10)]
    chunks = merge_and_split(paragraphs, target_words=800, overlap_words=50)
    # The tiny 10-word paragraph should not exist as its own separate chunk
    tiny_alone = [c for c in chunks if len(c.split()) < 50]
    assert tiny_alone == []


def test_chunk_document_full_pipeline(tmp_path):
    html = f"""
    <html><body>
        <p>{make_sentence(50)}</p>
        <p>{make_sentence(50)}</p>
        <p>Page 3</p>
        <p>{make_sentence(900)}</p>
    </body></html>
    """
    f = tmp_path / "NVDA_10-K_test_sample.htm"
    f.write_text(html, encoding="utf-8")

    chunks = chunk_document(f, company_ticker="NVDA", form="10-K", filing_date="2016-03-17")

    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.company_ticker == "NVDA" for c in chunks)
    assert all(c.form == "10-K" for c in chunks)
    assert all(c.filing_date == "2016-03-17" for c in chunks)
    # chunk_index should be sequential starting at 0
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # chunk_id should be deterministic and include key metadata
    assert chunks[0].chunk_id == "NVDA_10-K_2016-03-17_0000"


def test_chunk_document_missing_file_returns_empty_list():
    result = chunk_document(
        Path("does_not_exist.htm"),
        company_ticker="NVDA", form="10-K", filing_date="2016-03-17",
    )
    assert result == []


def test_chunk_document_empty_html_returns_empty_list(tmp_path):
    f = tmp_path / "empty.htm"
    f.write_text("<html><body></body></html>", encoding="utf-8")
    result = chunk_document(f, company_ticker="NVDA", form="10-K", filing_date="2016-01-01")
    assert result == []

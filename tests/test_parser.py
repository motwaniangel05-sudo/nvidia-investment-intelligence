"""Tests for HTML filing text extraction and cleaning."""

import pytest
from bs4 import BeautifulSoup

from rag.parser import (
    ParsingError,
    clean_text,
    extract_paragraphs,
    extract_text,
    is_boilerplate,
    strip_unwanted_tags,
)


def test_strip_unwanted_tags_removes_script_and_style():
    html = "<html><script>alert(1)</script><style>.x{color:red}</style><p>Real text</p></html>"
    soup = BeautifulSoup(html, "lxml")
    soup = strip_unwanted_tags(soup)
    text = soup.get_text()
    assert "alert" not in text
    assert "color:red" not in text
    assert "Real text" in text


def test_strip_unwanted_tags_removes_tables():
    html = "<html><table><tr><td>123</td><td>456</td></tr></table><p>Narrative text</p></html>"
    soup = BeautifulSoup(html, "lxml")
    soup = strip_unwanted_tags(soup)
    text = soup.get_text()
    assert "123" not in text
    assert "Narrative text" in text


def test_extract_paragraphs_preserves_structure():
    html = "<div><p>First paragraph.</p><p>Second paragraph.</p></div>"
    soup = BeautifulSoup(html, "lxml")
    paragraphs = extract_paragraphs(soup)
    assert "First paragraph." in paragraphs
    assert "Second paragraph." in paragraphs


def test_is_boilerplate_flags_short_fragments():
    assert is_boilerplate("12") is True
    assert is_boilerplate("Page 42") is True
    assert is_boilerplate("...........") is True
    assert is_boilerplate("- - -") is True


def test_is_boilerplate_allows_real_content():
    real = "NVIDIA designs graphics processing units for various markets."
    assert is_boilerplate(real) is False


def test_clean_text_removes_checkbox_glyphs():
    assert clean_text("Yes ý No o") == "Yes No"
    assert clean_text("Yes o No ý") == "Yes No"


def test_clean_text_does_not_affect_normal_words():
    text = "This is normal text with the word yesterday in it."
    assert clean_text(text) == text
    text2 = "No changes to report."
    assert clean_text(text2) == text2


def test_clean_text_collapses_whitespace():
    messy = "Too    many   spaces"
    assert clean_text(messy) == "Too many spaces"


def test_clean_text_collapses_excess_newlines():
    messy = "Para one\n\n\n\n\nPara two"
    assert clean_text(messy) == "Para one\n\nPara two"


def test_extract_text_full_pipeline(tmp_path):
    html = """
    <html>
        <head><title>ignored</title></head>
        <body>
            <script>var x = 1;</script>
            <p>NVIDIA Corporation designs GPUs.</p>
            <table><tr><td>999999</td></tr></table>
            <p>Page 12</p>
            <p>The company reported strong revenue growth this year.</p>
        </body>
    </html>
    """
    f = tmp_path / "sample.htm"
    f.write_text(html, encoding="utf-8")

    result = extract_text(f)
    assert "NVIDIA Corporation designs GPUs." in result
    assert "999999" not in result
    assert "Page 12" not in result
    assert "revenue growth" in result


def test_extract_text_missing_file_raises():
    with pytest.raises(ParsingError):
        extract_text("does_not_exist.htm")


def test_extract_text_empty_html_returns_empty_string(tmp_path):
    f = tmp_path / "empty.htm"
    f.write_text("<html><body></body></html>", encoding="utf-8")
    result = extract_text(f)
    assert result == ""

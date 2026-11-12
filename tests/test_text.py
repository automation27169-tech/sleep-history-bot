"""Tests for text utilities."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.text import (
    slugify, chunk_text, split_sentences, count_words,
    similarity, estimate_duration, format_srt_time, wrap_text
)


def test_slugify():
    assert slugify("Hello World!") == "hello-world"
    assert slugify("Test@#$%") == "test"
    assert slugify("A" * 100) == "a" * 60  # max_len


def test_chunk_text():
    text = "Para1\n\nPara2\n\nPara3"
    chunks = chunk_text(text, max_chars=20)
    assert len(chunks) >= 1


def test_split_sentences():
    text = "First sentence. Second sentence! Third?"
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0].text == "First sentence."
    assert sentences[0].word_start == 0
    assert sentences[0].word_end == 2


def test_count_words():
    assert count_words("Hello world") == 2
    assert count_words("") == 0


def test_similarity():
    assert similarity("hello", "hello") == 1.0
    assert similarity("hello", "world") < 0.5
    assert 0 <= similarity("test", "test") <= 1


def test_estimate_duration():
    assert estimate_duration(140, 140) == 1.0  # 1 min
    assert estimate_duration(700, 140) == 5.0  # 5 min


def test_format_srt_time():
    assert format_srt_time(0) == "00:00:00,000"
    assert format_srt_time(61.5) == "00:01:01,500"
    assert format_srt_time(3661) == "01:01:01,000"


def test_wrap_text():
    text = "This is a long sentence that should be wrapped to multiple lines"
    wrapped = wrap_text(text, width=20)
    lines = wrapped.split('\n')
    assert all(len(line) <= 20 for line in lines)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])

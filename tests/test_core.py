from rag_system.core import Document, chunks, content_hash, unique_documents, metrics, percentile
import pytest


def test_invalid_document():
    with pytest.raises(ValueError):
        Document.from_dict({"id": "a", "text": " "})


def test_chunks_preserve_offsets():
    doc = Document("a", "one two three four")
    tokenizer = lambda *a, **kw: {"offset_mapping": [(0,3),(4,7),(8,13),(14,18)]}
    result = list(chunks(doc, tokenizer, 3, 1))
    assert [r["text"] for r in result] == ["one two three", "three four"]
    assert result == list(chunks(doc, tokenizer, 3, 1))


def test_metrics():
    result = metrics(["x", "a", "a"], {"a": 1, "b": 1}, 3)
    assert result["recall"] == .5
    assert result["precision"] == 1/3
    assert result["mrr"] == .5
    assert result["hit_rate"] == 1
    assert metrics(["a", "b"], {"a": 2, "b": 1}, 2)["ndcg"] == 1


def test_unique_and_percentile():
    assert unique_documents([{"doc_id":"a"}, {"doc_id":"a"}, {"doc_id":"b"}], 2) == ["a","b"]
    assert percentile([1, 3], 50) == 2
    assert percentile([], 95) is None


def test_hash():
    assert content_hash(Document("a", "hello")) == content_hash(Document("b", "hello"))
    assert content_hash(Document("a", "hello")) != content_hash(Document("a", "world"))

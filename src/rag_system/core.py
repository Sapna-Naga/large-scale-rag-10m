"""Pure functions shared by ingestion, search and evaluation."""
import hashlib
import math
import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    title: str = ""
    source: str = ""

    @classmethod
    def from_dict(cls, row):
        doc_id = row.get("_id", row.get("id"))
        text = row.get("text")
        if doc_id is None or not str(doc_id).strip() or not isinstance(text, str) or not text.strip():
            raise ValueError("Each document needs a nonempty id and text")
        return cls(str(doc_id), text, str(row.get("title") or ""), str(row.get("source") or ""))


def chunks(doc, tokenizer, size=220, overlap=32):
    """Split the original text using tokenizer offsets, preserving the source text."""
    if not 0 <= overlap < size:
        raise ValueError("Require 0 <= overlap < size")
    offsets = tokenizer(doc.text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False, verbose=False)["offset_mapping"]
    for start in range(0, len(offsets), size - overlap):
        end = min(start + size, len(offsets))
        a, b = offsets[start][0], offsets[end - 1][1]
        text = doc.text[a:b]
        if text.strip():
            yield {"doc_id": doc.id, "title": doc.title, "source": doc.source,
                   "text": text, "start": a, "end": b,
                   "chunk_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc.id}:{start}"))}
        if end == len(offsets):
            break


def content_hash(doc):
    return hashlib.sha256((doc.title + "\0" + doc.text + "\0" + doc.source).encode()).hexdigest()


def unique_documents(hits, k):
    return list(dict.fromkeys(str(hit["doc_id"]) for hit in hits))[:k]


def metrics(ranked, relevance, k):
    ranked = list(dict.fromkeys(ranked))[:k]
    positives = {d for d, r in relevance.items() if r > 0}
    flags = [d in positives for d in ranked]
    dcg = sum((2 ** max(0, relevance.get(d, 0)) - 1) / math.log2(i + 2)
              for i, d in enumerate(ranked))
    ideal = sum((2 ** r - 1) / math.log2(i + 2)
                for i, r in enumerate(sorted((r for r in relevance.values() if r > 0), reverse=True)[:k]))
    return {"recall": sum(flags) / len(positives) if positives else 0.0,
            "precision": sum(flags) / k,
            "mrr": next((1 / (i + 1) for i, flag in enumerate(flags) if flag), 0.0),
            "ndcg": dcg / ideal if ideal else 0.0, "hit_rate": float(any(flags))}


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p / 100
    lo = int(pos)
    return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (pos - lo)

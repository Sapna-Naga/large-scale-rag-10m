"""Small, testable building blocks shared by the pipeline."""
import hashlib
import json
import math
import uuid
from pathlib import Path


def records(path):
    """Read JSONL without loading the entire corpus into memory."""
    with Path(path).open(encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('Expected a JSON object')
                yield row
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{path}:{line_no}: {exc}') from exc


def document(row):
    doc_id = row.get('id', row.get('_id'))
    text = row.get('text')
    title = row.get('title', '')
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError('Document id must be a nonempty string')
    if not isinstance(text, str) or not text.strip() or not isinstance(title, str):
        raise ValueError(f'Invalid text or title for {doc_id}')
    return doc_id, title.strip(), text.strip()


def digest(title, text):
    return hashlib.sha256(json.dumps([title, text], ensure_ascii=False).encode()).hexdigest()


def point_id(doc_id, chunk_no):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, json.dumps([doc_id, chunk_no])))


def chunks(text, tokenizer, size=256, overlap=32):
    """Use character offsets so citations preserve original text."""
    if not 0 <= overlap < size:
        raise ValueError('Require 0 <= overlap < size')
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False, verbose=False)['offset_mapping']
    start = 0
    while start < len(offsets):
        end = min(start + size, len(offsets))
        left, right = offsets[start][0], offsets[end-1][1]
        yield text[left:right], left, right
        if end == len(offsets):
            break
        start = end - overlap


def rrf(rankings, constant=60):
    scores, items = {}, {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            key = item['point_id']
            items[key] = item
            scores[key] = scores.get(key, 0) + 1 / (constant + rank)
    return [dict(items[key], score=scores[key]) for key in sorted(scores, key=lambda k: (-scores[k], k))]


def unique_documents(hits):
    return list(dict.fromkeys(hit['doc_id'] for hit in hits))


def metrics(ranking, relevance, k):
    if k < 1:
        raise ValueError('k must be positive')
    ranking = list(dict.fromkeys(ranking))[:k]
    relevant = {key for key, value in relevance.items() if value > 0}
    if not relevant:
        raise ValueError('Query has no positive relevance labels')
    hits = [int(key in relevant) for key in ranking]
    dcg = sum((2**relevance.get(key, 0)-1)/math.log2(i+2) for i,key in enumerate(ranking))
    ideal = sum((2**v-1)/math.log2(i+2) for i,v in enumerate(sorted(relevance.values(), reverse=True)[:k]))
    return {'recall': sum(hits)/len(relevant), 'precision': sum(hits)/k,
            'mrr': next((1/(i+1) for i,v in enumerate(hits) if v), 0),
            'ndcg': dcg/ideal, 'hit_rate': int(any(hits))}

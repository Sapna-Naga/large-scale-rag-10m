"""Evaluate real relevance judgments and measure retrieval after warm-up."""
import csv
import hashlib
import json
import math
import platform
import statistics
import time
from pathlib import Path
from urllib.parse import urlsplit
from .core import records, metrics, unique_documents


def file_hash(path):
    """Identify the exact input file used in the report."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_cases(queries, qrels):
    """Validate all judged query IDs before selecting a development subset."""
    labels = {}
    with open(qrels, encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        if not {'query-id', 'corpus-id', 'score'} <= set(reader.fieldnames or []):
            raise ValueError('Invalid qrels header')
        for row in reader:
            qid, doc = row['query-id'], row['corpus-id']
            score = int(row['score'])
            if not qid or not doc or score < 0:
                raise ValueError('Invalid relevance judgment')
            relevant = labels.setdefault(qid, {})
            if doc in relevant:
                raise ValueError(f'Duplicate qrels pair: {qid}, {doc}')
            relevant[doc] = score
    if not labels:
        raise ValueError('No relevance judgments found')
    cases, seen = [], set()
    for row in records(queries):
        qid = row.get('_id', row.get('id'))
        query = row.get('text')
        if not isinstance(qid, str) or not qid.strip() or qid in seen:
            raise ValueError(f'Invalid or duplicate query ID: {qid!r}')
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f'Invalid query text: {qid}')
        seen.add(qid)
        if qid in labels and any(v > 0 for v in labels[qid].values()):
            cases.append((qid, query, labels[qid]))
    missing = labels.keys() - seen
    if missing:
        raise ValueError(f'{len(missing)} qrels query IDs missing from queries file')
    if not cases:
        raise ValueError('No queries have positive relevance labels')
    return cases, len(labels) - len(cases)


def evaluate(engine, queries, qrels, output, k=10, mode='hybrid', rerank=False,
             candidates=100, ef=128, limit=None, warmup_query=None):
    if not 1 <= k <= candidates <= 1000 or ef < 1:
        raise ValueError('Require 1 <= k <= candidates <= 1000 and ef >= 1')
    if limit is not None and limit < 1:
        raise ValueError('Evaluation limit must be positive')
    cases, no_positive = load_cases(queries, qrels)
    total_cases = len(cases)
    if limit is not None:
        cases = cases[:limit]
    cfg = engine.cfg
    points_before = engine.db.count(cfg.collection, exact=True).count
    if points_before == 0:
        raise ValueError('Cannot evaluate an empty collection')
    # Use the same retrieval options for warm-up and every measured query.
    options = dict(k=candidates, candidates=candidates, mode=mode, rerank=rerank, ef=ef)
    # A separate, real training query can be supplied for held-out evaluation.
    warmup = cases[0][1] if warmup_query is None else warmup_query
    if not isinstance(warmup, str) or not warmup.strip():
        raise ValueError('Warm-up query must not be empty')
    start = time.perf_counter()
    engine.search(warmup, **options)
    warmup_seconds = time.perf_counter() - start
    print(f'Warm-up completed in {warmup_seconds:.3f} s; excluded from measured latency.', flush=True)
    results, times, details = [], [], []
    for qid, query, relevance in cases:
        start = time.perf_counter()
        hits = engine.search(query, **options)
        elapsed = time.perf_counter() - start
        ranking = unique_documents(hits)
        result = metrics(ranking, relevance, k)
        relevant = {doc for doc, score in relevance.items() if score > 0}
        times.append(elapsed)
        results.append(result)
        details.append({
            'query_id': qid, 'latency_seconds': elapsed,
            'returned_chunks': len(hits), 'unique_documents_in_pool': len(ranking),
            'top_k_document_ids': ranking[:k],
            'returned_pool_document_ids': ranking,
            'first_relevant_rank_in_pool': next((i for i, doc in enumerate(ranking, 1) if doc in relevant), None),
            'returned_pool_recall': len(relevant.intersection(ranking)) / len(relevant),
            'metrics': result,
        })
        if len(results) % 10 == 0 or len(results) == len(cases):
            print(f'Evaluated {len(results)}/{len(cases)} queries', flush=True)
    points_after = engine.db.count(cfg.collection, exact=True).count
    if points_after != points_before:
        raise RuntimeError('Collection point count changed during evaluation; rerun on a fixed collection')
    ordered = sorted(times)
    # Record the endpoint without credentials, query parameters or fragments.
    endpoint = urlsplit(cfg.url)
    host = endpoint.hostname or ''
    if ':' in host:
        host = f'[{host}]'
    safe_endpoint = f'{endpoint.scheme}://{host}' + (f':{endpoint.port}' if endpoint.port else '')
    report = {
        'mode': mode, 'rerank': rerank, 'k': k, 'candidates': candidates, 'ef': ef,
        'queries': len(results), 'eligible_queries': total_cases,
        'queries_without_positive_labels': no_positive,
        'query_selection': 'first eligible queries in file order' if limit is not None else 'all eligible queries',
        'queries_file': str(queries), 'qrels_file': str(qrels),
        'queries_sha256': file_hash(queries), 'qrels_sha256': file_hash(qrels),
        'qdrant_endpoint': safe_endpoint,
        'chunk_size': cfg.chunk_size, 'chunk_overlap': cfg.overlap,
        'collection': cfg.collection, 'embedding': cfg.embedding,
        'embedding_revision': cfg.revision, 'reranker': cfg.reranker if rerank else None,
        'device': cfg.device, 'indexed_chunks': points_after,
        'warmup_seconds': warmup_seconds,
        'warmup_source': 'first evaluated query' if warmup_query is None else 'explicit query',
        'warmup_query': warmup,
        'warmup_matches_evaluated_query': any(q == warmup for _, q, _ in cases),
        'metrics': {key: statistics.mean(r[key] for r in results) for key in results[0]},
        'mean_returned_pool_recall': statistics.mean(d['returned_pool_recall'] for d in details),
        'queries_with_fewer_than_k_unique_documents': sum(d['unique_documents_in_pool'] < k for d in details),
        'latency_seconds': {f'p{p}': ordered[math.ceil(p / 100 * len(ordered)) - 1] for p in (50, 95, 99)},
        'serial_queries_per_second': len(times) / sum(times),
        'latency_scope': 'Engine.search including query encoding, database calls and optional reranking; warm-up excluded',
        'platform': platform.platform(), 'python': platform.python_version(),
        'per_query': details,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    # The full ranking diagnostics remain in JSON rather than flooding the console.
    return {key: value for key, value in report.items() if key != 'per_query'} | {'report_file': str(path)}

"""Compare verified dense search paths using the same real SciFact query vectors."""
import argparse
import json
import math
import os
import platform
import random
import statistics
import time
from pathlib import Path

import httpx

from .core import metrics
from .evaluate import file_hash, load_cases
from .index_experiment import (BASE, SOURCE, TARGET, EXPECTED, COUNTERS,
                               check_source, check_target_config, info, count,
                               request, telemetry)

DATA = Path('data/scifact')
PROFILE = Path('configs/scifact-retrieval-v1.json')
MODEL = 'BAAI/bge-small-en-v1.5'
PREFIX = 'Represent this sentence for searching relevant passages: '
SEED = 42
SIZE = 100
WARMUP = 3


def workload():
    """Use only real training queries; keep three warm-up texts separate."""
    manifest = json.loads(PROFILE.read_text(encoding='utf-8'))
    conf = manifest['configuration']
    if (conf['collection'], conf['embedding'], conf['embedding_revision'],
            conf['indexed_chunks'], conf['candidates'], conf['ef']) != (
            SOURCE, MODEL, None, EXPECTED, 200, 256):
        raise RuntimeError('Frozen profile differs from the benchmark design')
    hashes = {name: file_hash(DATA / name) for name in ('queries.jsonl', 'qrels/train.tsv')}
    if any(manifest['dataset_sha256'][name] != value for name, value in hashes.items()):
        raise RuntimeError('Training dataset checksum mismatch')
    cases, _ = load_cases(DATA / 'queries.jsonl', DATA / 'qrels/train.tsv')
    if len(cases) != 809:
        raise RuntimeError('Expected 809 eligible training queries')
    # Shuffle once, then retain distinct text values so warm-up cannot repeat a measured query.
    random.Random(SEED).shuffle(cases)
    distinct = []
    seen = set()
    for case in cases:
        if case[1] not in seen:
            seen.add(case[1])
            distinct.append(case)
        if len(distinct) == SIZE + WARMUP:
            break
    if len(distinct) != SIZE + WARMUP:
        raise RuntimeError('Not enough distinct training queries')
    return distinct, hashes, file_hash(PROFILE)


def check_vector(vector):
    if (not isinstance(vector, list) or len(vector) != 384 or
            any(type(v) not in (int, float) or not math.isfinite(v) for v in vector) or
            not math.isclose(math.sqrt(sum(v*v for v in vector)), 1.0, abs_tol=1e-4)):
        raise ValueError('Expected a finite, normalized 384-dimensional query vector')
    return vector


def write_new(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print('REPORT', path, flush=True)


def prepare(output):
    cases, hashes, profile_hash = workload()
    # Import the model only in this process. No model is loaded by the run command.
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL, revision=None, device='cpu',
                                trust_remote_code=False, local_files_only=True)
    rows = []
    start = time.perf_counter()
    for qid, text, _ in cases:
        # Match the existing Engine._dense_search instruction and normalization.
        vector = check_vector(model.encode(PREFIX + text, normalize_embeddings=True,
                                           show_progress_bar=False).tolist())
        rows.append({'query_id': qid, 'vector': vector})
        if len(rows) % 20 == 0 or len(rows) == len(cases):
            print('ENCODED_QUERIES', len(rows), flush=True)
    write_new(output, {'model': MODEL, 'revision': None, 'query_prefix': PREFIX,
                       'seed': SEED, 'dataset_sha256': hashes, 'profile_sha256': profile_hash,
                       'encoding_seconds': time.perf_counter()-start, 'rows': rows})


def load_vectors(path):
    cases, hashes, profile_hash = workload()
    data = json.loads(path.read_text(encoding='utf-8'))
    if (data['model'], data['revision'], data['query_prefix'], data['seed'],
            data['dataset_sha256'], data['profile_sha256']) != (
            MODEL, None, PREFIX, SEED, hashes, profile_hash):
        raise RuntimeError('Prepared query vector metadata mismatch')
    rows = data['rows']
    if [row['query_id'] for row in rows] != [case[0] for case in cases]:
        raise RuntimeError('Prepared query IDs do not match the frozen workload')
    return [(qid, relevance, check_vector(row['vector']))
            for (qid, _, relevance), row in zip(cases, rows, strict=True)], hashes, profile_hash


def variants():
    # At depth 200, ef below 200 does not yield a distinct graph beam width.
    for depth, efs in ((10, (32, 64, 128, 256)), (200, (256,))):
        yield f'plain_k{depth}', SOURCE, depth, False, 256, 'unfiltered_plain'
        yield f'exact_k{depth}', TARGET, depth, True, 256, 'unfiltered_exact'
        for ef in efs:
            yield f'hnsw_k{depth}_ef{ef}', TARGET, depth, False, ef, 'unfiltered_hnsw'


def search(client, collection, vector, depth, exact, ef):
    body = {'query': vector, 'using': 'dense', 'limit': depth,
            'params': {'exact': exact, 'hnsw_ef': ef, 'quantization': {'ignore': True}},
            'with_payload': ['doc_id'], 'with_vector': False}
    start = time.perf_counter()
    points = request(client, 'POST', f'/collections/{collection}/points/query', json=body)['points']
    elapsed = time.perf_counter() - start
    ids, docs, scores = [], [], []
    for point in points:
        pid, doc, score = point['id'], point['payload']['doc_id'], point['score']
        if not isinstance(pid, str) or not pid or not isinstance(doc, str) or not doc:
            raise RuntimeError('Invalid search hit identity')
        if type(score) not in (int, float) or not math.isfinite(score):
            raise RuntimeError('Invalid search score')
        ids.append(pid)
        docs.append(doc)
        scores.append(score)
    if len(ids) != depth or len(set(ids)) != depth:
        raise RuntimeError('Unexpected hit count or duplicate point ID')
    return {'point_ids': ids, 'document_ids': docs, 'scores': scores, 'seconds': elapsed}


def layout(snapshot):
    return sorted((s['info']['num_points'], s['info']['vector_data']['dense']['num_vectors'],
                   s['info']['vector_data']['dense']['num_indexed_vectors'])
                  for s in snapshot['segments'])


def check_layout(snapshot, collection):
    segments = layout(snapshot)
    if (sum(s[0] for s in segments), sum(s[1] for s in segments),
            sum(s[2] for s in segments)) != (EXPECTED, EXPECTED, EXPECTED if collection == TARGET else 0):
        raise RuntimeError('Unexpected dense segment coverage')


def checked_delta(before, after, expected_counter, n):
    if (not before['process_id'] or before['process_id'] != after['process_id'] or
            layout(before) != layout(after)):
        raise RuntimeError('Qdrant process or segment layout changed during benchmark')
    delta = {k: after['counts'][k] - before['counts'][k] for k in COUNTERS}
    if any(v < 0 for v in delta.values()) or delta[expected_counter] < n:
        raise RuntimeError(f'Expected search path not observed: {delta}')
    # Plain operations may also occur for the empty appendable segment.
    for key in ('unfiltered_exact', 'unfiltered_hnsw'):
        if key != expected_counter and delta[key] != 0:
            raise RuntimeError(f'Unexpected search path or concurrent traffic: {delta}')
    return delta


def neighbor_recall(ids, reference):
    if not reference or len(ids) != len(reference) or len(set(reference)) != len(reference) or len(set(ids)) != len(ids):
        raise ValueError('Need equal nonempty unique point rankings')
    return len(set(ids) & set(reference)) / len(reference)


def summarize(rows):
    times = sorted(row['seconds'] for row in rows)
    return {'queries': len(rows),
            'latency_seconds': {f'p{p}': times[math.ceil(p/100*len(times))-1] for p in (50, 95, 99)},
            'serial_qps_from_sum_of_request_times': len(times)/sum(times),
            'mean_point_overlap_with_exact': statistics.mean(row['point_recall_vs_exact'] for row in rows),
            'exact_order_matches': sum(row['exact_order_match'] for row in rows),
            'mean_unique_documents_in_pool': statistics.mean(row['unique_documents'] for row in rows),
            'document_metrics_at_10': {k: statistics.mean(row['metrics'][k] for row in rows) for k in rows[0]['metrics']},
            'mean_document_pool_recall': statistics.mean(row['pool_recall'] for row in rows)}


def run(client, vector_file, limit):
    rows, hashes, profile_hash = load_vectors(vector_file)
    measured, warmup = rows[:limit], rows[SIZE:]
    source = check_source(client)
    target = info(client, TARGET)
    check_target_config(target)
    if (target['status'] != 'green' or target['optimizer_status'] != 'ok' or
            count(client, TARGET) != EXPECTED or
            target['config']['params'] != source['config']['params']):
        raise RuntimeError('Experimental collection is not ready or differs from source vector configuration')
    results = {}
    for label, collection, depth, exact, ef, counter in variants():
        print('VARIANT', label, flush=True)
        for _, _, vector in warmup:
            search(client, collection, vector, depth, exact, ef)
        before = telemetry(client, collection)
        check_layout(before, collection)
        details = []
        for qid, relevance, vector in measured:
            row = search(client, collection, vector, depth, exact, ef)
            docs = list(dict.fromkeys(row['document_ids']))
            row.update(query_id=qid, unique_documents=len(docs), metrics=metrics(docs, relevance, 10),
                       pool_recall=metrics(docs, relevance, max(1, len(docs)))['recall'])
            details.append(row)
        after = telemetry(client, collection)
        check_layout(after, collection)
        delta = checked_delta(before, after, counter, len(measured))
        print('SEARCH_COUNTER_DELTA', json.dumps(delta), flush=True)
        results[label] = {'collection': collection, 'depth': depth, 'exact': exact, 'ef': ef,
                          'telemetry_before': before, 'telemetry_after': after,
                          'counter_delta': delta, 'per_query': details}
    if count(client, SOURCE) != EXPECTED or count(client, TARGET) != EXPECTED:
        raise RuntimeError('Collection count changed during benchmark')
    for label, result in results.items():
        reference = results[f'exact_k{result["depth"]}']['per_query']
        for row, ref in zip(result['per_query'], reference, strict=True):
            if row['query_id'] != ref['query_id']:
                raise RuntimeError('Reference query order mismatch')
            row['point_recall_vs_exact'] = neighbor_recall(row['point_ids'], ref['point_ids'])
            row['exact_order_match'] = row['point_ids'] == ref['point_ids']
        result['summary'] = summarize(result['per_query'])
        print('SUMMARY', label, json.dumps(result['summary']), flush=True)
    return {'dataset_sha256': hashes, 'profile_sha256': profile_hash,
            'query_vector_sha256': file_hash(vector_file), 'seed': SEED,
            'measured_queries': limit, 'measured_requests': limit * len(results),
            'warmup_requests': len(warmup) * len(results), 'source_info': source, 'target_info': target,
            'warmup_query_ids': [r[0] for r in warmup], 'platform': platform.platform(),
            'latency_scope': 'Sequential Qdrant HTTP query, response decoding; no embedding, fusion, or FastAPI',
            'reference': 'Exact dense search on the copied HNSW collection, separately at each depth',
            'results': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--vectors', type=Path, default=DATA / 'index_queries100.json')
    parser.add_argument('--limit', type=int, default=SIZE)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.limit <= SIZE:
        parser.error('Limit must be 1 through 100')
    if args.action == 'prepare':
        if args.output is not None or args.vectors.exists():
            parser.error('Prepare writes a new --vectors file; do not overwrite an existing file')
        prepare(args.vectors)
    else:
        if args.output is None or args.output.exists():
            parser.error('Run requires a new --output report filename')
        headers = {'api-key': os.environ['QDRANT_API_KEY']} if os.getenv('QDRANT_API_KEY') else {}
        with httpx.Client(base_url=BASE, headers=headers, timeout=120, trust_env=False) as client:
            report = run(client, args.vectors, args.limit)
        write_new(args.output, report)
        print('TOTAL_MEASURED_REQUESTS', report['measured_requests'], flush=True)


if __name__ == '__main__':
    main()

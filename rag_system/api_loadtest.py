"""Measure the frozen SciFact retrieval API with real, reproducible training queries."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time
from datetime import datetime, timezone, timedelta

import httpx

from .core import records, document
from .runtime_profile import profile_options


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def workload(profile, queries_path, report_path, limit, seed):
    # The frozen manifest identifies both the exact queries and reference rankings.
    if sha256(queries_path) != profile['dataset_sha256']['queries.jsonl']:
        raise ValueError('Query dataset checksum mismatch')
    if sha256(report_path) != profile['training_report_sha256']:
        raise ValueError('Reference report checksum mismatch')
    report = json.loads(Path(report_path).read_text(encoding='utf-8'))
    queries = {}
    for row in records(queries_path):
        qid, _, text = document(row)
        if qid in queries:
            raise ValueError('Duplicate query ID')
        queries[qid] = text
    cases = []
    seen = set()
    for row in report['per_query']:
        qid = row['query_id']
        if qid in seen:
            raise ValueError('Duplicate reference query ID')
        seen.add(qid)
        if qid not in queries:
            raise ValueError('Reference query not in dataset')
        ids = row['top_k_document_ids']
        if len(ids) != profile['configuration']['k'] or len(set(ids)) != len(ids):
            raise ValueError('Invalid reference ranking')
        cases.append((qid, queries[qid], ids))
    if limit < 1 or limit + 3 > len(cases):
        raise ValueError('Require measured queries plus three separate warm-up queries')
    random.Random(seed).shuffle(cases)
    return cases[:limit], cases[limit:limit + 3]


def readiness(client, profile):
    r = client.get('/health')
    r.raise_for_status()
    data = r.json()
    c = profile['configuration']
    expected = {'status': 'ready', 'collection': c['collection'],
                'indexed_chunks': c['indexed_chunks'], 'retrieval_profile': profile['profile'],
                'retrieval': profile_options(profile), 'experimental_generation_enabled': False}
    if any(data.get(key) != value for key, value in expected.items()):
        raise ValueError('API readiness does not match the frozen retrieval profile')
    return data


def request(client, case, k):
    qid, query, expected = case
    started = time.perf_counter()
    status = None
    ids = []
    error = None
    try:
        response = client.post('/search', json={'query': query, 'k': k})
        status = response.status_code
        if status != 200:
            error = 'http_status'
        else:
            result = response.json()
            if not isinstance(result, list) or len(result) != k:
                raise ValueError('Invalid result count')
            for hit in result:
                if (not isinstance(hit, dict) or not isinstance(hit.get('doc_id'), str)
                        or not hit['doc_id'] or not isinstance(hit.get('point_id'), str)
                        or not hit['point_id'] or not isinstance(hit.get('text'), str)
                        or not hit['text'].strip() or not isinstance(hit.get('title'), str)
                        or type(hit.get('start')) is not int or type(hit.get('end')) is not int
                        or hit['start'] < 0 or hit['end'] - hit['start'] != len(hit['text'])
                        or type(hit.get('score')) not in (int, float)
                        or not math.isfinite(hit['score'])):
                    raise ValueError('Invalid passage payload')
                ids.append(hit['doc_id'])
            if len(set(ids)) != k:
                raise ValueError('Duplicate document IDs')
    except httpx.HTTPError as exc:
        error = type(exc).__name__
    except (ValueError, TypeError, KeyError):
        error = 'invalid_payload'
    return {'query_id': qid, 'http_status': status, 'error': error,
            'latency_seconds': time.perf_counter() - started,
            'actual_doc_ids': ids, 'expected_doc_ids': expected,
            'ranking_match': error is None and ids == expected}


def summarize(results, elapsed):
    valid = [r for r in results if r['error'] is None]
    latencies = sorted(r['latency_seconds'] for r in valid)
    matches = sum(r['ranking_match'] for r in valid)
    return {'requests': len(results), 'http_200': sum(r['http_status'] == 200 for r in results),
            'valid_responses': len(valid), 'failures': len(results) - len(valid),
            'ranking_matches': matches, 'ranking_mismatches': len(valid) - matches,
            'elapsed_seconds': elapsed, 'valid_qps': len(valid) / elapsed,
            'matching_qps': matches / elapsed,
            'valid_latency_seconds': {f'p{p}': latencies[math.ceil(p / 100 * len(latencies)) - 1]
                                      for p in (50, 95, 99)} if latencies else {}}


def cache_snapshot(client):
    response = client.get('/cache')
    response.raise_for_status()
    data = response.json()
    if type(data.get('enabled')) is not bool or any(
            type(data.get(k)) is not int or data[k] < 0
            for k in ('hits', 'misses', 'evictions', 'expirations', 'skipped', 'entries', 'encoded_bytes')):
        raise ValueError('Invalid cache statistics')
    return data


def run(client, profile, cases, warmup, workers, rounds, *, track_cache=False):
    runs = []
    k = profile['configuration']['k']
    for round_no in range(1, rounds + 1):
        # Reverse the order in alternate rounds to reduce order bias.
        order = workers if round_no % 2 else list(reversed(workers))
        for n in order:
            print(f'Round {round_no}: {n} concurrent clients', flush=True)
            cache_before = cache_snapshot(client) if track_cache else None
            warm = [request(client, case, k) for case in warmup]
            if any(r['error'] for r in warm):
                raise RuntimeError('Warm-up request failed; check API before benchmarking')
            cache_after_warmup = cache_snapshot(client) if track_cache else None
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=n) as pool:
                results = list(pool.map(lambda case: request(client, case, k), cases))
            elapsed = time.perf_counter() - started
            summary = summarize(results, elapsed)
            runs.append({'round': round_no, 'workers': n, 'warmup': warm,
                         **summary, 'per_request': results})
            if track_cache:
                cache_after = cache_snapshot(client)
                counters = ('hits', 'misses', 'evictions', 'expirations', 'skipped')
                delta = {key: cache_after[key] - cache_after_warmup[key] for key in counters}
                if any(v < 0 for v in delta.values()) or len({cache_before['enabled'], cache_after_warmup['enabled'], cache_after['enabled']}) != 1:
                    raise RuntimeError('Cache was reset or configuration changed during the run')
                runs[-1]['cache'] = {'before_warmup': cache_before, 'after_warmup': cache_after_warmup,
                                     'after_measurement': cache_after, 'measured_delta': delta}
                print('CACHE_MEASURED_DELTA', json.dumps(delta), flush=True)
            print(json.dumps(summary, indent=2), flush=True)
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', default='configs/scifact-retrieval-v1.json')
    p.add_argument('--queries', default='data/scifact/queries.jsonl')
    p.add_argument('--reference', default='benchmarks/scifact/full_train/train809_c384_o32_hybrid.json')
    p.add_argument('--limit', type=int, default=100)
    p.add_argument('--workers', type=int, nargs='+', default=[1, 2, 4, 8])
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', required=True)
    p.add_argument('--cache-stats', action='store_true', help='Record cache counters outside measured intervals')
    a = p.parse_args()
    if not 1 <= a.rounds <= 10 or any(not 1 <= n <= 16 for n in a.workers) or len(set(a.workers)) != len(a.workers):
        p.error('Use 1–10 rounds and unique worker counts in 1–16')
    path = Path(a.output)
    if path.exists():
        p.error('Output exists; use a new report filename')
    profile = json.loads(Path(a.profile).read_text(encoding='utf-8'))
    cases, warmup = workload(profile, a.queries, a.reference, a.limit, a.seed)
    timestamp = datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat()
    # One shared, thread-safe HTTP client; the server still uses one process.
    with httpx.Client(base_url='http://127.0.0.1:8000', timeout=180,
                      trust_env=False, limits=httpx.Limits(max_connections=max(a.workers),
                      max_keepalive_connections=max(a.workers))) as client:
        before = readiness(client, profile)
        runs = run(client, profile, cases, warmup, a.workers, a.rounds, track_cache=a.cache_stats)
        after = readiness(client, profile)
    report = {'timestamp_ist': timestamp, 'profile_sha256': sha256(a.profile),
              'reference_sha256': sha256(a.reference), 'queries_sha256': sha256(a.queries),
              'seed': a.seed, 'query_ids': [c[0] for c in cases],
              'warmup_query_ids': [c[0] for c in warmup], 'workers': a.workers,
              'rounds': a.rounds, 'client_platform': platform.platform(),
              'client_python': platform.python_version(), 'health_before': before, 'health_after': after,
              'latency_scope': 'Client HTTP round trip plus JSON/payload validation; closed-loop, warm-up excluded',
              'runs': runs}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print('REPORT', path)
    print('TOTAL_MEASURED_REQUESTS', sum(r['requests'] for r in runs))
    print('TOTAL_FAILURES', sum(r['failures'] for r in runs))
    print('TOTAL_RANKING_MISMATCHES', sum(r['ranking_mismatches'] for r in runs))
    # Keep measured failures in the report and signal an unsuccessful benchmark.
    return int(any(r['failures'] or r['ranking_mismatches'] for r in runs))


if __name__ == '__main__':
    raise SystemExit(main())

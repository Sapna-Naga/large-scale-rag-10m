"""Build a separate SciFact HNSW test collection and verify its search path."""
import argparse
import json
import math
import os
import time
from pathlib import Path

import httpx

SOURCE = 'rag_documents_v1_c384_o32'
TARGET = 'rag_documents_v1_c384_o32_hnsw'
EXPECTED = 6547
BASE = 'http://127.0.0.1:6333'
COUNTERS = ('unfiltered_plain', 'unfiltered_hnsw', 'unfiltered_exact')


def request(client, method, path, **kwargs):
    response = client.request(method, path, **kwargs)
    response.raise_for_status()
    body = response.json()
    if body['status'] != 'ok':
        raise RuntimeError(f'Qdrant rejected {method} {path}: {body["status"]}')
    return body['result']


def info(client, name):
    return request(client, 'GET', f'/collections/{name}')


def count(client, name):
    return request(client, 'POST', f'/collections/{name}/points/count',
                   json={'exact': True})['count']


def check_source(client):
    source = info(client, SOURCE)
    config = source['config']
    params = config['params']
    if (source['status'] != 'green' or source['optimizer_status'] != 'ok'
            or count(client, SOURCE) != EXPECTED
            or params['vectors'] != {'dense': {'size': 384, 'distance': 'Cosine', 'on_disk': True}}
            or params['sparse_vectors'] != {'bm25': {'index': {'on_disk': True}, 'modifier': 'idf'}}
            or params['shard_number'] != 1 or params['replication_factor'] != 1
            or config['quantization_config'] is not None
            or source['payload_schema'].get('doc_id', {}).get('data_type') != 'keyword'):
        raise RuntimeError('Source configuration or count differs from assessed baseline')
    return source


def batches(client, name):
    offset = None
    seen = set()
    while True:
        body = {'limit': 64, 'with_payload': True, 'with_vector': True}
        if offset is not None:
            body['offset'] = offset
        result = request(client, 'POST', f'/collections/{name}/points/scroll', json=body)
        points = result['points']
        for point in points:
            key = point['id']
            if key in seen:
                raise RuntimeError('Repeated point in scroll')
            seen.add(key)
        if points:
            yield points
        next_offset = result.get('next_page_offset')
        if next_offset is None:
            break
        if not points or next_offset == offset:
            raise RuntimeError('Scroll did not advance')
        offset = next_offset
    if len(seen) != EXPECTED:
        raise RuntimeError(f'Expected {EXPECTED} points; saw {len(seen)}')


def comparable_point(a, b):
    # Cosine vectors can be normalized again on upsert. Compare with a small
    # numeric tolerance, but require the same payload and sparse components.
    if a['id'] != b['id'] or a['payload'] != b['payload']:
        return False
    av, bv = a['vector'], b['vector']
    if set(av) != {'dense', 'bm25'} or set(bv) != set(av):
        return False
    if len(av['dense']) != 384 or len(bv['dense']) != 384:
        return False
    if not all(math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-7)
               for x, y in zip(av['dense'], bv['dense'])):
        return False
    def sparse(v):
        s = v['bm25']
        if len(s['indices']) != len(s['values']) or len(set(s['indices'])) != len(s['indices']):
            raise ValueError('Invalid sparse vector')
        return sorted(zip(s['indices'], s['values']))
    return sparse(av) == sparse(bv)


def create(client, source):
    # Never replace or update an existing collection, including a partial copy.
    names = {item['name'] for item in request(client, 'GET', '/collections')['collections']}
    if TARGET in names:
        raise RuntimeError(f'{TARGET} already exists; inspect it before proceeding')
    params = source['config']['params']
    request(client, 'PUT', f'/collections/{TARGET}', json={
        'vectors': params['vectors'], 'sparse_vectors': params['sparse_vectors'],
        'shard_number': 1, 'replication_factor': 1, 'on_disk_payload': True,
        'hnsw_config': {'m': 16, 'ef_construct': 100, 'full_scan_threshold': 1,
                        'max_indexing_threads': 2, 'on_disk': True},
        'optimizers_config': {'indexing_threshold': 128, 'default_segment_number': 2,
                              'max_optimization_threads': 1},
    })
    request(client, 'PUT', f'/collections/{TARGET}/index', params={'wait': 'true'},
            json={'field_name': 'doc_id', 'field_schema': 'keyword'})
    copied = 0
    start = time.perf_counter()
    for points in batches(client, SOURCE):
        request(client, 'PUT', f'/collections/{TARGET}/points', params={'wait': 'true'},
                json={'points': [{k: p[k] for k in ('id', 'payload', 'vector')} for p in points]})
        copied += len(points)
        if copied % 1024 == 0 or copied == EXPECTED:
            print('COPIED_POINTS', copied, flush=True)
    print('COPY_SECONDS', round(time.perf_counter() - start, 3), flush=True)


def wait_ready(client):
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        target = info(client, TARGET)
        if target['optimizer_status'] != 'ok' or target['status'] == 'red':
            raise RuntimeError(f'Optimizer failure: {target["optimizer_status"]}')
        if target['status'] == 'green' and count(client, TARGET) == EXPECTED:
            return target
        print('WAITING_FOR_OPTIMIZER', target['status'], flush=True)
        time.sleep(5)
    raise RuntimeError('Collection did not become ready within 600 seconds')


def verify_copy(client):
    checked = 0
    for points in batches(client, SOURCE):
        result = request(client, 'POST', f'/collections/{TARGET}/points', json={
            'ids': [p['id'] for p in points], 'with_payload': True, 'with_vector': True})
        by_id = {p['id']: p for p in result}
        if len(result) != len(points) or any(p['id'] not in by_id or
                not comparable_point(p, by_id[p['id']]) for p in points):
            raise RuntimeError('Copied point, payload or vector mismatch')
        checked += len(points)
    if count(client, SOURCE) != EXPECTED or count(client, TARGET) != EXPECTED:
        raise RuntimeError('Collection count changed during verification')
    print('VERIFIED_POINTS', checked, flush=True)
    return checked


def telemetry(client, name):
    # Qdrant v1.14.1 includes segment telemetry only at detail level 4.
    data = request(client, 'GET', '/telemetry', params={'details_level': 4, 'anonymize': 'false'})
    matches = [c for c in data['collections']['collections'] if c.get('id') == name]
    if len(matches) != 1:
        raise RuntimeError(f'Telemetry collection missing: {name}')
    segments = [seg for shard in (matches[0].get('shards') or [])
                for seg in ((shard.get('local') or {}).get('segments') or [])]
    if not segments:
        raise RuntimeError('Detailed segment telemetry is unavailable')
    counts = dict.fromkeys(COUNTERS, 0)
    for seg in segments:
        dense = [index for index in (seg.get('vector_index_searches') or [])
                 if index.get('index_name') == 'dense']
        if len(dense) != 1:
            raise RuntimeError('Expected one dense index telemetry entry per segment')
        for key in COUNTERS:
            # Qdrant omits empty operation statistics. Missing counters are
            # zero only after the dense index entry has been identified.
            value = dense[0].get(key, {'count': 0}).get('count')
            if type(value) is not int or value < 0:
                raise RuntimeError(f'Invalid telemetry count: {key}')
            counts[key] += value
    return {'counts': counts, 'segments': segments, 'process_id': data.get('id')}


def probe(client, name):
    # Use an actual stored corpus vector only to diagnose the search path.
    # This is not a retrieval-quality or latency benchmark.
    sample = request(client, 'POST', f'/collections/{SOURCE}/points/scroll',
                     json={'limit': 1, 'with_vector': True, 'with_payload': False})['points'][0]
    before = telemetry(client, name)
    for _ in range(5):
        hits = request(client, 'POST', f'/collections/{name}/points/query', json={
            'query': sample['vector']['dense'], 'using': 'dense', 'limit': 10,
            'params': {'exact': False, 'hnsw_ef': 256}, 'with_payload': False})['points']
        if len(hits) != 10:
            raise RuntimeError('Probe returned unexpected number of results')
    after = telemetry(client, name)
    if before['process_id'] != after['process_id']:
        raise RuntimeError('Qdrant process changed during the probe')
    if len(before['segments']) != len(after['segments']):
        raise RuntimeError('Segment count changed during the probe')
    delta = {key: after['counts'][key] - before['counts'][key] for key in COUNTERS}
    if any(value < 0 for value in delta.values()):
        raise RuntimeError('Telemetry counters reset during the probe')
    if sum(delta.values()) < 5:
        raise RuntimeError('Insufficient observed search operations; search path remains unverified')
    print('SEARCH_COUNTER_DELTA', json.dumps(delta), flush=True)
    return {'collection': name, 'query_point_id': sample['id'], 'requests': 5,
            'before': before, 'after': after, 'counter_delta': delta,
            'hnsw_observed': delta['unfiltered_hnsw'] > 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['baseline', 'create', 'verify'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output report already exists; choose a new filename')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    headers = {'api-key': os.environ['QDRANT_API_KEY']} if os.getenv('QDRANT_API_KEY') else {}
    with httpx.Client(base_url=BASE, headers=headers, timeout=120, trust_env=False) as client:
        source = check_source(client)
        report = {'source_info': source, 'action': args.action}
        if args.action == 'baseline':
            report['probe'] = probe(client, SOURCE)
        else:
            if args.action == 'create':
                create(client, source)
            target = wait_ready(client)
            config = target['config']
            h = config['hnsw_config']
            if (h['m'], h['ef_construct'], h['full_scan_threshold'], h['on_disk'],
                config['optimizer_config']['indexing_threshold'], config['quantization_config']) != (16, 100, 1, True, 128, None):
                raise RuntimeError('Experimental collection configuration mismatch')
            report['target_info'] = target
            report['verified_points'] = verify_copy(client)
            report['probe'] = probe(client, TARGET)
        with args.output.open('x', encoding='utf-8') as f:
            json.dump(report, f, indent=2, allow_nan=False)
        print('REPORT', args.output, flush=True)
        print('HNSW_OBSERVED', report['probe']['hnsw_observed'], flush=True)
        if args.action != 'baseline' and not report['probe']['hnsw_observed']:
            raise SystemExit('HNSW traversal not yet observed; inspect telemetry before benchmarking')


if __name__ == '__main__':
    main()

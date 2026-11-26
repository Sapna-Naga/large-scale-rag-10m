import json
import httpx
import pytest
from rag_system.api_loadtest import request, summarize, readiness, workload, run, sha256

CASE = ('q1', 'example', ['d1'])
HIT = {'doc_id': 'd1', 'point_id': 'p1', 'title': 'Title', 'text': 'text',
       'start': 0, 'end': 4, 'score': 0.5}


def client(handler):
    return httpx.Client(base_url='http://test', transport=httpx.MockTransport(handler))


@pytest.mark.parametrize('status,body,error,match', [
    (200, [HIT], None, True),
    (200, [dict(HIT, doc_id='d2')], None, False),
    (200, [], 'invalid_payload', False),
    (200, [dict(HIT, end=5)], 'invalid_payload', False),
    (503, {}, 'http_status', False),
])
def test_request(status, body, error, match):
    with client(lambda r: httpx.Response(status, json=body)) as c:
        result = request(c, CASE, 1)
    assert result['error'] == error
    assert result['ranking_match'] is match


def test_timeout():
    def handler(r):
        raise httpx.ReadTimeout('timeout')
    with client(handler) as c:
        assert request(c, CASE, 1)['error'] == 'ReadTimeout'


def test_summary():
    results = [dict(error=None, ranking_match=True, latency_seconds=.1, http_status=200),
               dict(error=None, ranking_match=False, latency_seconds=.3, http_status=200),
               dict(error='http_status', ranking_match=False, latency_seconds=.4, http_status=503)]
    s = summarize(results, 1)
    assert (s['valid_responses'], s['failures'], s['ranking_matches'], s['ranking_mismatches']) == (2,1,1,1)
    assert s['valid_latency_seconds'] == {'p50': .1, 'p95': .3, 'p99': .3}


def test_workload(tmp_path):
    q = tmp_path/'q.jsonl'
    q.write_text('\n'.join(json.dumps({'_id': str(i), 'text': f'query {i}'}) for i in range(6)))
    r = tmp_path/'r.json'
    r.write_text(json.dumps({'per_query': [{'query_id': str(i), 'top_k_document_ids': ['d1']} for i in range(5)]}))
    p = {'dataset_sha256': {'queries.jsonl': sha256(q)}, 'training_report_sha256': sha256(r), 'configuration': {'k': 1}}
    a,b = workload(p,q,r,2,42)
    assert workload(p,q,r,2,42) == (a,b)
    assert len(a)==2 and len(b)==3 and not {x[0] for x in a} & {x[0] for x in b}
    q.write_text('changed')
    with pytest.raises(ValueError, match='checksum'):
        workload(p,q,r,2,42)


def test_readiness_rejects_mismatch():
    p = {'profile':'p', 'configuration': dict(collection='c', indexed_chunks=1,k=1,mode='hybrid',candidates=2,ef=1,rerank=False,parallel=True,fusion='document',rrf_constant=10,dense_weight=4)}
    with client(lambda r: httpx.Response(200,json={'status':'ready'})) as c:
        with pytest.raises(ValueError,match='readiness'):
            readiness(c,p)


def test_round_order():
    with client(lambda r: httpx.Response(200,json=[HIT])) as c:
        runs = run(c, {'configuration':{'k':1}}, [CASE], [CASE], [1,2], 2)
    assert [r['workers'] for r in runs] == [1,2,2,1]
    assert all(r['requests']==r['ranking_matches']==1 for r in runs)

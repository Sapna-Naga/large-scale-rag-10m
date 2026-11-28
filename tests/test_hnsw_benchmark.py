"""Controlled fixtures test the harness, not real retrieval performance."""
import json
from pathlib import Path

import httpx
import pytest

from rag_system import hnsw_benchmark as b


def snapshot(plain=0, hnsw=0, exact=0, process='fixture', indexed=6547):
    return {'process_id': process,
            'counts': dict(zip(b.COUNTERS, (plain, hnsw, exact))),
            'segments': [{'info': {'num_points': 6547, 'vector_data': {'dense': {
                'num_vectors': 6547, 'num_indexed_vectors': indexed}}}}]}


def test_variants():
    variants = list(b.variants())
    assert len(variants) == 9
    assert [v[4] for v in variants if v[2] == 10 and v[-1] == 'unfiltered_hnsw'] == [32,64,128,256]
    assert all(v[4] >= v[2] for v in variants)


@pytest.mark.parametrize('vector', [[1.0], [0.0]*384, [float('nan')]*384, [True]+[0.0]*383])
def test_invalid_vector(vector):
    with pytest.raises(ValueError):
        b.check_vector(vector)


def test_valid_vector():
    assert b.check_vector([1.0]+[0.0]*383)[0] == 1.0


def test_neighbor_overlap():
    assert b.neighbor_recall(['p1','p3'], ['p1','p2']) == 0.5
    with pytest.raises(ValueError):
        b.neighbor_recall(['p1','p1'], ['p1','p2'])


@pytest.mark.parametrize('counter,after', [
    ('unfiltered_hnsw', snapshot(plain=5,hnsw=5)),
    ('unfiltered_exact', snapshot(plain=5,exact=5)),
    ('unfiltered_plain', snapshot(plain=5)),
])
def test_telemetry_valid(counter, after):
    assert b.checked_delta(snapshot(), after, counter, 5)[counter] == 5


@pytest.mark.parametrize('after', [snapshot(hnsw=4), snapshot(hnsw=5,exact=1),
    snapshot(hnsw=5,process='changed'), snapshot(hnsw=5,indexed=0)])
def test_telemetry_failure(after):
    with pytest.raises(RuntimeError):
        b.checked_delta(snapshot(), after, 'unfiltered_hnsw', 5)


def test_layout():
    b.check_layout(snapshot(), b.TARGET)
    b.check_layout(snapshot(indexed=0), b.SOURCE)
    with pytest.raises(RuntimeError):
        b.check_layout(snapshot(indexed=0), b.TARGET)


@pytest.mark.parametrize('duplicate', [False, True])
def test_search_http(duplicate):
    def handler(request):
        assert request.method == 'POST'
        assert request.url.path == f'/collections/{b.TARGET}/points/query'
        data = json.loads(request.content)
        assert data['using'] == 'dense'
        assert data['limit'] == 2
        assert data['params'] == {'exact': True, 'hnsw_ef': 256, 'quantization': {'ignore': True}}
        assert data['with_payload'] == ['doc_id']
        points = [{'id': 'p1', 'payload': {'doc_id': 'd1'}, 'score': 0.9},
                  {'id': 'p1' if duplicate else 'p2', 'payload': {'doc_id': 'd2'}, 'score': 0.8}]
        return httpx.Response(200, json={'status':'ok','result':{'points':points}})
    with httpx.Client(base_url=b.BASE, transport=httpx.MockTransport(handler)) as client:
        if duplicate:
            with pytest.raises(RuntimeError, match='duplicate'):
                b.search(client,b.TARGET,[1.0]+[0.0]*383,2,True,256)
        else:
            result = b.search(client,b.TARGET,[1.0]+[0.0]*383,2,True,256)
            assert result['point_ids'] == ['p1','p2']
            assert result['document_ids'] == ['d1','d2']


def test_no_overwrite(tmp_path):
    path = tmp_path/'report.json'
    b.write_new(path, {'fixture': True})
    with pytest.raises(FileExistsError):
        b.write_new(path, {'fixture': False})
    assert json.loads(path.read_text()) == {'fixture': True}


def test_summary():
    rows = [{'seconds': s, 'point_recall_vs_exact': r, 'exact_order_match': r == 1,
             'unique_documents': 2, 'metrics': {'recall': r}, 'pool_recall': r}
            for s,r in [(0.1,1.0),(0.2,0.5)]]
    out = b.summarize(rows)
    assert out['mean_point_overlap_with_exact'] == 0.75
    assert out['exact_order_matches'] == 1
    assert out['latency_seconds'] == {'p50':0.1,'p95':0.2,'p99':0.2}
    assert out['serial_qps_from_sum_of_request_times'] == pytest.approx(2/0.3)


def test_prepared_metadata_and_ids(tmp_path,monkeypatch):
    cases = [('q1','fixture text',{'d1':1})]
    monkeypatch.setattr(b,'workload',lambda:(cases,{'queries.jsonl':'fixture'},'profile'))
    path = tmp_path/'vectors.json'
    data = {'model':b.MODEL,'revision':None,'query_prefix':b.PREFIX,'seed':b.SEED,
            'dataset_sha256':{'queries.jsonl':'fixture'},'profile_sha256':'profile',
            'rows':[{'query_id':'q1','vector':[1.0]+[0.0]*383}]}
    path.write_text(json.dumps(data))
    assert b.load_vectors(path)[0][0][0] == 'q1'
    data['rows'][0]['query_id'] = 'other'
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError,match='IDs'):
        b.load_vectors(path)

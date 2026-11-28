import copy
import pytest
import httpx
from rag_system import index_experiment as exp


def point():
    return {'id': 'a', 'payload': {'doc_id': '1'}, 'vector': {
        'dense': [0.01] * 384, 'bm25': {'indices': [3, 7], 'values': [0.5, 1.0]}}}


def test_vector_comparison():
    a = point()
    b = copy.deepcopy(a)
    b['vector']['dense'][0] += 1e-8
    b['vector']['bm25'] = {'indices': [7, 3], 'values': [1.0, 0.5]}
    assert exp.comparable_point(a, b)


@pytest.mark.parametrize('field', ['id', 'payload', 'dense', 'bm25'])
def test_mismatch(field):
    a = point()
    b = copy.deepcopy(a)
    if field in ('id', 'payload'):
        b[field] = None
    elif field == 'dense':
        b['vector'][field][0] = 0.5
    else:
        b['vector'][field]['values'][0] = 0.9
    assert not exp.comparable_point(a, b)


def test_existing_target_protected():
    def handler(r):
        assert r.method == 'GET'
        assert r.url.path == '/collections'
        return httpx.Response(200, json={'status': 'ok', 'result': {'collections': [{'name': exp.TARGET}]}})
    with httpx.Client(base_url=exp.BASE, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match='already exists'):
            exp.create(client, {})


def test_telemetry_scopes_collection_and_vector():
    def collection(name, count):
        return {'id': name, 'shards': [{'local': {'segments': [{'vector_index_searches': [
            {'index_name': 'dense', 'unfiltered_hnsw': {'count': count}},
            {'index_name': 'bm25', 'unfiltered_hnsw': {'count': 99}}]}]}}]}
    def handler(r):
        assert r.url.path == '/telemetry'
        assert r.url.params['details_level'] == '4'
        assert r.url.params['anonymize'] == 'false'
        return httpx.Response(200, json={'status': 'ok', 'result': {'collections': {
            'collections': [collection(exp.SOURCE, 5), collection(exp.TARGET, 15)]}}})
    with httpx.Client(base_url=exp.BASE, transport=httpx.MockTransport(handler)) as client:
        assert exp.telemetry(client, exp.TARGET)['counts']['unfiltered_hnsw'] == 15


def test_missing_segment_telemetry_rejected():
    def handler(r):
        return httpx.Response(200, json={'status': 'ok', 'result': {'collections': {
            'collections': [{'id': exp.SOURCE, 'shards': [{'local': {}}]}]}}})
    with httpx.Client(base_url=exp.BASE, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match='Detailed segment telemetry'):
            exp.telemetry(client, exp.SOURCE)


@pytest.mark.parametrize('entries', [[], [{'index_name': 'bm25'}],
    [{'index_name': 'dense'}, {'index_name': 'dense'}]])
def test_missing_or_ambiguous_dense_telemetry_rejected(entries):
    def handler(r):
        return httpx.Response(200, json={'status': 'ok', 'result': {'collections': {
            'collections': [{'id': exp.SOURCE, 'shards': [{'local': {'segments': [
                {'vector_index_searches': entries}]}}]}]}}})
    with httpx.Client(base_url=exp.BASE, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match='one dense index'):
            exp.telemetry(client, exp.SOURCE)


@pytest.mark.parametrize('plain,hnsw,expected', [(5, 0, False), (0, 5, True), (5, 5, True), (0, 0, None)])
def test_probe_requires_observed_search_operations(monkeypatch, plain, hnsw, expected):
    def handler(r):
        if r.url.path.endswith('/scroll'):
            result = {'points': [point()]}
        else:
            assert r.url.path.endswith('/query')
            result = {'points': [{'id': str(i)} for i in range(10)]}
        return httpx.Response(200, json={'status': 'ok', 'result': result})
    before = {'counts': dict.fromkeys(exp.COUNTERS, 0), 'segments': [{}], 'process_id': 'test'}
    after = {'counts': {'unfiltered_plain': plain, 'unfiltered_hnsw': hnsw,
                       'unfiltered_exact': 0}, 'segments': [{}], 'process_id': 'test'}
    snapshots = iter([before, after])
    monkeypatch.setattr(exp, 'telemetry', lambda *args: next(snapshots))
    with httpx.Client(base_url=exp.BASE, transport=httpx.MockTransport(handler)) as client:
        if expected is None:
            with pytest.raises(RuntimeError, match='Insufficient observed'):
                exp.probe(client, exp.SOURCE)
        else:
            assert exp.probe(client, exp.SOURCE)['hnsw_observed'] is expected

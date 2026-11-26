"""API contract tests using controlled dependencies, not benchmark data."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rag_system import api


class FakeDB:
    exists = True
    points = 6547

    def collection_exists(self, name):
        return self.exists

    def count(self, *, collection_name, exact):
        assert exact
        return SimpleNamespace(count=self.points)


class FakeEngine:
    def __init__(self):
        self.cfg = SimpleNamespace(collection='test_index')
        self.db = FakeDB()
        self.calls = []

    def runtime_search(self, query, k):
        self.calls.append(('search', query, k))
        return [{'doc_id': 'fixture', 'text': 'Fixture passage', 'score': 0.5}]

    def answer(self, query, k):
        self.calls.append(('ask', query, k))
        return {'answer': 'Insufficient evidence.', 'citations': [], 'status': 'abstained'}


@pytest.fixture
def setup(monkeypatch):
    e = FakeEngine()
    monkeypatch.setattr(api, 'engine', lambda: e)
    monkeypatch.setattr(api, 'active_profile', lambda cfg: None)
    monkeypatch.delenv('RAG_ENABLE_EXPERIMENTAL_GENERATION', raising=False)
    with TestClient(api.app) as client:
        yield client, e


def test_live_without_engine(setup, monkeypatch):
    client, _ = setup
    monkeypatch.setattr(api, 'engine', lambda: pytest.fail('Liveness loaded engine'))
    assert client.get('/live').json() == {'status': 'alive'}


def test_health_profile(setup, monkeypatch):
    client, _ = setup
    profile = {'profile': 'fixture-profile', 'configuration': {'indexed_chunks': 6547}}
    monkeypatch.setattr(api, 'active_profile', lambda cfg: profile)
    monkeypatch.setattr(api, 'profile_options', lambda p: {'k': 10})
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ready', 'collection': 'test_index',
                               'indexed_chunks': 6547, 'retrieval_profile': 'fixture-profile',
                               'retrieval': {'k': 10}, 'experimental_generation_enabled': False}


@pytest.mark.parametrize('exists,count', [(False, 6547), (True, 0)])
def test_health_unavailable_index(setup, exists, count):
    client, e = setup
    e.db.exists, e.db.points = exists, count
    assert client.get('/health').status_code == 503


def test_health_count_mismatch(setup, monkeypatch):
    client, _ = setup
    monkeypatch.setattr(api, 'active_profile', lambda cfg: {'configuration': {'indexed_chunks': 1}})
    assert client.get('/health').status_code == 503


def test_health_profile_error(setup, monkeypatch):
    client, _ = setup
    def broken(cfg):
        raise ValueError('private configuration detail')
    monkeypatch.setattr(api, 'active_profile', broken)
    response = client.get('/health')
    assert response.status_code == 503
    assert 'private configuration detail' not in response.text


@pytest.mark.parametrize('k', [None, 3])
def test_search_contract(setup, k):
    client, e = setup
    payload = {'query': '  real query  '}
    if k is not None:
        payload['k'] = k
    response = client.post('/search', json=payload)
    assert response.status_code == 200
    assert response.json()[0]['doc_id'] == 'fixture'
    assert e.calls == [('search', 'real query', k)]


@pytest.mark.parametrize('payload', [ {}, {'query': ''}, {'query': '   '},
    {'query': 'x'*4001}, {'query': 5}, {'query': 'x', 'k': 0},
    {'query': 'x', 'k': 51}, {'query': 'x', 'k': True}, {'query': 'x', 'k': '3'},
    {'query': 'x', 'unexpected': 1}])
def test_bad_search_request(setup, payload):
    client, e = setup
    assert client.post('/search', json=payload).status_code == 422
    assert not e.calls


def test_search_failure_is_sanitized(setup, monkeypatch):
    client, e = setup
    def broken(*args):
        raise RuntimeError('private endpoint detail')
    monkeypatch.setattr(e, 'runtime_search', broken)
    response = client.post('/search', json={'query': 'x'})
    assert response.status_code == 503
    assert 'private endpoint detail' not in response.text


def test_generation_disabled_by_default(setup):
    client, e = setup
    assert client.post('/ask', json={'query': 'x'}).status_code == 503
    assert not e.calls


def test_experimental_generation_opt_in(setup, monkeypatch):
    client, e = setup
    monkeypatch.setenv('RAG_ENABLE_EXPERIMENTAL_GENERATION', '1')
    assert client.get('/health').json()['experimental_generation_enabled'] is True
    response = client.post('/ask', json={'query': 'x'})
    assert response.status_code == 200
    assert response.json()['status'] == 'abstained'
    assert e.calls == [('ask', 'x', None)]

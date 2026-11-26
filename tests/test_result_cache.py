import json
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from rag_system import api
from rag_system.result_cache import ResultCache, cache_key, from_environment
from rag_system.api_loadtest import run
import httpx


def test_snapshot_and_hit():
    c = ResultCache(2)
    assert c.get(b'a') is None
    r = [{'text':'original'}]
    c.put(b'a', r)
    r[0]['text'] = 'modified'
    assert c.get(b'a') == [{'text':'original'}]
    assert c.stats()['hits'] == c.stats()['misses'] == 1


def test_lru():
    c = ResultCache(2)
    c.put(b'a', [1]); c.put(b'b', [2]); c.get(b'a'); c.put(b'c', [3])
    assert c.get(b'b') is None and c.get(b'a') == [1]
    assert c.stats()['evictions'] == 1


def test_byte_budget_and_oversized():
    c = ResultCache(3, max_bytes=10)
    c.put(b'a', [123]); c.put(b'b', [456]); c.put(b'c', ['x'*20])
    assert c.stats()['encoded_bytes'] <= 10
    assert c.stats()['evictions'] == c.stats()['skipped'] == 1
    assert c.get(b'c') is None


def test_ttl_not_sliding():
    now = [0]
    c = ResultCache(2, ttl=2, clock=lambda: now[0])
    c.put(b'a', [1]); now[0] = 1; assert c.get(b'a') == [1]
    now[0] = 2; assert c.get(b'a') is None
    assert c.stats()['expirations'] == 1 and c.stats()['encoded_bytes'] == 0


def test_empty_not_cached():
    c = ResultCache(2); c.put(b'a', [])
    assert c.get(b'a') is None and c.stats()['skipped'] == 1


def test_key_isolation():
    p = {'profile':'p','configuration':{'collection':'c','k':10}}
    assert len({cache_key(p,'q',10),cache_key(p,'q',3),cache_key(p,'Q',10),
                cache_key({'profile':'p','configuration':{'collection':'d','k':10}},'q',10)}) == 4


def test_default_disabled(monkeypatch):
    for key in ('RAG_SEARCH_CACHE_ENTRIES','RAG_SEARCH_CACHE_BYTES','RAG_SEARCH_CACHE_TTL_SECONDS'):
        monkeypatch.delenv(key, raising=False)
    assert not from_environment().enabled


@pytest.mark.parametrize('args', [(-1, 10, 10), (1025, 10, 10), (1, 0, 10), (1, 10, 0)])
def test_invalid_limits(args):
    with pytest.raises(ValueError): ResultCache(*args)


@pytest.fixture
def cached_api(monkeypatch):
    cache = ResultCache(128)
    calls = []
    profile = {'profile':'fixture','configuration':{'k':10,'collection':'c'}}
    def search(query,k):
        calls.append((query,k))
        return [{'doc_id':'fixture','text':query}]
    e = SimpleNamespace(cfg=object(),runtime_search=search)
    monkeypatch.setattr(api,'engine',lambda:e)
    monkeypatch.setattr(api,'search_cache',lambda:cache)
    monkeypatch.setattr(api,'active_profile',lambda cfg:profile)
    monkeypatch.setattr(api,'profile_options',lambda p,k=None:{'k':10 if k is None else k})
    with TestClient(api.app) as client: yield client, calls, profile, e


def test_api_cache_parity_and_k(cached_api):
    c,calls,_,_ = cached_api
    first=c.post('/search',json={'query':' q '})
    second=c.post('/search',json={'query':'q','k':10})
    assert first.status_code == second.status_code == 200 and first.json() == second.json()
    assert calls == [('q',None)]
    c.post('/search',json={'query':'q','k':3})
    s=c.get('/cache').json()
    assert (s['hits'],s['misses'],s['entries']) == (1,2,2)


def test_api_profile_change(cached_api):
    c,calls,p,_=cached_api
    c.post('/search',json={'query':'q'})
    p['configuration']['collection']='new'
    c.post('/search',json={'query':'q'})
    assert len(calls)==2


def test_api_cache_requires_profile(cached_api,monkeypatch):
    c,calls,_,_=cached_api
    monkeypatch.setattr(api,'active_profile',lambda cfg:None)
    assert c.post('/search',json={'query':'q'}).status_code==503
    assert not calls


def test_api_failure_not_cached(cached_api,monkeypatch):
    c,_,_,e=cached_api
    def broken(*args): raise RuntimeError('unavailable')
    monkeypatch.setattr(e,'runtime_search',broken)
    for _ in range(2): assert c.post('/search',json={'query':'q'}).status_code==503
    s=c.get('/cache').json()
    assert s['entries']==0 and s['misses']==2


def test_cache_tracking_excludes_warmup():
    cache=ResultCache(10)
    hit={'doc_id':'d1','point_id':'p1','title':'Title','text':'text','start':0,'end':4,'score':.5}
    def handler(r):
        if r.url.path=='/cache': return httpx.Response(200,json=cache.stats())
        key=json.loads(r.content)['query'].encode()
        if cache.get(key) is None: cache.put(key,[hit])
        return httpx.Response(200,json=[hit])
    with httpx.Client(base_url='http://test',transport=httpx.MockTransport(handler)) as c:
        runs=run(c,{'configuration':{'k':1}},[('q','measured',['d1'])],[('w','warm',['d1'])],[1],2,track_cache=True)
    assert runs[0]['cache']['measured_delta']['misses']==1
    assert runs[0]['cache']['measured_delta']['hits']==0
    assert runs[1]['cache']['measured_delta']['hits']==1

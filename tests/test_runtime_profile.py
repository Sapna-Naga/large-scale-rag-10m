"""Test profile integration without loading models or contacting Qdrant."""
import json
from unittest.mock import Mock
import pytest
from rag_system.config import Config
from rag_system.engine import Engine
from rag_system.runtime_profile import load_profile, profile_options


@pytest.fixture
def profile(tmp_path, monkeypatch):
    # Use a controlled profile so tests are independent of local benchmark files.
    c = dict(collection='rag_documents_v1_c384_o32', chunk_size=384,
             chunk_overlap=32, embedding='BAAI/bge-small-en-v1.5',
             embedding_revision=None, embedding_input='chunk', device='cpu',
             qdrant_endpoint='http://127.0.0.1:6333', mode='hybrid', parallel=True,
             fusion='document', rerank=False, sparse_weight=1.0,
             sparse_embedding_input='chunk', dense_weight=4.0, rrf_constant=10,
             k=10, candidates=200, ef=256, indexed_chunks=6547)
    p = tmp_path / 'profile.json'
    p.write_text(json.dumps(dict(profile='fixture', configuration=c)))
    monkeypatch.setenv('RAG_RETRIEVAL_PROFILE', str(p))
    cfg = Config(url=c['qdrant_endpoint'], collection=c['collection'],
                 chunk_size=384, overlap=32, embedding=c['embedding'], revision=None,
                 embedding_input='chunk', device='cpu')
    return p, cfg


def test_runtime_matches_frozen_options(profile):
    p, cfg = profile
    engine = Engine.__new__(Engine)
    engine.cfg = cfg
    engine.search = Mock(return_value=[])
    engine.runtime_search('test', for_answer=True)
    engine.search.assert_called_once_with('test', k=10, mode='hybrid', candidates=200,
        ef=256, rerank=False, parallel=True, fusion='document', rrf_constant=10,
        dense_weight=4.0)


def test_runtime_mismatch_rejected(profile):
    from dataclasses import replace
    p, cfg = profile
    with pytest.raises(ValueError, match='chunk_size'):
        load_profile(replace(cfg, chunk_size=256), p)


def test_explicit_k(profile):
    p, cfg = profile
    assert profile_options(load_profile(cfg, p), 5)['k'] == 5
    with pytest.raises(ValueError):
        profile_options(load_profile(cfg, p), 201)


def test_legacy_answer_retrieval(monkeypatch):
    monkeypatch.delenv('RAG_RETRIEVAL_PROFILE', raising=False)
    e = Engine.__new__(Engine)
    e.cfg = Config()
    e.search = Mock(return_value=[])
    e.runtime_search('test', for_answer=True)
    e.search.assert_called_once_with('test', k=8, candidates=50, rerank=True)


def test_invalid_profile_rejected(profile):
    p, cfg = profile
    data = json.loads(p.read_text())
    data['configuration']['rerank'] = True
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Unsupported'):
        load_profile(cfg, p)


def test_answer_empty_pool_uses_frozen_profile(profile):
    _, cfg = profile
    e = Engine.__new__(Engine)
    e.cfg = cfg
    e.search = Mock(return_value=[])
    assert e.answer('test')['status'] == 'abstained'
    assert e.search.call_args.kwargs['rerank'] is False
    assert e.search.call_args.kwargs['k'] == 10

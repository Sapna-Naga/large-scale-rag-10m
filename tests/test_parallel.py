"""Controlled fixtures test concurrent execution, not real benchmark quality."""
from threading import Barrier
from types import SimpleNamespace
import pytest
from rag_system.engine import Engine
from rag_system.core import rrf


def fixture_engine():
    # Avoid downloading models or connecting to a real database in unit tests.
    engine = Engine.__new__(Engine)
    engine.dense = object()
    engine.sparse = object()
    return engine


def rankings():
    return ([{'point_id': 'a', 'doc_id': 'd1', 'text': 'one', 'score': .8},
             {'point_id': 'b', 'doc_id': 'd2', 'text': 'two', 'score': .7}],
            [{'point_id': 'b', 'doc_id': 'd2', 'text': 'two', 'score': 5.},
             {'point_id': 'c', 'doc_id': 'd3', 'text': 'three', 'score': 4.}])


def test_parallel_overlaps_and_preserves_fusion():
    engine = fixture_engine()
    dense, sparse = rankings()
    engine._dense_search = lambda *a, **kw: dense
    engine._sparse_search = lambda *a: sparse
    baseline = engine.search('query', k=3, candidates=3)
    gate = Barrier(2, timeout=5)

    def dense_branch(query, candidates, ef):
        assert (query, candidates, ef) == ('query', 3, 128)
        gate.wait()
        return dense

    def sparse_branch(query, candidates):
        assert (query, candidates) == ('query', 3)
        gate.wait()
        return sparse

    engine._dense_search = dense_branch
    engine._sparse_search = sparse_branch
    # The barrier requires both workers to start before either can finish.
    assert engine.search('query', k=3, candidates=3, parallel=True) == baseline == rrf([dense, sparse])


def test_parallel_preserves_reranker_order():
    engine = fixture_engine()
    dense, sparse = rankings()
    engine._dense_search = lambda *a, **kw: dense
    engine._sparse_search = lambda *a: sparse
    engine.reranker = SimpleNamespace(predict=lambda pairs, **kw: [float(i) for i in range(len(pairs))])
    assert engine.search('query', k=2, candidates=3, rerank=True, parallel=True) == engine.search('query', k=2, candidates=3, rerank=True)


def test_branch_failure_is_not_silently_ignored():
    engine = fixture_engine()
    def fail(*args):
        raise RuntimeError('branch failed')
    engine._dense_search = fail
    engine._sparse_search = lambda *a: rankings()[1]
    with pytest.raises(RuntimeError, match='branch failed'):
        engine.search('query', parallel=True)


@pytest.mark.parametrize('mode', ['dense', 'exact', 'bm25'])
def test_parallel_rejects_nonhybrid_mode(mode):
    with pytest.raises(ValueError, match='requires hybrid'):
        fixture_engine().search('query', mode=mode, parallel=True)


def test_single_modes_preserve_search_options():
    engine = fixture_engine()
    dense, sparse = rankings()
    calls = []
    def dense_branch(query, candidates, ef, exact=False):
        calls.append((query, candidates, ef, exact))
        return dense
    engine._dense_search = dense_branch
    engine._sparse_search = lambda *a: sparse
    assert engine.search('query', mode='dense', k=1, candidates=3, ef=64) == dense[:1]
    assert engine.search('query', mode='exact', k=1, candidates=3, ef=64) == dense[:1]
    assert engine.search('query', mode='bm25', k=1, candidates=3) == sparse[:1]
    assert calls == [('query', 3, 64, False), ('query', 3, 64, True)]

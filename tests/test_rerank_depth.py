"""Test bounded reranking with controlled fixtures, not benchmark data."""
import json
from types import SimpleNamespace
import pytest
from rag_system.engine import Engine
from rag_system.evaluate import evaluate
from test_evaluate import FakeEngine, inputs


def fixture_engine():
    engine = Engine.__new__(Engine)
    engine.dense = engine.sparse = object()
    rows = [dict(point_id=str(i), doc_id=doc, text=f'chunk {i}', score=1.0)
            for i, doc in enumerate(['A', 'A', 'B', 'C', 'D'])]
    engine._dense_search = lambda *a, **kw: rows
    engine._sparse_search = lambda *a: rows
    return engine


def test_document_prefix_and_tail():
    engine = fixture_engine()
    args = dict(k=4, candidates=4, fusion='document')
    original = engine.search('query', **args)
    calls = []
    def predict(pairs, **kwargs):
        calls.extend(pairs)
        return [0.1, 0.9]
    engine.reranker = SimpleNamespace(predict=predict)
    result = engine.search('query', **args, rerank=True, rerank_depth=2, parallel=True)
    assert calls == [('query', original[0]['text']), ('query', original[1]['text'])]
    assert [h['doc_id'] for h in result] == ['B', 'A', 'C', 'D']
    assert [h['score_type'] for h in result] == ['cross_encoder']*2 + ['retrieval']*2
    assert result[0]['retrieval_score'] == original[1]['score']
    assert result[2]['score'] == original[2]['score']
    assert all('score_type' not in h for h in original)


def test_chunk_legacy_matches_full_explicit_depth():
    engine = fixture_engine()
    engine.reranker = SimpleNamespace(predict=lambda pairs, **kw: list(range(len(pairs))))
    args = dict(k=4, candidates=4, rerank=True)
    old = engine.search('query', **args)
    new = engine.search('query', **args, rerank_depth=4)
    assert [(h['point_id'], h['score']) for h in old] == [(h['point_id'], h['score']) for h in new]
    assert all('score_type' not in h for h in old)


def test_depth_limited_by_actual_pool_and_stable_ties():
    engine = fixture_engine()
    calls = []
    def predict(pairs, **kw):
        calls.extend(pairs)
        return [1.0]*len(pairs)
    engine.reranker = SimpleNamespace(predict=predict)
    args = dict(k=5, candidates=5, fusion='document')
    baseline = engine.search('query', **args)
    result = engine.search('query', **args, rerank=True, rerank_depth=5)
    assert len(calls) == 4
    assert [h['point_id'] for h in result] == [h['point_id'] for h in baseline]


@pytest.mark.parametrize('options', [
    {'rerank_depth': 10}, {'rerank': True, 'rerank_depth': 0},
    {'rerank': True, 'rerank_depth': 51}, {'rerank': True, 'rerank_depth': 2.5},
    {'rerank': True, 'rerank_depth': True},
])
def test_invalid_depth_rejected(options):
    with pytest.raises(ValueError):
        Engine.__new__(Engine).search('query', **options)


@pytest.mark.parametrize('scores', [[1.0], [float('nan'), 1.0], [float('inf'), 1.0]])
def test_invalid_scores_rejected(scores):
    engine = fixture_engine()
    engine.reranker = SimpleNamespace(predict=lambda *a, **kw: scores)
    with pytest.raises(ValueError, match='invalid scores'):
        engine.search('query', k=2, candidates=4, rerank=True, rerank_depth=2)


def test_evaluator_reports_prefix_and_pool_recall(tmp_path):
    queries, qrels = inputs(tmp_path)
    engine = FakeEngine()
    def search(query, **options):
        engine.calls.append((query, options))
        return [{'doc_id': doc} for doc in ['z', 'a', 'b']]
    engine.search = search
    output = tmp_path/'report.json'
    summary = evaluate(engine, queries, qrels, output, k=2, candidates=3,
                       fusion='document', rerank=True, rerank_depth=2)
    assert all(o['rerank_depth'] == 2 for _, o in engine.calls)
    assert summary['rerank_unit'] == 'unique documents'
    assert summary['rerank_depth'] == 2
    assert summary['mean_reranked_items'] == 2
    assert summary['mean_rerank_prefix_recall'] == .5
    assert summary['mean_returned_pool_recall'] == 1.0
    assert json.loads(output.read_text())['per_query'][1]['rerank_prefix_recall'] == 0

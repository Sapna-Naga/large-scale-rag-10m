"""Controlled fixtures verify fusion logic, not benchmark quality."""
import math
import json
import pytest
from rag_system.core import rrf
from rag_system.engine import Engine
from rag_system.evaluate import evaluate
from test_evaluate import FakeEngine, inputs


def hit(point, doc):
    return {'point_id': point, 'doc_id': doc, 'text': point, 'score': 1.0}


def test_chunk_default_matches_original():
    a, b, c = hit('a', 'A'), hit('b', 'B'), hit('c', 'C')
    ranks = [[a, b], [c, b]]
    scores = {'a': 1/61, 'b': 2/62, 'c': 1/61}
    expected = [dict(h, score=scores[h['point_id']]) for h in [b, a, c]]
    assert rrf(ranks) == expected
    assert rrf(ranks, weights=[1, 1], fusion='chunk') == expected


def test_document_fusion_matches_across_different_chunks():
    ranks = [[hit('a1', 'A'), hit('a2', 'A'), hit('b1', 'B')],
             [hit('c1', 'C'), hit('b2', 'B')]]
    result = rrf(ranks, fusion='document')
    assert [h['doc_id'] for h in result] == ['B', 'A', 'C']
    assert math.isclose(result[0]['score'], 2/62)
    assert result[0]['point_id'] == 'b1'
    assert math.isclose(result[1]['score'], 1/61)


def test_dense_weight_and_constant():
    a, b = hit('a', 'A'), hit('b', 'B')
    result = rrf([[b], [a]], constant=10, weights=[2, 1])
    assert [h['point_id'] for h in result] == ['b', 'a']
    assert math.isclose(result[0]['score'], 2/11)


def test_representative_from_strongest_branch():
    result = rrf([[hit('a1', 'A')], [hit('a2', 'A')]],
                 weights=[1, 2], fusion='document')
    assert result[0]['point_id'] == 'a2'
    assert math.isclose(result[0]['score'], 3/61)


def test_engine_document_fusion_parallel_and_limit():
    engine = Engine.__new__(Engine)
    engine.dense = engine.sparse = object()
    engine._dense_search = lambda *a, **kw: [hit('a1', 'A'), hit('a2', 'A')]
    engine._sparse_search = lambda *a: [hit('b1', 'B'), hit('a3', 'A')]
    args = dict(k=2, candidates=2, fusion='document', rrf_constant=10, dense_weight=2)
    sequential = engine.search('query', **args)
    assert sequential == engine.search('query', parallel=True, **args)
    assert [h['doc_id'] for h in sequential] == ['A', 'B']


@pytest.mark.parametrize('options', [
    {'mode': 'dense', 'fusion': 'document'},
    {'fusion': 'document', 'rerank': True},
    {'rrf_constant': 0},
    {'dense_weight': float('nan')},
    {'dense_weight': 0},
    {'fusion': 'invalid'},
])
def test_invalid_options(options):
    engine = Engine.__new__(Engine)
    with pytest.raises(ValueError):
        engine.search('query', **options)


def test_evaluation_metadata_and_options(tmp_path):
    queries, qrels = inputs(tmp_path)
    engine = FakeEngine()
    output = tmp_path / 'report.json'
    summary = evaluate(engine, queries, qrels, output, candidates=10, limit=1,
                       fusion='document', rrf_constant=10, dense_weight=2)
    assert summary['fusion'] == 'document'
    assert summary['rrf_constant'] == 10 and summary['dense_weight'] == 2
    assert summary['sparse_weight'] == 1.0
    assert summary['returned_pool_unit'] == 'unique documents'
    assert all(options['fusion'] == 'document' and options['rrf_constant'] == 10
               and options['dense_weight'] == 2 for _, options in engine.calls)
    assert json.loads(output.read_text(encoding='utf-8'))['fusion'] == 'document'

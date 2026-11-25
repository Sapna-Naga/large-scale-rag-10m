"""Check title-aware scoring with controlled fixtures, not benchmark data."""
import json
import sys
from types import SimpleNamespace
import pytest
from rag_system.engine import Engine
from rag_system.evaluate import evaluate
from rag_system import cli
from test_rerank_depth import fixture_engine
from test_evaluate import FakeEngine, inputs


@pytest.mark.parametrize('title', [None, '', '   '])
def test_empty_title_falls_back_to_chunk(title):
    assert Engine._rerank_text({'title': title, 'text': 'evidence'}, 'title_chunk') == 'evidence'


def test_missing_title_falls_back():
    assert Engine._rerank_text({'text': 'evidence'}, 'title_chunk') == 'evidence'


def test_title_is_not_used_by_default():
    assert Engine._rerank_text({'title': 'Title', 'text': 'evidence'}, 'chunk') == 'evidence'


def test_title_aware_prefix_preserves_original_payload_and_tail():
    engine = fixture_engine()
    rows = [dict(point_id=str(i), doc_id=str(i), title=f'Title {i}', text=f'Chunk {i}', score=1.0)
            for i in range(4)]
    engine._dense_search = lambda *a, **kw: rows
    engine._sparse_search = lambda *a, **kw: rows
    opts = dict(k=4, candidates=4, fusion='document', parallel=True)
    baseline = engine.search('query', **opts)
    calls = []
    def predict(pairs, **kw):
        calls.extend(pairs)
        return [0.1, 0.9]
    engine.reranker = SimpleNamespace(predict=predict)
    result = engine.search('query', **opts, rerank=True, rerank_depth=2, rerank_input='title_chunk')
    assert calls == [('query', h['title']+'\n\n'+h['text']) for h in baseline[:2]]
    assert [h['doc_id'] for h in result] == [h['doc_id'] for h in [baseline[1], baseline[0], *baseline[2:]]]
    assert all(h['text'] == next(b['text'] for b in baseline if b['doc_id']==h['doc_id']) for h in result)
    assert all('score_type' not in h for h in rows)


@pytest.mark.parametrize('options', [{'rerank_input': 'title_chunk'}, {'rerank': True, 'rerank_input': 'invalid'}])
def test_invalid_title_input_rejected(options):
    with pytest.raises(ValueError):
        Engine.__new__(Engine).search('query', **options)


def test_title_input_evaluation_metadata(tmp_path):
    queries, qrels = inputs(tmp_path)
    engine = FakeEngine()
    output = tmp_path/'report.json'
    summary = evaluate(engine, queries, qrels, output, k=2, candidates=3,
                       fusion='document', rerank=True, rerank_depth=2, rerank_input='title_chunk')
    assert all(o['rerank_input']=='title_chunk' for _, o in engine.calls)
    assert summary['reranker_input'] == 'title + representative chunk text'
    assert json.loads(output.read_text())['rerank_input_mode'] == 'title_chunk'


def test_cli_rejects_title_without_reranking(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['cli', 'search', 'query', '--rerank-input', 'title_chunk'])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2

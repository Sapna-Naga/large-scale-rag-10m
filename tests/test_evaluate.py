"""Synthetic unit fixtures test the evaluator, not retrieval quality."""
import json
from types import SimpleNamespace
import pytest
from rag_system.evaluate import evaluate, load_cases


def inputs(tmp_path):
    queries = tmp_path / 'queries.jsonl'
    queries.write_text('\n'.join(json.dumps({'_id': i, 'text': q}) for i, q in
                                 [('0', 'first'), ('1', 'second'), ('2', 'unused')]), encoding='utf-8')
    qrels = tmp_path / 'train.tsv'
    qrels.write_text('query-id\tcorpus-id\tscore\n0\ta\t1\n1\tb\t1\n', encoding='utf-8')
    return queries, qrels


class FakeEngine:
    cfg = SimpleNamespace(collection='unit-test', embedding='fixture', revision=None,
                          reranker='fixture', device='cpu')
    db = SimpleNamespace(count=lambda *args, **kwargs: SimpleNamespace(count=3))

    def __init__(self):
        self.calls = []

    def search(self, query, **options):
        self.calls.append((query, options))
        return [{'doc_id': x} for x in ['z', 'z', 'a']]


def test_evaluation_warmup_and_document_metrics(tmp_path):
    queries, qrels = inputs(tmp_path)
    engine = FakeEngine()
    output = tmp_path / 'report.json'
    summary = evaluate(engine, queries, qrels, output, k=2, candidates=3, limit=1,
                       warmup_query='separate warm-up')
    assert [q for q, _ in engine.calls] == ['separate warm-up', 'first']
    assert all(options['k'] == 3 for _, options in engine.calls)
    assert summary['queries'] == 1 and summary['eligible_queries'] == 2
    assert summary['metrics']['recall'] == 1 and summary['metrics']['mrr'] == .5
    assert summary['warmup_matches_evaluated_query'] is False
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['per_query'][0]['top_k_document_ids'] == ['z', 'a']
    assert report['per_query'][0]['first_relevant_rank_in_pool'] == 2
    assert report['per_query'][0]['returned_chunks'] == 3
    assert 'per_query' not in summary


def test_missing_qrels_query_fails_before_limit(tmp_path):
    queries, qrels = inputs(tmp_path)
    with qrels.open('a', encoding='utf-8') as stream:
        stream.write('missing\tc\t1\n')
    with pytest.raises(ValueError, match='missing from queries file'):
        evaluate(FakeEngine(), queries, qrels, tmp_path/'report.json', limit=1)


def test_duplicate_qrels_rejected(tmp_path):
    queries, qrels = inputs(tmp_path)
    with qrels.open('a', encoding='utf-8') as stream:
        stream.write('0\ta\t1\n')
    with pytest.raises(ValueError, match='Duplicate qrels pair'):
        load_cases(queries, qrels)


def test_collection_count_change_fails(tmp_path):
    queries, qrels = inputs(tmp_path)
    engine = FakeEngine()
    counts = iter([3, 4])
    engine.db = SimpleNamespace(count=lambda *args, **kwargs: SimpleNamespace(count=next(counts)))
    with pytest.raises(RuntimeError, match='point count changed'):
        evaluate(engine, queries, qrels, tmp_path/'report.json', limit=1)
    assert not (tmp_path/'report.json').exists()

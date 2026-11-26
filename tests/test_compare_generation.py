"""Controlled fixtures test the diagnostic; they are not benchmark data."""
import json
from types import SimpleNamespace
import pytest
from rag_system.compare_generation import bounded_request, bounded_validate, collect_evidence, accepted

SOURCES = [{'doc_id': str(i), 'citation': i} for i in range(1, 4)]


def reply(claims, finish='stop'):
    return {'choices': [{'finish_reason': finish, 'message': {'content': json.dumps({'claims': claims})}}]}


def test_bounded_schema():
    payload = bounded_request('fixture', 'question', ['[1] evidence'])
    claims = payload['response_format']['json_schema']['schema']['properties']['claims']
    assert claims['maxItems'] == 2
    assert claims['items']['properties']['citations']['maxItems'] == 1
    assert claims['items']['properties']['text']['maxLength'] == 240
    assert payload['max_tokens'] == 512


@pytest.mark.parametrize('claims', [
    [{'text': str(i), 'citations': [1]} for i in range(3)],
    [{'text': 'Finding', 'citations': [1, 2, 3]}],
    [{'text': 'x'*241, 'citations': [1]}],
    [{'text': 'Finding.', 'citations': [1]}, {'text': 'finding', 'citations': [2]}],
])
def test_limits_enforced_in_python(claims):
    with pytest.raises(ValueError):
        bounded_validate(reply(claims), SOURCES)


def test_valid_generation_and_abstention():
    result = bounded_validate(reply([{'text': 'Finding.', 'citations': [1]}]), SOURCES)
    assert result['answer'] == 'Finding. [1]'
    assert accepted('scientific_question', '1', result)
    abstained = bounded_validate(reply([]), SOURCES)
    assert accepted('out_of_domain', None, abstained)
    assert not accepted('scientific_question', '1', abstained)


def test_truncated_still_rejected():
    with pytest.raises(ValueError, match='complete'):
        bounded_validate(reply([], 'length'), SOURCES)


def test_evidence_packing_preserves_order_and_budget():
    engine = SimpleNamespace(cfg=SimpleNamespace(context_tokens=5),
                             llm_tokenizer=SimpleNamespace(encode=lambda s, **kw: s.split()))
    hits = [{'doc_id': 'a', 'text': 'first'}, {'doc_id': 'b', 'text': 'too many tokens to fit'},
            {'doc_id': 'c', 'text': 'last'}]
    blocks, sources = collect_evidence(engine, hits)
    assert blocks == ['[1] first', '[2] last']
    assert [s['doc_id'] for s in sources] == ['a', 'c']

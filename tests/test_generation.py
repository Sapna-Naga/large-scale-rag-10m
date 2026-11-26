"""Offline tests for claim-level generation and the stable public response."""
import json
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from rag_system.engine import Engine
from rag_system.generation import build_request, validate_response, ABSTENTION

SOURCES = [{'doc_id': 'fixture-a', 'citation': 1}, {'doc_id': 'fixture-b', 'citation': 2}]


def response(claims, finish='stop'):
    return {'choices': [{'finish_reason': finish, 'message': {'content': json.dumps({'claims': claims})}}]}


def test_schema_restricts_types_and_source_ids():
    request = build_request('fixture-model', 'question', ['[1] first', '[2] second'])
    schema = request['response_format']['json_schema']['schema']
    claim = schema['properties']['claims']['items']
    assert claim['properties']['citations']['items'] == {'type': 'integer', 'enum': [1, 2]}
    assert claim['properties']['citations']['minItems'] == 1
    assert claim['required'] == ['text', 'citations']
    assert schema['required'] == ['claims']
    assert not schema['additionalProperties'] and not claim['additionalProperties']
    assert json.loads(request['messages'][1]['content'])['evidence'] == ['[1] first', '[2] second']


def test_valid_cited_answer():
    assert validate_response(response([{'text': 'Finding.', 'citations': [2]}]), SOURCES) == {
        'answer': 'Finding. [2]', 'citations': [SOURCES[1]], 'status': 'generated'}


def test_valid_abstention():
    assert validate_response(response([]), SOURCES) == {
        'answer': ABSTENTION, 'citations': [], 'status': 'abstained'}


@pytest.mark.parametrize('text,ids', [
    ('Finding', [[1], [2]]), ('Finding', [999]), ('Finding', ['1']),
    ('Finding', [True]), ('Finding', [1, 1]), ('Finding', []),
    ('', [1]), (None, [1]), ('Finding', None), (ABSTENTION, [1]),
    ('Finding [1]', [1]), ('Finding [999]', [1]), ('Finding [1, 2]', [1]),
    ('Finding', [0]), ('Finding', [1.0]),
])
def test_invalid_output_rejected(text, ids):
    with pytest.raises(ValueError):
        validate_response(response([{'text': text, 'citations': ids}]), SOURCES)


def test_truncated_output_rejected():
    with pytest.raises(ValueError, match='complete'):
        validate_response(response([{'text': 'Finding', 'citations': [1]}], finish='length'), SOURCES)


def test_multiple_claims_preserve_source_association():
    claims = [{'text': 'First.', 'citations': [2]}, {'text': 'Second.', 'citations': [1, 2]}]
    result = validate_response(response(claims), SOURCES)
    assert result == {'answer': 'First. [2] Second. [1] [2]', 'citations': SOURCES, 'status': 'generated'}


@pytest.mark.parametrize('obj', [
    {'answer': 'Finding', 'citations': [1]},
    {'claims': [{'text': 'Finding'}]},
    {'claims': [{'text': 'Finding', 'citations': [1], 'extra': 0}]},
    {'claims': 'Finding'}, {'claims': [1]}, {'claims': [], 'extra': 1},
])
def test_bad_schema_rejected(obj):
    body = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(obj)}}]}
    with pytest.raises(ValueError):
        validate_response(body, SOURCES)


def test_duplicate_keys_rejected():
    body = {'choices': [{'finish_reason': 'stop', 'message': {'content': '{"claims":[],"claims":[]}'}}]}
    with pytest.raises(ValueError, match='Duplicate JSON key'):
        validate_response(body, SOURCES)


def test_invalid_later_claim_rejects_entire_answer():
    with pytest.raises(ValueError):
        validate_response(response([{'text': 'Valid', 'citations': [1]}, {'text': 'Invalid', 'citations': [3]}]), SOURCES)


def test_no_evidence_rejected():
    with pytest.raises(ValueError):
        build_request('fixture-model', 'question', [])


def test_engine_uses_schema_without_changing_retrieval(monkeypatch):
    e = Engine.__new__(Engine)
    e.cfg = SimpleNamespace(context_tokens=2000, llm_url='http://127.0.0.1:8001/v1', llm_model='fixture')
    e.runtime_search = Mock(return_value=[{'doc_id': 'fixture-a', 'text': 'Evidence'}])
    e.llm_tokenizer = SimpleNamespace(encode=lambda text, **kw: text.split())
    reply = Mock()
    reply.json.return_value = response([{'text': 'Finding.', 'citations': [1]}])
    post = Mock(return_value=reply)
    monkeypatch.setattr('httpx.post', post)
    result = e.answer('question')
    e.runtime_search.assert_called_once_with('question', None, for_answer=True)
    assert result['status'] == 'generated'
    assert result['answer'] == 'Finding. [1]'
    assert result['citations'][0]['doc_id'] == 'fixture-a'
    assert post.call_args.kwargs['json']['response_format']['type'] == 'json_schema'
    reply.json.return_value = response([{'text': 'Finding', 'citations': [[1]]}])
    assert e.answer('question')['status'] == 'invalid_model_output'

"""Offline tests for the generation contract, including observed failures."""
import json
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from rag_system.engine import Engine
from rag_system.generation import build_request, validate_response, ABSTENTION

SOURCES = [{'doc_id': 'fixture-a', 'citation': 1}, {'doc_id': 'fixture-b', 'citation': 2}]

def response(answer, ids, finish='stop'):
    return {'choices': [{'finish_reason': finish, 'message': {'content': json.dumps({'answer': answer, 'citations': ids})}}]}


def test_schema_restricts_types_and_source_ids():
    request = build_request('fixture-model', 'question', ['[1] first', '[2] second'])
    schema = request['response_format']['json_schema']['schema']
    assert schema['properties']['citations']['items'] == {'type': 'integer', 'enum': [1, 2]}
    assert schema['required'] == ['answer', 'citations']
    assert schema['additionalProperties'] is False
    assert json.loads(request['messages'][1]['content'])['evidence'] == ['[1] first', '[2] second']


def test_valid_cited_answer():
    assert validate_response(response('Finding [2].', [2]), SOURCES) == {
        'answer': 'Finding [2].', 'citations': [SOURCES[1]], 'status': 'generated'}


def test_valid_abstention():
    assert validate_response(response(ABSTENTION, []), SOURCES)['status'] == 'abstained'


@pytest.mark.parametrize('answer,ids', [
    ('Finding', [[1], [2]]), ('Finding', [1, 2]),
    ('Finding [999]', [999]), ('Finding [1]', ['1']),
    ('Finding [1]', [True]), ('Finding [1]', [2]),
    ('Finding [1]', [1, 1]), ('Finding', []),
    ('', []), (None, [1]), ('Finding [1]', None),
    (ABSTENTION, [1]),
])
def test_invalid_output_rejected(answer, ids):
    with pytest.raises(ValueError):
        validate_response(response(answer, ids), SOURCES)


def test_truncated_output_rejected():
    with pytest.raises(ValueError, match='complete'):
        validate_response(response('Finding [1]', [1], finish='length'), SOURCES)


def test_engine_uses_schema_without_changing_retrieval(monkeypatch):
    e = Engine.__new__(Engine)
    e.cfg = SimpleNamespace(context_tokens=2000, llm_url='http://127.0.0.1:8001/v1', llm_model='fixture')
    e.runtime_search = Mock(return_value=[{'doc_id': 'fixture-a', 'text': 'Evidence'}])
    e.llm_tokenizer = SimpleNamespace(encode=lambda text, **kw: text.split())
    reply = Mock()
    reply.json.return_value = response('Finding [1].', [1])
    post = Mock(return_value=reply)
    monkeypatch.setattr('httpx.post', post)
    result = e.answer('question')
    e.runtime_search.assert_called_once_with('question', None, for_answer=True)
    assert result['status'] == 'generated'
    assert result['citations'][0]['doc_id'] == 'fixture-a'
    assert post.call_args.kwargs['json']['response_format']['type'] == 'json_schema'
    reply.json.return_value = response('Finding', [[1]])
    assert e.answer('question')['status'] == 'invalid_model_output'

"""Controlled fixtures test selection mechanics, not scientific accuracy."""
import json
import pytest
from rag_system.evidence_selection import selection_request, validate_selection, development_cases

SOURCES = [{'citation': 1, 'doc_id': 'fixture1', 'text': 'First fixture passage.'},
           {'citation': 2, 'doc_id': 'fixture2', 'text': 'Second fixture passage.'}]


def response(value, finish='stop'):
    return {'choices': [{'finish_reason': finish, 'message': {'content': json.dumps(value)}}]}


def test_schema_contains_abstention_and_only_available_ids():
    payload = selection_request('fixture', 'query', ['[1] a', '[2] b'])
    assert payload['response_format']['json_schema']['schema']['properties']['source_id']['enum'] == [0, 1, 2]


def test_verbatim_selection():
    result = validate_selection(response({'source_id': 2}), SOURCES)
    assert result['status'] == 'evidence_selected'
    assert result['excerpt'] == SOURCES[1]['text']
    assert result['source'] == SOURCES[1]


def test_abstention():
    assert validate_selection(response({'source_id': 0}), SOURCES) == {'status': 'abstained', 'excerpt': None, 'source': None}


@pytest.mark.parametrize('value', [{'source_id': True}, {'source_id': '1'}, {'source_id': -1}, {'source_id': 3}, {'source_id': 1, 'answer': 'extra'}, [], {}, {'source_id': [1]}])
def test_invalid_selection(value):
    with pytest.raises(ValueError):
        validate_selection(response(value), SOURCES)


def test_incomplete_response():
    with pytest.raises(ValueError):
        validate_selection(response({'source_id': 1}, 'length'), SOURCES)


def test_duplicate_key_rejected():
    body = response({'source_id': 1})
    body['choices'][0]['message']['content'] = '{"source_id":1,"source_id":2}'
    with pytest.raises(ValueError):
        validate_selection(body, SOURCES)


def test_development_cases_skip_original_pilot(tmp_path):
    q, r = tmp_path/'q.jsonl', tmp_path/'r.tsv'
    q.write_text(''.join(json.dumps({'_id': str(i), 'text': f'Fixture query {i}'})+'\n' for i in range(25)))
    r.write_text('query-id\tcorpus-id\tscore\n'+''.join(f'{i}\td{i}\t1\n' for i in range(25)))
    assert development_cases(q, r, 2) == [('train_20', 'Fixture query 20', {'d20'}), ('train_21', 'Fixture query 21', {'d21'})]

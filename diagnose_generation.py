"""Trace two SciFact generation failures without changing the RAG engine."""
import json
import re
from unittest.mock import patch

import httpx
from rag_system.engine import Engine
from rag_system.runtime_profile import active_profile

CASES = [
    ('What role does crh-1 play in long-term associative memory in Caenorhabditis elegans?', '3202143'),
    ('76-85% of people with severe mental disorder receive no treatment in low and middle income countries.', '6490571'),
]


def show(label, value):
    print(label, json.dumps(value, indent=2, ensure_ascii=True, default=str), flush=True)


def inspect_output(response, source_count):
    """Explain the existing validator's requirements without changing them."""
    try:
        choice = response.json()['choices'][0]
        raw = choice['message']['content']
        show('FINISH_REASON', choice.get('finish_reason'))
        show('RAW_MODEL_OUTPUT', raw)
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise TypeError('Expected a JSON object')
        answer, ids = obj['answer'], obj['citations']
        if not isinstance(answer, str):
            raise TypeError('answer must be a string')
        if not isinstance(ids, list):
            raise TypeError('citations must be a list')
        inline = sorted({int(i) for i in re.findall(r'\[(\d+)\]', answer)})
        show('INLINE_CITATIONS', inline)
        show('DECLARED_CITATIONS', ids)
        errors = []
        if any(type(i) is not int or not 1 <= i <= source_count for i in ids):
            errors.append('Citation IDs must be integers in the allowed source range.')
        elif set(inline) != set(ids):
            errors.append('Inline citation IDs do not match the declared citation IDs.')
        show('VALIDATION_ERRORS', errors)
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        show('VALIDATION_ERRORS', [f'{type(exc).__name__}: {exc}'])


def diagnose(engine, question, expected_doc):
    """Observe the actual request sent by Engine.answer and its response."""
    hits = []
    original_search = engine.runtime_search
    original_post = httpx.post

    def trace_search(*args, **kwargs):
        result = original_search(*args, **kwargs)
        hits.extend(result)
        show('RETRIEVED_RANKS', [{'rank': n, 'doc_id': h['doc_id'], 'point_id': h.get('point_id'), 'title': h.get('title')} for n, h in enumerate(result, 1)])
        return result

    def trace_post(*args, **kwargs):
        payload = kwargs['json']
        user = json.loads(payload['messages'][-1]['content'])
        evidence = user['evidence']
        # Match the exact evidence string to the corresponding retrieved chunk.
        remaining = iter(hits)
        source_map = []
        for n, block in enumerate(evidence, 1):
            hit = next((h for h in remaining if block == f'[{n}] {h["text"]}'), None)
            if hit is None:
                raise RuntimeError('Evidence did not match the inspected engine format.')
            source_map.append({'citation': n, 'doc_id': hit['doc_id'], 'point_id': hit.get('point_id'), 'title': hit.get('title')})
        show('EVIDENCE_SOURCE_MAP', source_map)
        show('EXPECTED_DOC_IN_RETRIEVAL', any(str(h['doc_id']) == expected_doc for h in hits))
        show('EXPECTED_DOC_IN_CONTEXT', any(str(h['doc_id']) == expected_doc for h in source_map))
        # Only log the request body, never authorization headers.
        show('ACTUAL_LLM_REQUEST', payload)
        response = original_post(*args, **kwargs)
        show('HTTP_STATUS', response.status_code)
        response.raise_for_status()
        inspect_output(response, len(evidence))
        show('TOKEN_USAGE', response.json().get('usage'))
        return response

    show('QUESTION', question)
    show('EXPECTED_DOCUMENT', expected_doc)
    with patch.object(engine, 'runtime_search', side_effect=trace_search), patch.object(httpx, 'post', side_effect=trace_post):
        result = engine.answer(question)
    show('FINAL_APPLICATION_RESPONSE', result)


def main():
    engine = Engine()
    try:
        profile = active_profile(engine.cfg)
        if profile is None or engine.cfg.collection != 'rag_documents_v1_c384_o32':
            raise RuntimeError('Restore the frozen runtime profile before running diagnostics.')
        if engine.cfg.llm_model != 'Qwen/Qwen2.5-1.5B-Instruct' or engine.cfg.llm_url.rstrip('/') != 'http://127.0.0.1:8001/v1' or engine.cfg.context_tokens != 2000:
            raise RuntimeError('Restore the original local LLM settings and 2000-token evidence budget.')
        if engine.db.count(collection_name=engine.cfg.collection, exact=True).count != 6547:
            raise RuntimeError('Unexpected collection point count.')
        for question, expected_doc in CASES:
            diagnose(engine, question, expected_doc)
    finally:
        engine.db.close()


if __name__ == '__main__':
    main()

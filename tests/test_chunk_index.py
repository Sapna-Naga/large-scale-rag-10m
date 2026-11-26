"""Synthetic unit fixtures for the read-only chunking audit."""
import json
import sqlite3
from dataclasses import replace
from types import SimpleNamespace
import pytest
from rag_system.config import Config, index_signature
from rag_system.core import digest, point_id
from rag_system.verify_chunk_index import verify


def fixture(tmp_path):
    base = Config(collection='rag_documents_v1', embedding_input='chunk', chunk_size=256, overlap=32)
    cfg = replace(base, collection='rag_documents_v1_c128_o32', chunk_size=128)
    source = tmp_path/'corpus.jsonl'
    source.write_text(json.dumps(dict(id='a', title='Title', text='abc def'))+'\n', encoding='utf-8')
    for name, c, count in [('base',base,1),('new',cfg,2)]:
        with sqlite3.connect(tmp_path/name) as db:
            db.execute('create table settings(key text, value text)')
            db.execute('insert into settings values (?,?)',('index',index_signature(c)))
            db.execute('create table docs(id text, hash text, chunks int, pending int)')
            db.execute('insert into docs values (?,?,?,0)',('a',digest('Title','abc def'),count))
    parts = [dict(doc_id='a',title='Title',text='abc',start=0,end=3,chunk_no=0),
             dict(doc_id='a',title='Title',text='def',start=4,end=7,chunk_no=1)]
    points = [SimpleNamespace(id=point_id('a',i),payload=dict(p)) for i,p in enumerate(parts)]
    db = SimpleNamespace(count=lambda *a,**kw:SimpleNamespace(count=len(points)),
        scroll=lambda **kw:(points,None))
    engine = SimpleNamespace(cfg=cfg,db=db,make_chunks=lambda *a:iter(parts))
    return engine, source, tmp_path/'new', tmp_path/'base', points


def test_valid_chunk_index(tmp_path):
    e,s,n,b,_=fixture(tmp_path)
    r=verify(e,s,n,b)
    assert r['points']==2 and r['baseline_points']==1 and r['documents']==1


def test_payload_mismatch(tmp_path):
    e,s,n,b,points=fixture(tmp_path)
    points[0].payload['text']='wrong'
    with pytest.raises(ValueError,match='payload mismatch'):
        verify(e,s,n,b)


def test_missing_point(tmp_path):
    e,s,n,b,points=fixture(tmp_path)
    points.pop()
    with pytest.raises(ValueError,match='point count'):
        verify(e,s,n,b)


def test_changed_source(tmp_path):
    e,s,n,b,_=fixture(tmp_path)
    s.write_text('{"id":"a","title":"Title","text":"different"}\n')
    with pytest.raises(ValueError,match='Source hash mismatch'):
        verify(e,s,n,b)


def test_baseline_protection(tmp_path):
    e,s,n,b,_=fixture(tmp_path)
    e.cfg=replace(e.cfg,collection='rag_documents_v1')
    with pytest.raises(ValueError,match='separate chunking'):
        verify(e,s,n,b)

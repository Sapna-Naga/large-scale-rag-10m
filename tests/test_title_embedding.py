"""Controlled fixtures for indexing safety and dense input construction."""
import json
from dataclasses import replace
from types import SimpleNamespace
import pytest
from rag_system.config import Config, index_signature
from rag_system.engine import Engine
from rag_system.ingest import ingest, delete_document
from rag_system.evaluate import evaluate
from test_evaluate import FakeEngine as EvalEngine, inputs
from test_ingest import FakeEngine


@pytest.mark.parametrize('collection,mode', [('rag_documents_v1','title_chunk'),
                                              ('fixture_title_dense','chunk'),
                                              ('fixture','bad')])
def test_invalid_index_pair(collection, mode):
    with pytest.raises(ValueError):
        Config(collection=collection, embedding_input=mode)


def test_legacy_signature_preserved():
    cfg = Config(collection='rag_documents_v1', embedding_input='chunk')
    assert index_signature(cfg) == json.dumps([cfg.url,cfg.collection,cfg.embedding,
                                               cfg.revision,cfg.chunk_size,cfg.overlap])
    title = replace(cfg, collection='fixture_title_dense', embedding_input='title_chunk')
    assert json.loads(index_signature(title))[-1] == 'title_chunk'


class Vector(list):
    def tolist(self):
        return list(self)


def build_engine(mode, length=3):
    engine = Engine.__new__(Engine)
    engine.cfg = Config(collection='fixture_title_dense' if mode=='title_chunk' else 'fixture',
                        embedding_input=mode)
    calls = {}
    def encode(texts, **kw):
        calls['dense'] = list(texts)
        return [Vector([1.0, 0.0]) for _ in texts]
    def sparse(texts, **kw):
        calls['sparse'] = list(texts)
        return [SimpleNamespace(indices=Vector([1]),values=Vector([1.0])) for _ in texts]
    def tokenize(texts, **kw):
        calls['tokenize'] = list(texts)
        assert kw['truncation'] is False
        return {'input_ids': [[0]*length for _ in texts]}
    engine.dense = SimpleNamespace(encode=encode, tokenizer=tokenize,max_seq_length=8)
    engine.sparse = SimpleNamespace(embed=sparse)
    engine.m = SimpleNamespace(PointStruct=lambda **kw: kw,SparseVector=lambda **kw: kw,
                               WriteOrdering=SimpleNamespace(STRONG='strong'))
    engine.db = SimpleNamespace(upsert=lambda collection,points,**kw:calls.update(points=points))
    return engine,calls


@pytest.mark.parametrize('mode', ['chunk','title_chunk'])
def test_only_dense_input_changes(mode):
    engine,calls=build_engine(mode)
    rows=[dict(doc_id='a',chunk_no=0,title='Title',text='Evidence',start=0,end=8),
          dict(doc_id='b',chunk_no=0,title='',text='Other',start=0,end=5)]
    engine.upsert(rows)
    assert calls['dense']==(['Title\n\nEvidence','Other'] if mode=='title_chunk' else ['Evidence','Other'])
    assert calls['sparse']==['Evidence','Other']
    assert [p['payload'] for p in calls['points']]==rows
    assert rows[0]['text']=='Evidence'


def test_combined_token_overflow_rejected_before_write():
    engine,calls=build_engine('title_chunk',length=9)
    with pytest.raises(ValueError, match='exceeds embedding token limit'):
        engine.upsert([dict(doc_id='a',chunk_no=0,title='Title',text='Evidence')])
    assert 'dense' not in calls and 'points' not in calls


def test_checkpoint_mismatch_rejects_ingest_and_delete(tmp_path):
    e=FakeEngine()
    source=tmp_path/'docs.jsonl'; source.write_text('{"id":"a","text":"one"}\n')
    state=tmp_path/'checkpoint.sqlite'
    ingest(e,source,state)
    e.cfg=SimpleNamespace(**vars(e.cfg),embedding_input='title_chunk')
    with pytest.raises(ValueError, match='different index configuration'):
        ingest(e,source,state)
    with pytest.raises(ValueError, match='different index configuration'):
        delete_document(e,state,'a')


def test_new_title_checkpoint_requires_empty_collection(tmp_path):
    e=FakeEngine(); e.cfg=SimpleNamespace(**vars(e.cfg),embedding_input='title_chunk')
    e.db=SimpleNamespace(count=lambda *a,**kw: SimpleNamespace(count=1))
    source=tmp_path/'docs.jsonl'; source.write_text('{"id":"a","text":"one"}\n')
    with pytest.raises(ValueError, match='not empty'):
        ingest(e,source,tmp_path/'checkpoint.sqlite')
    assert not e.points


def test_report_includes_embedding_input(tmp_path):
    e=EvalEngine(); e.cfg=SimpleNamespace(**vars(e.cfg),embedding_input='title_chunk')
    q,r=inputs(tmp_path)
    report=evaluate(e,q,r,tmp_path/'report.json',limit=1)
    assert report['embedding_input']=='title_chunk' and report['sparse_embedding_input']=='chunk'


def test_compare_indexes_detects_changes(tmp_path):
    from rag_system.verify_title_index import compare_indexes
    from rag_system.core import point_id
    from test_migrate_endpoint import Point, FakeClient
    import sqlite3
    base = Config(collection='baseline', embedding_input='chunk')
    title = replace(base, collection='baseline_title_dense', embedding_input='title_chunk')
    paths = [tmp_path/'base.sqlite', tmp_path/'title.sqlite']
    for cfg, path in zip([base,title], paths):
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE settings(key TEXT PRIMARY KEY, value TEXT)')
            db.execute('INSERT INTO settings VALUES (?,?)',('index',index_signature(cfg)))
            db.execute('CREATE TABLE docs(id TEXT, hash TEXT, chunks INTEGER, pending INTEGER)')
            db.execute("INSERT INTO docs VALUES ('a','hash',1,0)")
    a = FakeClient(); b = FakeClient()
    a.points = a.points[:1]; b.points = b.points[:1]
    b.points[0].vector['dense'] = [0.2,0.1]
    client = SimpleNamespace(count=lambda name,**kw: (a if name=='baseline' else b).count(),
                             scroll=lambda collection_name,**kw: (a if collection_name=='baseline' else b).scroll(collection_name=collection_name,**kw))
    result=compare_indexes(client,title,'baseline',*paths)
    assert result==dict(documents=1,points=1,payload_mismatches=0,sparse_vector_mismatches=0,changed_dense_vectors=1)
    b.points[0].vector['bm25']['values']=[2.0]
    with pytest.raises(ValueError,match='did not isolate'):
        compare_indexes(client,title,'baseline',*paths)

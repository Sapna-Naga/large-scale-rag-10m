import json
from types import SimpleNamespace
import pytest
from rag_system.ingest import ingest, delete_document

class FakeEngine:
    """In-memory test double; never used as benchmark data."""
    cfg = SimpleNamespace(url='test',collection='test',embedding='test',revision=None,chunk_size=2,overlap=0,batch=3)
    def __init__(self):
        self.points={}
        self.fail=False
    def make_chunks(self,doc_id,title,text):
        for i,part in enumerate(text.split()):
            yield {'doc_id':doc_id,'chunk_no':i,'text':part}
    def upsert(self,rows):
        for row in rows:
            self.points[row['doc_id'],row['chunk_no']]=row
        if self.fail:
            raise RuntimeError('Injected failure')
    def delete(self,doc_id):
        self.points={k:v for k,v in self.points.items() if k[0]!=doc_id}


def test_replay_update_delete(tmp_path):
    e=FakeEngine()
    p=tmp_path/'docs.jsonl'; state=tmp_path/'state.sqlite'
    def write(rows):
        p.write_text('\n'.join(json.dumps({'id':key,'text':value}) for key,value in rows))
    write([('a','one two three four'),('b','five'),('c','five')])
    e.fail=True
    with pytest.raises(RuntimeError):
        ingest(e,p,state)
    e.fail=False
    report=ingest(e,p,state)
    assert report['checkpoint_documents']==2 and len(e.points)==5 and report['duplicate']==1
    assert ingest(e,p,state)['unchanged']==2
    write([('a','new')])
    assert ingest(e,p,state)['checkpoint_chunks']==2
    assert len(e.points)==2
    delete_document(e,state,'a')
    assert len(e.points)==1

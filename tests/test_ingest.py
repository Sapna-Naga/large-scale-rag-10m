import json
from types import SimpleNamespace
from rag_system.ingest import ingest


class FakeEngine:
    # Test fixture only: no fabricated benchmark results.
    def __init__(self):
        self.cfg = SimpleNamespace(url="test", collection="test", dense_model="test", dense_revision=None,
                                   chunk_size=3, overlap=1, batch=2)
        self.dense = SimpleNamespace(tokenizer=lambda text, **kw: {"offset_mapping": [(i,i+1) for i in range(len(text))]})
        self.points = {}
    def ensure_collection(self): pass
    def upsert(self, rows):
        self.points.update({r["chunk_id"]:r for r in rows})
    def delete(self, doc_id):
        self.points = {k:v for k,v in self.points.items() if v["doc_id"] != doc_id}


def test_replay_and_update(tmp_path):
    path = tmp_path / "corpus.jsonl"
    manifest = tmp_path / "manifest.sqlite"
    e = FakeEngine()
    path.write_text(json.dumps({"id":"a", "text":"abcdef"})+"\n")
    assert ingest(e,path,manifest)["indexed"] == 1
    assert len(e.points) == 3
    assert ingest(e,path,manifest)["unchanged"] == 1
    path.write_text(json.dumps({"id":"a", "text":"abc"})+"\n")
    assert ingest(e,path,manifest)["indexed"] == 1
    assert len(e.points) == 1

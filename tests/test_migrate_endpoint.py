"""Unit tests use simulated Qdrant responses, never benchmark data."""
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from rag_system.core import point_id
from rag_system.migrate_endpoint import migrate, signature, OLD_URL, NEW_URL


@dataclass
class Point:
    id: str
    payload: dict
    vector: dict

    def model_dump(self, **kwargs):
        return {'id': self.id, 'payload': self.payload, 'vector': self.vector}


class FakeClient:
    def __init__(self):
        self.points = [Point(point_id('a', i), {'doc_id': 'a', 'chunk_no': i},
                             {'dense': [0.1, 0.2], 'bm25': {'indices': [1], 'values': [1.0]}})
                       for i in range(2)]

    def count(self, *args, **kwargs):
        return SimpleNamespace(count=len(self.points))

    def get_collection(self, *args):
        return SimpleNamespace(config={'test': True})

    def scroll(self, *, collection_name, limit, offset, with_payload, with_vectors):
        # Force multiple pages to exercise the scroll loop.
        if offset is None:
            return self.points[:1], self.points[1].id if len(self.points) > 1 else None
        return self.points[1:], None


@pytest.fixture
def checkpoint(tmp_path):
    cfg = SimpleNamespace(url=NEW_URL, collection='fixture', embedding='fixture',
                          revision=None, chunk_size=256, overlap=32)
    path = tmp_path / 'checkpoint.sqlite'
    with closing(sqlite3.connect(path)) as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT)')
        db.execute('CREATE TABLE docs (id TEXT PRIMARY KEY, hash TEXT, chunks INTEGER, pending INTEGER)')
        db.execute("INSERT INTO settings VALUES ('index', ?)", (signature(cfg, OLD_URL),))
        db.execute("INSERT INTO docs VALUES ('a', 'hash', 2, 0)")
        db.commit()
    return path, cfg


def read_signature(path):
    with closing(sqlite3.connect(path)) as db:
        return db.execute("SELECT value FROM settings WHERE key='index'").fetchone()[0]


def test_validate_and_migrate(checkpoint):
    path, cfg = checkpoint
    assert migrate(path, cfg, FakeClient(), FakeClient())['status'] == 'validated_only'
    assert read_signature(path) == signature(cfg, OLD_URL)
    result = migrate(path, cfg, FakeClient(), FakeClient(), apply=True)
    assert result['status'] == 'migrated' and result['matching_endpoint_points'] == 2
    assert read_signature(result['backup']) == signature(cfg, OLD_URL)
    assert read_signature(path) == signature(cfg, NEW_URL)
    assert migrate(path, cfg, FakeClient(), FakeClient(), apply=True)['status'] == 'already_migrated'


def test_different_vector_rejected(checkpoint):
    path, cfg = checkpoint
    new = FakeClient()
    new.points[0].vector['dense'] = [0.2, 0.1]
    with pytest.raises(ValueError, match='different payloads or vectors'):
        migrate(path, cfg, FakeClient(), new, apply=True)
    assert read_signature(path) == signature(cfg, OLD_URL)
    assert not list(path.parent.glob('*.bak'))


def test_pending_rejected(checkpoint):
    path, cfg = checkpoint
    with closing(sqlite3.connect(path)) as db:
        db.execute('UPDATE docs SET pending=1')
        db.commit()
    with pytest.raises(ValueError, match='unfinished documents'):
        migrate(path, cfg, FakeClient(), FakeClient(), apply=True)
    assert read_signature(path) == signature(cfg, OLD_URL)


def test_point_mismatch_rejected(checkpoint):
    path, cfg = checkpoint
    new = FakeClient()
    new.points[0].payload['doc_id'] = 'wrong'
    with pytest.raises(ValueError, match='payload does not match'):
        migrate(path, cfg, FakeClient(), new, apply=True)
    assert read_signature(path) == signature(cfg, OLD_URL)


def test_other_configuration_rejected(checkpoint):
    path, cfg = checkpoint
    cfg.collection = 'another'
    with pytest.raises(ValueError, match='configuration does not match'):
        migrate(path, cfg, FakeClient(), FakeClient(), apply=True)
    assert json.loads(read_signature(path))[1] == 'fixture'

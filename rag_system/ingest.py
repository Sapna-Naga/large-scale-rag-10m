"""Single-writer ingestion with bounded batches and disk-backed checkpoints."""
import json
import sqlite3
from pathlib import Path
from .core import records, document, digest


def ingest(engine, source, state, limit=None):
    path = Path(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    try:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)')
        db.execute('CREATE TABLE IF NOT EXISTS docs (id TEXT PRIMARY KEY, hash TEXT NOT NULL, chunks INTEGER NOT NULL, pending INTEGER NOT NULL)')
        db.execute('CREATE INDEX IF NOT EXISTS docs_hash ON docs(hash)')
        signature = json.dumps([engine.cfg.url, engine.cfg.collection, engine.cfg.embedding, engine.cfg.revision, engine.cfg.chunk_size, engine.cfg.overlap])
        previous = db.execute("SELECT value FROM settings WHERE key='index'").fetchone()
        if previous and previous[0] != signature:
            raise ValueError('Checkpoint belongs to a different index configuration')
        db.execute("INSERT OR IGNORE INTO settings VALUES ('index', ?)", (signature,))
        db.commit()
        stats = dict(read=0, indexed=0, unchanged=0, duplicate=0, chunks=0)
        batch, completed = [], []

        def flush():
            # A document is marked complete only after every chunk is acknowledged.
            if batch:
                engine.upsert(batch)
                batch.clear()
            if completed:
                db.executemany('UPDATE docs SET pending=0, chunks=? WHERE id=?', completed)
                db.commit()
                stats['indexed'] += len(completed)
                stats['chunks'] += sum(count for count,_ in completed)
                completed.clear()

        for row in records(source):
            if limit is not None and stats['read'] >= limit:
                break
            stats['read'] += 1
            doc_id, title, text = document(row)
            h = digest(title, text)
            old = db.execute('SELECT hash,pending FROM docs WHERE id=?', (doc_id,)).fetchone()
            if old and old[1] and any(key==doc_id for _,key in completed):
                flush()
                old = db.execute('SELECT hash,pending FROM docs WHERE id=?', (doc_id,)).fetchone()
            if old and old == (h,0):
                stats['unchanged'] += 1
                continue
            if db.execute('SELECT 1 FROM docs WHERE hash=? AND id<>?', (h,doc_id)).fetchone():
                if old:
                    raise ValueError(f'Update would duplicate another document: {doc_id}')
                stats['duplicate'] += 1
                continue
            # Record pending status before any remote writes. Replay cleans partial chunks.
            db.execute('INSERT OR REPLACE INTO docs VALUES (?,?,0,1)', (doc_id,h))
            db.commit()
            if old:
                engine.delete(doc_id)
            count = 0
            for chunk in engine.make_chunks(doc_id,title,text):
                batch.append(chunk)
                count += 1
                if len(batch) >= engine.cfg.batch:
                    flush()
            if not count:
                raise ValueError(f'No tokens in document {doc_id}')
            completed.append((count,doc_id))
            if len(completed) >= engine.cfg.batch:
                flush()
            if stats['read'] % 1000 == 0:
                print(json.dumps(stats), flush=True)
        flush()
        stats['checkpoint_documents'], stats['checkpoint_chunks'] = db.execute('SELECT COUNT(*), COALESCE(SUM(chunks),0) FROM docs WHERE pending=0').fetchone()
        stats['pending_documents'] = db.execute('SELECT COUNT(*) FROM docs WHERE pending=1').fetchone()[0]
        return stats
    finally:
        db.close()


def delete_document(engine, state, doc_id):
    # Keep the checkpoint until remote deletion succeeds; retry after a failure.
    with sqlite3.connect(state) as db:
        signature = json.dumps([engine.cfg.url, engine.cfg.collection, engine.cfg.embedding, engine.cfg.revision, engine.cfg.chunk_size, engine.cfg.overlap])
        if db.execute("SELECT value FROM settings WHERE key='index'").fetchone() != (signature,):
            raise ValueError('Checkpoint belongs to a different index configuration')
        db.execute('UPDATE docs SET pending=1 WHERE id=?', (doc_id,))
        db.commit()
        engine.delete(doc_id)
        db.execute('DELETE FROM docs WHERE id=?', (doc_id,))

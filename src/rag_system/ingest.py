"""Restartable single-writer ingestion with an on-disk document manifest."""
import json
import sqlite3
import time
from pathlib import Path
from .core import Document, chunks, content_hash


def ingest(engine, path, manifest, limit=None):
    """Replay the input safely after a crash. Do not run concurrent writers."""
    Path(manifest).parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    stats = dict(read=0, indexed=0, unchanged=0, chunks_written=0)
    engine.ensure_collection()
    with sqlite3.connect(manifest) as db, open(path, encoding="utf-8") as f:
        db.execute("CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, digest TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        fingerprint = json.dumps([engine.cfg.url, engine.cfg.collection, engine.cfg.dense_model,
            engine.cfg.dense_revision, engine.cfg.chunk_size, engine.cfg.overlap])
        old = db.execute("SELECT value FROM metadata WHERE key='config'").fetchone()
        if old and old[0] != fingerprint:
            raise ValueError("Manifest belongs to a different index configuration")
        db.execute("INSERT OR IGNORE INTO metadata VALUES ('config', ?)", (fingerprint,))
        db.commit()
        # Commit a group only after all its chunks have been acknowledged by Qdrant.
        pending, rows = {}, []
        def flush():
            if rows:
                engine.upsert(rows)
                stats["chunks_written"] += len(rows)
                rows.clear()
            if pending:
                db.executemany("INSERT OR REPLACE INTO documents VALUES (?, ?)", pending.items())
                db.commit()
                stats["indexed"] += len(pending)
                pending.clear()
        for line_number, line in enumerate(f, 1):
            if limit is not None and stats["read"] >= limit:
                break
            if not line.strip():
                continue
            try:
                doc = Document.from_dict(json.loads(line))
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"Invalid document at line {line_number}") from exc
            stats["read"] += 1
            digest = content_hash(doc)
            if doc.id in pending:
                flush()
            previous = db.execute("SELECT digest FROM documents WHERE id=?", (doc.id,)).fetchone()
            if previous and previous[0] == digest:
                stats["unchanged"] += 1
                continue
            # Clear previous chunks before rewriting an updated document. A failed rewrite
            # is retried on replay. Updates are not atomic; use an offline rebuild if needed.
            if previous:
                db.execute("DELETE FROM documents WHERE id=?", (doc.id,))
                db.commit()
            if previous:
                engine.delete(doc.id)
            # A prior interrupted initial write can leave stale chunks if input changes.
            # Resume requires the same immutable input file until a run completes.
            chunk_count = 0
            for chunk in chunks(doc, engine.dense.tokenizer, engine.cfg.chunk_size, engine.cfg.overlap):
                chunk_count += 1
                rows.append(chunk)
                if len(rows) >= engine.cfg.batch:
                    engine.upsert(rows)
                    stats["chunks_written"] += len(rows)
                    rows.clear()
            if not chunk_count:
                raise ValueError(f"Document {doc.id} yielded no tokens")
            pending[doc.id] = digest
            if len(pending) >= engine.cfg.batch:
                flush()
        flush()
        stats["manifest_documents"] = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    stats["seconds"] = time.perf_counter() - start
    stats["documents_per_second"] = stats["read"] / max(stats["seconds"], 1e-9)
    return stats


def delete_document(engine, manifest, doc_id):
    # Invalidate the checkpoint first so interrupted deletes cannot cause skipped reingestion.
    with sqlite3.connect(manifest) as db:
        db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        db.commit()
    engine.delete(doc_id)

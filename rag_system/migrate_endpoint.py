"""Safely move a local SciFact checkpoint from localhost to IPv4.

Stop ingestion, deletion and other collection writes before running.
This tool only reads Qdrant. By default it also leaves SQLite unchanged.
"""
import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from .config import Config, index_signature
from .core import point_id

OLD_URL = 'http://localhost:6333'
NEW_URL = 'http://127.0.0.1:6333'


def signature(cfg, url):
    # Match the signature used by both ingestion and document deletion.
    return index_signature(cfg, url)


def fingerprint(client, collection, expected):
    """Check every expected chunk and hash its payload and both vectors."""
    if client.count(collection, exact=True).count != len(expected):
        raise ValueError('Collection count differs from checkpoint')
    found = {}
    offset = None
    while True:
        points, next_offset = client.scroll(collection_name=collection,
                                            limit=1000, offset=offset,
                                            with_payload=True, with_vectors=True)
        for point in points:
            key = str(point.id)
            payload = point.payload or {}
            if key not in expected or key in found:
                raise ValueError(f'Unexpected or repeated point ID: {key}')
            if (payload.get('doc_id'), payload.get('chunk_no')) != expected[key]:
                raise ValueError(f'Point payload does not match checkpoint: {key}')
            if not isinstance(point.vector, dict) or not {'dense', 'bm25'} <= point.vector.keys():
                raise ValueError(f'Missing named vectors: {key}')
            # Pydantic converts sparse vectors and UUIDs to JSON-compatible values.
            data = point.model_dump(mode='json', include={'id', 'payload', 'vector'})
            encoded = json.dumps(data, sort_keys=True, ensure_ascii=False,
                                 separators=(',', ':'), allow_nan=False).encode('utf-8')
            found[key] = hashlib.sha256(encoded).hexdigest()
        if next_offset is None:
            break
        if not points or next_offset == offset:
            raise ValueError('Qdrant pagination did not advance')
        offset = next_offset
    if found.keys() != expected.keys():
        raise ValueError('Some checkpoint points are missing from the collection')
    if client.count(collection, exact=True).count != len(expected):
        raise ValueError('Collection count changed during validation')
    return found


def backup_checkpoint(path):
    # Create a unique backup without overwriting an earlier backup.
    fd, name = tempfile.mkstemp(prefix=path.name + '.before-ipv4-',
                                suffix='.bak', dir=path.parent)
    os.close(fd)
    backup = Path(name)
    try:
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as source:
            with closing(sqlite3.connect(backup)) as target:
                source.backup(target)
                if target.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
                    raise ValueError('Backup integrity check failed')
    except Exception:
        backup.unlink(missing_ok=True)
        raise
    return backup


def migrate(path, cfg, old_client, new_client, apply=False):
    """Verify the local migration and optionally commit only its URL change."""
    if cfg.url != NEW_URL:
        raise ValueError(f'Configure QDRANT_URL={NEW_URL} first')
    path = Path(path).resolve(strict=True)
    old_signature, new_signature = signature(cfg, OLD_URL), signature(cfg, NEW_URL)
    # An immediate transaction prevents other SQLite writers while validating.
    # Read-only mode is used when no migration was requested.
    uri = path.as_uri() + ('?mode=rw' if apply else '?mode=ro')
    with closing(sqlite3.connect(uri, uri=True, timeout=10)) as db:
        try:
            db.execute('BEGIN IMMEDIATE' if apply else 'BEGIN')
            if db.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
                raise ValueError('Checkpoint integrity check failed')
            previous = db.execute("SELECT value FROM settings WHERE key='index'").fetchone()
            if previous not in ((old_signature,), (new_signature,)):
                raise ValueError('Checkpoint configuration does not match the expected local index')
            rows = db.execute('SELECT id, chunks, pending FROM docs').fetchall()
            if not rows or any(pending != 0 or chunks < 1 for _, chunks, pending in rows):
                raise ValueError('Checkpoint is empty or contains unfinished documents')
            expected = {point_id(doc, i): (doc, i)
                        for doc, chunks, _ in rows for i in range(chunks)}
            if len(expected) != sum(chunks for _, chunks, _ in rows):
                raise ValueError('Duplicate expected point IDs')
            if old_client.get_collection(cfg.collection).config != new_client.get_collection(cfg.collection).config:
                raise ValueError('The endpoints have different collection configurations')
            print('Checking localhost points and vectors...', flush=True)
            old = fingerprint(old_client, cfg.collection, expected)
            print('Checking IPv4 points and vectors...', flush=True)
            new = fingerprint(new_client, cfg.collection, expected)
            if old != new:
                raise ValueError('The endpoints returned different payloads or vectors')
            result = {'documents': len(rows), 'points': len(expected),
                      'matching_endpoint_points': len(old), 'pending_documents': 0,
                      'old_url': OLD_URL, 'new_url': NEW_URL}
            if previous == (new_signature,):
                result['status'] = 'already_migrated'
            elif apply:
                # SQLite backup copies committed data, including committed WAL contents.
                backup = backup_checkpoint(path)
                db.execute("UPDATE settings SET value=? WHERE key='index'", (new_signature,))
                if db.execute("SELECT value FROM settings WHERE key='index'").fetchone() != (new_signature,):
                    raise RuntimeError('Checkpoint update could not be verified')
                db.commit()
                result.update(status='migrated', backup=str(backup))
            else:
                result['status'] = 'validated_only'
            return result
        finally:
            # Roll back an uncommitted transaction if validation or backup failed.
            if db.in_transaction:
                db.rollback()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', required=True)
    parser.add_argument('--apply', action='store_true', help='Commit migration after all checks')
    args = parser.parse_args()
    from qdrant_client import QdrantClient
    cfg = Config()
    with closing(QdrantClient(url=OLD_URL, api_key=os.getenv('QDRANT_API_KEY'), timeout=120)) as old:
        with closing(QdrantClient(url=NEW_URL, api_key=os.getenv('QDRANT_API_KEY'), timeout=120)) as new:
            print(json.dumps(migrate(args.state, cfg, old, new, args.apply), indent=2))


if __name__ == '__main__':
    main()

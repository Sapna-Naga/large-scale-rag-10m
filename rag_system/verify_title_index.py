"""Read-only comparison of two small experimental collections (not a 10M audit)."""
import argparse
import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path

from .config import Config, index_signature
from .core import point_id


def checkpoint_rows(path, cfg):
    # Read-only mode prevents accidental creation or modification of checkpoints.
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)) as db:
        if db.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
            raise ValueError('Checkpoint integrity check failed')
        if db.execute("SELECT value FROM settings WHERE key='index'").fetchone() != (index_signature(cfg),):
            raise ValueError('Checkpoint does not match collection configuration')
        rows = db.execute('SELECT id, hash, chunks, pending FROM docs ORDER BY id').fetchall()
        if not rows or any(n < 1 or pending for _, _, n, pending in rows):
            raise ValueError('Checkpoint is empty or contains unfinished documents')
        return rows


def hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def collection_hashes(client, name, expected):
    if client.count(name, exact=True).count != len(expected):
        raise ValueError(f'{name}: point count differs from checkpoint')
    hashes, offset = {}, None
    while True:
        points, next_offset = client.scroll(collection_name=name, limit=500, offset=offset,
                                            with_payload=True, with_vectors=True)
        for point in points:
            key = str(point.id)
            data = point.model_dump(mode='json', include={'payload','vector'})
            payload, vectors = data['payload'], data['vector']
            if key not in expected or key in hashes or (payload['doc_id'], payload['chunk_no']) != expected[key]:
                raise ValueError(f'{name}: unexpected, repeated or mismatched point: {key}')
            sparse = vectors['bm25']
            # Normalize the order of sparse terms before comparing values.
            terms = sorted(zip(sparse['indices'], sparse['values'], strict=True))
            hashes[key] = (hash_json(payload), hash_json(terms), hash_json(vectors['dense']))
        if next_offset is None:
            break
        if not points or next_offset == offset:
            raise ValueError('Pagination did not advance')
        offset = next_offset
    if hashes.keys() != expected.keys() or client.count(name, exact=True).count != len(expected):
        raise ValueError(f'{name}: missing points or collection changed during verification')
    return hashes


def compare_indexes(client, cfg, baseline, baseline_state, title_state):
    if cfg.embedding_input != 'title_chunk':
        raise ValueError('Enable title-aware embedding mode before verification')
    control = replace(cfg, collection=baseline, embedding_input='chunk')
    rows = checkpoint_rows(baseline_state, control)
    if rows != checkpoint_rows(title_state, cfg):
        raise ValueError('Source document hashes or chunk counts differ')
    expected = {point_id(doc, i): (doc, i) for doc, _, n, _ in rows for i in range(n)}
    print('Verifying baseline payloads and vectors...', flush=True)
    old = collection_hashes(client, baseline, expected)
    print('Verifying title-aware payloads and vectors...', flush=True)
    new = collection_hashes(client, cfg.collection, expected)
    payload_changes = sum(old[k][0] != new[k][0] for k in old)
    sparse_changes = sum(old[k][1] != new[k][1] for k in old)
    dense_changes = sum(old[k][2] != new[k][2] for k in old)
    result = dict(documents=len(rows), points=len(expected), payload_mismatches=payload_changes,
                  sparse_vector_mismatches=sparse_changes, changed_dense_vectors=dense_changes)
    print(json.dumps(result, indent=2), flush=True)
    if payload_changes or sparse_changes or not dense_changes:
        raise ValueError('The new index did not isolate the dense embedding change')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='rag_documents_v1')
    parser.add_argument('--baseline-state', required=True)
    parser.add_argument('--title-state', required=True)
    args = parser.parse_args()
    from .engine import Engine
    engine = Engine()
    try:
        compare_indexes(engine.db, engine.cfg, args.baseline, args.baseline_state, args.title_state)
    finally:
        engine.db.close()


if __name__ == '__main__':
    main()

"""Read-only audit of a small SciFact chunking experiment against its source."""
import argparse
import json
from dataclasses import replace

from .core import records, document, digest, point_id
from .verify_title_index import checkpoint_rows


def verify(engine, source, state, baseline_state):
    cfg = engine.cfg
    if cfg.embedding_input != 'chunk':
        raise ValueError('This experiment requires chunk-only embeddings')
    baseline = replace(cfg, collection='rag_documents_v1', chunk_size=256, overlap=32)
    expected_name = f'rag_documents_v1_c{cfg.chunk_size}_o{cfg.overlap}'
    if cfg.collection != expected_name or (cfg.chunk_size, cfg.overlap) == (256, 32):
        raise ValueError('Select a separate chunking experiment collection')
    old = checkpoint_rows(baseline_state, baseline)
    current = checkpoint_rows(state, cfg)
    if [(d,h) for d,h,_,_ in old] != [(d,h) for d,h,_,_ in current]:
        raise ValueError('Document IDs or content hashes differ from baseline')

    # Reconstruct the expected chunks using the same tokenizer and settings.
    # This audit is deliberately for the small SciFact corpus, not 10M documents.
    expected = {}
    seen = set()
    checkpoints = {d: (h,n) for d,h,n,_ in current}
    for row in records(source):
        doc_id, title, text = document(row)
        if doc_id in seen or doc_id not in checkpoints:
            raise ValueError(f'Unexpected or duplicate source document: {doc_id}')
        seen.add(doc_id)
        source_hash, count = checkpoints[doc_id]
        if digest(title, text) != source_hash:
            raise ValueError(f'Source hash mismatch: {doc_id}')
        parts = list(engine.make_chunks(doc_id, title, text))
        if len(parts) != count:
            raise ValueError(f'Chunk count mismatch: {doc_id}')
        for part in parts:
            if not (0 <= part['start'] < part['end'] <= len(text)) or part['text'] != text[part['start']:part['end']]:
                raise ValueError(f'Invalid evidence offsets: {doc_id}')
            key = point_id(doc_id, part['chunk_no'])
            if key in expected:
                raise ValueError(f'Duplicate point ID: {key}')
            expected[key] = part
    if seen != checkpoints.keys():
        raise ValueError('Checkpoint includes documents absent from the source')

    total = len(expected)
    if engine.db.count(cfg.collection, exact=True).count != total:
        raise ValueError('Database point count differs from expected chunks')
    offset = None
    while True:
        points, next_offset = engine.db.scroll(collection_name=cfg.collection,
            limit=500, offset=offset, with_payload=True, with_vectors=False)
        for point in points:
            key = str(point.id)
            if key not in expected or point.payload != expected.pop(key):
                raise ValueError(f'Unexpected point or payload mismatch: {key}')
        if next_offset is None:
            break
        if not points or next_offset == offset:
            raise ValueError('Pagination did not advance')
        offset = next_offset
    if expected or engine.db.count(cfg.collection, exact=True).count != total:
        raise ValueError('Missing points or collection changed during verification')
    counts = [n for _,_,n,_ in current]
    return dict(collection=cfg.collection, chunk_size=cfg.chunk_size, overlap=cfg.overlap,
                documents=len(current), points=total, baseline_points=sum(n for _,_,n,_ in old),
                mean_chunks_per_document=total/len(current), max_chunks_per_document=max(counts),
                pending_documents=0, payload_mismatches=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--state', required=True)
    parser.add_argument('--baseline-state', required=True)
    args = parser.parse_args()
    from .engine import Engine
    engine = Engine()
    try:
        print(json.dumps(verify(engine, args.source, args.state, args.baseline_state), indent=2))
    finally:
        engine.db.close()


if __name__ == '__main__':
    main()

"""Load a frozen retrieval profile for interactive search and generation.

Benchmark commands remain independent of this runtime profile.
"""
import json
import os
from pathlib import Path


def load_profile(cfg, path):
    """Reject a profile that does not describe the active index."""
    profile = json.loads(Path(path).read_text(encoding='utf-8'))
    c = profile['configuration']
    mapping = {'collection': 'collection', 'chunk_size': 'chunk_size',
               'chunk_overlap': 'overlap', 'embedding': 'embedding',
               'embedding_revision': 'revision', 'embedding_input': 'embedding_input',
               'device': 'device', 'qdrant_endpoint': 'url'}
    for field, attr in mapping.items():
        if c[field] != getattr(cfg, attr):
            raise ValueError(f'Runtime profile mismatch: {field}. Restore the matching environment.')
    # This version supports the frozen non-reranked hybrid profile only.
    if (c['mode'] != 'hybrid' or c['parallel'] is not True or
            c['fusion'] != 'document' or c['rerank'] is not False or
            c['sparse_weight'] != 1.0 or c['sparse_embedding_input'] != 'chunk'):
        raise ValueError('Unsupported runtime retrieval profile')
    from .core import validate_fusion
    validate_fusion(c['mode'], c['rerank'], c['fusion'], c['rrf_constant'], c['dense_weight'], candidates=c['candidates'])
    if (type(c['k']) is not int or type(c['candidates']) is not int or
            not 1 <= c['k'] <= c['candidates'] <= 1000 or
            type(c['ef']) is not int or c['ef'] < 1 or
            type(c['indexed_chunks']) is not int or c['indexed_chunks'] < 1):
        raise ValueError('Invalid runtime profile limits')
    return profile


def active_profile(cfg):
    path = os.getenv('RAG_RETRIEVAL_PROFILE')
    return load_profile(cfg, path) if path else None


def profile_options(profile, k=None):
    c = profile['configuration']
    k = c['k'] if k is None else k
    if type(k) is not int or not 1 <= k <= c['candidates']:
        raise ValueError('Requested k exceeds runtime profile candidate limit')
    return dict(k=k, mode=c['mode'], candidates=c['candidates'], ef=c['ef'],
                rerank=c['rerank'], parallel=c['parallel'], fusion=c['fusion'],
                rrf_constant=c['rrf_constant'], dense_weight=c['dense_weight'])


def main():
    """Read-only preflight: check configuration and live point count."""
    from .engine import Engine
    e = Engine()
    try:
        p = active_profile(e.cfg)
        if p is None:
            raise ValueError('Set RAG_RETRIEVAL_PROFILE before running preflight')
        n = e.db.count(e.cfg.collection, exact=True).count
        if n != p['configuration']['indexed_chunks']:
            raise ValueError(f'Point count mismatch: {n}')
        print(json.dumps({'profile': p['profile'], 'collection': e.cfg.collection,
                          'indexed_chunks': n, 'retrieval': profile_options(p)}, indent=2))
    finally:
        e.db.close()


if __name__ == '__main__':
    main()

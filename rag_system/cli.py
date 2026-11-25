"""Command-line entry point for the project."""
import argparse
import json
import time
from .engine import Engine


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init')
    init.add_argument('--quantize', action='store_true')
    ing = sub.add_parser('ingest')
    ing.add_argument('source')
    ing.add_argument('--state', default='data/checkpoint.sqlite')
    ing.add_argument('--limit', type=int)
    delete = sub.add_parser('delete')
    delete.add_argument('id')
    delete.add_argument('--state', default='data/checkpoint.sqlite')
    for name in ('search','ask'):
        q = sub.add_parser(name)
        q.add_argument('query', nargs='?')
        q.add_argument('--k', type=int, default=8)
        if name=='search':
            q.add_argument('--mode', choices=['dense','exact','bm25','hybrid'], default='hybrid')
            q.add_argument('--rerank', action='store_true')
            q.add_argument('--candidates', type=int, default=50)
            q.add_argument('--ef', type=int, default=128)
    ev = sub.add_parser('evaluate')
    ev.add_argument('queries')
    ev.add_argument('qrels')
    ev.add_argument('--output', required=True)
    ev.add_argument('--mode', choices=['dense','exact','bm25','hybrid'], default='hybrid')
    ev.add_argument('--rerank', action='store_true')
    ev.add_argument('--k', type=int, default=10)
    ev.add_argument('--candidates', type=int, default=100)
    ev.add_argument('--ef', type=int, default=128)
    sub.add_parser('count')
    a = p.parse_args()
    if getattr(a,'limit',None) is not None and a.limit<1:
        p.error('--limit must be positive')
    engine = Engine()
    start = time.perf_counter()
    if a.command=='init':
        engine.initialize(a.quantize)
        result = {'collection':engine.cfg.collection,'status':'created'}
    elif a.command=='ingest':
        from .ingest import ingest
        result = ingest(engine,a.source,a.state,a.limit)
    elif a.command=='delete':
        from .ingest import delete_document
        delete_document(engine,a.state,a.id)
        result = {'deleted':a.id}
    elif a.command=='search':
        result = engine.search(a.query or input('Query: '),a.k,a.mode,a.candidates,a.rerank,a.ef)
    elif a.command=='ask':
        result = engine.answer(a.query or input('Question: '),a.k)
    elif a.command=='evaluate':
        from .evaluate import evaluate
        result = evaluate(engine,a.queries,a.qrels,a.output,a.k,a.mode,a.rerank,a.candidates,a.ef)
    else:
        result = {'indexed_chunks':engine.db.count(engine.cfg.collection,exact=True).count}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f'Elapsed seconds: {time.perf_counter()-start:.3f}')

if __name__=='__main__':
    main()

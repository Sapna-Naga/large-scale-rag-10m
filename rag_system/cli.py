"""Command-line entry point for the project."""
import argparse
import json
import time
from .engine import Engine
from .core import validate_fusion


def add_fusion_options(parser):
    """Expose optional fusion experiments without changing existing defaults."""
    parser.add_argument("--rerank-depth", type=int,
                        help="Rerank only this many initial items; requires --rerank")
    parser.add_argument("--rerank-input", choices=["chunk", "title_chunk"], default="chunk",
                        help="Use chunk text or stored title plus chunk text; requires --rerank for title_chunk")
    parser.add_argument("--fusion", choices=["chunk", "document"], default="chunk")
    parser.add_argument("--rrf-constant", type=int, default=60)
    parser.add_argument("--dense-weight", type=float, default=1.0,
                        help="Dense rank contribution relative to sparse weight 1.0")


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
        q.add_argument('--k', type=int, default=8 if name=='search' else None)
        if name=='search':
            q.add_argument('--mode', choices=['dense','exact','bm25','hybrid'], default='hybrid')
            q.add_argument('--rerank', action='store_true')
            q.add_argument('--parallel', action='store_true', help='Run both hybrid retrieval branches concurrently')
            q.add_argument('--candidates', type=int, default=50)
            q.add_argument('--ef', type=int, default=128)
            add_fusion_options(q)
    ev = sub.add_parser('evaluate')
    ev.add_argument('queries')
    ev.add_argument('qrels')
    ev.add_argument('--output', required=True)
    ev.add_argument('--mode', choices=['dense','exact','bm25','hybrid'], default='hybrid')
    ev.add_argument('--rerank', action='store_true')
    ev.add_argument('--parallel', action='store_true', help='Run both hybrid retrieval branches concurrently')
    ev.add_argument('--k', type=int, default=10)
    ev.add_argument('--candidates', type=int, default=100)
    ev.add_argument('--ef', type=int, default=128)
    add_fusion_options(ev)
    ev.add_argument('--limit', type=int, help='Evaluate only the first N eligible queries (development only)')
    ev.add_argument('--warmup-query', help='Separate real query for model and retrieval warm-up')
    sub.add_parser('count')
    a = p.parse_args()
    if getattr(a,'limit',None) is not None and a.limit<1:
        p.error('--limit must be positive')
    if getattr(a, 'parallel', False) and a.mode != 'hybrid':
        p.error('--parallel requires --mode hybrid')
    if a.command in ('search', 'evaluate'):
        try:
            validate_fusion(a.mode, a.rerank, a.fusion, a.rrf_constant, a.dense_weight,
                            rerank_depth=a.rerank_depth, candidates=a.candidates, rerank_input=a.rerank_input)
        except ValueError as exc:
            p.error(str(exc))
    engine = Engine()
    start = time.perf_counter()
    if a.command=='init':
        engine.initialize(a.quantize)
        result = {'collection':engine.cfg.collection,'status':'created', 'embedding_input':engine.cfg.embedding_input}
    elif a.command=='ingest':
        from .ingest import ingest
        result = ingest(engine,a.source,a.state,a.limit)
    elif a.command=='delete':
        from .ingest import delete_document
        delete_document(engine,a.state,a.id)
        result = {'deleted':a.id}
    elif a.command=='search':
        result = engine.search(a.query or input('Query: '),a.k,a.mode,a.candidates,a.rerank,a.ef,parallel=a.parallel,
                               fusion=a.fusion, rrf_constant=a.rrf_constant, dense_weight=a.dense_weight,
                               rerank_depth=a.rerank_depth, rerank_input=a.rerank_input)
    elif a.command=='ask':
        result = engine.answer(a.query or input('Question: '),a.k)
    elif a.command=='evaluate':
        from .evaluate import evaluate
        result = evaluate(engine,a.queries,a.qrels,a.output,a.k,a.mode,a.rerank,a.candidates,a.ef,
                          limit=a.limit, warmup_query=a.warmup_query, parallel=a.parallel,
                               fusion=a.fusion, rrf_constant=a.rrf_constant, dense_weight=a.dense_weight,
                               rerank_depth=a.rerank_depth, rerank_input=a.rerank_input)
    else:
        result = {'indexed_chunks':engine.db.count(engine.cfg.collection,exact=True).count}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f'Elapsed seconds: {time.perf_counter()-start:.3f}')

if __name__=='__main__':
    main()

"""Command-line entry points for all main workflows."""
import argparse
import json
from .engine import Engine
from .ingest import ingest, delete_document
from .evaluation import evaluate, benchmark, write_report


def main():
    parser = argparse.ArgumentParser(prog="rag")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("ingest")
    p.add_argument("corpus")
    p.add_argument("--manifest", default="data/manifest.sqlite")
    p.add_argument("--limit", type=int)
    p = commands.add_parser("delete")
    p.add_argument("id")
    p.add_argument("--manifest", default="data/manifest.sqlite")
    for name in ("search", "answer"):
        p = commands.add_parser(name)
        p.add_argument("query")
        p.add_argument("--k", type=int, default=8)
        p.add_argument("--rerank", action="store_true")
        if name == "search":
            p.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    p = commands.add_parser("evaluate")
    p.add_argument("queries")
    p.add_argument("qrels")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--candidates", type=int, default=100)
    p.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--exact", action="store_true")
    p.add_argument("--output", required=True)
    p = commands.add_parser("benchmark")
    p.add_argument("queries")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--generate", action="store_true")
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    engine = Engine()
    if args.command == "ingest":
        result = ingest(engine, args.corpus, args.manifest, args.limit)
    elif args.command == "delete":
        delete_document(engine, args.manifest, args.id)
        result = {"deleted": args.id}
    elif args.command == "search":
        result = engine.search(args.query, k=args.k, mode=args.mode, rerank=args.rerank)
    elif args.command == "answer":
        result = engine.answer(args.query, k=args.k, rerank=args.rerank)
    elif args.command == "evaluate":
        result = write_report(engine, evaluate(engine, args.queries, args.qrels, args.k,
                              args.candidates, args.mode, args.rerank, args.exact), args.output)
    else:
        result = write_report(engine, benchmark(engine, args.queries, args.workers, args.repeat,
                                                args.generate), args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

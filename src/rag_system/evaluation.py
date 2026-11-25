"""Evaluate real relevance labels and measure actual request timings."""
import csv
import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path
from statistics import mean
from .core import metrics, percentile, unique_documents


def read_queries(path):
    with open(path, encoding="utf-8") as f:
        return {str(r.get("_id", r.get("id"))): r["text"] for r in map(json.loads, f) }


def evaluate(engine, queries_path, qrels_path, k=10, candidates=100, mode="hybrid", rerank=False, exact=False):
    if not 1 <= k <= candidates <= 1000:
        raise ValueError("Require 1 <= k <= candidates <= 1000")
    queries = read_queries(queries_path)
    qrels = defaultdict(dict)
    with open(qrels_path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qrels[row["query-id"]][row["corpus-id"]] = int(row["score"])
    results, latencies = [], []
    for qid, relevance in qrels.items():
        if not any(r > 0 for r in relevance.values()):
            continue
        start = time.perf_counter()
        # Retrieve extra chunks so several chunks of one document do not dominate evaluation.
        result = engine.search(queries[qid], k=candidates, candidates=candidates,
                               mode=mode, rerank=rerank, exact=exact)
        latencies.append((time.perf_counter() - start) * 1000)
        results.append(metrics(unique_documents(result["hits"], k), relevance, k))
    if not results:
        raise ValueError("No positively judged queries")
    return {"kind": "retrieval", "mode": mode, "rerank": rerank, "exact": exact, "k": k,
            "candidates": candidates, "queries": len(results),
            **{name: mean(r[name] for r in results) for name in results[0]},
            **{f"p{p}_ms": percentile(latencies, p) for p in (50, 95, 99)}}


def benchmark(engine, queries_path, workers=1, repeat=1, generate=False):
    import psutil
    queries = list(read_queries(queries_path).values()) * repeat
    if not queries or workers < 1 or repeat < 1:
        raise ValueError("Need queries, workers and repeat >= 1")
    def run(query):
        start = time.perf_counter()
        try:
            engine.answer(query) if generate else engine.search(query)
            return (time.perf_counter() - start) * 1000, None
        except Exception as exc:
            return (time.perf_counter() - start) * 1000, type(exc).__name__
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(run, queries))
    elapsed = time.perf_counter() - start
    successes = [ms for ms, error in outcomes if error is None]
    errors = defaultdict(int)
    for _, error in outcomes:
        if error:
            errors[error] += 1
    return {"kind": "load", "workers": workers, "generation": generate,
            "requests": len(outcomes), "successes": len(successes), "errors": dict(errors),
            "failure_rate": 1 - len(successes)/len(outcomes), "successful_qps": len(successes)/elapsed,
            "seconds": elapsed, "client_rss_bytes_after": psutil.Process().memory_info().rss,
            **{f"p{p}_ms": percentile(successes, p) for p in (50, 95, 99)}}


def write_report(engine, result, output):
    import importlib.metadata
    from datetime import datetime, timezone
    cfg = engine.cfg
    result = {**result, "timestamp": datetime.now(timezone.utc).isoformat(),
              "collection": cfg.collection, "indexed_chunks": engine.client.count(cfg.collection, exact=True).count,
              "dense_model": cfg.dense_model, "dense_revision": cfg.dense_revision,
              "reranker": cfg.reranker, "reranker_revision": cfg.reranker_revision,
              "llm_model": cfg.llm_model, "device": cfg.device,
              "shards_requested": cfg.shards, "replicas_requested": cfg.replicas,
              "chunk_size": cfg.chunk_size, "overlap": cfg.overlap, "hnsw_ef": cfg.hnsw_ef,
              "platform": platform.platform(), "python": platform.python_version(),
              "packages": {p: importlib.metadata.version(p) for p in
                           ("qdrant-client", "fastembed", "sentence-transformers")}}
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result

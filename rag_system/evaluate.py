"""Evaluate real qrels and record measured timings; no invented benchmark values."""
import csv
import json
import platform
import statistics
import time
from pathlib import Path
from .core import records, metrics, unique_documents


def evaluate(engine, queries, qrels, output, k=10, mode='hybrid', rerank=False, candidates=100, ef=128):
    labels = {}
    with open(qrels, encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            labels.setdefault(row['query-id'], {})[row['corpus-id']] = int(row['score'])
    results, times = [], []
    for row in records(queries):
        qid = row.get('_id', row.get('id'))
        if qid not in labels or not any(v>0 for v in labels[qid].values()):
            continue
        start = time.perf_counter()
        hits = engine.search(row['text'], k=candidates, candidates=candidates, mode=mode, rerank=rerank, ef=ef)
        times.append(time.perf_counter()-start)
        results.append(metrics(unique_documents(hits), labels[qid], k))
    if not results:
        raise ValueError('No evaluated queries; check query IDs and qrels')
    ordered = sorted(times)
    import math
    report = {'mode':mode,'rerank':rerank,'k':k,'candidates':candidates,'ef':ef,'queries':len(results),
              'collection':engine.cfg.collection,'embedding':engine.cfg.embedding,'embedding_revision':engine.cfg.revision,
              'indexed_chunks':engine.db.count(engine.cfg.collection, exact=True).count,
              'metrics':{key:statistics.mean(r[key] for r in results) for key in results[0]},
              'latency_seconds':{f'p{p}':ordered[math.ceil(p/100*len(ordered))-1] for p in (50,95,99)},
              'serial_queries_per_second':len(times)/sum(times),'platform':platform.platform(),'python':platform.python_version()}
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report

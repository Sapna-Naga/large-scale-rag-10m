"""Measure concurrent HTTP search requests using real query records."""
import argparse
import concurrent.futures
import json
import math
import time
from pathlib import Path
from .core import records


def main():
    import httpx
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('queries')
    p.add_argument('--url',default='http://127.0.0.1:8000/search')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--limit',type=int,default=100)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.workers<1 or a.limit<1:
        p.error('workers and limit must be positive')
    from itertools import islice
    queries=[r['text'] for r in islice(records(a.queries),a.limit)]
    if not queries:
        raise ValueError('No queries')
    with httpx.Client(timeout=180) as client:
        def request(query):
            start=time.perf_counter()
            try:
                response=client.post(a.url,json={'query':query})
                response.raise_for_status()
                return time.perf_counter()-start,None
            except httpx.HTTPError as exc:
                return time.perf_counter()-start,str(exc)
        start=time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            results=list(pool.map(request,queries))
        elapsed=time.perf_counter()-start
    latencies=sorted(t for t,error in results if error is None)
    errors=[error for _,error in results if error is not None]
    report={'requests':len(results),'workers':a.workers,'elapsed_seconds':elapsed,'successes':len(latencies),'failures':len(errors),'successful_qps':len(latencies)/elapsed,'errors':errors,'successful_latency_seconds':{f'p{p}':latencies[math.ceil(p/100*len(latencies))-1] for p in (50,95,99)} if latencies else {}}
    path=Path(a.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()

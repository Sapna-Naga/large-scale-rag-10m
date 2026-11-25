"""Generate a static GitHub Pages report from actual evaluation JSON files."""
import argparse
import html
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reports', nargs='*')
    p.add_argument('--output',default='docs/index.html')
    a=p.parse_args()
    rows=[]
    for name in a.reports:
        report=json.loads(Path(name).read_text(encoding='utf-8'))
        fields=[name,report['mode'],report['rerank'],report['indexed_chunks'],report['queries'],report['k'],report['metrics']['recall'],report['metrics']['ndcg'],report['latency_seconds']['p95']]
        rows.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in fields)+'</tr>')
    body=''.join(rows) or '<tr><td colspan="9">No measured results yet.</td></tr>'
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Large-scale RAG benchmarks</title><style>body{font:16px system-ui;max-width:1100px;margin:3rem auto;padding:1rem;color:#182332}table{border-collapse:collapse;width:100%}td,th{padding:.7rem;text-align:left;border-bottom:1px solid #ccc}section{overflow:auto}p{color:#475569}</style><h1>Large-scale RAG benchmarks</h1><p>Measured retrieval results. Indexed chunks are not source documents.</p><section><table><thead><tr><th>Report</th><th>Mode</th><th>Rerank</th><th>Chunks</th><th>Queries</th><th>K</th><th>Recall</th><th>nDCG</th><th>P95 seconds</th></tr></thead><tbody>'''+body+'</tbody></table></section></html>'
    path=Path(a.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(page,encoding='utf-8')
    print(f'Wrote {path}')

if __name__=='__main__':
    main()

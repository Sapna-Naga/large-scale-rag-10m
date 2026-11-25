"""Publish only actual JSON reports; render an empty state if none exist."""
import html
import json
from pathlib import Path


def main():
    reports = sorted(Path("benchmarks/results").glob("*.json"))
    sections = []
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        rows = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in report.items())
        sections.append(f"<section><h2>{html.escape(path.stem)}</h2><table>{rows}</table></section>")
    body = "".join(sections) or "<p>No benchmark reports have been recorded.</p>"
    Path("docs").mkdir(exist_ok=True)
    Path("docs/index.html").write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Large Scale RAG | Benchmarks</title>
<style>body{{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 20px;color:#19283b;background:#f6f8fb}}h1{{font-size:2rem}}section{{background:white;padding:24px;margin:20px 0;border:1px solid #dce2ea;border-radius:12px}}table{{width:100%;border-collapse:collapse;overflow-wrap:anywhere}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #eee}}th{{width:35%;color:#52647c}}</style>
<h1>Large Scale RAG</h1><p>Measured benchmark results · 10M document design target</p>{body}</html>''', encoding="utf-8")


if __name__ == "__main__":
    main()

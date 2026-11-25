# Large-scale RAG — 10 million document target

A Python reference implementation for scalable retrieval-augmented generation. **10 million documents is a design target, not a measured achievement.** No dataset or benchmark results are invented. Unit-test fixtures are synthetic and used only to verify code behavior.

## Setup (Python 3.11 recommended)

Install Python, Git and Docker Engine with Compose. From this repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m rag_system.console -- python -m pip install -r requirements.txt
python -m rag_system.console -- docker compose up -d
python -m rag_system.console -- python -m pytest -q
```

On Linux/macOS activate with `source .venv/bin/activate`. The pinned top-level dependency versions are proposed, not an experimentally resolved lockfile. Model downloads and Docker images require internet access. No commercial API keys are required for local execution.

## First real dataset and search

Review the dataset's original terms before downloading or redistributing. The BEIR SciFact archive is obtained from the BEIR distribution server. See https://github.com/beir-cellar/beir and https://github.com/allenai/scifact. Do not commit downloaded corpora to GitHub.

```powershell
python -m rag_system.console -- python -m rag_system.dataset
python -m rag_system.console -- python -m rag_system.cli init
python -m rag_system.console -- python -m rag_system.cli ingest data/scifact/corpus.jsonl
python -m rag_system.console -- python -m rag_system.cli count
python -m rag_system.console -- python -m rag_system.cli search
```

The last command prompts for a query. Other supported corpora are UTF-8 JSONL records with a unique string `id` (or `_id`), `text` and optional `title`. Convert real source documents to that format; this version does not parse PDFs or crawl Wikipedia. Use `--limit 1000` for early trials. Remove the limit to resume by replaying the same corpus. Deduplication is exact on title + text; keep separate accounting of duplicates. For BEIR, deduplicating can remove judged IDs; results may undercount relevance in that case. Audit duplicates before interpreting scores. The text is stripped before chunking; offsets refer to the stripped text.

## Retrieval experiments

```powershell
python -m rag_system.console -- python -m rag_system.cli evaluate data/scifact/queries.jsonl data/scifact/qrels/test.tsv --mode bm25 --output benchmarks/bm25.json
python -m rag_system.console -- python -m rag_system.cli evaluate data/scifact/queries.jsonl data/scifact/qrels/test.tsv --mode dense --output benchmarks/dense.json
python -m rag_system.console -- python -m rag_system.cli evaluate data/scifact/queries.jsonl data/scifact/qrels/test.tsv --mode exact --output benchmarks/exact.json
python -m rag_system.console -- python -m rag_system.cli evaluate data/scifact/queries.jsonl data/scifact/qrels/test.tsv --mode hybrid --output benchmarks/hybrid.json
python -m rag_system.console -- python -m rag_system.cli evaluate data/scifact/queries.jsonl data/scifact/qrels/test.tsv --mode hybrid --rerank --output benchmarks/reranked.json
python -m rag_system.console -- python -m rag_system.dashboard benchmarks/bm25.json benchmarks/dense.json benchmarks/exact.json benchmarks/hybrid.json benchmarks/reranked.json
```

Enable GitHub Pages in repository Settings → Pages → Deploy from a branch → main → /docs. Commit only measured reports cleared of confidential data. The HTML report is static, not an inference service. Evaluation reports document-level Recall, Precision, MRR, nDCG and hit rate, plus serial query latency and throughput. Results include model startup if it occurs during a query. The exact baseline scans original dense vectors; use on feasible subsets. Candidate retrieval is chunk-level and document ranking collapses repeated IDs. Incomplete qrels and subset corpora affect interpretation.

## Generation and API

Run an OpenAI-compatible local server supporting JSON-object output. For example, on a compatible Linux GPU host, install vLLM in a separate environment following https://docs.vllm.ai/ and run:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8001
```

Then run:

```powershell
python -m rag_system.console -- python -m rag_system.cli ask
python -m rag_system.console -- python -m uvicorn rag_system.api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs for API testing. The API is local-only by default; it has no authentication. Generation validates citation IDs and abstains when evidence is absent or model output is invalid. Citation validity is not evidence entailment or a guarantee of factual correctness. The context budget counts evidence tokens, not the complete prompt; configure the server context window accordingly.

## Daily IST logging

Prefix commands with `python -m rag_system.console --`. The wrapper records command arguments, typed/piped stdin lines, combined stdout/stderr and exit status. Logs append to `logs/DDMonYYYY.log` (e.g. `25Nov2026.log`) with UTC+05:30 timestamps. Date is checked for every record, including across midnight. The example filename is illustrative, not an actual run date. Commands executed outside the wrapper are not captured. It is a line-oriented pipe-based logger, not a full terminal recorder: password prompts, full-screen terminal programs, HTTP request bodies and GUI input are not supported. Avoid passing secrets as arguments or stdin. Logs may contain document contents and are gitignored. Separate processes can interleave log lines. Do not use the wrapper for interactive password entry.

## Configuration

Set environment variables before launching each command (PowerShell: `$env:RAG_DEVICE='cuda'`; Bash: `export RAG_DEVICE=cuda`). Defaults are in `rag_system/config.py`.

- QDRANT_URL, QDRANT_API_KEY, RAG_COLLECTION
- RAG_EMBEDDING, RAG_EMBEDDING_REVISION, RAG_RERANKER, RAG_DEVICE
- RAG_CHUNK_SIZE, RAG_OVERLAP, RAG_BATCH
- RAG_SHARDS, RAG_REPLICAS (create-time; replicas require multiple Qdrant peers)
- RAG_LLM_URL, RAG_LLM_MODEL, RAG_LLM_API_KEY, RAG_CONTEXT_TOKENS

Use a new collection and separate checkpoint for every changed embedding/chunking configuration. Pin model revisions for reproducible runs. Do not reset the database without also resetting the matching checkpoint. Only one ingestion/deletion writer per collection; no concurrent updates. Updates delete old chunks before replacement, so reads can observe partial updates; for strict consistency use a separately built collection and controlled cutover. Interrupted writes are repaired by replaying the same corpus. Source IDs must be stable.

## Scale plan and limitations

Qdrant provides disk-backed dense HNSW and sparse indexes, IDF-based BM25, optional scalar quantization (`init --quantize`), sharding and replication. The included Compose file is single-node only. Configure a real cluster following https://qdrant.tech/documentation/guides/distributed_deployment/ before multi-node tests. The API serializes model access initially to bound resource use. Ingestion batches chunks across documents, uses disk-backed SQLite checkpoints and streams JSONL, but tokenizes each individual document in memory. For 10M documents, measure corpus/chunk distribution, disk and memory, ingestion throughput and index overhead on representative subsets before allocating hardware. No verified 10M unique-source corpus is bundled. The ingestion checkpoint is not a distributed transaction or distributed job scheduler.

Not implemented or validated here: distributed deployment automation, multi-writer ingestion, structure-aware parsing, caching, Prometheus, generation quality benchmarks, executed concurrent load benchmarks, failure recovery of real clusters, ANN recall comparison, actual 10M benchmarking. These are follow-on work; no performance or error-free production claim is made.

## Upstream documentation and licenses

Qdrant (Apache-2.0): https://github.com/qdrant/qdrant
FastEmbed (Apache-2.0): https://github.com/qdrant/fastembed
Sentence Transformers (Apache-2.0): https://github.com/UKPLab/sentence-transformers
BGE-small model card (MIT): https://huggingface.co/BAAI/bge-small-en-v1.5
Reranker model card: https://huggingface.co/BAAI/bge-reranker-base
Qwen model card: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
Qdrant hybrid setup: https://qdrant.tech/course/beginners/module-3/hybrid-search-in-qdrant/
Review all model and dataset terms before redistribution. No model weights are bundled.

## Verification

The delivered code was compiled and unit-tested in Python 3.13.5 in the authoring environment. The tests use no external services. Python 3.11 is the intended deployment version; CI tests core code on that version after upload. The dependency installation, actual model downloads, Qdrant network integration, vLLM server and full-scale benchmark were not executed in the authoring environment.

With the API running, measure concurrent requests using real queries:

```powershell
python -m rag_system.console -- python -m rag_system.loadtest data/scifact/queries.jsonl --workers 4 --limit 100 --output benchmarks/load4.json
```

The load generator is implemented but has not been run against a live server here.

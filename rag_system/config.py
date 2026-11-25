"""Configuration is explicit and shared by ingestion and search."""
import os
import json
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    url: str = os.getenv('QDRANT_URL', 'http://127.0.0.1:6333')
    collection: str = os.getenv('RAG_COLLECTION', 'rag_documents_v1')
    embedding: str = os.getenv('RAG_EMBEDDING', 'BAAI/bge-small-en-v1.5')
    revision: str | None = os.getenv('RAG_EMBEDDING_REVISION')
    reranker: str = os.getenv('RAG_RERANKER', 'BAAI/bge-reranker-base')
    device: str = os.getenv('RAG_DEVICE', 'cpu')
    llm_url: str = os.getenv('RAG_LLM_URL', 'http://localhost:8001/v1')
    llm_model: str = os.getenv('RAG_LLM_MODEL', 'Qwen/Qwen2.5-7B-Instruct')
    chunk_size: int = int(os.getenv('RAG_CHUNK_SIZE', '256'))
    overlap: int = int(os.getenv('RAG_OVERLAP', '32'))
    shards: int = int(os.getenv('RAG_SHARDS', '1'))
    replicas: int = int(os.getenv('RAG_REPLICAS', '1'))
    batch: int = int(os.getenv('RAG_BATCH', '64'))
    context_tokens: int = int(os.getenv('RAG_CONTEXT_TOKENS', '3000'))

    embedding_input: str = os.getenv('RAG_EMBEDDING_INPUT', 'chunk')

    def __post_init__(self):
        # Reserve a collection suffix for title-aware dense vectors. Do not
        # allow an experimental mode to write into the original collection.
        if self.embedding_input not in ('chunk', 'title_chunk'):
            raise ValueError('RAG_EMBEDDING_INPUT must be chunk or title_chunk')
        if self.collection.endswith('_title_dense') != (self.embedding_input == 'title_chunk'):
            raise ValueError('Title-aware mode requires a collection ending in _title_dense; chunk mode must not use that suffix')


def index_signature(cfg, url=None):
    """Keep the legacy checkpoint valid and distinguish new vector inputs."""
    values = [cfg.url if url is None else url, cfg.collection, cfg.embedding,
              cfg.revision, cfg.chunk_size, cfg.overlap]
    mode = getattr(cfg, 'embedding_input', 'chunk')
    if mode != 'chunk':
        values.append(mode)
    return json.dumps(values)

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection: str = os.getenv("RAG_COLLECTION", "rag_v1")
    key: str | None = os.getenv("QDRANT_API_KEY")
    dense_model: str = os.getenv("DENSE_MODEL", "BAAI/bge-small-en-v1.5")
    dense_revision: str | None = os.getenv("DENSE_REVISION")
    reranker: str = os.getenv("RERANKER", "cross-encoder/ms-marco-MiniLM-L6-v2")
    reranker_revision: str | None = os.getenv("RERANKER_REVISION")
    llm_url: str = os.getenv("LLM_URL", "http://localhost:8001/v1")
    llm_model: str = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    llm_key: str = os.getenv("LLM_API_KEY", "")
    device: str = os.getenv("MODEL_DEVICE", "cpu")
    shards: int = int(os.getenv("SHARDS", "1"))
    replicas: int = int(os.getenv("REPLICAS", "1"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "220"))
    overlap: int = int(os.getenv("CHUNK_OVERLAP", "32"))
    batch: int = int(os.getenv("BATCH_SIZE", "64"))
    hnsw_m: int = int(os.getenv("HNSW_M", "16"))
    hnsw_ef: int = int(os.getenv("HNSW_EF", "128"))
    context_chars: int = int(os.getenv("CONTEXT_CHARS", "12000"))

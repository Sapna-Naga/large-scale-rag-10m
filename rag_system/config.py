"""Configuration is explicit and shared by ingestion and search."""
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    url: str = os.getenv('QDRANT_URL', 'http://localhost:6333')
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

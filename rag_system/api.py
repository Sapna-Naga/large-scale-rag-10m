"""Read-only API. Deploy behind authentication and rate limiting."""
from functools import lru_cache
from threading import Lock
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .engine import Engine

app = FastAPI(title='Large-scale RAG')
lock = Lock()

@lru_cache(maxsize=1)
def engine():
    return Engine()

class Query(BaseModel):
    query: str = Field(min_length=1,max_length=4000)
    k: int | None = Field(default=None,ge=1,le=50)

@app.get('/health')
def health():
    if not engine().db.collection_exists(engine().cfg.collection):
        raise HTTPException(503,'Index not initialized')
    return {'status':'ready'}

@app.post('/search')
def search(request: Query):
    # Serialize model access initially; benchmark before raising concurrency.
    with lock:
        return engine().runtime_search(request.query,request.k)

@app.post('/ask')
def ask(request: Query):
    with lock:
        return engine().answer(request.query,request.k)

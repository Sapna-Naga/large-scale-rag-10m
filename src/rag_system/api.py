"""Read-only API. Bind to localhost unless protected by an authenticated gateway."""
from contextlib import asynccontextmanager
import logging
import time
from typing import Literal
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, make_asgi_app
from .engine import Engine

REQUESTS = Counter("rag_requests_total", "Completed requests", ["route", "status"])
LATENCY = Histogram("rag_request_seconds", "Request duration", ["route"])


@asynccontextmanager
async def lifespan(app):
    app.state.engine = Engine()
    yield
    app.state.engine.client.close()


app = FastAPI(title="Large Scale RAG", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


class Query(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    k: int = Field(default=8, ge=1, le=50)
    mode: Literal["dense", "bm25", "hybrid"] = "hybrid"
    rerank: bool = False


@app.get("/health")
def health(request: Request):
    try:
        request.app.state.engine.client.get_collection(request.app.state.engine.cfg.collection)
        return {"status": "ready"}
    except Exception:
        raise HTTPException(503, "Index unavailable")


def execute(request, route, operation):
    start = time.perf_counter()
    status = "error"
    try:
        result = operation(request.app.state.engine)
        status = "ok"
        return result
    except Exception:
        logging.exception("RAG request failed")
        raise HTTPException(502, "Request failed; inspect server logs")
    finally:
        REQUESTS.labels(route, status).inc()
        LATENCY.labels(route).observe(time.perf_counter() - start)


@app.post("/search")
def search(body: Query, request: Request):
    return execute(request, "search", lambda e: e.search(body.query, k=body.k, mode=body.mode, rerank=body.rerank))


@app.post("/answer")
def answer(body: Query, request: Request):
    return execute(request, "answer", lambda e: e.answer(body.query, k=body.k, rerank=body.rerank))

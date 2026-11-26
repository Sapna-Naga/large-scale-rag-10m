"""Read-only local API; do not expose publicly without deployment safeguards."""
import logging
import os
from functools import lru_cache
from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict, field_validator

from .engine import Engine
from .runtime_profile import active_profile, profile_options
from .result_cache import cache_key, from_environment

app = FastAPI(title='Large-scale RAG', version='0.3.0')
lock = Lock()
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def engine():
    # Call under the shared lock so simultaneous first requests cannot load
    # multiple copies of the embedding models.
    return Engine()


@lru_cache(maxsize=1)
def search_cache():
    return from_environment()


class Query(BaseModel):
    model_config = ConfigDict(extra='forbid')
    query: str = Field(min_length=1, max_length=4000, strict=True)
    k: int | None = Field(default=None, ge=1, le=50, strict=True)

    @field_validator('query')
    @classmethod
    def nonblank(cls, value):
        value = value.strip()
        if not value:
            raise ValueError('Query must not be blank')
        return value


def generation_enabled():
    # This is an explicit experimental opt-in, not a quality certification.
    return os.getenv('RAG_ENABLE_EXPERIMENTAL_GENERATION') == '1'


def unavailable(operation, exc):
    # Do not send infrastructure URLs, tokens or internal tracebacks to clients.
    logger.warning('%s failed (%s)', operation, type(exc).__name__)
    return HTTPException(503, 'RAG service unavailable')


@app.get('/live')
def live():
    """Process liveness does not depend on Qdrant or model loading."""
    return {'status': 'alive'}


@app.get('/health')
def health():
    """Check index readiness, not inference quality or model warm-up."""
    with lock:
        try:
            e = engine()
            profile = active_profile(e.cfg)
            if not e.db.collection_exists(e.cfg.collection):
                raise HTTPException(503, 'Index not initialized')
            count = e.db.count(collection_name=e.cfg.collection, exact=True).count
            if count <= 0:
                raise HTTPException(503, 'Index is empty')
            if profile is not None and count != profile['configuration']['indexed_chunks']:
                raise HTTPException(503, 'Index point count does not match profile')
            return {'status': 'ready', 'collection': e.cfg.collection,
                    'indexed_chunks': count,
                    'retrieval_profile': profile['profile'] if profile else None,
                    'retrieval': profile_options(profile) if profile else None,
                    'experimental_generation_enabled': generation_enabled()}
        except HTTPException:
            raise
        except Exception as exc:
            raise unavailable('Index readiness', exc) from None


@app.get('/cache')
def cache_status():
    # Local diagnostic only; no query text or cached passages are returned.
    with lock:
        try:
            return search_cache().stats()
        except Exception as exc:
            raise unavailable('Cache status', exc) from None


@app.post('/search')
def search(request: Query):
    # Keep the existing ranked passage response and serialize model access.
    with lock:
        try:
            e = engine()
            cache = search_cache()
            if not cache.enabled:
                return e.runtime_search(request.query, request.k)
            # Opt-in only for an immutable, frozen index. TTL is not a substitute
            # for cache invalidation when documents or model weights change.
            profile = active_profile(e.cfg)
            if profile is None:
                raise ValueError('Search caching requires a frozen retrieval profile')
            k = profile_options(profile, request.k)['k']
            key = cache_key(profile, request.query, k)
            cached = cache.get(key)
            if cached is not None:
                return cached
            result = e.runtime_search(request.query, request.k)
            cache.put(key, result)
            return result
        except Exception as exc:
            raise unavailable('Search', exc) from None


@app.post('/ask')
def ask(request: Query):
    # Never expose the current local model as a validated answer generator.
    if not generation_enabled():
        raise HTTPException(503, 'Experimental answer generation is disabled; use /search')
    with lock:
        try:
            return engine().answer(request.query, request.k)
        except Exception as exc:
            raise unavailable('Experimental generation', exc) from None

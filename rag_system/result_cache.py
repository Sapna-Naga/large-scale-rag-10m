"""Small process-local cache for a read-only, versioned retrieval index.

The API lock protects all calls. The byte limit covers encoded keys and values,
not Python object overhead or temporary response allocations.
"""
from collections import OrderedDict
import hashlib
import json
import os
import time


class ResultCache:
    def __init__(self, max_entries=0, max_bytes=4*1024*1024, ttl=300, clock=time.monotonic):
        if not 0 <= max_entries <= 1024 or not 1 <= max_bytes <= 32*1024*1024 or not 1 <= ttl <= 3600:
            raise ValueError('Invalid search cache limits')
        self.max_entries, self.max_bytes, self.ttl = max_entries, max_bytes, ttl
        self.clock = clock
        self.items = OrderedDict()
        self.bytes = self.hits = self.misses = self.evictions = self.expirations = self.skipped = 0

    @property
    def enabled(self):
        return self.max_entries > 0

    def _remove(self, key):
        _, data = self.items.pop(key)
        self.bytes -= len(key) + len(data)

    def _expire(self):
        now = self.clock()
        for key, (deadline, _) in list(self.items.items()):
            if deadline <= now:
                self._remove(key)
                self.expirations += 1

    def get(self, key):
        self._expire()
        if key not in self.items:
            self.misses += 1
            return None
        self.hits += 1
        self.items.move_to_end(key)
        return json.loads(self.items[key][1])

    def put(self, key, result):
        if not self.enabled:
            return
        self._expire()
        # Cache serialized snapshots so a caller cannot mutate stored results.
        data = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
        size = len(key) + len(data)
        if key in self.items:
            self._remove(key)
        if not result or size > self.max_bytes:
            self.skipped += 1
            return
        while len(self.items) >= self.max_entries or self.bytes + size > self.max_bytes:
            self._remove(next(iter(self.items)))
            self.evictions += 1
        self.items[key] = (self.clock() + self.ttl, data)
        self.bytes += size

    def stats(self):
        self._expire()
        return dict(enabled=self.enabled, entries=len(self.items), encoded_bytes=self.bytes,
                    max_entries=self.max_entries, max_bytes=self.max_bytes, ttl_seconds=self.ttl,
                    hits=self.hits, misses=self.misses, evictions=self.evictions,
                    expirations=self.expirations, skipped=self.skipped)


def cache_key(profile, query, k):
    # Include the entire validated configuration and effective result count.
    data = json.dumps([profile['profile'], profile['configuration'], query, k],
                      sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(data).digest()


def from_environment():
    return ResultCache(int(os.getenv('RAG_SEARCH_CACHE_ENTRIES', '0')),
                       int(os.getenv('RAG_SEARCH_CACHE_BYTES', str(4*1024*1024))),
                       int(os.getenv('RAG_SEARCH_CACHE_TTL_SECONDS', '300')))

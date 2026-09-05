"""
src/cache/redis_cache.py
=========================
Thin Redis wrapper used for three layers of caching:
  1. Full-response cache   (graphs.py / nodes.py — cache_check_node)
  2. Tool-level RAG cache  (nodes.py — _call_medicine / _call_lab / etc.)
  3. LLM classification cache (router.py — detect_criticality / detect_plan)

Design choices:
  - If Redis is unreachable, every function degrades to a no-op (cache_get
    returns None, cache_set does nothing) and logs a warning ONCE. The app
    keeps working without caching rather than crashing — caching should
    never be a hard dependency.
  - Keys are built from a normalized (lowercased, stripped) dict of the
    inputs that actually determine the output, hashed to a fixed-length
    string. This means "Cold, cough and fever" and "cold,cough and fever"
    hit the same cache entry.
"""

import hashlib
import json
import os

import redis

from src.logger import get_logger

logger = get_logger(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

_client = None
_connection_warned = False


def get_client():
    """Lazily connects and reuses a single Redis client. Returns None if
    Redis isn't reachable — callers must handle that (all functions below
    already do)."""
    global _client, _connection_warned

    if _client is not None:
        return _client

    try:
        client = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
            decode_responses=True, socket_connect_timeout=1,
        )
        client.ping()
        _client = client
        logger.info("Redis cache connected: %s:%s (db=%s)", REDIS_HOST, REDIS_PORT, REDIS_DB)
        return _client
    except redis.exceptions.RedisError as e:
        if not _connection_warned:
            logger.warning(
                "Redis not reachable at %s:%s (%s) — caching disabled, all calls run live. "
                "Check your container is up: docker ps",
                REDIS_HOST, REDIS_PORT, e,
            )
            _connection_warned = True
        return None


def make_key(prefix: str, **kwargs) -> str:
    """Builds a stable cache key from a prefix + normalized kwargs.
    Example: make_key('tool:medicine', symptoms='Cold, cough', age=25)
    """
    normalized = {
        k: (str(v).strip().lower() if v is not None else "")
        for k, v in sorted(kwargs.items())
    }
    raw = json.dumps(normalized, sort_keys=True)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}:{digest}"


def cache_get(key: str):
    """Returns the cached string value, or None on miss / Redis unavailable."""
    client = get_client()
    if client is None:
        return None
    try:
        value = client.get(key)
        if value is not None:
            logger.info("Cache HIT — %s", key)
        return value
    except redis.exceptions.RedisError as e:
        logger.warning("Redis GET failed for %s: %s", key, e)
        return None


def cache_set(key: str, value: str, ttl: int = 86400):
    """Stores value under key with a TTL in seconds (default 24h). Silently
    no-ops if Redis is unavailable — never raises into calling code."""
    client = get_client()
    if client is None:
        return
    try:
        client.set(key, value, ex=ttl)
    except redis.exceptions.RedisError as e:
        logger.warning("Redis SET failed for %s: %s", key, e)


def cache_delete_prefix(prefix: str):
    """Deletes all keys under a prefix — useful for invalidating tool
    caches after you re-run `python -m src.rag.ingest` with new data."""
    client = get_client()
    if client is None:
        return
    try:
        cursor = 0
        deleted = 0
        while True:
            cursor, keys = client.scan(cursor=cursor, match=f"{prefix}*", count=200)
            if keys:
                client.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break
        logger.info("Cache — deleted %d keys under prefix '%s'", deleted, prefix)
    except redis.exceptions.RedisError as e:
        logger.warning("Redis prefix delete failed for %s: %s", prefix, e)
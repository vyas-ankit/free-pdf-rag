# src/core/cache.py
"""
Semantic query cache backed by a JSON file on disk.

Design:
- Max 10 entries (configurable). LRU eviction when full.
- Each entry stores the original query, its embedding (384 floats), the
  cached answer, and a last_accessed timestamp.
- On lookup: embed the incoming query, compute cosine similarity against all
  cached embeddings. If best match >= threshold, return the cached answer
  and update last_accessed. Cache miss otherwise.
- On write: add the new entry. If at capacity, evict the LRU entry first.
- Persisted to disk (JSON) after every write and every LRU update so the
  cache survives process restarts.

Only the FIRST turn of a conversation is eligible for caching. Callers
(rag_simple.py) enforce this by only calling get/set on turn 1.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.core.config import get_query_cache_config


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SemanticQueryCache:
    """JSON-file-backed semantic cache with LRU eviction."""

    def __init__(self):
        cfg = get_query_cache_config()
        self.enabled             = bool(cfg.get("enabled", True))
        self.max_entries         = int(cfg.get("max_entries", 10))
        self.similarity_threshold = float(cfg.get("similarity_threshold", 0.90))
        self.cache_path          = Path(cfg.get("cache_path", "data/cache/query_cache.json"))
        self._entries: List[dict] = []
        self._embeddings_model   = None   # lazy-loaded
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────

    def _load(self) -> None:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._entries = json.load(f)
                print(f"[cache] Loaded {len(self._entries)} entries from {self.cache_path}")
            except Exception as exc:
                print(f"[cache] Could not load cache file: {exc} — starting fresh")
                self._entries = []
        else:
            self._entries = []

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._entries, f, indent=2)

    # ── Embeddings ────────────────────────────────────────────────────────

    def _get_embeddings_model(self):
        if self._embeddings_model is None:
            from src.core.vector_store import get_embeddings
            self._embeddings_model = get_embeddings()
        return self._embeddings_model

    def _embed(self, text: str) -> List[float]:
        model = self._get_embeddings_model()
        return model.embed_query(text)

    # ── Public API ────────────────────────────────────────────────────────

    def get(self, query: str) -> Optional[str]:
        """Return cached answer if a semantically similar query exists, else None."""
        if not self.enabled or not self._entries:
            return None

        query_embedding = self._embed(query)
        best_score = -1.0
        best_idx   = -1

        for i, entry in enumerate(self._entries):
            score = _cosine_similarity(query_embedding, entry["embedding"])
            if score > best_score:
                best_score = score
                best_idx   = i

        if best_score >= self.similarity_threshold:
            print(f"[cache] HIT  score={best_score:.3f}  query={query[:60]!r}")
            # Update LRU timestamp and persist
            self._entries[best_idx]["last_accessed"] = _now_iso()
            self._save()
            return self._entries[best_idx]["answer"]

        print(f"[cache] MISS score={best_score:.3f}  query={query[:60]!r}")
        return None

    def set(self, query: str, answer: str) -> None:
        """Store a new query-answer pair. Evicts LRU entry if at capacity."""
        if not self.enabled:
            return

        query_embedding = self._embed(query)

        if len(self._entries) >= self.max_entries:
            # Evict the entry with the oldest last_accessed timestamp (LRU)
            lru_idx = min(
                range(len(self._entries)),
                key=lambda i: self._entries[i]["last_accessed"],
            )
            evicted = self._entries.pop(lru_idx)
            print(f"[cache] EVICT (LRU)  query={evicted['query'][:60]!r}")

        entry = {
            "query":         query,
            "embedding":     query_embedding,
            "answer":        answer,
            "cached_at":     _now_iso(),
            "last_accessed": _now_iso(),
        }
        self._entries.append(entry)
        self._save()
        print(f"[cache] SET   entries={len(self._entries)}/{self.max_entries}  "
              f"query={query[:60]!r}")

    def clear(self) -> None:
        """Empty the cache and delete the file."""
        self._entries = []
        if self.cache_path.exists():
            os.remove(self.cache_path)
        print("[cache] Cleared")

    @property
    def size(self) -> int:
        return len(self._entries)


# Module-level singleton — one cache instance for the whole process
_cache: Optional[SemanticQueryCache] = None


def get_cache() -> SemanticQueryCache:
    global _cache
    if _cache is None:
        _cache = SemanticQueryCache()
    return _cache

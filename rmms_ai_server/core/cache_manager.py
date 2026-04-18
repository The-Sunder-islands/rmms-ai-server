from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CacheManager:
    def __init__(self):
        self._cache: dict[str, dict] = {}

    def _compute_key(self, file_hash: str, params: dict) -> str:
        params_str = json.dumps(params, sort_keys=True)
        combined = f"{file_hash}:{params_str}"
        return hashlib.sha256(combined.encode()).hexdigest()[:32]

    @staticmethod
    def compute_file_hash(file_path: str) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:32]

    def get(self, file_hash: str, params: dict) -> Optional[dict]:
        key = self._compute_key(file_hash, params)
        return self._cache.get(key)

    def put(self, file_hash: str, params: dict, result: dict) -> None:
        key = self._compute_key(file_hash, params)
        self._cache[key] = result
        logger.debug(f"Cache put: {key}")

    def invalidate(self, file_hash: str, params: dict) -> None:
        key = self._compute_key(file_hash, params)
        self._cache.pop(key, None)

    def clear(self) -> None:
        self._cache.clear()


cache_manager = CacheManager()

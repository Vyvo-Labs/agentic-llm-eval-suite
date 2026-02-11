from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ResponseCache:
    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def key_for(self, payload: dict[str, Any]) -> str:
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _path_for_key(self, key: str) -> Path:
        subdir = self._cache_dir / key[:2]
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path_for_key(key)
        if not path.exists():
            return None

        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def put(self, key: str, value: dict[str, Any]) -> None:
        path = self._path_for_key(key)
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(path)


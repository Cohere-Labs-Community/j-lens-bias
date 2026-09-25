"""Append-only cache for intervention runs, so a sweep resumes instead of restarting.

One JSONL line per (item, condition). Keys are built from the fields that change
the result — item, layers, mode, alpha, variant — so adding items or conditions
only computes what is new. Values are the scalars and short profiles we analyse;
residuals are not stored (they are large and cheap to recompute when needed).

    cache = RunCache(path)
    rec = cache.get(key) or cache.put(key, compute())
"""

from __future__ import annotations

import json
from pathlib import Path


class RunCache:
    def __init__(self, path: Path, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled
        self.rows: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue                      # half-written last line
                    self.rows[rec["_key"]] = rec
        self.hits = self.misses = 0

    @staticmethod
    def key(**parts) -> str:
        """Stable key from the fields that determine a run."""
        return "|".join(f"{k}={parts[k]}" for k in sorted(parts))

    def get(self, key: str) -> dict | None:
        if not self.enabled:
            return None
        rec = self.rows.get(key)
        self.hits += rec is not None
        self.misses += rec is None
        return rec

    def put(self, key: str, value: dict) -> dict:
        rec = {"_key": key, **value}
        self.rows[key] = rec
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        return rec

    def __len__(self) -> int:
        return len(self.rows)

    def __repr__(self) -> str:
        return f"RunCache({self.path.name}, {len(self)} rows, {self.hits} hits / {self.misses} misses)"

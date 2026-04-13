"""Semantic Scholar API client with SQLite response caching.

Fetches paper metadata and references by arXiv ID. The cache key is the arXiv
ID (not S2's paperId) because that's what the rest of the pipeline uses. Cache
is a single SQLite file; writes are idempotent, so re-running build scripts
incurs zero HTTP traffic after the first pass.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

import requests

S2_BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = "paperId,title,externalIds,references.paperId,references.title,references.externalIds"
DEFAULT_CACHE_PATH = Path("data/cache/s2_cache.db")
DEFAULT_REQ_DELAY = 0.1  # 10 req/s with an API key (S2 allows 100/s, be polite)


class S2Client:
    def __init__(
        self,
        cache_path: Path | str = DEFAULT_CACHE_PATH,
        api_key: str | None = None,
        req_delay: float = DEFAULT_REQ_DELAY,
        session: requests.Session | None = None,
    ):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key if api_key is not None else os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        self.req_delay = req_delay
        self.session = session or requests.Session()
        self._last_request_time = 0.0
        self.http_calls = 0
        self.cache_hits = 0
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    response_json TEXT NOT NULL,
                    fetched_at INTEGER NOT NULL
                )
                """
            )

    def _cache_get(self, arxiv_id: str) -> dict | None:
        with sqlite3.connect(self.cache_path) as conn:
            row = conn.execute(
                "SELECT response_json FROM papers WHERE paper_id = ?", (arxiv_id,)
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def _cache_put(self, arxiv_id: str, response: dict) -> None:
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO papers (paper_id, response_json, fetched_at) VALUES (?, ?, ?)",
                (arxiv_id, json.dumps(response), int(time.time())),
            )

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self.req_delay:
            time.sleep(self.req_delay - elapsed)
        self._last_request_time = time.time()

    def _fetch(self, arxiv_id: str, max_retries: int = 6) -> dict | None:
        """HTTP GET with 429 backoff. Returns None on 404. Raises after max_retries."""
        url = f"{S2_BASE}/paper/ARXIV:{arxiv_id}"
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        backoff = 2.0
        for attempt in range(max_retries):
            self._throttle()
            try:
                resp = self.session.get(url, params={"fields": FIELDS}, headers=headers, timeout=30)
            except requests.RequestException:
                if attempt == max_retries - 1:
                    raise
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            self.http_calls += 1
            if resp.status_code == 404:
                return None
            if resp.status_code == 429:
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"S2 API: exhausted retries for {arxiv_id}")

    def get_paper(self, arxiv_id: str) -> dict | None:
        """Return the S2 paper record for an arXiv ID. Cached. None if unknown to S2."""
        cached = self._cache_get(arxiv_id)
        if cached is not None:
            self.cache_hits += 1
            # Sentinel for cached 404s
            if cached.get("_s2_not_found"):
                return None
            return cached

        response = self._fetch(arxiv_id)
        if response is None:
            self._cache_put(arxiv_id, {"_s2_not_found": True})
            return None
        self._cache_put(arxiv_id, response)
        return response

    def get_references(self, arxiv_id: str) -> list[dict]:
        """Return the list of references (possibly empty) for an arXiv ID.

        Each reference is a dict with paperId, title, externalIds. Returns []
        if the paper is unknown to S2 or has no references field.
        """
        paper = self.get_paper(arxiv_id)
        if paper is None:
            return []
        return paper.get("references") or []

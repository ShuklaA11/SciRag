"""Tests for S2Client cache layer + HTTP error handling.

Uses a real temp SQLite file and a mocked requests.Session so the tests
exercise the actual cache code path without touching the network.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
import requests

from src.pipeline.s2_client import S2Client


def make_response(status_code: int, body: dict | None = None) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = body or {}
    if status_code >= 400 and status_code != 429 and status_code != 404:
        resp.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status_code}")
    else:
        resp.raise_for_status.return_value = None
    return resp


def make_client(tmp_path, responses: list):
    session = MagicMock(spec=requests.Session)
    session.get.side_effect = responses
    return S2Client(
        cache_path=tmp_path / "s2.db",
        api_key="test-key",
        req_delay=0.0,  # no throttling in tests
        session=session,
    ), session


def sample_paper(paper_id="abc", refs=None):
    return {
        "paperId": paper_id,
        "title": "A Paper",
        "externalIds": {"ArXiv": "1234.5678"},
        "references": refs
        or [
            {"paperId": "ref1", "title": "Ref 1", "externalIds": {"ArXiv": "1111.1111"}},
            {"paperId": "ref2", "title": "Ref 2", "externalIds": {"ArXiv": "2222.2222"}},
        ],
    }


def test_cache_miss_then_hit(tmp_path):
    """First call hits HTTP, second call hits cache."""
    paper = sample_paper()
    client, session = make_client(tmp_path, [make_response(200, paper)])

    result1 = client.get_paper("1234.5678")
    assert result1 == paper
    assert client.http_calls == 1
    assert client.cache_hits == 0

    result2 = client.get_paper("1234.5678")
    assert result2 == paper
    assert client.http_calls == 1  # unchanged — no new HTTP
    assert client.cache_hits == 1
    assert session.get.call_count == 1


def test_idempotency_across_instances(tmp_path):
    """A second client instance using the same cache file reads without HTTP."""
    paper = sample_paper()
    client1, _ = make_client(tmp_path, [make_response(200, paper)])
    client1.get_paper("1234.5678")

    session2 = MagicMock(spec=requests.Session)
    client2 = S2Client(
        cache_path=tmp_path / "s2.db",
        api_key="test-key",
        req_delay=0.0,
        session=session2,
    )
    result = client2.get_paper("1234.5678")
    assert result == paper
    assert session2.get.call_count == 0
    assert client2.cache_hits == 1


def test_429_backoff_then_success(tmp_path):
    """A 429 triggers a retry; the next 200 succeeds."""
    paper = sample_paper()
    client, session = make_client(
        tmp_path, [make_response(429), make_response(200, paper)]
    )
    result = client.get_paper("1234.5678")
    assert result == paper
    assert session.get.call_count == 2
    assert client.http_calls == 2


def test_404_is_cached_as_not_found(tmp_path):
    """A 404 returns None, and subsequent lookups don't re-hit HTTP."""
    client, session = make_client(tmp_path, [make_response(404)])

    assert client.get_paper("9999.9999") is None
    assert session.get.call_count == 1

    # Second lookup — cached sentinel, no HTTP
    assert client.get_paper("9999.9999") is None
    assert session.get.call_count == 1
    assert client.cache_hits == 1


def test_get_references_returns_list(tmp_path):
    paper = sample_paper()
    client, _ = make_client(tmp_path, [make_response(200, paper)])
    refs = client.get_references("1234.5678")
    assert len(refs) == 2
    assert refs[0]["externalIds"]["ArXiv"] == "1111.1111"


def test_get_references_empty_for_unknown_paper(tmp_path):
    client, _ = make_client(tmp_path, [make_response(404)])
    assert client.get_references("9999.9999") == []


def test_get_references_handles_null_references(tmp_path):
    """S2 returns references: null for some papers — client should normalize to []."""
    paper = sample_paper()
    paper["references"] = None
    client, _ = make_client(tmp_path, [make_response(200, paper)])
    assert client.get_references("1234.5678") == []

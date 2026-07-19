"""sqllab — a text-to-SQL layer over SciRAG's evaluation results.

Phase 1 materializes ``eval/results/*.json`` run summaries and their per-question
``*.jsonl`` records into a single DuckDB file, so benchmark runs can be queried
with SQL (and, in later phases, natural language) instead of grepping JSON.
"""

from .ingest import build_eval_db
from .schema import DEFAULT_DB_PATH, connect

__all__ = ["build_eval_db", "connect", "DEFAULT_DB_PATH"]

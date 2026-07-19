"""DuckDB schema for the evaluation-results warehouse.

Three tables, kept deliberately clean so generated SQL stays simple:

* ``runs``               — one row per ``*_summary.json`` (curated columns + raw_json)
* ``retrieval_results``  — one row per question, per QASPER retrieval run
* ``scifact_results``    — one row per claim, per SciFact NLI run

Family-specific metrics live as nullable columns on ``runs`` (a retrieval run has
no ``accuracy``; an NLI run has no ``mean_recall_at_k_fuzzy``). Anything not
promoted to a first-class column is preserved verbatim in ``raw_json`` and can be
reached with DuckDB's ``json_extract``.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

DEFAULT_DB_PATH = Path("data/eval.duckdb")

# --- DDL ---------------------------------------------------------------------

RUNS_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_name                TEXT PRIMARY KEY,
    task                    TEXT,      -- 'retrieval' | 'scifact_nli' | 'other'
    week                    TEXT,      -- 'week3'... derived from run_name
    is_smoke                BOOLEAN,   -- small debug run (name ends in 'smoke')
    -- config
    embedder                TEXT,
    retriever               TEXT,
    mode                    TEXT,
    k                       INTEGER,
    retrieve_k              INTEGER,
    rerank                  BOOLEAN,
    rerank_model            TEXT,
    router_threshold        DOUBLE,
    router_top_n            INTEGER,
    multihop                BOOLEAN,
    model_name              TEXT,      -- NLI model, for scifact runs
    nei_threshold           DOUBLE,
    git_commit              TEXT,
    -- sizes
    n_processed             INTEGER,
    n_with_evidence         INTEGER,
    n_pairs                 INTEGER,
    -- retrieval metrics
    mean_recall_at_k_fuzzy  DOUBLE,
    mean_recall_at_k_strict DOUBLE,
    mean_recall_at_k        DOUBLE,
    mean_answer_f1          DOUBLE,
    -- nli metrics
    accuracy                DOUBLE,
    macro_f1                DOUBLE,
    retrieval_recall_at_k   DOUBLE,
    hit_accuracy            DOUBLE,
    -- timing
    latency_total_p50       DOUBLE,
    latency_total_p95       DOUBLE,
    runtime_sec             DOUBLE,
    -- provenance
    results_file            TEXT,
    source_file             TEXT,
    raw_json                TEXT
);
"""

RETRIEVAL_RESULTS_DDL = """
CREATE TABLE IF NOT EXISTS retrieval_results (
    run_name            TEXT,
    question_id         TEXT,
    paper_id            TEXT,
    question            TEXT,
    recall_at_k         DOUBLE,
    recall_at_k_strict  DOUBLE,
    mode                TEXT,
    rerank              BOOLEAN,
    n_retrieved         INTEGER,
    top_section_type    TEXT,
    retrieve_latency_ms DOUBLE,
    rerank_latency_ms   DOUBLE
);
"""

SCIFACT_RESULTS_DDL = """
CREATE TABLE IF NOT EXISTS scifact_results (
    run_name        TEXT,
    claim_id        INTEGER,
    doc_id          INTEGER,
    gold_label      TEXT,
    pred_label      TEXT,
    correct         BOOLEAN,
    support_prob    DOUBLE,
    contradict_prob DOUBLE,
    nei_prob        DOUBLE
);
"""

TABLES = ("runs", "retrieval_results", "scifact_results")


def connect(db_path: str | Path = DEFAULT_DB_PATH, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open (or create) the warehouse. Parent dirs are created on write."""
    path = Path(db_path)
    if not read_only:
        path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


def create_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create all tables if absent. Idempotent."""
    con.execute(RUNS_DDL)
    con.execute(RETRIEVAL_RESULTS_DDL)
    con.execute(SCIFACT_RESULTS_DDL)


def drop_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Drop all tables, so a rebuild starts clean."""
    for table in TABLES:
        con.execute(f"DROP TABLE IF EXISTS {table}")

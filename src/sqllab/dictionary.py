"""Generate a data dictionary for the eval warehouse.

The dictionary is produced from the *live* database (so it never drifts from the
real schema) and combines three things the LLM needs to write correct SQL:

1. table names + purposes,
2. column names, types, and short descriptions,
3. the distinct values of low-cardinality "enum" columns (e.g. ``task``,
   ``embedder``, ``gold_label``) — so the model uses real category names.

This same text doubles as human-readable documentation of the warehouse.
"""

from __future__ import annotations

import duckdb

# Hand-written purpose per table (structure comes from the DB, meaning from here).
_TABLE_PURPOSE = {
    "runs": "One row per benchmark run (a *_summary.json). Config + headline metrics.",
    "retrieval_results": "One row per QASPER question, per retrieval run.",
    "scifact_results": "One row per SciFact claim, per NLI run.",
}

_COLUMN_DOC = {
    # runs
    "run_name": "Unique run id, e.g. 'week6_rerank_k10'.",
    "task": "Run family: 'retrieval' | 'scifact_nli' | 'other'.",
    "week": "Dev week the run belongs to, e.g. 'week6'.",
    "is_smoke": "True for small debug runs; filter these out for real comparisons.",
    "embedder": "Embedding model used for retrieval (e.g. 'bge', 'specter2').",
    "retriever": "Retriever used for the SciFact evidence step (e.g. 'bm25').",
    "mode": "Retrieval mode, e.g. 'flat', 'router', 'oracle'.",
    "k": "Top-k passages scored for the metric.",
    "retrieve_k": "Candidate pool size fetched before reranking.",
    "rerank": "Whether cross-encoder reranking was applied.",
    "rerank_model": "Cross-encoder model name, when rerank is true.",
    "router_threshold": "Section-router probability threshold.",
    "router_top_n": "Max section types the router may select.",
    "multihop": "Whether multi-hop question decomposition was used.",
    "model_name": "NLI/LLM model name (mostly for scifact_nli runs).",
    "nei_threshold": "Not-Enough-Info probability cutoff for NLI.",
    "git_commit": "Repo commit the run was produced at.",
    "n_processed": "Questions actually evaluated in the run.",
    "n_with_evidence": "Questions that had gold evidence.",
    "n_pairs": "Claim/doc pairs evaluated (scifact).",
    "mean_recall_at_k_fuzzy": "PRIMARY retrieval metric: mean recall@k, fuzzy match.",
    "mean_recall_at_k_strict": "Mean recall@k with strict substring match.",
    "mean_recall_at_k": "Mean recall@k for end-to-end idea runs.",
    "mean_answer_f1": "Mean answer token-F1 for end-to-end runs.",
    "accuracy": "PRIMARY NLI metric: classification accuracy.",
    "macro_f1": "Macro-averaged F1 across NLI classes.",
    "retrieval_recall_at_k": "Evidence-retrieval recall@k inside a scifact run.",
    "hit_accuracy": "NLI accuracy on claims whose evidence was retrieved.",
    "latency_total_p50": "Median end-to-end query latency (ms).",
    "latency_total_p95": "95th-percentile end-to-end query latency (ms).",
    "runtime_sec": "Wall-clock seconds for the whole run.",
    "results_file": "Path to the per-record .jsonl this summary points at.",
    "source_file": "Path to the summary .json this row came from.",
    "raw_json": "Full original summary as JSON text (for rare fields).",
    # retrieval_results
    "question_id": "QASPER question id.",
    "paper_id": "arXiv id of the paper the question is about.",
    "question": "Natural-language research question.",
    "recall_at_k": "1.0 if a gold answer was found in top-k, else 0.0 (fuzzy).",
    "recall_at_k_strict": "Strict-match version of recall_at_k.",
    "n_retrieved": "Number of passages retrieved for this question.",
    "top_section_type": "Section type of the top-ranked retrieved passage.",
    "retrieve_latency_ms": "Retrieval latency for this question (ms).",
    "rerank_latency_ms": "Rerank latency for this question (ms).",
    # scifact_results
    "claim_id": "SciFact claim id.",
    "doc_id": "SciFact corpus document id.",
    "gold_label": "True label: 'SUPPORT' | 'CONTRADICT' | 'NEI'.",
    "pred_label": "Predicted label: 'SUPPORT' | 'CONTRADICT' | 'NEI'.",
    "correct": "True when pred_label == gold_label.",
    "support_prob": "Model probability of SUPPORT.",
    "contradict_prob": "Model probability of CONTRADICT.",
    "nei_prob": "Model probability of NEI.",
}

# Columns worth enumerating distinct values for (low cardinality, category-like).
_ENUM_COLUMNS = {
    "runs": ("task", "week", "embedder", "retriever", "mode", "rerank", "is_smoke"),
    "retrieval_results": ("mode", "top_section_type"),
    "scifact_results": ("gold_label", "pred_label"),
}

_MAX_ENUM_VALUES = 15


def _columns(con: duckdb.DuckDBPyConnection, table: str) -> list[tuple[str, str]]:
    rows = con.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = ? ORDER BY ordinal_position",
        [table],
    ).fetchall()
    return [(name, dtype) for name, dtype in rows]


def _distinct_values(con: duckdb.DuckDBPyConnection, table: str, column: str) -> list | None:
    """Return distinct non-null values if there are few enough to enumerate."""
    rows = con.execute(
        f"SELECT DISTINCT {column} FROM {table} "
        f"WHERE {column} IS NOT NULL ORDER BY 1 LIMIT {_MAX_ENUM_VALUES + 1}"
    ).fetchall()
    if len(rows) > _MAX_ENUM_VALUES:
        return None
    return [r[0] for r in rows]


def build_data_dictionary(con: duckdb.DuckDBPyConnection) -> str:
    """Render the warehouse schema + descriptions + enum values as prompt text."""
    tables = [
        r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()
    ]

    lines: list[str] = ["# Eval warehouse schema (DuckDB)\n"]
    for table in tables:
        purpose = _TABLE_PURPOSE.get(table, "")
        lines.append(f"## Table: {table}")
        if purpose:
            lines.append(purpose)
        for name, dtype in _columns(con, table):
            doc = _COLUMN_DOC.get(name, "")
            lines.append(f"  - {name} ({dtype}) — {doc}".rstrip(" —"))
        for column in _ENUM_COLUMNS.get(table, ()):  # enumerate category values
            values = _distinct_values(con, table, column)
            if values:
                rendered = ", ".join(repr(v) for v in values)
                lines.append(f"  * {column} values: {rendered}")
        lines.append("")

    lines.append(
        "Join child tables to runs on run_name. All queries must be read-only SELECTs."
    )
    return "\n".join(lines)

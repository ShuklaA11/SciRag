"""Load ``eval/results`` JSON artifacts into the DuckDB warehouse.

Ingestion is driven off the run summaries: each ``*_summary.json`` becomes a row
in ``runs``, and its ``results_file`` pointer (when present) is followed to load
the matching per-question / per-claim ``.jsonl`` into the right child table.

Rebuilds are full and idempotent — tables are dropped and repopulated, so the
warehouse is always a clean reflection of what is on disk.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb

from .schema import DEFAULT_DB_PATH, connect, create_tables, drop_tables

DEFAULT_RESULTS_DIR = Path("eval/results")

_WEEK_RE = re.compile(r"^(week\d+)")

# Column order for `runs` inserts — must stay in sync with schema.RUNS_DDL.
_RUN_COLUMNS = (
    "run_name", "task", "week", "is_smoke",
    "embedder", "retriever", "mode", "k", "retrieve_k", "rerank", "rerank_model",
    "router_threshold", "router_top_n", "multihop", "model_name", "nei_threshold",
    "git_commit",
    "n_processed", "n_with_evidence", "n_pairs",
    "mean_recall_at_k_fuzzy", "mean_recall_at_k_strict", "mean_recall_at_k",
    "mean_answer_f1",
    "accuracy", "macro_f1", "retrieval_recall_at_k", "hit_accuracy",
    "latency_total_p50", "latency_total_p95", "runtime_sec",
    "results_file", "source_file", "raw_json",
)


def _classify_task(d: dict) -> str:
    if "accuracy" in d or "n_pairs" in d:
        return "scifact_nli"
    if any(key in d for key in ("mean_recall_at_k_fuzzy", "mean_recall_at_k")):
        return "retrieval"
    return "other"


def _run_row(d: dict, source_file: Path) -> tuple:
    """Map a summary dict to a `runs` row tuple in `_RUN_COLUMNS` order."""
    run_name = d.get("run_name") or source_file.stem.replace("_summary", "")
    week_match = _WEEK_RE.match(run_name)
    latency = d.get("latency_ms") or {}
    values = {
        "run_name": run_name,
        "task": _classify_task(d),
        "week": week_match.group(1) if week_match else None,
        "is_smoke": "smoke" in run_name,
        "embedder": d.get("embedder"),
        "retriever": d.get("retriever"),
        "mode": d.get("mode"),
        "k": d.get("k"),
        "retrieve_k": d.get("retrieve_k"),
        "rerank": d.get("rerank"),
        "rerank_model": d.get("rerank_model"),
        "router_threshold": d.get("router_threshold"),
        "router_top_n": d.get("router_top_n"),
        "multihop": d.get("multihop"),
        "model_name": d.get("model_name") or d.get("llm_model"),
        "nei_threshold": d.get("nei_threshold"),
        "git_commit": d.get("git_commit"),
        "n_processed": d.get("n_processed") or d.get("n_evaluated"),
        "n_with_evidence": d.get("n_with_evidence"),
        "n_pairs": d.get("n_pairs"),
        "mean_recall_at_k_fuzzy": d.get("mean_recall_at_k_fuzzy"),
        "mean_recall_at_k_strict": d.get("mean_recall_at_k_strict"),
        "mean_recall_at_k": d.get("mean_recall_at_k"),
        "mean_answer_f1": d.get("mean_answer_f1"),
        "accuracy": d.get("accuracy"),
        "macro_f1": d.get("macro_f1"),
        "retrieval_recall_at_k": d.get("retrieval_recall_at_k"),
        "hit_accuracy": d.get("hit_accuracy"),
        "latency_total_p50": latency.get("total_p50"),
        "latency_total_p95": latency.get("total_p95"),
        "runtime_sec": d.get("runtime_sec"),
        "results_file": d.get("results_file"),
        "source_file": str(source_file),
        "raw_json": json.dumps(d),
    }
    return tuple(values[col] for col in _RUN_COLUMNS)


def _retrieval_row(rec: dict, run_name: str) -> tuple:
    section_types = rec.get("retrieved_section_types") or []
    chunk_ids = rec.get("retrieved_chunk_ids") or []
    return (
        run_name,
        rec.get("question_id"),
        rec.get("paper_id"),
        rec.get("question"),
        rec.get("recall_at_k"),
        rec.get("recall_at_k_strict"),
        rec.get("mode"),
        rec.get("rerank"),
        len(chunk_ids),
        section_types[0] if section_types else None,
        rec.get("retrieve_latency_ms"),
        rec.get("rerank_latency_ms"),
    )


def _scifact_row(rec: dict, run_name: str) -> tuple:
    gold, pred = rec.get("gold_label"), rec.get("pred_label")
    return (
        run_name,
        rec.get("claim_id"),
        rec.get("doc_id"),
        gold,
        pred,
        (gold == pred) if gold is not None and pred is not None else None,
        rec.get("support_prob"),
        rec.get("contradict_prob"),
        rec.get("nei_prob"),
    )


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _resolve_results_file(results_file: str | None, results_dir: Path) -> Path | None:
    """Resolve a summary's `results_file` pointer to an existing .jsonl on disk."""
    if not results_file:
        return None
    candidate = results_dir / Path(results_file).name
    return candidate if candidate.exists() else None


def build_eval_db(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    rebuild: bool = True,
) -> dict[str, int]:
    """Build the warehouse from summary + jsonl files. Returns row counts per table."""
    results_dir = Path(results_dir)
    summaries = sorted(results_dir.glob("*summary*.json"))

    run_rows: list[tuple] = []
    retrieval_rows: list[tuple] = []
    scifact_rows: list[tuple] = []

    for summary_path in summaries:
        d = json.loads(summary_path.read_text())
        if not isinstance(d, dict) or "manifest_version" in d:
            continue  # skip the baseline manifest — it is not a run
        run_rows.append(_run_row(d, summary_path))

        run_name = d.get("run_name") or summary_path.stem.replace("_summary", "")
        jsonl_path = _resolve_results_file(d.get("results_file"), results_dir)
        if jsonl_path is None:
            continue
        task = _classify_task(d)
        records = _read_jsonl(jsonl_path)
        if task == "scifact_nli":
            scifact_rows.extend(_scifact_row(r, run_name) for r in records)
        elif task == "retrieval":
            retrieval_rows.extend(_retrieval_row(r, run_name) for r in records)

    con = connect(db_path)
    try:
        if rebuild:
            drop_tables(con)
        create_tables(con)
        _insert(con, "runs", _RUN_COLUMNS, run_rows)
        _insert(con, "retrieval_results", _RETRIEVAL_COLUMNS, retrieval_rows)
        _insert(con, "scifact_results", _SCIFACT_COLUMNS, scifact_rows)
        counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                  for t in ("runs", "retrieval_results", "scifact_results")}
    finally:
        con.close()
    return counts


_RETRIEVAL_COLUMNS = (
    "run_name", "question_id", "paper_id", "question", "recall_at_k",
    "recall_at_k_strict", "mode", "rerank", "n_retrieved", "top_section_type",
    "retrieve_latency_ms", "rerank_latency_ms",
)

_SCIFACT_COLUMNS = (
    "run_name", "claim_id", "doc_id", "gold_label", "pred_label", "correct",
    "support_prob", "contradict_prob", "nei_prob",
)


def _insert(con: duckdb.DuckDBPyConnection, table: str, columns: tuple, rows: list[tuple]) -> None:
    if not rows:
        return
    placeholders = ", ".join("?" for _ in columns)
    col_list = ", ".join(columns)
    con.executemany(f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})", rows)

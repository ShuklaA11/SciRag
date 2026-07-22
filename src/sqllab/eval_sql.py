"""Execution-match evaluation for the text-to-SQL layer.

For each gold (question, reference SQL) pair we generate SQL from the question,
execute both, and compare *result sets* — not the SQL strings. Two queries that
differ syntactically but return the same rows count as correct (the standard
Spider/BIRD execution-match metric). Result sets are compared order-insensitively
(rows sorted) with floats rounded, so trivial formatting differences don't fail.

This is model-agnostic: point it at any LLM provider via the injected client to
benchmark providers head-to-head.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import duckdb

from ..llm.client import LLMClient
from .dictionary import build_data_dictionary
from .text_to_sql import GuardError, SQLResult, answer

DEFAULT_GOLD_PATH = Path("eval/sql_gold.jsonl")

_FLOAT_NDIGITS = 6


def _normalize(rows: list[tuple] | tuple[tuple, ...]) -> list[tuple]:
    """Canonicalize a result set for order-insensitive comparison."""
    def cell(v: object) -> object:
        return round(v, _FLOAT_NDIGITS) if isinstance(v, float) else v

    return sorted((tuple(cell(c) for c in row) for row in rows), key=repr)


_MAX_PERMUTED_COLS = 6


def result_sets_match(pred: list[tuple] | tuple, gold: list[tuple] | tuple) -> bool:
    """True if gold's answer is recoverable from pred, tolerating row and column
    order AND extra columns.

    Rationale: strict shape-matching (Spider exec) penalizes a stronger model
    that returns the right answer *plus* helpful context — e.g. gold asks for
    `max(accuracy)` and the model returns `(run_name, accuracy)` of the top run.
    Both answer the question. We accept pred when some ordered selection of its
    columns reproduces gold's rows exactly (values must match; row count must
    match), so genuinely wrong values still fail. Bounded to small results.
    """
    gold_norm = _normalize(gold)
    if _normalize(pred) == gold_norm:
        return True
    if not gold:
        return not pred
    if not pred:
        return False
    n_gold, n_pred = len(gold[0]), len(pred[0])
    if n_pred < n_gold or n_pred > _MAX_PERMUTED_COLS:
        return False
    for cols in permutations(range(n_pred), n_gold):  # subsumes reorder + projection
        projected = [tuple(row[i] for i in cols) for row in pred]
        if _normalize(projected) == gold_norm:
            return True
    return False


@dataclass(frozen=True)
class CaseResult:
    id: str
    question: str
    gold_sql: str
    pred_sql: str | None
    correct: bool
    repaired: bool
    error: str | None


def load_gold(path: str | Path = DEFAULT_GOLD_PATH) -> list[dict]:
    with Path(path).open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def evaluate_case(case: dict, con: duckdb.DuckDBPyConnection, client: LLMClient, dictionary: str) -> CaseResult:
    gold_sql = case["sql"]
    gold_rows = con.execute(gold_sql).fetchall()
    try:
        pred: SQLResult = answer(case["question"], con, client, dictionary=dictionary)
    except (GuardError, duckdb.Error) as exc:
        return CaseResult(case["id"], case["question"], gold_sql, None, False, False, str(exc))

    correct = result_sets_match(pred.rows, gold_rows)
    return CaseResult(
        id=case["id"],
        question=case["question"],
        gold_sql=gold_sql,
        pred_sql=pred.sql,
        correct=correct,
        repaired=pred.repaired,
        error=None,
    )


def evaluate(
    con: duckdb.DuckDBPyConnection,
    client: LLMClient,
    *,
    gold_path: str | Path = DEFAULT_GOLD_PATH,
) -> dict:
    """Run the full gold set and return a summary dict (existing eval style)."""
    dictionary = build_data_dictionary(con)
    gold = load_gold(gold_path)
    cases = [evaluate_case(c, con, client, dictionary) for c in gold]

    n = len(cases)
    n_correct = sum(c.correct for c in cases)
    n_repaired = sum(c.repaired for c in cases)
    n_errors = sum(c.error is not None for c in cases)
    return {
        "n_cases": n,
        "n_correct": n_correct,
        "execution_accuracy": (n_correct / n) if n else 0.0,
        "n_repaired": n_repaired,
        "n_guard_or_sql_errors": n_errors,
        "cases": [c.__dict__ for c in cases],
    }

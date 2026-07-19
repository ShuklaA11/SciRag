"""Natural-language question -> validated read-only SQL, over the eval warehouse.

The valuable part here is not the prompt — it is the safety net around it:

* **schema-grounding**: the data dictionary is injected into the prompt so the
  model references only real tables, columns, and category values.
* **read-only guard**: the generated SQL is parsed and rejected unless it is a
  single ``SELECT``/``WITH`` statement (no INSERT/DROP/ATTACH/PRAGMA/...).
* **validate -> repair**: the query is dry-run with ``EXPLAIN``; on a DuckDB
  error the message is fed back to the model for exactly one repair attempt.

Execution additionally runs on a read-only connection as defense in depth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import duckdb

from ..llm.client import LLMClient
from .dictionary import build_data_dictionary

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|copy|install|"
    r"load|pragma|call|export|import|replace|truncate|grant|revoke|vacuum|set)\b",
    re.IGNORECASE,
)
_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)

_SYSTEM = """You are a text-to-SQL assistant for a DuckDB database of ML benchmark results.
Translate the user's question into ONE read-only SQL SELECT query for DuckDB.

Rules:
- Output ONLY the SQL. No prose, no explanation, no markdown fences.
- Exactly one statement. It must start with SELECT or WITH.
- Never write INSERT, UPDATE, DELETE, DROP, or any statement that changes data.
- Use only the tables, columns, and category values listed in the schema below.
- Exclude smoke runs (is_smoke = true) when comparing runs, unless asked otherwise.
- Prefer a single table. Only JOIN when you need columns from another table.
  Every run-level metric (mean_recall_at_k_fuzzy, accuracy, macro_f1, latency_*,
  runtime_sec, k, rerank, embedder, ...) lives on the `runs` table — never join
  to a child table to read them.
- Identify a specific run with `run_name = '<exact id>'`. Never infer filters by
  parsing a run's name into embedder/model/week (e.g. do NOT write model_name = 'large').

{dictionary}
"""


class GuardError(ValueError):
    """Raised when generated SQL is not a safe, single read-only SELECT."""


@dataclass(frozen=True)
class SQLResult:
    question: str
    sql: str
    columns: tuple[str, ...]
    rows: tuple[tuple, ...]
    repaired: bool


def extract_sql(text: str) -> str:
    """Pull SQL out of a model response, tolerating markdown fences."""
    fenced = _FENCE.search(text)
    candidate = fenced.group(1) if fenced else text
    return candidate.strip().rstrip(";").strip()


def assert_read_only(sql: str) -> None:
    """Reject anything that is not a single read-only SELECT/WITH statement."""
    stripped = _BLOCK_COMMENT.sub(" ", _LINE_COMMENT.sub(" ", sql)).strip()
    if not stripped:
        raise GuardError("empty query")
    if ";" in stripped.rstrip(";"):
        raise GuardError("multiple statements are not allowed")
    if not re.match(r"(?is)^\s*(select|with)\b", stripped):
        raise GuardError("query must start with SELECT or WITH")
    match = _FORBIDDEN.search(stripped)
    if match:
        raise GuardError(f"forbidden keyword: {match.group(1).upper()}")


def _repair_hint(previous_sql: str, error: str) -> str:
    return (
        f"\n\nYour previous query failed. Fix it and return only the corrected SQL.\n"
        f"Previous SQL:\n{previous_sql}\n\nDuckDB error:\n{error}"
    )


def generate_sql(
    question: str,
    dictionary: str,
    client: LLMClient,
    *,
    previous_sql: str | None = None,
    error: str | None = None,
) -> str:
    """Ask the model for one guarded SELECT. Raises GuardError if it's unsafe."""
    system = _SYSTEM.format(dictionary=dictionary)
    user = question
    if previous_sql is not None and error is not None:
        user = question + _repair_hint(previous_sql, error)
    raw = client.generate(system, user, temperature=0.0, max_tokens=512, num_ctx=8192)
    sql = extract_sql(raw)
    assert_read_only(sql)
    return sql


def answer(
    question: str,
    con: duckdb.DuckDBPyConnection,
    client: LLMClient,
    *,
    dictionary: str | None = None,
) -> SQLResult:
    """Full path: NL question -> guarded SQL -> (one repair) -> executed rows."""
    dictionary = dictionary if dictionary is not None else build_data_dictionary(con)

    sql = generate_sql(question, dictionary, client)
    repaired = False
    try:
        con.execute(f"EXPLAIN {sql}")
    except duckdb.Error as first_error:
        sql = generate_sql(
            question, dictionary, client, previous_sql=sql, error=str(first_error)
        )
        con.execute(f"EXPLAIN {sql}")  # if this still fails, surface it to the caller
        repaired = True

    cursor = con.execute(sql)
    columns = tuple(desc[0] for desc in cursor.description)
    rows = tuple(cursor.fetchall())
    return SQLResult(question=question, sql=sql, columns=columns, rows=rows, repaired=repaired)

"""Ask the eval warehouse a question in natural language.

Usage:
    python scripts/ask_sql.py "which retrieval run had the best fuzzy recall?"
    python scripts/ask_sql.py --summary "how many runs used reranking?"
    python scripts/ask_sql.py --sql-only "average recall by embedder"

Shows the generated SQL, then the result rows. With --summary, the model also
writes a one-line natural-language answer from the rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.client import get_client
from src.sqllab.schema import DEFAULT_DB_PATH, connect
from src.sqllab.text_to_sql import SQLResult, answer

_MAX_ROWS_SHOWN = 50


def _render_table(result: SQLResult) -> str:
    if not result.rows:
        return "(no rows)"
    cols = result.columns
    shown = result.rows[:_MAX_ROWS_SHOWN]
    widths = [
        max(len(str(cols[i])), *(len(str(row[i])) for row in shown))
        for i in range(len(cols))
    ]
    header = "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cols))
    sep = "  ".join("-" * w for w in widths)
    body = "\n".join(
        "  ".join(str(row[i]).ljust(widths[i]) for i in range(len(cols))) for row in shown
    )
    extra = f"\n... ({len(result.rows) - _MAX_ROWS_SHOWN} more rows)" if len(result.rows) > _MAX_ROWS_SHOWN else ""
    return f"{header}\n{sep}\n{body}{extra}"


def _summarize(question: str, result: SQLResult, client) -> str:
    table = _render_table(result)
    system = (
        "You answer questions about ML benchmark results. You are given the user's "
        "question and the SQL result rows. Answer in one concise sentence using only the rows."
    )
    user = f"Question: {question}\n\nResult rows:\n{table}"
    return client.generate(system, user, temperature=0.0, max_tokens=200).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="+", help="Natural-language question")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--summary", action="store_true", help="Add a NL answer from the rows")
    parser.add_argument("--sql-only", action="store_true", help="Print SQL and exit (no execution shown)")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"error: {args.db} not found — run scripts/build_eval_db.py first")
        return 1

    question = " ".join(args.question)
    client = get_client(args.provider)
    con = connect(args.db, read_only=True)
    try:
        result = answer(question, con, client)
    finally:
        con.close()

    print(f"\nSQL{' (repaired)' if result.repaired else ''}:\n  {result.sql}\n")
    if args.sql_only:
        return 0
    print(_render_table(result))
    if args.summary:
        print(f"\nAnswer: {_summarize(question, result, client)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

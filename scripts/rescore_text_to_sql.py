"""Re-score a text-to-SQL result file under the *current* execution matcher.

The LLM predictions in a result file are frozen (generation is non-deterministic
at temperature > 0). When the matcher in `src.sqllab.eval_sql.result_sets_match`
changes, the honest way to update the numbers is to re-execute each stored
`pred_sql` / `gold_sql` against the warehouse and re-apply the matcher — not to
regenerate, which would change the predictions themselves.

Usage:
    python scripts/rescore_text_to_sql.py eval/results/sql_t2s_anthropic_sonnet46.json
    python scripts/rescore_text_to_sql.py <file> --out <other>   # don't overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sqllab.eval_sql import result_sets_match
from src.sqllab.schema import DEFAULT_DB_PATH, connect


def rescore(summary: dict, con: duckdb.DuckDBPyConnection) -> tuple[dict, list[tuple[str, bool, bool]]]:
    """Return a new summary with `correct` re-evaluated, plus the list of flips."""
    flips: list[tuple[str, bool, bool]] = []
    cases = []
    for case in summary["cases"]:
        old = case["correct"]
        if case["pred_sql"] is None:  # guard/SQL error at generation time — stays failed
            new = False
        else:
            try:
                gold_rows = con.execute(case["gold_sql"]).fetchall()
                pred_rows = con.execute(case["pred_sql"]).fetchall()
                new = result_sets_match(pred_rows, gold_rows)
            except duckdb.Error:
                new = False
        if new != old:
            flips.append((case["id"], old, new))
        cases.append({**case, "correct": new})

    n = summary["n_cases"]
    n_correct = sum(c["correct"] for c in cases)
    return {
        **summary,
        "n_correct": n_correct,
        "execution_accuracy": (n_correct / n) if n else 0.0,
        "cases": cases,
    }, flips


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_file", type=Path)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--out", type=Path, default=None, help="Write here instead of overwriting")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"error: {args.db} not found — run scripts/build_eval_db.py first")
        return 1

    summary = json.loads(args.result_file.read_text())
    con = connect(args.db, read_only=True)
    try:
        rescored, flips = rescore(summary, con)
    finally:
        con.close()

    old_n, n = summary["n_correct"], summary["n_cases"]
    new_n = rescored["n_correct"]
    print(f"{args.result_file.name}: {old_n}/{n} = {old_n / n:.1%}  ->  {new_n}/{n} = {new_n / n:.1%}")
    for cid, old, new in flips:
        print(f"  flip {cid}: {old} -> {new}")

    out = args.out or args.result_file
    out.write_text(json.dumps(rescored, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Benchmark the text-to-SQL layer over the gold question set.

Usage:
    python scripts/eval_text_to_sql.py                       # default provider (Ollama)
    SCIRAG_LLM_PROVIDER=anthropic python scripts/eval_text_to_sql.py
    python scripts/eval_text_to_sql.py --out eval/results/sql_t2s_ollama.json

Requires a built warehouse (run scripts/build_eval_db.py first) and a reachable
LLM provider. Prints per-case pass/fail and overall execution accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.client import get_client
from src.sqllab.eval_sql import DEFAULT_GOLD_PATH, evaluate
from src.sqllab.schema import DEFAULT_DB_PATH, connect


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_PATH)
    parser.add_argument("--provider", default=None, help="Override SCIRAG_LLM_PROVIDER")
    parser.add_argument("--out", type=Path, default=None, help="Write summary JSON here")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"error: {args.db} not found — run scripts/build_eval_db.py first")
        return 1

    client = get_client(args.provider)
    con = connect(args.db, read_only=True)
    try:
        summary = evaluate(con, client, gold_path=args.gold)
    finally:
        con.close()

    for case in summary["cases"]:
        mark = "PASS" if case["correct"] else "FAIL"
        tag = " (repaired)" if case["repaired"] else ""
        print(f"  [{mark}] {case['id']}{tag}")
        if not case["correct"]:
            detail = case["error"] or f"pred: {case['pred_sql']}"
            print(f"         {detail}")

    acc = summary["execution_accuracy"]
    print(f"\nExecution accuracy: {summary['n_correct']}/{summary['n_cases']} = {acc:.1%}")
    print(f"Repaired: {summary['n_repaired']}   Errors: {summary['n_guard_or_sql_errors']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

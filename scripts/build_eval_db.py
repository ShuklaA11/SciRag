"""Build the DuckDB eval-results warehouse from eval/results/*.json[l].

Usage:
    python scripts/build_eval_db.py                 # build data/eval.duckdb
    python scripts/build_eval_db.py --db /tmp/e.db  # custom location

After building, prints row counts and runs a self-check: a known run's fuzzy
recall in the DB must match its source summary JSON exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sqllab.ingest import DEFAULT_RESULTS_DIR, build_eval_db
from src.sqllab.schema import DEFAULT_DB_PATH, connect

_CHECK_RUN = "week6_rerank_k10"
_CHECK_COLUMN = "mean_recall_at_k_fuzzy"


def _self_check(db_path: Path, results_dir: Path) -> bool:
    """Cross-check one run's metric in the DB against its summary JSON on disk."""
    summary = results_dir / f"{_CHECK_RUN}_summary.json"
    if not summary.exists():
        print(f"[check] skipped — {summary} not found")
        return True
    expected = json.loads(summary.read_text())[_CHECK_COLUMN]

    con = connect(db_path, read_only=True)
    try:
        row = con.execute(
            f"SELECT {_CHECK_COLUMN} FROM runs WHERE run_name = ?", [_CHECK_RUN]
        ).fetchone()
    finally:
        con.close()

    if row is None:
        print(f"[check] FAIL — run {_CHECK_RUN!r} missing from DB")
        return False
    actual = row[0]
    ok = actual == expected
    verdict = "OK" if ok else "FAIL"
    print(f"[check] {verdict} — {_CHECK_RUN}.{_CHECK_COLUMN}: db={actual} json={expected}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    counts = build_eval_db(args.results_dir, args.db)
    print(f"Built {args.db}")
    for table, n in counts.items():
        print(f"  {table:20s} {n:>7,d} rows")

    return 0 if _self_check(args.db, args.results_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())

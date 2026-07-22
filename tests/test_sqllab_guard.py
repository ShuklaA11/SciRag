"""Unit tests for the text-to-SQL guardrails and execution-match comparator.

These cover the security-critical path (rejecting non-read-only SQL) and the
result-set comparison logic without needing an LLM or a populated warehouse.
"""

import pytest

from src.sqllab.eval_sql import result_sets_match
from src.sqllab.text_to_sql import GuardError, assert_read_only, extract_sql


@pytest.mark.unit
@pytest.mark.parametrize("raw,expected", [
    ("SELECT 1", "SELECT 1"),
    ("```sql\nSELECT 1\n```", "SELECT 1"),
    ("```\nSELECT 1\n```", "SELECT 1"),
    ("SELECT 1;", "SELECT 1"),
    ("  SELECT 1 ;  ", "SELECT 1"),
])
def test_extract_sql_strips_fences_and_semicolons(raw, expected):
    assert extract_sql(raw) == expected


@pytest.mark.unit
@pytest.mark.parametrize("sql", [
    "SELECT run_name FROM runs",
    "select * from runs where rerank",
    "WITH x AS (SELECT 1) SELECT * FROM x",
    "SELECT count(*) FROM runs -- a comment",
])
def test_assert_read_only_accepts_selects(sql):
    assert_read_only(sql)  # should not raise


@pytest.mark.unit
@pytest.mark.parametrize("sql", [
    "DROP TABLE runs",
    "DELETE FROM runs",
    "UPDATE runs SET k = 1",
    "INSERT INTO runs (run_name) VALUES ('x')",
    "ATTACH 'evil.db'",
    "PRAGMA database_list",
    "SELECT 1; DROP TABLE runs",
    "",
    "   ",
    "EXPLAIN SELECT 1",  # not a bare SELECT/WITH
])
def test_assert_read_only_rejects_unsafe(sql):
    with pytest.raises(GuardError):
        assert_read_only(sql)


@pytest.mark.unit
def test_result_sets_match_is_order_insensitive():
    a = [("bge", 0.771), ("specter2", 0.739)]
    b = [("specter2", 0.739), ("bge", 0.771)]
    assert result_sets_match(a, b)


@pytest.mark.unit
def test_result_sets_match_rounds_floats():
    assert result_sets_match([(0.7712192192,)], [(0.7712192,)])


@pytest.mark.unit
def test_result_sets_match_detects_difference():
    assert not result_sets_match([("bge",)], [("specter2",)])


@pytest.mark.unit
def test_result_sets_match_tolerates_column_order():
    pred = [(0.771, "bge"), (0.739, "specter2")]      # (avg, embedder)
    gold = [("bge", 0.771), ("specter2", 0.739)]      # (embedder, avg)
    assert result_sets_match(pred, gold)


@pytest.mark.unit
def test_result_sets_match_tolerates_extra_context_columns():
    # gold asks for the top run name; pred returns name + its accuracy (helpful
    # context). Both answer the question, so the extra column must not fail it.
    pred = [("week4_bge", 0.771)]
    gold = [("week4_bge",)]
    assert result_sets_match(pred, gold)


@pytest.mark.unit
def test_result_sets_match_rejects_extra_columns_with_wrong_values():
    # extra columns don't buy a pass when the gold values aren't present
    pred = [("specter2", 0.739)]
    gold = [("bge",)]
    assert not result_sets_match(pred, gold)

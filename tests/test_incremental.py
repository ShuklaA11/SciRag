"""Tests for src/wiki/incremental.py — pure functions, no LLM."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.wiki.incremental import (
    STATE_FILENAME,
    clear_stale_marker,
    decide,
    flag_stale_concepts,
    hash_tei,
    load_state,
    save_state,
    update_state,
)


# ----------------------------------------------------------------- hashing


def test_hash_tei_deterministic() -> None:
    assert hash_tei("hello") == hash_tei("hello")


def test_hash_tei_changes_with_content() -> None:
    assert hash_tei("a") != hash_tei("b")


# ----------------------------------------------------------------- state IO


def test_load_state_missing_returns_empty(tmp_path: Path) -> None:
    assert load_state(tmp_path) == {}


def test_load_state_corrupt_returns_empty(tmp_path: Path) -> None:
    (tmp_path / STATE_FILENAME).write_text("{not json")
    assert load_state(tmp_path) == {}


def test_load_state_non_dict_returns_empty(tmp_path: Path) -> None:
    (tmp_path / STATE_FILENAME).write_text("[1, 2, 3]")
    assert load_state(tmp_path) == {}


def test_save_then_load_roundtrip(tmp_path: Path) -> None:
    state = {"bert": "deadbeef", "elmo": "cafebabe"}
    save_state(tmp_path, state)
    assert load_state(tmp_path) == state


def test_save_state_creates_parent_dir(tmp_path: Path) -> None:
    nested = tmp_path / "wiki"
    save_state(nested, {"x": "y"})
    assert (nested / STATE_FILENAME).exists()


def test_update_state_is_immutable() -> None:
    original = {"bert": "h1"}
    new = update_state(original, "elmo", "tei content")
    assert "elmo" not in original
    assert new["bert"] == "h1"
    assert new["elmo"] == hash_tei("tei content")


# ------------------------------------------------------------------ decide


def test_decide_new_paper() -> None:
    d = decide("bert", "tei", {}, md_path_exists=False, force_rebuild=False)
    assert d.needs_compile is True
    assert d.reason == "new"


def test_decide_unchanged_skipped() -> None:
    state = {"bert": hash_tei("tei")}
    d = decide("bert", "tei", state, md_path_exists=True, force_rebuild=False)
    assert d.needs_compile is False
    assert d.reason == "unchanged"


def test_decide_changed_recompiles() -> None:
    state = {"bert": hash_tei("old tei")}
    d = decide("bert", "new tei", state, md_path_exists=True, force_rebuild=False)
    assert d.needs_compile is True
    assert d.reason == "changed"


def test_decide_md_missing_recompiles_even_if_hash_matches() -> None:
    state = {"bert": hash_tei("tei")}
    d = decide("bert", "tei", state, md_path_exists=False, force_rebuild=False)
    assert d.needs_compile is True
    assert d.reason == "new"


def test_decide_bootstrap_md_exists_no_state() -> None:
    # Pre-incremental world: md was compiled before .state.json existed.
    # Should adopt it (no recompile, but backfill state at the caller).
    d = decide("bert", "tei", {}, md_path_exists=True, force_rebuild=False)
    assert d.needs_compile is False
    assert d.reason == "bootstrap"


def test_decide_force_rebuild_always_compiles() -> None:
    state = {"bert": hash_tei("tei")}
    d = decide("bert", "tei", state, md_path_exists=True, force_rebuild=True)
    assert d.needs_compile is True
    assert d.reason == "force_rebuild"


# ----------------------------------------------------------- stale flagging


@pytest.fixture
def concepts(tmp_path: Path) -> Path:
    d = tmp_path / "concepts"
    d.mkdir()
    (d / "masked_lm.md").write_text(
        '---\nconcept: "MLM"\ngenerated_by: x\ngenerated_at: x\n'
        'status: ok\ndepends_on: [bert, 1601.06738]\n---\n\n## Definition\nX\n'
    )
    (d / "transformer.md").write_text(
        '---\nconcept: "Transformer"\ngenerated_by: x\ngenerated_at: x\n'
        'status: ok\ndepends_on: [attention_is_all_you_need]\n---\n\n## Definition\nY\n'
    )
    return d


def test_flag_stale_concepts_marks_dependent(concepts: Path) -> None:
    written = flag_stale_concepts(concepts, {"bert"})
    assert len(written) == 1
    assert written[0].name == "masked_lm.stale"
    payload = json.loads(written[0].read_text())
    assert payload["concept_slug"] == "masked_lm"
    assert payload["triggered_by"] == ["bert"]


def test_flag_stale_concepts_no_overlap(concepts: Path) -> None:
    written = flag_stale_concepts(concepts, {"some_other_paper"})
    assert written == []


def test_flag_stale_concepts_empty_changed_set(concepts: Path) -> None:
    assert flag_stale_concepts(concepts, set()) == []


def test_flag_stale_concepts_missing_dir(tmp_path: Path) -> None:
    assert flag_stale_concepts(tmp_path / "no_such_dir", {"bert"}) == []


def test_flag_stale_concepts_multiple_triggers(concepts: Path) -> None:
    written = flag_stale_concepts(concepts, {"bert", "1601.06738"})
    assert len(written) == 1
    payload = json.loads(written[0].read_text())
    assert payload["triggered_by"] == ["1601.06738", "bert"]


def test_clear_stale_marker_removes_file(concepts: Path) -> None:
    flag_stale_concepts(concepts, {"bert"})
    md = concepts / "masked_lm.md"
    assert md.with_suffix(".stale").exists()
    clear_stale_marker(md)
    assert not md.with_suffix(".stale").exists()


def test_clear_stale_marker_no_op_when_absent(concepts: Path) -> None:
    md = concepts / "masked_lm.md"
    # no marker exists; should not raise
    clear_stale_marker(md)

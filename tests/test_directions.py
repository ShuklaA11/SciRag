"""SB-D2: gap detection + direction proposal. Fake LLM (no model). Guards the
NOVEL→gap filter and the propose/parse/fallback contract."""

from __future__ import annotations

import json

from src.brainstorm import DirectionProposer, gaps_from_verdicts
from src.ideas import CONTRADICTED, ENTAILED, NOVEL, ClaimVerdict


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.last_user: str | None = None

    def generate(self, system, user, **kwargs) -> str:
        self.last_user = user
        return self.response


def _v(claim: str, bucket: str) -> ClaimVerdict:
    return ClaimVerdict(claim, bucket, 0.0, 0.0, 0, None)


def _dirs_json(*dirs: str) -> str:
    return json.dumps({"directions": list(dirs)})


# --- gaps_from_verdicts -----------------------------------------------------


def test_gaps_are_only_novel_verdicts():
    verdicts = [_v("done", ENTAILED), _v("open1", NOVEL), _v("refuted", CONTRADICTED), _v("open2", NOVEL)]
    assert gaps_from_verdicts(verdicts) == ["open1", "open2"]


def test_no_novel_means_no_gaps():
    assert gaps_from_verdicts([_v("a", ENTAILED), _v("b", CONTRADICTED)]) == []


# --- DirectionProposer ------------------------------------------------------


def test_propose_returns_parsed_directions_and_passes_gaps():
    llm = FakeLLM(_dirs_json("Probe X on low-resource data.", "Test Y under distribution shift."))
    out = DirectionProposer(llm).propose("sparse attention", ["X untested", "Y untested"])
    assert out == ["Probe X on low-resource data.", "Test Y under distribution shift."]
    assert "sparse attention" in llm.last_user
    assert "X untested" in llm.last_user


def test_no_gaps_short_circuits_without_calling_llm():
    llm = FakeLLM(_dirs_json("should not appear"))
    assert DirectionProposer(llm).propose("topic", []) == []
    assert llm.last_user is None


def test_caps_at_max_directions():
    llm = FakeLLM(_dirs_json(*[f"dir {i}" for i in range(10)]))
    out = DirectionProposer(llm, max_directions=3).propose("t", ["gap"])
    assert out == ["dir 0", "dir 1", "dir 2"]


def test_malformed_json_proposes_nothing():
    assert DirectionProposer(FakeLLM("not json")).propose("t", ["gap"]) == []


def test_non_string_items_dropped():
    llm = FakeLLM('{"directions": ["Real direction.", 7, "", "  "]}')
    assert DirectionProposer(llm).propose("t", ["gap"]) == ["Real direction."]


def test_object_items_coerced_to_direction_field():
    # Local models often return {title, description} objects despite the string
    # instruction — coerce to the most direction-like field instead of dropping.
    llm = FakeLLM(json.dumps({"directions": [
        {"title": "Label", "description": "Investigate X under distribution shift."},
        {"direction": "Probe Y in low-resource settings."},
        {"unusable": "no known key"},
    ]}))
    assert DirectionProposer(llm).propose("t", ["gap"]) == [
        "Investigate X under distribution shift.",
        "Probe Y in low-resource settings.",
    ]

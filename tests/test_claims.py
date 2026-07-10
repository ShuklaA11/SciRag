"""SB-C1: idea → atomic claims. Fake LLM (structural _LLMLike), so no model
download. Guards the decompose contract: split, dedup, caps, and the
conservative fallbacks that keep the engine always-runnable."""

from __future__ import annotations

import json

from src.ideas import ClaimDecomposer


class FakeLLM:
    """Structural _LLMLike stub. Returns a canned response; records the last
    (system, user) so tests can assert the idea reached the prompt."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.last_user: str | None = None

    def generate(self, system, user, **kwargs) -> str:
        self.last_user = user
        return self.response


def _claims_json(*claims: str) -> str:
    return json.dumps({"claims": list(claims)})


def test_splits_idea_into_atomic_claims():
    llm = FakeLLM(_claims_json("Sparse attention reduces memory.", "It matches dense accuracy."))
    out = ClaimDecomposer(llm).decompose("Sparse attention saves memory and matches accuracy.")
    assert out == ["Sparse attention reduces memory.", "It matches dense accuracy."]
    assert llm.last_user == "Sparse attention saves memory and matches accuracy."


def test_single_claim_passthrough():
    llm = FakeLLM(_claims_json("Contrastive pretraining improves low-resource NER."))
    assert ClaimDecomposer(llm).decompose("contrastive helps NER") == [
        "Contrastive pretraining improves low-resource NER."
    ]


def test_dedups_case_insensitively_preserving_order():
    llm = FakeLLM(_claims_json("Claim A.", "claim a.", "Claim B.", "CLAIM A."))
    assert ClaimDecomposer(llm).decompose("idea") == ["Claim A.", "Claim B."]


def test_caps_at_max_claims():
    llm = FakeLLM(_claims_json(*[f"Distinct claim {i}." for i in range(10)]))
    out = ClaimDecomposer(llm, max_claims=3).decompose("idea")
    assert out == ["Distinct claim 0.", "Distinct claim 1.", "Distinct claim 2."]


def test_blank_idea_yields_no_claims_without_calling_llm():
    llm = FakeLLM(_claims_json("should not be used"))
    assert ClaimDecomposer(llm).decompose("   ") == []
    assert llm.last_user is None


def test_malformed_json_falls_back_to_whole_idea():
    assert ClaimDecomposer(FakeLLM("not json at all")).decompose("My idea") == ["My idea"]


def test_empty_claim_list_falls_back_to_whole_idea():
    assert ClaimDecomposer(FakeLLM(_claims_json())).decompose("My idea") == ["My idea"]


def test_non_string_items_are_dropped():
    llm = FakeLLM('{"claims": ["Real claim.", 42, "", "  "]}')
    assert ClaimDecomposer(llm).decompose("idea") == ["Real claim."]

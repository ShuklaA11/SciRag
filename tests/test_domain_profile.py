"""SB-A1: DomainProfile foundation + NLP profile equivalence.

The key guard is `test_nlp_patterns_match_legacy_chunker`: it proves the
section patterns extracted into the NLP profile classify heads identically to
the pre-Phase-A `section_chunker.section_type_for_head`, so SB-A2 can point the
chunker at the profile without behavior change.
"""

from __future__ import annotations

import pytest

from src.domain import active_profile, available, get_profile
from src.domain.profile import DomainProfile
from src.domain.profiles import NLP_ML
from src.pipeline.section_chunker import SECTION_TYPES as LEGACY_TYPES
from src.pipeline.section_chunker import section_type_for_head


def test_nlp_profile_registered_and_default():
    assert "nlp_ml" in available()
    assert get_profile("nlp_ml") is NLP_ML
    assert active_profile().name == "nlp_ml"


def test_nlp_section_types_match_legacy():
    assert NLP_ML.section_types == tuple(LEGACY_TYPES)


@pytest.mark.parametrize(
    "head",
    [
        "Abstract",
        "Introduction",
        "Motivation",
        "Related Work",
        "Background",
        "Prior Work",
        "Proposed Method",
        "Model Architecture",
        "Our Approach",
        "Experimental Setup",
        "Implementation Details",
        "Datasets",
        "Results",
        "Evaluation",
        "Ablation Study",
        "Conclusion",
        "Discussion",
        "Future Work",
        "Limitations",
        "Acknowledgements",  # -> other
        "References",  # -> other
        "Model Overview",  # method wins over introduction (order matters)
    ],
)
def test_nlp_patterns_match_legacy_chunker(head):
    # First-match over the profile's compiled patterns; fall back to "other".
    label = "other"
    for name, pat in NLP_ML.compiled_patterns():
        if pat.search(head):
            label = name
            break
    assert label == section_type_for_head(head)


def test_active_profile_respects_env(monkeypatch):
    monkeypatch.setenv("SCIRAG_DOMAIN", "nlp_ml")
    assert active_profile().name == "nlp_ml"
    monkeypatch.setenv("SCIRAG_DOMAIN", "does_not_exist")
    with pytest.raises(KeyError):
        active_profile()


def test_get_profile_unknown_raises():
    with pytest.raises(KeyError):
        get_profile("nope")


def test_post_init_rejects_pattern_label_not_in_types():
    with pytest.raises(ValueError):
        DomainProfile(
            name="bad",
            section_types=("abstract", "other"),
            section_patterns=(("method", r"\bmethod\b"),),  # 'method' not in types
            embedder_name="bge",
            verification_strategy="nli",
            verification_model="x",
        )

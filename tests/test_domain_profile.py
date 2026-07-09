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
from src.domain.profiles import BIOMEDICAL, NLP_ML
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


def test_biomedical_registered_and_selectable(monkeypatch):
    assert "biomedical" in available()
    assert get_profile("biomedical") is BIOMEDICAL
    monkeypatch.setenv("SCIRAG_DOMAIN", "biomedical")
    assert active_profile().name == "biomedical"


def test_biomedical_taxonomy_differs_from_nlp():
    assert BIOMEDICAL.section_types != NLP_ML.section_types
    assert "discussion" in BIOMEDICAL.section_types
    assert "related_work" not in BIOMEDICAL.section_types
    assert "experiments" not in BIOMEDICAL.section_types


@pytest.mark.parametrize(
    "head,expected",
    [
        ("Abstract", "abstract"),
        ("Introduction", "introduction"),
        ("Background", "background"),
        ("Materials and Methods", "methods"),
        ("Statistical Analysis", "methods"),
        ("Results", "results"),
        ("Primary Outcome", "results"),
        ("Discussion", "discussion"),
        ("Conclusion", "conclusion"),
        ("Related Work", "other"),  # not a biomedical bucket
    ],
)
def test_biomedical_head_classification(head, expected):
    label = "other"
    for name, pat in BIOMEDICAL.compiled_patterns():
        if pat.search(head):
            label = name
            break
    assert label == expected


@pytest.mark.parametrize(
    "head,nlp_label,bio_label",
    [
        ("Discussion", "conclusion", "discussion"),
        ("Related Work", "related_work", "other"),
        ("Materials and Methods", "method", "methods"),
    ],
)
def test_active_profile_switches_chunker_classification(monkeypatch, head, nlp_label, bio_label):
    # Same head, different bucket depending on the active domain — proves the
    # chunker honors active_profile() at call time (the SB-A2 wiring generalizes).
    monkeypatch.setenv("SCIRAG_DOMAIN", "nlp_ml")
    assert section_type_for_head(head) == nlp_label
    monkeypatch.setenv("SCIRAG_DOMAIN", "biomedical")
    assert section_type_for_head(head) == bio_label


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

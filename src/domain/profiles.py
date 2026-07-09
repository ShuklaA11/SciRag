"""Built-in domain profiles. Imported by ``src.domain`` to self-register.

The NLP/ML profile holds the *exact* section taxonomy and head-matching rules
that lived in ``section_chunker`` and ``router.tfidf_classifier`` before Phase A,
so extracting them here is behavior-preserving (guarded by the existing
``test_section_chunker`` / ``test_tfidf_classifier`` suites once consumers read
from the profile in SB-A2/A3).
"""

from __future__ import annotations

from src.domain.profile import DomainProfile, register

# Order matters: more-specific patterns first; "introduction" runs late so
# heads like "Model Overview" claim "method" before introduction's "overview".
_NLP_SECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("abstract", r"\babstract\b"),
    ("related_work", r"\brelated\s+work\b|\bbackground\b|\bprior\s+work\b|\bliterature\b"),
    ("method", r"\bmethod|\bapproach|\barchitecture|\bmodel\b|\balgorithm|\bformulation|\bproposed\b"),
    ("experiments", r"\bexperiment|\bsetup\b|\bimplementation\b|\btraining\b|\bdataset"),
    ("results", r"\bresult|\bevaluation\b|\bfinding|\banalysis\b|\bablation"),
    ("conclusion", r"\bconclusion|\bdiscussion\b|\bfuture\s+work\b|\blimitation"),
    ("introduction", r"\bintroduction\b|\bmotivation\b|\boverview\b"),
)

NLP_ML = DomainProfile(
    name="nlp_ml",
    section_types=(
        "abstract",
        "introduction",
        "related_work",
        "method",
        "experiments",
        "results",
        "conclusion",
        "other",
    ),
    section_patterns=_NLP_SECTION_PATTERNS,
    embedder_name="bge",  # canonical baseline embedder (see eval/baseline_v2.json)
    verification_strategy="nli",
    verification_model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    data_sources=("arxiv", "acl_anthology", "semantic_scholar", "papers_with_code"),
    eval_benchmark="qasper",
)

register(NLP_ML)


# --- Biomedical -------------------------------------------------------------
# A deliberately different taxonomy: clinical/biomed papers are strict IMRaD
# with Background/Methods/Discussion buckets and no ML-isms (related_work,
# experiments, ablation). This is what makes the profile do real work — the
# same head classifies differently depending on the active domain (e.g.
# "Discussion" -> conclusion under NLP, -> discussion here).
_BIOMED_SECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("abstract", r"\babstract\b"),
    ("background", r"\bbackground\b"),
    ("methods", r"\bmethod|\bmaterials\b|\bstudy\s+design\b|\bparticipants\b|\bpatients\b|\bprocedure|\bstatistical\s+analysis\b"),
    ("results", r"\bresult|\bfinding|\boutcome"),
    ("discussion", r"\bdiscussion\b|\binterpretation\b"),
    ("conclusion", r"\bconclusion|\blimitation"),
    ("introduction", r"\bintroduction\b|\bmotivation\b"),
)

BIOMEDICAL = DomainProfile(
    name="biomedical",
    section_types=(
        "abstract",
        "introduction",
        "background",
        "methods",
        "results",
        "discussion",
        "conclusion",
        "other",
    ),
    section_patterns=_BIOMED_SECTION_PATTERNS,
    embedder_name="specter2",  # scientific-general, citation-based; already available
    verification_strategy="nli",
    verification_model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",  # validated on SciFact (eval/baseline_v2.json)
    data_sources=("pubmed", "pmc", "semantic_scholar", "clinicaltrials_gov"),
    eval_benchmark="scifact",
)

register(BIOMEDICAL)

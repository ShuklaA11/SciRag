"""Tests for the multi-label TF-IDF router."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.router.tfidf_classifier import SECTION_TYPES, TfidfRouter


@pytest.fixture
def synthetic_data() -> tuple[list[str], list[list[str]]]:
    questions = [
        "what dataset do they use for training",
        "which dataset is used in the experiments",
        "what corpus do the authors train on",
        "how many parameters does the model have",
        "what is the architecture of their model",
        "which model is proposed in the paper",
        "what F1 score do they achieve",
        "what is the accuracy on the test set",
        "how does it perform compared to baselines",
        "what is the main contribution of the paper",
        "what problem do the authors solve",
        "what is the goal of this work",
    ]
    labels = [
        ["experiments"],
        ["experiments"],
        ["experiments"],
        ["method"],
        ["method"],
        ["method"],
        ["results"],
        ["results"],
        ["results"],
        ["introduction"],
        ["introduction"],
        ["introduction"],
    ]
    return questions, labels


def test_fit_and_predict_returns_known_classes(synthetic_data):
    questions, labels = synthetic_data
    router = TfidfRouter(min_df=1).fit(questions, labels)
    preds = router.predict(["what dataset is used"], threshold=0.5, top_n=1)
    assert len(preds) == 1
    for label in preds[0].labels:
        assert label in SECTION_TYPES
    assert set(preds[0].probabilities.keys()) == set(SECTION_TYPES)


def test_predict_separates_obvious_classes(synthetic_data):
    questions, labels = synthetic_data
    router = TfidfRouter(min_df=1).fit(questions, labels)
    preds = router.predict(
        [
            "what dataset do they evaluate on",
            "what is the accuracy",
        ],
        threshold=0.5,
        top_n=1,
    )
    assert "experiments" in preds[0].labels
    assert "results" in preds[1].labels


def test_save_load_roundtrip(synthetic_data, tmp_path: Path):
    questions, labels = synthetic_data
    router = TfidfRouter(min_df=1).fit(questions, labels)
    out = tmp_path / "router.joblib"
    router.save(out)

    loaded = TfidfRouter.load(out)
    before = router.predict_proba(["what dataset is used"])
    after = loaded.predict_proba(["what dataset is used"])
    assert before.shape == after.shape
    assert (abs(before - after) < 1e-9).all()


def test_predict_without_fit_raises():
    router = TfidfRouter()
    with pytest.raises(RuntimeError):
        router.predict_proba(["anything"])


def test_top_n_union_with_threshold(synthetic_data):
    questions, labels = synthetic_data
    router = TfidfRouter(min_df=1).fit(questions, labels)
    high = router.predict(["random unrelated text xyzzy"], threshold=0.99, top_n=2)
    assert len(high[0].labels) >= 2

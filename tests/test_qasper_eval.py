"""Pure-logic unit tests for src/evaluation/qasper_eval.py.

No I/O, no LLM, no SPECTER2 — just scoring and normalization.
"""

from __future__ import annotations

import pytest

from src.evaluation.qasper_eval import (
    aggregate_results,
    extract_gold_answers,
    extract_gold_evidence,
    max_token_f1,
    normalize_answer,
    recall_at_k,
    token_f1,
)


class TestNormalize:
    def test_lowercases(self):
        assert normalize_answer("HELLO") == "hello"

    def test_strips_punctuation(self):
        assert normalize_answer("hello, world!") == "hello world"

    def test_drops_articles(self):
        assert normalize_answer("a cat in the hat") == "cat in hat"

    def test_collapses_whitespace(self):
        assert normalize_answer("hello   world\n\n\tfoo") == "hello world foo"


class TestTokenF1:
    def test_exact_match(self):
        assert token_f1("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert token_f1("cat dog", "bird fish") == 0.0

    def test_partial_overlap(self):
        # pred: {hello, world}, gold: {hello, there}
        # common=1, precision=1/2, recall=1/2, F1=0.5
        assert token_f1("hello world", "hello there") == pytest.approx(0.5)

    def test_normalization_applied(self):
        # "The cat." vs "a cat" -> both normalize to "cat" -> F1=1.0
        assert token_f1("The cat.", "a cat") == 1.0

    def test_yes_vs_yes_punct(self):
        assert token_f1("yes", "yes.") == 1.0

    def test_yes_vs_no(self):
        assert token_f1("yes", "no") == 0.0

    def test_both_empty(self):
        assert token_f1("", "") == 1.0

    def test_one_empty(self):
        assert token_f1("hello", "") == 0.0

    def test_max_over_multi_gold(self):
        # pred matches the 2nd gold exactly
        assert max_token_f1("exact match", ["nope nope", "exact match", "bar"]) == 1.0

    def test_max_empty_gold_list(self):
        assert max_token_f1("anything", []) == 0.0


class TestRecallAtK:
    def test_full_coverage(self):
        chunks = ["the quick brown fox jumps over the lazy dog"]
        gold = ["quick brown fox", "lazy dog"]
        assert recall_at_k(chunks, gold) == 1.0

    def test_partial(self):
        chunks = ["the quick brown fox"]
        gold = ["quick brown fox", "lazy dog"]
        assert recall_at_k(chunks, gold) == 0.5

    def test_no_match(self):
        chunks = ["nothing relevant here"]
        gold = ["quick brown fox", "lazy dog"]
        assert recall_at_k(chunks, gold) == 0.0

    def test_empty_evidence_returns_none(self):
        assert recall_at_k(["whatever"], []) is None

    def test_whitespace_normalized(self):
        chunks = ["we    propose\na novel\tmethod for retrieval"]
        gold = ["we propose a novel method"]
        assert recall_at_k(chunks, gold) == 1.0

    def test_multi_chunk_any_covers(self):
        chunks = ["first chunk about dogs", "second chunk about cats"]
        gold = ["about cats"]
        assert recall_at_k(chunks, gold) == 1.0

    def test_blank_gold_sentence_ignored(self):
        chunks = ["hello world"]
        gold = ["   ", "hello"]
        # blank sentence filtered from denom; 1/1 matches
        assert recall_at_k(chunks, gold) == 1.0


class TestExtractGold:
    def test_unanswerable(self):
        answers = [{"answer": {"unanswerable": True, "yes_no": None,
                                "extractive_spans": [], "free_form_answer": "",
                                "evidence": [], "highlighted_evidence": []}}]
        assert extract_gold_answers(answers) == ["Unanswerable"]

    def test_yes_no(self):
        ann_yes = {"answer": {"unanswerable": False, "yes_no": True,
                               "extractive_spans": [], "free_form_answer": "",
                               "evidence": [], "highlighted_evidence": []}}
        ann_no = {"answer": {"unanswerable": False, "yes_no": False,
                              "extractive_spans": [], "free_form_answer": "",
                              "evidence": [], "highlighted_evidence": []}}
        assert extract_gold_answers([ann_yes, ann_no]) == ["Yes", "No"]

    def test_extractive_spans(self):
        ann = {"answer": {"unanswerable": False, "yes_no": None,
                          "extractive_spans": ["spanA", "spanB"],
                          "free_form_answer": "",
                          "evidence": [], "highlighted_evidence": []}}
        assert extract_gold_answers([ann]) == ["spanA spanB"]

    def test_free_form(self):
        ann = {"answer": {"unanswerable": False, "yes_no": None,
                          "extractive_spans": [],
                          "free_form_answer": "a long prose answer",
                          "evidence": [], "highlighted_evidence": []}}
        assert extract_gold_answers([ann]) == ["a long prose answer"]

    def test_evidence_prefers_highlighted(self):
        ann = {"answer": {"highlighted_evidence": ["sent one.", "sent two."],
                          "evidence": ["some fallback paragraph"]}}
        out = extract_gold_evidence([ann])
        assert out == ["sent one.", "sent two."]

    def test_evidence_fallback_to_paragraph(self):
        ann = {"answer": {"highlighted_evidence": [],
                          "evidence": ["fallback paragraph"]}}
        assert extract_gold_evidence([ann]) == ["fallback paragraph"]

    def test_evidence_dedup_across_annotations(self):
        ann1 = {"answer": {"highlighted_evidence": ["same sentence."]}}
        ann2 = {"answer": {"highlighted_evidence": ["same sentence.", "new one."]}}
        out = extract_gold_evidence([ann1, ann2])
        assert out == ["same sentence.", "new one."]


class TestAggregate:
    def test_skips_none_recall(self):
        results = [
            {"recall_at_k": 1.0, "answer_f1": 0.8},
            {"recall_at_k": None, "answer_f1": 0.6},
            {"recall_at_k": 0.5, "answer_f1": 0.4},
        ]
        agg = aggregate_results(results)
        assert agg["n_evaluated"] == 3
        assert agg["n_with_evidence"] == 2
        assert agg["mean_recall_at_k"] == pytest.approx(0.75)
        assert agg["mean_answer_f1"] == pytest.approx(0.6)

    def test_empty(self):
        agg = aggregate_results([])
        assert agg["n_evaluated"] == 0
        assert agg["mean_recall_at_k"] is None
        assert agg["mean_answer_f1"] is None

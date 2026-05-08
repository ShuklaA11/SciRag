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


class TestRecallAtKStrict:
    """Legacy substring-match behavior, preserved under strict=True."""

    def test_full_coverage(self):
        chunks = ["the quick brown fox jumps over the lazy dog"]
        gold = ["quick brown fox", "lazy dog"]
        assert recall_at_k(chunks, gold, strict=True) == 1.0

    def test_partial(self):
        chunks = ["the quick brown fox"]
        gold = ["quick brown fox", "lazy dog"]
        assert recall_at_k(chunks, gold, strict=True) == 0.5

    def test_no_match(self):
        chunks = ["nothing relevant here"]
        gold = ["quick brown fox", "lazy dog"]
        assert recall_at_k(chunks, gold, strict=True) == 0.0

    def test_empty_evidence_returns_none(self):
        assert recall_at_k(["whatever"], [], strict=True) is None

    def test_whitespace_normalized(self):
        chunks = ["we    propose\na novel\tmethod for retrieval"]
        gold = ["we propose a novel method"]
        assert recall_at_k(chunks, gold, strict=True) == 1.0

    def test_multi_chunk_any_covers(self):
        chunks = ["first chunk about dogs", "second chunk about cats"]
        gold = ["about cats"]
        assert recall_at_k(chunks, gold, strict=True) == 1.0

    def test_blank_gold_sentence_ignored(self):
        chunks = ["hello world"]
        gold = ["   ", "hello"]
        assert recall_at_k(chunks, gold, strict=True) == 1.0


class TestRecallAtKFuzzy:
    """Default token-coverage behavior with citation stripping."""

    def test_bibref_vs_author_year_matches(self):
        gold = [
            "We compare against multilingual NMT BIBREF19 and cross-lingual "
            "transfer BIBREF16."
        ]
        chunks = [
            "we compare against multilingual nmt (johnson et al. 2016) and "
            "cross-lingual transfer (kim et al. 2017)."
        ]
        # Citations strip from both sides; remaining tokens overlap fully.
        assert recall_at_k(chunks, gold) == 1.0

    def test_paren_year_only_citation_stripped(self):
        chunks = ["foo bar baz"]
        gold = ["foo bar baz (2020)."]
        assert recall_at_k(chunks, gold) == 1.0

    def test_partial_overlap_below_threshold_zero(self):
        # gold has 10 distinct tokens; chunk shares 3 of them. 3/10 < 0.7.
        gold = ["alpha beta gamma delta epsilon zeta eta theta iota kappa"]
        chunks = ["alpha beta gamma plus a bunch of unrelated filler words"]
        assert recall_at_k(chunks, gold) == 0.0

    def test_partial_overlap_above_threshold_one(self):
        # 8 of 10 gold tokens present in the chunk -> 0.8 >= 0.7.
        gold = ["alpha beta gamma delta epsilon zeta eta theta iota kappa"]
        chunks = ["alpha beta gamma delta epsilon zeta eta theta plus filler"]
        assert recall_at_k(chunks, gold) == 1.0

    def test_whitespace_and_case_variation(self):
        chunks = ["WE   propose\nA novel\tMETHOD for retrieval"]
        gold = ["we propose a novel method for retrieval"]
        assert recall_at_k(chunks, gold) == 1.0

    def test_empty_gold_returns_none(self):
        assert recall_at_k(["anything"], []) is None

    def test_blank_gold_sentence_ignored_in_denom(self):
        chunks = ["hello world foo bar"]
        gold = ["   ", "hello world foo bar"]
        assert recall_at_k(chunks, gold) == 1.0

    def test_punctuation_stripped_both_sides(self):
        chunks = ["hello world"]
        gold = ["hello, world!"]
        assert recall_at_k(chunks, gold) == 1.0

    def test_threshold_param_respected(self):
        # 7 of 10 gold tokens in chunk -> 0.7 exactly.
        gold = ["alpha beta gamma delta epsilon zeta eta theta iota kappa"]
        chunks = ["alpha beta gamma delta epsilon zeta eta plus some filler"]
        assert recall_at_k(chunks, gold, threshold=0.5) == 1.0
        assert recall_at_k(chunks, gold, threshold=0.95) == 0.0

    def test_token_set_helper(self):
        from src.evaluation.qasper_eval import _token_set

        # BIBREF and parenthetical citations both stripped; punctuation gone.
        assert _token_set("Foo bar BIBREF12 (Smith 2020), baz!") == {"foo", "bar", "baz"}


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

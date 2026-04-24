"""QASPER eval: retrieval recall@k + generative answer F1.

Metric definitions follow the official QASPER evaluation
(https://github.com/allenai/qasper-led-baseline). The normalize_answer
and token_f1 helpers are ported from that repo's scoring code to stay
comparable with published QASPER numbers.

Aggregation notes:
  - Questions whose paper is not in our processed corpus are skipped
    (reported separately as n_skipped_no_corpus).
  - Questions with empty gold evidence are skipped from the recall
    aggregate but still scored for answer F1.
  - Each question may have multiple gold answers (one per annotator);
    we take the max F1 across them.
"""

from __future__ import annotations

import re
import string
import time
from collections import Counter
from typing import Any

ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
WHITESPACE = re.compile(r"\s+")


def normalize_answer(s: str) -> str:
    """SQuAD/QASPER-style normalization: lowercase, strip punctuation,
    drop articles, collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = ARTICLES.sub(" ", s)
    s = WHITESPACE.sub(" ", s).strip()
    return s


def _tokenize(s: str) -> list[str]:
    return normalize_answer(s).split()


def token_f1(pred: str, gold: str) -> float:
    """Token-level F1 between two strings."""
    pred_tokens = _tokenize(pred)
    gold_tokens = _tokenize(gold)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def max_token_f1(pred: str, golds: list[str]) -> float:
    """Max token-F1 across a list of gold answers."""
    if not golds:
        return 0.0
    return max(token_f1(pred, g) for g in golds)


def _normalize_for_match(s: str) -> str:
    return WHITESPACE.sub(" ", s.lower()).strip()


def recall_at_k(
    retrieved_texts: list[str], gold_evidence: list[str]
) -> float | None:
    """Sentence-coverage recall@k.

    A gold evidence sentence is "covered" if its normalized form appears
    as a substring of any retrieved chunk's normalized text. Returns
    None if gold_evidence is empty (signals "skip in aggregate").
    """
    if not gold_evidence:
        return None
    norm_chunks = [_normalize_for_match(t) for t in retrieved_texts]
    covered = 0
    for sent in gold_evidence:
        n = _normalize_for_match(sent)
        if not n:
            continue
        if any(n in c for c in norm_chunks):
            covered += 1
    denom = sum(1 for s in gold_evidence if _normalize_for_match(s))
    if denom == 0:
        return None
    return covered / denom


def extract_gold_answers(qas_answers: list[dict]) -> list[str]:
    """Collapse QASPER's multi-annotator answer structure into a list of
    gold answer strings (one per annotation).

    Priority within an annotation:
      unanswerable -> "Unanswerable"
      yes_no       -> "Yes" / "No"
      extractive_spans -> space-joined spans
      free_form_answer -> that string
    """
    out: list[str] = []
    for ann in qas_answers:
        a = ann.get("answer", ann)
        if a.get("unanswerable"):
            out.append("Unanswerable")
            continue
        yn = a.get("yes_no")
        if yn is True:
            out.append("Yes")
            continue
        if yn is False:
            out.append("No")
            continue
        spans = a.get("extractive_spans") or []
        if spans:
            out.append(" ".join(spans))
            continue
        free = a.get("free_form_answer") or ""
        if free.strip():
            out.append(free)
    return out


def extract_gold_evidence(qas_answers: list[dict]) -> list[str]:
    """Collect all highlighted_evidence sentences across annotations.

    Falls back to `evidence` (paragraph-level) if highlighted is empty.
    Duplicates removed, order preserved.
    """
    seen: set[str] = set()
    out: list[str] = []
    for ann in qas_answers:
        a = ann.get("answer", ann)
        sents = a.get("highlighted_evidence") or a.get("evidence") or []
        for s in sents:
            key = _normalize_for_match(s)
            if key and key not in seen:
                seen.add(key)
                out.append(s)
    return out


PROMPT_SYSTEM = (
    "You are a scientific paper QA assistant. Answer the question using ONLY "
    "the provided context passages. If the context does not contain enough "
    "information, reply with exactly: Unanswerable. Keep answers short — a "
    "phrase or one sentence."
)

PROMPT_USER_TEMPLATE = """Question: {question}

Context:
{context}

Answer:"""


def _build_context(retrieved: list[dict]) -> str:
    parts = []
    for i, r in enumerate(retrieved, 1):
        parts.append(f"[{i}] {r['text']}")
    return "\n\n".join(parts)


EVAL_NUM_CTX = 4096


def evaluate_question(
    question: str,
    question_id: str,
    paper_id: str,
    gold_answers: list[str],
    gold_evidence: list[str],
    flat_index,
    llm_client,
    *,
    k: int = 5,
    max_answer_tokens: int = 128,
    num_ctx: int = EVAL_NUM_CTX,
) -> dict[str, Any]:
    """Run retrieval + LLM on a single QASPER question, return scored dict.

    num_ctx: Ollama context window override. Default 4096 covers
    question + 5 chunks (~512 tokens each) + system prompt + answer
    headroom. Ollama's default of 2048 silently truncates this.
    """
    t0 = time.time()
    retrieved = flat_index.search(question, k=k, paper_ids={paper_id})
    t1 = time.time()
    context = _build_context(retrieved)
    user_prompt = PROMPT_USER_TEMPLATE.format(question=question, context=context)
    predicted = llm_client.generate(
        system=PROMPT_SYSTEM,
        user=user_prompt,
        max_tokens=max_answer_tokens,
        temperature=0.0,
        num_ctx=num_ctx,
    ).strip()
    t2 = time.time()

    retrieved_texts = [r["text"] for r in retrieved]
    r_at_k = recall_at_k(retrieved_texts, gold_evidence)
    f1 = max_token_f1(predicted, gold_answers) if gold_answers else 0.0

    return {
        "question_id": question_id,
        "paper_id": paper_id,
        "question": question,
        "gold_answers": gold_answers,
        "gold_evidence": gold_evidence,
        "retrieved_chunk_ids": [r["chunk_id"] for r in retrieved],
        "retrieved_arxiv_ids": [r["arxiv_id"] for r in retrieved],
        "recall_at_k": r_at_k,
        "predicted_answer": predicted,
        "answer_f1": f1,
        "latency_ms": {
            "retrieval": int((t1 - t0) * 1000),
            "llm": int((t2 - t1) * 1000),
        },
    }


def aggregate_results(results: list[dict]) -> dict[str, Any]:
    """Compute mean recall@k and mean answer F1 across per-question results.
    Questions with recall_at_k=None are excluded from the recall mean."""
    recall_vals = [r["recall_at_k"] for r in results if r["recall_at_k"] is not None]
    f1_vals = [r["answer_f1"] for r in results]
    return {
        "n_evaluated": len(results),
        "n_with_evidence": len(recall_vals),
        "mean_recall_at_k": (sum(recall_vals) / len(recall_vals)) if recall_vals else None,
        "mean_answer_f1": (sum(f1_vals) / len(f1_vals)) if f1_vals else None,
    }

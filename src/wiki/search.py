"""Wiki search — BM25 over the compiled wiki (v2 W11).

Makes the LLM-compiled wiki queryable: paper summaries and concept articles are
flattened into searchable documents and indexed with BM25 (adequate and fast for
the ~60-doc wiki; no model needed). Reuses the wiki loaders and the evidence
retriever's tokenizer. Consumed by the CLI and the Streamlit search page.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.verification.evidence_retriever import tokenize
from src.wiki.concept_extractor import Summary, load_summaries_dir
from src.wiki.indices import Concept, load_concepts_dir

DEFAULT_WIKI_ROOT = Path("wiki")


@dataclass(frozen=True)
class WikiDoc:
    """A searchable wiki entry (a paper summary or a concept article)."""

    kind: str  # "paper" | "concept"
    ident: str  # arxiv_id | concept slug
    title: str
    text: str  # flattened searchable body


@dataclass(frozen=True)
class WikiHit:
    kind: str
    ident: str
    title: str
    score: float


def _summary_doc(s: Summary) -> WikiDoc:
    body = " ".join([s.title, *s.sections.values()])
    return WikiDoc("paper", s.arxiv_id, s.title, body)


def _concept_doc(c: Concept) -> WikiDoc:
    body = " ".join(x for x in (c.name, c.definition, c.open_questions) if x)
    return WikiDoc("concept", c.slug, c.name, body)


class WikiSearchIndex:
    def __init__(self, docs: list[WikiDoc]) -> None:
        from rank_bm25 import BM25Okapi

        self._docs = docs
        self._bm25 = BM25Okapi([tokenize(d.text) for d in docs]) if docs else None

    @classmethod
    def from_wiki(cls, wiki_root: str | Path = DEFAULT_WIKI_ROOT) -> "WikiSearchIndex":
        root = Path(wiki_root)
        docs = [_summary_doc(s) for s in load_summaries_dir(root / "papers")]
        docs += [_concept_doc(c) for c in load_concepts_dir(root / "concepts")]
        return cls(docs)

    def __len__(self) -> int:
        return len(self._docs)

    def search(self, query: str, k: int = 5) -> list[WikiHit]:
        if not self._docs or k <= 0:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            WikiHit(self._docs[i].kind, self._docs[i].ident, self._docs[i].title, float(scores[i]))
            for i in order
        ]

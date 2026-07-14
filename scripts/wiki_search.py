"""Wiki search CLI (v2 W11). Query the compiled wiki from the terminal.

    python scripts/wiki_search.py "bert language model"
    python scripts/wiki_search.py "machine translation" -k 10
"""

from __future__ import annotations

import argparse

from src.wiki.search import WikiSearchIndex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="search terms")
    ap.add_argument("-k", type=int, default=5, help="number of results")
    ap.add_argument("--wiki-root", default="wiki")
    args = ap.parse_args()

    index = WikiSearchIndex.from_wiki(args.wiki_root)
    hits = index.search(args.query, k=args.k)
    if not hits:
        print(f"no results in {len(index)} wiki entries")
        return
    print(f"top {len(hits)} of {len(index)} wiki entries for {args.query!r}:\n")
    for h in hits:
        print(f"  [{h.kind:7}] {h.ident:30} {h.title[:52]}  ({h.score:.1f})")


if __name__ == "__main__":
    main()

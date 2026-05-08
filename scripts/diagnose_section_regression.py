"""Week 5 Sub-task F: diagnose the 8pt regression of the section-aware
index vs the flat index on QASPER recall@5.

Joins three locked per-question runs on ``question_id``:
  - flat_bge (no sections, anchor)        : eval/results/week4_bge_full_retrieval.jsonl
  - flat_bge_sectioned, no routing        : eval/results/week4_bge_section_noroute.jsonl
  - flat_bge_sectioned, oracle routing    : eval/results/week4_bge_section_oracle.jsonl

Slices the per-question deltas to localize where the sectioned index
loses retrieval power, and where oracle routing recovers it. Also dumps
chunk token-count stats per index so we can see if sectioning is
fragmenting context.

No FAISS, no embedder. Reads JSONL only. ~5s.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def _by_qid(rows: list[dict]) -> dict[str, dict]:
    return {r["question_id"]: r for r in rows}


def _quantiles(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {}
    xs_sorted = sorted(xs)
    n = len(xs_sorted)

    def q(p: float) -> float:
        i = max(0, min(n - 1, int(round(p * (n - 1)))))
        return xs_sorted[i]

    return {
        "n": n,
        "mean": round(sum(xs) / n, 4),
        "median": round(q(0.5), 4),
        "p10": round(q(0.10), 4),
        "p90": round(q(0.90), 4),
        "min": round(xs_sorted[0], 4),
        "max": round(xs_sorted[-1], 4),
    }


def _chunk_stats(path: Path) -> dict:
    by_section: dict[str, list[int]] = defaultdict(list)
    all_tokens: list[int] = []
    by_paper: dict[str, list[int]] = defaultdict(list)
    n_chunks = 0
    with path.open() as f:
        for line in f:
            c = json.loads(line)
            tc = int(c["token_count"])
            all_tokens.append(tc)
            by_paper[c["arxiv_id"]].append(tc)
            st = c.get("section_type", "<unsectioned>")
            by_section[st].append(tc)
            n_chunks += 1
    overall = _quantiles([float(x) for x in all_tokens])
    per_section = {
        st: _quantiles([float(x) for x in vs])
        for st, vs in sorted(by_section.items(), key=lambda kv: -len(kv[1]))
    }
    chunks_per_paper = [len(v) for v in by_paper.values()]
    return {
        "n_chunks": n_chunks,
        "n_papers": len(by_paper),
        "tokens_overall": overall,
        "chunks_per_paper": _quantiles([float(x) for x in chunks_per_paper]),
        "tokens_by_section": per_section,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--flat", type=Path,
        default=Path("eval/results/week4_bge_full_retrieval.jsonl"),
    )
    p.add_argument(
        "--sect-noroute", type=Path,
        default=Path("eval/results/week4_bge_section_noroute.jsonl"),
    )
    p.add_argument(
        "--sect-oracle", type=Path,
        default=Path("eval/results/week4_bge_section_oracle.jsonl"),
    )
    p.add_argument(
        "--flat-chunks", type=Path,
        default=Path("data/index/flat_bge/chunks.jsonl"),
    )
    p.add_argument(
        "--sect-chunks", type=Path,
        default=Path("data/index/flat_bge_sectioned/chunks.jsonl"),
    )
    p.add_argument(
        "--out", type=Path,
        default=Path("eval/results/week5_section_diagnosis.json"),
    )
    p.add_argument("--regression-eps", type=float, default=0.05)
    p.add_argument("--top-failures", type=int, default=20)
    args = p.parse_args()

    flat = _by_qid(_load_jsonl(args.flat))
    sect_n = _by_qid(_load_jsonl(args.sect_noroute))
    sect_o = _by_qid(_load_jsonl(args.sect_oracle))

    common = sorted(set(flat) & set(sect_n) & set(sect_o))
    print(f"[diagnose] joined questions: {len(common)} "
          f"(flat={len(flat)} sect_noroute={len(sect_n)} sect_oracle={len(sect_o)})")

    # Per-question deltas, restricted to questions with non-null recalls
    # everywhere (i.e. with-evidence subset).
    rows: list[dict] = []
    for qid in common:
        rf, rn, ro = flat[qid], sect_n[qid], sect_o[qid]
        if any(r.get("recall_at_k") is None for r in (rf, rn, ro)):
            continue
        rows.append({
            "qid": qid,
            "paper_id": rf["paper_id"],
            "question": rf["question"],
            "r_flat": rf["recall_at_k"],
            "r_sect_noroute": rn["recall_at_k"],
            "r_sect_oracle": ro["recall_at_k"],
            "oracle_section_types": ro.get("oracle_section_types") or [],
            "n_oracle_types": len(ro.get("oracle_section_types") or []),
        })
    n = len(rows)
    print(f"[diagnose] with-evidence joined: {n}")

    means = {
        "flat": round(sum(r["r_flat"] for r in rows) / n, 4),
        "sect_noroute": round(sum(r["r_sect_noroute"] for r in rows) / n, 4),
        "sect_oracle": round(sum(r["r_sect_oracle"] for r in rows) / n, 4),
    }
    print(f"\n[diagnose] mean recall@5 fuzzy")
    for k, v in means.items():
        print(f"  {k}: {v}")
    print(f"  delta sect_noroute - flat   : {round(means['sect_noroute']-means['flat'], 4)}")
    print(f"  delta sect_oracle - flat    : {round(means['sect_oracle']-means['flat'], 4)}")
    print(f"  delta sect_oracle - sect_nr : {round(means['sect_oracle']-means['sect_noroute'], 4)}")

    # 1. delta distribution: regressed / stable / improved
    eps = args.regression_eps
    n_regress = sum(1 for r in rows if r["r_sect_noroute"] - r["r_flat"] < -eps)
    n_stable = sum(1 for r in rows if abs(r["r_sect_noroute"] - r["r_flat"]) <= eps)
    n_improve = sum(1 for r in rows if r["r_sect_noroute"] - r["r_flat"] > eps)
    print(f"\n[diagnose] delta distribution (sect_noroute - flat, eps={eps}): "
          f"regress={n_regress} stable={n_stable} improve={n_improve}")
    deltas_nr = [r["r_sect_noroute"] - r["r_flat"] for r in rows]
    print(f"  delta quantiles: {_quantiles(deltas_nr)}")

    # 2. slice by oracle section_type
    by_type: dict[str, list[float]] = defaultdict(list)
    by_type_oracle: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for st in r["oracle_section_types"]:
            by_type[st].append(r["r_sect_noroute"] - r["r_flat"])
            by_type_oracle[st].append(r["r_sect_oracle"] - r["r_sect_noroute"])
    print(f"\n[diagnose] mean delta by oracle section_type")
    print(f"  {'section':<14} {'n':>5} {'sect_nr-flat':>14} "
          f"{'sect_oracle-sect_nr':>22}")
    for st in sorted(by_type, key=lambda s: -len(by_type[s])):
        nr = by_type[st]
        oc = by_type_oracle[st]
        print(f"  {st:<14} {len(nr):>5} {sum(nr)/len(nr):>14.4f} "
              f"{sum(oc)/len(oc):>22.4f}")

    # 3. slice by # oracle types
    by_count: dict[int, list[float]] = defaultdict(list)
    by_count_oracle: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        c = r["n_oracle_types"]
        by_count[c].append(r["r_sect_noroute"] - r["r_flat"])
        by_count_oracle[c].append(r["r_sect_oracle"] - r["r_sect_noroute"])
    print(f"\n[diagnose] mean delta by # oracle types (multi-section evidence?)")
    print(f"  {'n_types':>8} {'n':>5} {'sect_nr-flat':>14} "
          f"{'sect_oracle-sect_nr':>22}")
    for c in sorted(by_count):
        nr = by_count[c]
        oc = by_count_oracle[c]
        print(f"  {c:>8} {len(nr):>5} {sum(nr)/len(nr):>14.4f} "
              f"{sum(oc)/len(oc):>22.4f}")

    # 4. chunk token stats
    print(f"\n[diagnose] chunk token-count stats: flat_bge")
    flat_stats = _chunk_stats(args.flat_chunks)
    print(f"  n_chunks={flat_stats['n_chunks']} n_papers={flat_stats['n_papers']}")
    print(f"  tokens_overall={flat_stats['tokens_overall']}")
    print(f"  chunks_per_paper={flat_stats['chunks_per_paper']}")

    print(f"\n[diagnose] chunk token-count stats: flat_bge_sectioned")
    sect_stats = _chunk_stats(args.sect_chunks)
    print(f"  n_chunks={sect_stats['n_chunks']} n_papers={sect_stats['n_papers']}")
    print(f"  tokens_overall={sect_stats['tokens_overall']}")
    print(f"  chunks_per_paper={sect_stats['chunks_per_paper']}")
    print(f"  by section_type:")
    for st, q in sect_stats["tokens_by_section"].items():
        print(f"    {st:<14} n={q['n']:>5} mean={q['mean']:>6.1f} "
              f"median={q['median']:>5} p10={q['p10']:>4} p90={q['p90']:>5}")

    # 5. top-N worst regressions for human eyeballing
    rows_sorted = sorted(rows, key=lambda r: r["r_sect_noroute"] - r["r_flat"])
    worst = rows_sorted[: args.top_failures]
    print(f"\n[diagnose] top {len(worst)} regressions (lowest sect_noroute - flat)")
    for r in worst:
        d = r["r_sect_noroute"] - r["r_flat"]
        do = r["r_sect_oracle"] - r["r_flat"]
        print(f"  Δ_nr={d:+.2f}  Δ_or={do:+.2f}  "
              f"types={r['oracle_section_types']}  q={r['question'][:80]!r}")

    summary = {
        "n_joined": len(common),
        "n_with_evidence": n,
        "means": means,
        "deltas": {
            "sect_noroute_minus_flat": round(means["sect_noroute"] - means["flat"], 4),
            "sect_oracle_minus_flat": round(means["sect_oracle"] - means["flat"], 4),
            "sect_oracle_minus_sect_noroute": round(
                means["sect_oracle"] - means["sect_noroute"], 4
            ),
        },
        "delta_distribution_sect_noroute_minus_flat": {
            "regress": n_regress, "stable": n_stable, "improve": n_improve,
            "eps": eps,
            "quantiles": _quantiles(deltas_nr),
        },
        "by_section_type": {
            st: {
                "n": len(by_type[st]),
                "mean_delta_sect_noroute_minus_flat": round(
                    sum(by_type[st]) / len(by_type[st]), 4
                ),
                "mean_delta_sect_oracle_minus_sect_noroute": round(
                    sum(by_type_oracle[st]) / len(by_type_oracle[st]), 4
                ),
            }
            for st in by_type
        },
        "by_n_oracle_types": {
            str(c): {
                "n": len(by_count[c]),
                "mean_delta_sect_noroute_minus_flat": round(
                    sum(by_count[c]) / len(by_count[c]), 4
                ),
                "mean_delta_sect_oracle_minus_sect_noroute": round(
                    sum(by_count_oracle[c]) / len(by_count_oracle[c]), 4
                ),
            }
            for c in by_count
        },
        "flat_chunk_stats": flat_stats,
        "sect_chunk_stats": sect_stats,
        "top_regressions": [
            {
                "qid": r["qid"],
                "paper_id": r["paper_id"],
                "question": r["question"],
                "r_flat": r["r_flat"],
                "r_sect_noroute": r["r_sect_noroute"],
                "r_sect_oracle": r["r_sect_oracle"],
                "oracle_section_types": r["oracle_section_types"],
            }
            for r in worst
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\n[diagnose] summary -> {args.out}")


if __name__ == "__main__":
    main()

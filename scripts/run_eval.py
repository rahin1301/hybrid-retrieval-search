"""Evaluate every retrieval mode against the graded relevance judgements.

Reports nDCG@10 and Recall@10 per system, overall and broken down by query
family, plus median and p95 latency. Caching is disabled so latency reflects
real retrieval work rather than Redis hits.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --k 10 --out data/eval/results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.dependencies import build_search_service  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.domain.metrics import mean, ndcg_at_k, recall_at_k  # noqa: E402
from app.domain.models import SearchMode  # noqa: E402

log = logging.getLogger("run_eval")

MODE_LABELS = {
    SearchMode.BM25: "BM25 (Elasticsearch)",
    SearchMode.DENSE: "Dense (bge-base + FAISS)",
    SearchMode.HYBRID: "Hybrid (RRF)",
    SearchMode.HYBRID_RERANK: "Hybrid + cross-encoder rerank",
}


@dataclass
class ModeResult:
    mode: SearchMode
    ndcg: float
    recall: float
    latency_p50_ms: float
    latency_p95_ms: float
    by_family: dict[str, dict[str, float]] = field(default_factory=dict)
    per_query: dict[str, dict[str, float]] = field(default_factory=dict)


def evaluate_mode(service, mode, queries, qrels, k) -> ModeResult:
    ndcg_scores: list[float] = []
    recall_scores: list[float] = []
    latencies: list[float] = []
    family_scores: dict[str, list[tuple[float, float]]] = defaultdict(list)
    per_query: dict[str, dict[str, float]] = {}

    for query in queries:
        judgements = qrels.get(query["qid"], {})
        if not judgements:
            log.warning("No judgements for %s; skipping", query["qid"])
            continue

        started = time.perf_counter()
        ranked = service.rank(query["text"], mode, k)
        latencies.append((time.perf_counter() - started) * 1000)

        ranked_ids = [doc.doc_id for doc in ranked]
        ndcg = ndcg_at_k(ranked_ids, judgements, k)
        recall = recall_at_k(ranked_ids, judgements, k)

        ndcg_scores.append(ndcg)
        recall_scores.append(recall)
        family_scores[query.get("family", "unknown")].append((ndcg, recall))
        per_query[query["qid"]] = {"ndcg": round(ndcg, 4), "recall": round(recall, 4)}

    return ModeResult(
        mode=mode,
        ndcg=mean(ndcg_scores),
        recall=mean(recall_scores),
        latency_p50_ms=statistics.median(latencies) if latencies else 0.0,
        latency_p95_ms=(
            statistics.quantiles(latencies, n=20)[-1]
            if len(latencies) >= 20
            else max(latencies, default=0.0)
        ),
        by_family={
            family: {
                "ndcg": round(mean([s[0] for s in scores]), 4),
                "recall": round(mean([s[1] for s in scores]), 4),
                "queries": len(scores),
            }
            for family, scores in sorted(family_scores.items())
        },
        per_query=per_query,
    )


def render_markdown(results: list[ModeResult], k: int, n_queries: int) -> str:
    lines = [
        f"### Retrieval quality ({n_queries} queries, k={k})",
        "",
        f"| System | nDCG@{k} | Recall@{k} | p50 latency | p95 latency |",
        "| --- | --- | --- | --- | --- |",
    ]
    baseline = results[0]
    for result in results:
        delta = ""
        if result is not baseline and baseline.ndcg > 0:
            change = (result.ndcg - baseline.ndcg) / baseline.ndcg * 100
            delta = f" ({change:+.1f}%)"
        lines.append(
            f"| {MODE_LABELS[result.mode]} | {result.ndcg:.4f}{delta} | "
            f"{result.recall:.4f} | {result.latency_p50_ms:.0f} ms | "
            f"{result.latency_p95_ms:.0f} ms |"
        )

    families = sorted({family for r in results for family in r.by_family})
    lines += [
        "",
        f"### nDCG@{k} by query family",
        "",
        "| System | " + " | ".join(families) + " |",
        "| --- " * (len(families) + 1) + "|",
    ]
    for result in results:
        cells = [
            f"{result.by_family.get(family, {}).get('ndcg', 0.0):.4f}" for family in families
        ]
        lines.append(f"| {MODE_LABELS[result.mode]} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    settings = get_settings()
    # Caching would make repeat runs report Redis latency instead of retrieval latency.
    settings = settings.model_copy(update={"cache_enabled": False})

    queries = json.loads((settings.eval_dir / "queries.json").read_text())
    qrels_path = settings.eval_dir / "qrels.json"
    if not qrels_path.exists():
        raise SystemExit(
            f"No judgements at {qrels_path}. Build a pool with scripts/make_pool.py "
            "and label it first."
        )
    qrels = json.loads(qrels_path.read_text())

    judged = [q for q in queries if qrels.get(q["qid"])]
    log.info("Evaluating %d judged queries at k=%d", len(judged), args.k)

    service = build_search_service(settings, with_reranker=True)

    results = [
        evaluate_mode(service, mode, judged, qrels, args.k)
        for mode in (
            SearchMode.BM25,
            SearchMode.DENSE,
            SearchMode.HYBRID,
            SearchMode.HYBRID_RERANK,
        )
    ]

    markdown = render_markdown(results, args.k, len(judged))
    print("\n" + markdown + "\n")

    out_path = args.out or settings.eval_dir / "results.json"
    out_path.write_text(
        json.dumps(
            {
                "k": args.k,
                "queries": len(judged),
                "embed_model": settings.embed_model,
                "rerank_model": settings.rerank_model,
                "candidate_k": settings.candidate_k,
                "rerank_depth": settings.rerank_depth,
                "rrf_k": settings.rrf_k,
                "systems": [
                    {
                        "mode": r.mode.value,
                        "label": MODE_LABELS[r.mode],
                        "ndcg_at_k": round(r.ndcg, 4),
                        "recall_at_k": round(r.recall, 4),
                        "latency_p50_ms": round(r.latency_p50_ms, 1),
                        "latency_p95_ms": round(r.latency_p95_ms, 1),
                        "by_family": r.by_family,
                        "per_query": r.per_query,
                    }
                    for r in results
                ],
            },
            indent=2,
        )
    )
    (settings.eval_dir / "results.md").write_text(markdown + "\n")
    log.info("Wrote %s and %s", out_path, settings.eval_dir / "results.md")


if __name__ == "__main__":
    main()

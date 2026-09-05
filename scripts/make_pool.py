"""Build a judgement pool for the evaluation queries.

Pooling follows standard TREC practice: **every system under test** contributes
its top-``depth`` results, the union is de-duplicated, and only that pool is
judged. Documents outside the pool are assumed non-relevant.

Pooling from the first-stage retrievers alone is not sufficient and silently
corrupts the comparison. Fusion and reranking reorder a deeper candidate list,
so they surface documents that neither retriever returned in its own top-k. If
those are never judged, they score as non-relevant by default and the deeper
system is penalised for reaching further — an artefact of the pool, not a real
quality difference. Measured here before the fix: 26% of the reranker's top-10
was unjudged, versus 0% for BM25 and dense.

Re-running is incremental: existing judgements in qrels.json are preserved, and
``scripts/show_pool.py --unjudged`` lists only what still needs a grade.

Usage:
    python scripts/make_pool.py --depth 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.dependencies import build_search_service  # noqa: E402
from app.config import get_settings  # noqa: E402

log = logging.getLogger("make_pool")

ABSTRACT_PREVIEW_CHARS = 400


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--depth", type=int, default=20, help="per-system pool depth (must be >= eval k)"
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    settings = get_settings().model_copy(update={"cache_enabled": False})
    queries = json.loads((settings.eval_dir / "queries.json").read_text())
    out_path = args.out or settings.eval_dir / "pool.jsonl"

    service = build_search_service(settings, with_reranker=True)
    modes = service.supported_modes
    log.info(
        "Pooling %d systems at depth %d: %s",
        len(modes),
        args.depth,
        [m.value for m in modes],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_candidates = 0

    with out_path.open("w") as handle:
        for query in queries:
            found_by: dict[str, list[str]] = {}
            for mode in modes:
                for hit in service.rank(query["text"], mode, args.depth):
                    found_by.setdefault(hit.doc_id, []).append(mode.value)

            candidates = []
            for doc_id, systems in found_by.items():
                document = service.store.get(doc_id)
                if document is None:
                    continue
                candidates.append(
                    {
                        "doc_id": doc_id,
                        "title": document.title,
                        "abstract": document.abstract[:ABSTRACT_PREVIEW_CHARS],
                        "found_by": systems,
                    }
                )

            total_candidates += len(candidates)
            handle.write(
                json.dumps(
                    {"qid": query["qid"], "query": query["text"], "candidates": candidates}
                )
                + "\n"
            )
            log.info("%s: pooled %d candidates", query["qid"], len(candidates))

    log.info(
        "Wrote %d queries / %d candidates to %s", len(queries), total_candidates, out_path
    )


if __name__ == "__main__":
    main()

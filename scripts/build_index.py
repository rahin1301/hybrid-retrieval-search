"""Build both retrieval indexes from data/corpus.jsonl.

Writes the Elasticsearch BM25 index and the FAISS dense index. Both are
rebuilt from scratch, so the script is idempotent and safe to re-run.

Usage:
    python scripts/build_index.py             # build both
    python scripts/build_index.py --only bm25
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.adapters.corpus_store import load_documents  # noqa: E402
from app.adapters.elastic_bm25 import ElasticIndexer, create_client  # noqa: E402
from app.adapters.encoder import SentenceTransformerEncoder  # noqa: E402
from app.adapters.faiss_dense import build_index, save_index  # noqa: E402
from app.config import get_settings  # noqa: E402

log = logging.getLogger("build_index")


def build_bm25(documents, settings) -> None:
    client = create_client(settings.es_url)
    if not client.ping():
        raise SystemExit(
            f"Elasticsearch is not reachable at {settings.es_url}. "
            "Start it with: docker compose up -d elasticsearch"
        )

    indexer = ElasticIndexer(client, settings.index_name)
    indexer.recreate_index()

    started = time.perf_counter()
    count = indexer.index_documents(documents)
    log.info(
        "Indexed %d documents into '%s' in %.1fs",
        count,
        settings.index_name,
        time.perf_counter() - started,
    )


def build_dense(documents, settings) -> None:
    encoder = SentenceTransformerEncoder(
        settings.embed_model,
        query_instruction=settings.query_instruction,
        batch_size=settings.embed_batch_size,
    )

    started = time.perf_counter()
    # Documents are embedded without the query instruction prefix: bge is
    # trained asymmetrically and prefixing documents degrades retrieval.
    embeddings = encoder.encode_documents(
        [doc.to_text() for doc in documents], show_progress=True
    )
    log.info(
        "Embedded %d documents (dim=%d) in %.1fs",
        len(documents),
        embeddings.shape[1],
        time.perf_counter() - started,
    )

    index = build_index(embeddings)
    save_index(
        index,
        [doc.id for doc in documents],
        settings.faiss_index_path,
        settings.faiss_ids_path,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["bm25", "dense"], help="build a single index")
    args = parser.parse_args()

    settings = get_settings()
    documents = load_documents(settings.corpus_path)
    if not documents:
        raise SystemExit("Corpus is empty; run scripts/fetch_corpus.py first")
    log.info("Loaded %d documents", len(documents))

    if args.only in (None, "bm25"):
        build_bm25(documents, settings)
    if args.only in (None, "dense"):
        build_dense(documents, settings)

    log.info("Done.")


if __name__ == "__main__":
    main()

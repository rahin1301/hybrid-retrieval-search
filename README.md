# Hybrid Retrieval Search

A search API over ~5.6k arXiv CS abstracts that compares four retrieval setups:

1. BM25 (Elasticsearch)
2. Dense retrieval (`bge-base-en-v1.5` + FAISS)
3. Hybrid, using Reciprocal Rank Fusion (RRF)
4. Hybrid + cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`)

The main goal was to measure how much hybrid search actually helps on a small corpus, so
most of the work went into the evaluation harness rather than the API.

## Results

32 hand-written queries, 1,316 graded relevance judgements, k = 10. Caching was off for
the latency numbers.

| System | nDCG@10 | Recall@10 | p50 latency |
| --- | --- | --- | --- |
| BM25 | 0.521 | 0.236 | 12 ms |
| Dense | 0.709 | 0.289 | 13 ms |
| Hybrid (RRF) | 0.727 | 0.294 | 35 ms |
| Hybrid + rerank | 0.732 | 0.293 | 698 ms |

The API defaults to hybrid RRF. Reranking is about 0.8% better on nDCG@10 but roughly 20x
slower, and with only 32 queries that difference is within noise. You can still turn it
on per request with `mode=hybrid_rerank`.

A couple of notes on reading this:

- Each query has ~32 relevant documents but only 10 slots, so the best possible
  Recall@10 is about 0.33. Hybrid gets ~90% of that.
- Only p50 is reported. With 32 queries, p95 is basically one data point and it changed by
  more than 2x between identical runs. Numbers are from an M-series MacBook with ES and
  Redis running in Docker.

### By query type

The queries are split into three groups: `lexical` (wording likely appears in the paper),
`semantic` (paraphrased, little word overlap), and `mixed`.

| System | lexical (13) | mixed (13) | semantic (6) |
| --- | --- | --- | --- |
| BM25 | 0.703 | 0.503 | 0.164 |
| Dense | 0.778 | 0.722 | 0.529 |
| Hybrid (RRF) | 0.809 | 0.753 | 0.493 |
| Hybrid + rerank | 0.793 | 0.804 | 0.446 |

This breakdown says more than the overall table:

- BM25 does very badly on paraphrased queries (0.164), which is expected.
- On semantic queries, dense alone beats hybrid. Since RRF weights both retrievers
  equally, the weak BM25 ranking drags the dense one down. This is only 6 queries though,
  so it's more of a hint than a solid result.
- Reranking mostly helps on mixed queries and slightly hurts on lexical ones.

So no single config wins everywhere. A simple query router (e.g. send paraphrase-like
queries to dense only) would probably do better than any of these. That's the next thing
to try.

## A bug in the first evaluation

The first run showed reranking *hurting* nDCG by 14% (0.709 → 0.607), and a bigger reranker
made it even worse. That seemed wrong, so the eval was checked before blaming the models.

The issue was the judgement pool. It only had the top 15 from BM25 and dense. Hybrid
and reranking pull up documents from deeper in the candidate list, and those documents
were never judged, so they counted as non-relevant. About 26% of the reranker's top 10 was
unjudged.

The fix was to pool the top 20 from all four systems (the usual TREC approach) and judge
the 489 extra documents. After that, reranking went from 0.607 to 0.732.
`scripts/make_pool.py` now does this by default.

Two rerankers compared on the fixed pool:

| Reranker | Params | nDCG@10 | p50 latency |
| --- | --- | --- | --- |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | 0.732 | 698 ms |
| `BAAI/bge-reranker-base` | 278M | 0.678 | 2898 ms |

The smaller model was both better and faster here. (The pool was built using MiniLM, so
bge has a small bias against it, but only 2.2% of its top 10 was unjudged.)

## How it works

```
query -> FastAPI /search -> SearchService
                              |-- Elasticsearch BM25   (top 100)
                              |-- FAISS + bge-base     (top 100)
                              v
                         RRF fusion, k=60  (top 50)
                              v
                  cross-encoder rerank (optional)
                              v
                           top-k hits
```

Redis caches query embeddings and results. If Redis is down, search still works, just slower.

Design choices:

- **RRF instead of score normalisation.** BM25 scores and cosine similarities are on
  completely different scales. RRF only uses ranks, so there's nothing to tune.
- **Exact FAISS index (`IndexFlatIP`).** At 5.6k docs a brute-force scan is a few ms, so
  there's no reason to use an approximate index. Embeddings are normalised so inner
  product = cosine.
- **Rerank only the top 50.** The cross-encoder needs one forward pass per document, so
  depth directly controls latency.

## Evaluation setup

- 32 queries: 13 lexical, 13 mixed, 6 semantic
- Relevance graded 0–3 by hand, from title + abstract
- Pool = top 20 from each of the four systems
- nDCG@10 (gain `2^rel - 1`, log2 discount) and Recall@10 (rel ≥ 1). Both are in
  `app/domain/metrics.py` with unit tests.

### Limitations

- Single annotator, so there's no agreement score. The line between grade 0 and 1
  is the fuzziest.
- 32 queries is enough to see BM25 vs dense, but not enough to say anything about
  differences under ~1%.
- A new system not in the pool could still be under-scored.
- The corpus is the most recent papers in six CS categories, so it's heavy on LLM/RAG topics.
- Judged on abstracts only, not full papers.

## Running it

Needs Python 3.12+ and Docker.

```bash
docker compose up -d elasticsearch redis

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python scripts/fetch_corpus.py    # ~2 min
python scripts/build_index.py     # ~15 min on CPU / MPS

uvicorn app.api.main:app --reload
```

```bash
curl "http://localhost:8000/search?q=cross+encoder+reranking&k=5"
```

`mode` can be `bm25`, `dense`, `hybrid` (default) or `hybrid_rerank`. There's also
`GET /health`, and Swagger docs at http://localhost:8000/docs.

To re-run the evaluation:

```bash
python scripts/make_pool.py --depth 20
python scripts/show_pool.py --unjudged   # anything left to grade
python scripts/run_eval.py               # writes data/eval/results.json and results.md
```

Settings like `CANDIDATE_K`, `RERANK_DEPTH`, `RRF_K`, `EMBED_MODEL` and `RERANK_MODEL` can
be changed through env variables; see `.env.example`.

## Project structure

```
app/
  domain/     RRF, metrics, models (no I/O)
  adapters/   Elasticsearch, FAISS, Redis, sentence-transformers
  api/        FastAPI routes + dependency wiring
  ports.py    interfaces the service depends on
  service.py  retrieval logic
scripts/      data fetching, indexing, pooling, eval
tests/        unit tests using in-memory fakes
```

The service only talks to the interfaces in `ports.py`, so the tests run without Docker:

```bash
pytest
ruff check .
```

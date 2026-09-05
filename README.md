# Hybrid Retrieval Search

Search over 5,607 arXiv CS abstracts, comparing four retrieval strategies against 1,316
graded relevance judgements: **BM25**, **dense**, **hybrid (RRF)**, and
**hybrid + cross-encoder reranking**.

The point of the project is not that hybrid search works — it is *measuring* how much it
works, and where. Every number below comes from `scripts/run_eval.py` and is reproducible
with one command.

## Results

32 queries, k=10, 1,316 graded judgements. Latency is measured with caching disabled.

| System | nDCG@10 | Recall@10 | p50 latency | p95 latency |
| --- | --- | --- | --- | --- |
| BM25 (Elasticsearch) | 0.5205 | 0.2355 | 12 ms | 30 ms |
| Dense (bge-base-en-v1.5 + FAISS) | 0.7087 (+36.2%) | 0.2890 | 13 ms | 100 ms |
| **Hybrid (RRF)** — *default* | **0.7267** (+39.6%) | **0.2936** | **35 ms** | 48 ms |
| Hybrid + cross-encoder rerank | 0.7322 (+40.7%) | 0.2930 | 698 ms | 2266 ms |

**Hybrid RRF is the default the API ships with.** Reranking scores marginally higher
(+0.8% nDCG@10) but costs **~20× the median latency** — 35 ms to 698 ms — and at 32
queries that 0.8% is inside the noise. Paying 20× for a difference the harness cannot
reliably measure is not a trade worth making, so reranking stays available per request
(`?mode=hybrid_rerank`) rather than on by default.

> **Reading Recall@10.** There are ~32 relevant documents per query but only 10 result
> slots, so the highest achievable Recall@10 is **0.327**. Hybrid reaches 0.294 — about
> 90% of the ceiling. Absolute recall looks low; relative to what is attainable it is not.

> **Reading the latency columns.** Quote the p50, not the p95. With 32 queries the p95 is
> the 30th–31st sorted value — effectively a single observation, and one slow query moves
> it by tens of milliseconds. Across five warm repeat runs, hybrid p50 stayed at 25–27 ms
> while its p95 swung between 29 ms and 64 ms. The p95 column is reported for completeness,
> but only the p50 is stable enough to compare systems on. Measured on an M-series laptop
> with Elasticsearch and Redis in Docker; treat these as relative, not absolute.

### Where each system actually wins

| System | lexical (n=13) | mixed (n=13) | semantic (n=6) |
| --- | --- | --- | --- |
| BM25 | 0.7027 | 0.5028 | 0.1640 |
| Dense | 0.7784 | 0.7219 | **0.5289** |
| Hybrid (RRF) | **0.8087** | 0.7526 | 0.4932 |
| Hybrid + rerank | 0.7927 | **0.8040** | 0.4455 |

This breakdown is the most useful table in the project, and it contradicts the aggregate:

- **BM25 collapses on paraphrase queries** (0.164). *"How to stop chatbots from leaking
  private training data"* shares almost no vocabulary with the differential-privacy and
  memorization papers that answer it, and lexical matching has nothing to grip.
- **Hybrid is not uniformly better.** On semantic queries, dense alone (0.529) beats
  hybrid (0.493): equal-weight RRF lets a retriever that is near-useless on those queries
  dilute a strong one. The aggregate win comes entirely from lexical and mixed queries.
  Note the sample size: this rests on 6 queries, so treat it as a signal worth
  investigating rather than a settled result.
- **Reranking pays for itself only on mixed queries** (0.752 → 0.804), and costs ~660 ms
  of p50 latency to do it. On lexical queries it is a slight net negative.

The honest summary is not "hybrid + rerank is best" — it is that the right configuration
depends on the query, and a router that sent paraphrase-style queries to dense-only would
likely beat every fixed configuration measured here.

## What the evaluation caught

The first version of this harness reported that cross-encoder reranking **hurt** quality
by 14% (0.709 → 0.607). Swapping in a stronger reranker (`bge-reranker-base`) made it
worse still, which is what prompted a closer look at the harness rather than the models.

The cause was pooling depth. Judgements were pooled from BM25 top-15 and dense top-15
only. BM25 and dense are then evaluated on a pool built from their own output, but fusion
and reranking reorder a 50-document candidate list and surface documents neither retriever
ranked in its own top-15. Those documents were unjudged, and unjudged counts as
non-relevant — so the systems that reached deeper were penalised for doing so:

| System | share of top-10 that was unjudged |
| --- | --- |
| BM25 | 0% |
| Dense | 0% |
| Hybrid (RRF) | 11.6% |
| Hybrid + rerank | **26.2%** |

A quarter of the reranker's results were being scored as non-relevant by default. After
re-pooling from **all four systems** at depth 20 and judging the 489 additional documents,
the ordering became monotonic and reranking came out ahead:

| | before fix | after fix |
| --- | --- | --- |
| Hybrid + rerank nDCG@10 | 0.6074 (−14% vs hybrid) | 0.7322 (+0.8% vs hybrid) |

The measurement was wrong, not the model. `scripts/make_pool.py` now pools from every
system under test, which is what standard TREC practice requires and what the first
version got wrong.

### Reranker choice

Re-measured on the corrected pool, the larger reranker is both worse and far slower:

| Reranker | params | nDCG@10 | p50 latency |
| --- | --- | --- | --- |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | **0.7322** | 698 ms |
| `BAAI/bge-reranker-base` | 278M | 0.6779 | 2898 ms |

Reproduce with `RERANK_MODEL=BAAI/bge-reranker-base python scripts/run_eval.py`. The
pool was built with the MiniLM reranker, so `bge-reranker-base` carries a small residual
version of the bias described above — but only 2.2% of its top-10 is unjudged, so the gap
is real rather than an artefact. The bigger model is not the better one here.

## Architecture

```
                         ┌──────────────────────────┐
   query ───────────────▶│      FastAPI /search     │
                         └────────────┬─────────────┘
                                      │
                        ┌─────────────▼─────────────┐
                        │       SearchService       │
                        └──┬────────────────────┬───┘
                           │                    │
              ┌────────────▼───────┐  ┌─────────▼──────────┐
              │  Elasticsearch     │  │  FAISS IndexFlatIP │
              │  BM25, english     │  │  bge-base-en-v1.5  │
              │  analyzer          │  │  (cosine)          │
              └────────────┬───────┘  └─────────┬──────────┘
                           │  top-100           │  top-100
                           └────────┬───────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │ Reciprocal Rank Fusion (k=60) │
                    └───────────────┬───────────────┘
                                    │  top-50
                                    ▼
                    ┌───────────────────────────────┐
                    │  Cross-encoder rerank         │
                    │  ms-marco-MiniLM-L-6-v2       │
                    └───────────────┬───────────────┘
                                    ▼
                                 top-k hits

   Redis caches query embeddings and result lists; cache faults degrade
   latency, never correctness.
```

### Design decisions

**Reciprocal Rank Fusion rather than score normalisation.** BM25 scores are unbounded and
corpus-dependent; cosine similarities sit in `[-1, 1]`. Combining them by weighted sum
needs per-corpus normalisation constants that do not transfer between queries. RRF discards
scores and fuses on *ranks*, so it needs no tuning and cannot be destabilised by one
retriever's score distribution. The cost is visible in the semantic column above: rank
fusion has no way to know that BM25's ranking is worthless for that query.

**Cross-encoder over the top 50 only.** A bi-encoder embeds query and document
independently and cannot model term interactions between them. A cross-encoder scores the
pair jointly and is more accurate, but costs one forward pass per candidate — hence a
shallow candidate set. At depth 50 that is ~660 ms of added p50 latency for +0.8% nDCG
overall, which is a bad trade at this corpus size and a defensible one only for the mixed
query family.

**Exact FAISS index (`IndexFlatIP`).** At 5.6k documents an exact scan takes single-digit
milliseconds. An approximate index (IVF/HNSW) would trade recall for a speedup that is not
needed; the retriever interface is unchanged if the corpus grows.

**Embeddings normalised at encode time**, so inner product equals cosine similarity and the
index can be a plain `IndexFlatIP`. Queries carry bge's instruction prefix; documents do
not — the model is trained asymmetrically.

## Evaluation methodology

- **32 queries** in three families, so the comparison is diagnostic rather than a single
  average:
  - `lexical` — wording likely appears verbatim in relevant papers
    (*"parameter efficient fine tuning with low rank adaptation"*)
  - `semantic` — paraphrase sharing little vocabulary with its relevant documents
    (*"making neural networks smaller and cheaper to run without losing accuracy"*)
  - `mixed` — natural phrasing, partial term overlap
- **Pooled judgements** — every system under test contributes its top 20; the
  de-duplicated union is judged. 1,316 judgements across 32 queries.
- **Graded relevance**, judged from title and abstract:

  | Grade | Meaning |
  | --- | --- |
  | 3 | Directly on topic — the paper is *about* what the query asks for |
  | 2 | Substantially relevant — addresses the topic as a major component |
  | 1 | Marginally relevant — related area, applies or mentions the technique |
  | 0 | Not relevant |

- **Metrics** — nDCG@10 with exponential gain (`2^rel − 1`) and log₂ discount, normalised
  against the ideal ordering of judged documents; Recall@10 over documents graded ≥ 1.
  Both are pure functions in `app/domain/metrics.py` with unit tests.

### Limitations

1. **Single annotator.** No second judge, so there is no inter-annotator agreement figure.
   The grade-1 vs grade-0 boundary is the least reliable.
2. **Residual pooling bias.** Fixed for the four systems measured here, since all four
   contribute to the pool. A *fifth* system could still surface relevant documents outside
   it and be under-credited.
3. **32 queries is small.** Enough to see the BM25-vs-dense gap (+36%); not enough for
   significance on the hybrid-vs-rerank gap (+0.8%). Treat sub-1% differences as noise —
   including the headline reranking win.
4. **Recent-corpus skew.** The corpus is the most recent ~5.6k papers across six CS
   categories, so it is dense in current topics (LLMs, RAG, agents) and thin elsewhere.
5. **Judged from title and abstract**, not full text.
6. **Tail latency is not measurable at this sample size.** 32 queries makes p95 a single
   observation; it varies by more than 2× between identical runs. Only the p50 is quoted
   as a comparison figure, and a real tail measurement would need thousands of queries.

## Quick start

```bash
# 1. Infrastructure
docker compose up -d elasticsearch redis

# 2. Environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# 3. Data and indexes  (fetch ~2 min; embedding ~15 min on CPU/MPS)
python scripts/fetch_corpus.py
python scripts/build_index.py

# 4. Serve
uvicorn app.api.main:app --reload
```

```bash
# Defaults to hybrid RRF; add &mode=hybrid_rerank to opt into reranking.
curl "http://localhost:8000/search?q=cross+encoder+reranking&k=5"
```

Interactive docs at http://localhost:8000/docs.

### Reproducing the evaluation

```bash
python scripts/make_pool.py --depth 20   # pool from all systems under test
python scripts/show_pool.py --unjudged   # list anything still needing a grade
python scripts/run_eval.py               # score all four systems
```

`run_eval.py` writes `data/eval/results.json` (per-query scores included) and
`data/eval/results.md`.

## API

| Endpoint | Description |
| --- | --- |
| `GET /health` | Liveness plus the modes this instance can serve |
| `GET /search?q=&mode=&k=` | `mode` ∈ `bm25`, `dense`, `hybrid`, `hybrid_rerank` |

```json
{
  "query": "cross encoder reranking",
  "mode": "hybrid_rerank",
  "took_ms": 84.2,
  "cached": false,
  "hits": [{"doc_id": "2609.04083v1", "score": 8.31, "title": "...", "abstract": "..."}]
}
```

## Layout

```
app/
  domain/        # models, RRF, metrics, errors — pure, no I/O
  adapters/      # Elasticsearch, FAISS, Redis, sentence-transformers
  api/           # FastAPI routes and the composition root
  ports.py       # Protocols the service depends on
  service.py     # retrieval orchestration
scripts/         # fetch_corpus, build_index, make_pool, show_pool, run_eval
tests/           # unit tests with in-memory fakes — no infrastructure needed
```

The service layer depends only on the protocols in `ports.py`, never on Elasticsearch,
FAISS or Redis directly. `app/api/dependencies.py` is the single place binding a concrete
adapter to each port, which is what lets the whole pipeline — fusion order, rerank depth,
cache behaviour, error mapping — be tested without infrastructure:

```bash
pytest                       # 51 tests, no Docker required
pytest --cov=app --cov-report=term-missing
ruff check .
```

## Configuration

Environment variables (see `.env.example`): `CANDIDATE_K` (per-retriever depth before
fusion, default 100), `RERANK_DEPTH` (candidates scored by the cross-encoder, default 50),
`RRF_K` (fusion damping, default 60), `EMBED_MODEL`, `RERANK_MODEL`, `CACHE_ENABLED`, plus
connection strings.

### Retrieval quality (32 queries, k=10)

| System | nDCG@10 | Recall@10 | p50 latency | p95 latency |
| --- | --- | --- | --- | --- |
| BM25 (Elasticsearch) | 0.5205 | 0.2355 | 8 ms | 27 ms |
| Dense (bge-base + FAISS) | 0.7087 (+36.2%) | 0.2890 | 13 ms | 106 ms |
| Hybrid (RRF) | 0.7267 (+39.6%) | 0.2936 | 30 ms | 39 ms |
| Hybrid + cross-encoder rerank | 0.6779 (+30.2%) | 0.2907 | 2898 ms | 17546 ms |

### nDCG@10 by query family

| System | lexical | mixed | semantic |
| --- | --- | --- | --- |
| BM25 (Elasticsearch) | 0.7027 | 0.5028 | 0.1640 |
| Dense (bge-base + FAISS) | 0.7784 | 0.7219 | 0.5289 |
| Hybrid (RRF) | 0.8087 | 0.7526 | 0.4932 |
| Hybrid + cross-encoder rerank | 0.7483 | 0.7164 | 0.4417 |

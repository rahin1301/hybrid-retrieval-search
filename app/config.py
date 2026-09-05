"""Application settings.

Settings are resolved from environment variables (or a local ``.env``) and
handed to components explicitly rather than read from a module-level global,
so tests can construct a component with a tailored ``Settings`` instance.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # -- infrastructure ---------------------------------------------------
    es_url: str = "http://localhost:9200"
    redis_url: str = "redis://localhost:6379/0"
    index_name: str = "arxiv"

    # -- models -----------------------------------------------------------
    embed_model: str = "BAAI/bge-base-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # bge-v1.5 is trained with an asymmetric objective: queries carry this
    # instruction prefix, documents are embedded bare. Dropping it costs a
    # measurable amount of nDCG.
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    embed_batch_size: int = 64

    # -- retrieval tuning -------------------------------------------------
    candidate_k: int = Field(
        default=100, description="Depth pulled from each retriever before fusion"
    )
    rerank_depth: int = Field(
        default=50, description="Number of fused candidates scored by the cross-encoder"
    )
    rrf_k: int = Field(default=60, description="RRF damping constant")

    # -- caching ----------------------------------------------------------
    cache_ttl_s: int = 3600
    cache_enabled: bool = True

    # -- paths ------------------------------------------------------------
    corpus_path: Path = ROOT / "data" / "corpus.jsonl"
    faiss_index_path: Path = ROOT / "data" / "index" / "dense.faiss"
    faiss_ids_path: Path = ROOT / "data" / "index" / "dense_ids.json"
    eval_dir: Path = ROOT / "data" / "eval"


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings, cached so env parsing happens once."""
    return Settings()

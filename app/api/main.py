"""FastAPI application entrypoint.

Models and indexes load once during startup rather than per request: the
embedding model alone takes several seconds to initialise, and loading it
lazily would put that cost on an unlucky user's first query.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.dependencies import build_search_service
from app.api.routes import router
from app.config import get_settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    log.info("Starting up: building search service")
    app.state.settings = settings
    app.state.search_service = build_search_service(settings)
    log.info("Startup complete")
    yield
    log.info("Shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Hybrid Retrieval Search",
        description=(
            "BM25 + dense retrieval with reciprocal rank fusion and "
            "cross-encoder reranking over arXiv abstracts."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()

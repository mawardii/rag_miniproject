"""FastAPI app for serving the RAG pipeline."""

import logging
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.config import Settings, get_settings
from rag.pipeline import RAGPipeline, RAGResult
from rag.retrieval.vectorstore import FINGERPRINT_FILE

logger = logging.getLogger(__name__)

app = FastAPI(title="Financial Filings RAG API", version="0.1.0")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class SourceResponse(BaseModel):
    source: str
    page: int | None = Field(
        default=None,
        description="One-indexed page number when available.",
    )
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    latency_s: float
    input_tokens: int | None
    output_tokens: int | None


class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    retrieval_mode: str
    rerank_enabled: bool
    top_k: int


@lru_cache
def get_app_settings() -> Settings:
    return get_settings()


@lru_cache
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(get_app_settings())


SETTINGS_DEP = Depends(get_app_settings)
PIPELINE_DEP = Depends(get_pipeline)


def _to_query_response(result: RAGResult) -> QueryResponse:
    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceResponse(
                source=source.source,
                page=source.page + 1 if source.page is not None else None,
                snippet=source.snippet,
            )
            for source in result.sources
        ],
        latency_s=result.latency_s,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


@app.get("/healthz", response_model=HealthResponse)
def healthz(settings: Settings = SETTINGS_DEP) -> HealthResponse:
    return HealthResponse(
        status="ok",
        index_ready=(settings.index_dir / FINGERPRINT_FILE).exists(),
        retrieval_mode=settings.retrieval_mode,
        rerank_enabled=settings.rerank_enabled,
        top_k=settings.top_k,
    )


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    pipeline: RAGPipeline = PIPELINE_DEP,
) -> QueryResponse:
    try:
        return _to_query_response(pipeline.answer(request.question))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("query failed")
        raise HTTPException(status_code=500, detail="query failed") from exc

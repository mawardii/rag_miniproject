"""The RAG pipeline facade.

Every consumer — Streamlit UI, the future FastAPI service, the eval harness —
goes through this class, so they all exercise the identical pipeline. The steps
(retrieve -> rerank -> format -> generate) are explicit rather than hidden inside
a nested chain: each stage can be called, logged, and evaluated on its own.
"""

import logging
import time
from dataclasses import dataclass, field
from functools import cached_property

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from rag.config import Settings, get_settings
from rag.generation.prompts import SYSTEM_PROMPT, build_user_message
from rag.retrieval.rerank import llm_rerank
from rag.retrieval.retrievers import HybridRetriever
from rag.retrieval.vectorstore import load_or_build_index

logger = logging.getLogger(__name__)


@dataclass
class SourceRef:
    """A retrieved chunk, in citation order ([1] == sources[0])."""

    source: str
    page: int | None
    snippet: str


@dataclass
class RAGResult:
    answer: str
    sources: list[SourceRef]
    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    # Full retrieved documents — used by the eval judge; API layers should drop this.
    context_docs: list[Document] = field(default_factory=list)


class RAGPipeline:
    def __init__(self, settings: Settings | None = None, force_rebuild: bool = False):
        self.settings = settings or get_settings()
        self._index = load_or_build_index(self.settings, force_rebuild=force_rebuild)
        self._hybrid = (
            HybridRetriever(self._index) if self.settings.retrieval_mode == "hybrid" else None
        )
        self._llm = ChatOpenAI(
            model=self.settings.chat_model,
            api_key=self.settings.deepseek_api_key.get_secret_value(),
            base_url=self.settings.chat_base_url,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            timeout=60,
            max_retries=2,
            # v4 models default to thinking mode, which spends the max_tokens budget
            # on hidden reasoning and can return empty content. Disable explicitly —
            # this matches the behavior of the retired deepseek-chat alias.
            extra_body={"thinking": {"type": "disabled"}},
        )

    def _utility_llm(self, max_tokens: int) -> ChatOpenAI:
        """Deterministic short-output LLM for rerank/judge duties."""
        return ChatOpenAI(
            model=self.settings.chat_model,
            api_key=self.settings.deepseek_api_key.get_secret_value(),
            base_url=self.settings.chat_base_url,
            temperature=0,
            max_tokens=max_tokens,
            timeout=30,
            max_retries=2,
            extra_body={"thinking": {"type": "disabled"}},
        )

    @cached_property
    def _rerank_llm(self) -> ChatOpenAI:
        return self._utility_llm(max_tokens=150)

    @cached_property
    def judge_llm(self) -> ChatOpenAI:
        return self._utility_llm(max_tokens=100)

    def retrieve(self, question: str) -> list[Document]:
        """Retrieval (+ optional rerank) only — graded directly by the eval harness."""
        s = self.settings
        fetch_k = s.fetch_k if s.rerank_enabled else s.top_k
        if self._hybrid is not None:
            candidates = self._hybrid.retrieve(question, fetch_k)
        else:
            candidates = self._index.similarity_search(question, k=fetch_k)
        if s.rerank_enabled:
            return llm_rerank(self._rerank_llm, question, candidates, s.top_k)
        return candidates[: s.top_k]

    def answer(self, question: str) -> RAGResult:
        t0 = time.perf_counter()
        docs = self.retrieve(question)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=build_user_message(question, docs)),
        ]
        response = self._llm.invoke(messages)
        latency = time.perf_counter() - t0

        usage = response.usage_metadata or {}
        sources = [
            SourceRef(
                source=d.metadata.get("source", "unknown"),
                page=d.metadata.get("page"),
                snippet=d.page_content[:200],
            )
            for d in docs
        ]
        logger.info(
            "answered in %.2fs (in=%s out=%s tokens, k=%d)",
            latency,
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            len(docs),
        )
        return RAGResult(
            answer=str(response.content),
            sources=sources,
            latency_s=latency,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            context_docs=docs,
        )

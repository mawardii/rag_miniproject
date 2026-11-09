"""Evaluation harness: `rag-eval [--answers] [--golden path]`.

Retrieval metrics (fast, embeddings-only):
- filing hit@k and filing MRR (rank over unique filenames, for historical comparison)
- evidence-page hit@k, recall@k, and MRR (rank over actual retrieved chunks)
  Page matches do not guarantee that a chunk contains the answer-bearing text.

Answer metrics (--answers; one generation + one judge LLM call per question):
- faithfulness: is every claim in the answer supported by the retrieved context?
- relevance:    does the answer actually address the question?

Every run prints the pipeline config it measured — a metric without its config
is meaningless, since the whole point is comparing configurations.
"""

import argparse
import json
import logging
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from rag.config import PROJECT_ROOT, Settings, get_settings
from rag.generation.prompts import format_context
from rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

DEFAULT_GOLDEN = PROJECT_ROOT / "evals" / "golden.jsonl"

JUDGE_SYSTEM = """\
You are a strict evaluator of RAG answers. Score the answer on three axes, 1-5:
- faithfulness: 5 = every claim is directly supported by the context excerpts;
  1 = contradicts or invents beyond the context. Saying "the context does not
  contain this" when it truly does not is faithful (5), not a failure.
- relevance: 5 = fully addresses the question; 1 = off-topic.
- correctness: compare against the reference answer written by a human analyst:
  5 = same conclusion and figures (minor rounding/wording differences are fine);
  3 = partially correct or incomplete; 1 = wrong conclusion or wrong figures.
  If no reference answer is provided, omit this key.
Respond with ONLY a JSON object:
{"faithfulness": <int>, "relevance": <int>, "correctness": <int>}"""

JUDGE_TEMPLATE = """\
Question: {question}

Context excerpts given to the system:
{context}

Answer to evaluate:
{answer}

Reference answer (ground truth):
{reference}"""


@dataclass
class RetrievalRow:
    question: str
    hit: bool
    reciprocal_rank: float
    retrieved: list[str]
    evidence_page_hit: bool | None = None
    evidence_page_recall: float | None = None
    evidence_page_reciprocal_rank: float | None = None
    expected_pages: list[tuple[str, int]] = field(default_factory=list)
    retrieved_chunks: list[dict] = field(default_factory=list)
    retrieval_latency_s: float = 0.0


def load_golden(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise RuntimeError(f"Golden set is empty: {path}")
    for row in rows:
        for evidence in row.get("evidence", []):
            if (
                not isinstance(evidence.get("source"), str)
                or not evidence["source"]
                or type(evidence.get("page")) is not int
                or evidence["page"] < 1
                or not isinstance(evidence.get("text"), str)
                or not evidence["text"].strip()
            ):
                raise ValueError("Evidence requires a source, a one-based integer page, and text")
    return rows


def evaluate_retrieval(pipeline: RAGPipeline, golden: list[dict]) -> list[RetrievalRow]:
    rows = []
    for item in golden:
        started = time.perf_counter()
        docs = pipeline.retrieve(item["question"])
        latency = time.perf_counter() - started
        # Rank unique sources in retrieval order (several chunks may share a source).
        seen: list[str] = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            if source not in seen:
                seen.append(source)
        expected = set(item["expected_sources"])
        rank = next((i for i, s in enumerate(seen, 1) if s in expected), None)
        expected_pages = {(e["source"], e["page"]) for e in item.get("evidence", [])}
        chunks = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata["page"] + 1 if doc.metadata.get("page") is not None else None,
                "text": doc.page_content,
            }
            for doc in docs
        ]
        pages = [(chunk["source"], chunk["page"]) for chunk in chunks]
        evidence_rank = next((i for i, page in enumerate(pages, 1) if page in expected_pages), None)
        rows.append(
            RetrievalRow(
                question=item["question"],
                hit=rank is not None,
                reciprocal_rank=1.0 / rank if rank else 0.0,
                retrieved=seen,
                evidence_page_hit=evidence_rank is not None if expected_pages else None,
                evidence_page_recall=(
                    len(expected_pages.intersection(pages)) / len(expected_pages)
                    if expected_pages
                    else None
                ),
                evidence_page_reciprocal_rank=(1.0 / evidence_rank if evidence_rank else 0.0)
                if expected_pages
                else None,
                expected_pages=sorted(expected_pages),
                retrieved_chunks=chunks,
                retrieval_latency_s=latency,
            )
        )
    return rows


def summarize_retrieval(rows: list[RetrievalRow]) -> dict:
    annotated = [row for row in rows if row.evidence_page_hit is not None]
    return {
        "questions": len(rows),
        "annotated_questions": len(annotated),
        "mean_retrieval_latency_s": statistics.mean(row.retrieval_latency_s for row in rows),
        "filing_hit_rate": statistics.mean(row.hit for row in rows),
        "filing_mrr_unique_sources": statistics.mean(row.reciprocal_rank for row in rows),
        "evidence_page_hit_rate": (
            statistics.mean(row.evidence_page_hit for row in annotated) if annotated else None
        ),
        "evidence_page_recall": (
            statistics.mean(row.evidence_page_recall for row in annotated) if annotated else None
        ),
        "evidence_page_mrr": (
            statistics.mean(row.evidence_page_reciprocal_rank for row in annotated)
            if annotated
            else None
        ),
    }


def judge_answer(pipeline: RAGPipeline, question: str, reference: str | None) -> tuple[dict, str]:
    result = pipeline.answer(question)
    prompt = JUDGE_TEMPLATE.format(
        question=question,
        context=format_context(result.context_docs),
        answer=result.answer,
        reference=reference or "(none provided)",
    )
    response = pipeline.judge_llm.bind(response_format={"type": "json_object"}).invoke(
        [SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=prompt)]
    )
    try:
        scores = json.loads(str(response.content))
    except ValueError:
        match = re.search(r"\{.*\}", str(response.content), re.DOTALL)
        scores = json.loads(match.group()) if match else {}
    return scores, result.answer


def describe_config(settings: Settings) -> str:
    return (
        f"retrieval_mode={settings.retrieval_mode} rerank={settings.rerank_enabled} "
        f"top_k={settings.top_k} fetch_k={settings.fetch_k} "
        f"chunk={settings.chunk_size}/{settings.chunk_overlap} "
        f"embeddings={settings.embedding_model}"
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument(
        "--output", type=Path, help="Save retrieval metrics and full chunks as JSON."
    )
    parser.add_argument(
        "--answers",
        action="store_true",
        help="Also generate and LLM-judge answers (costs one gen + one judge call per question).",
    )
    args = parser.parse_args()

    settings = get_settings()
    pipeline = RAGPipeline(settings)
    golden = load_golden(args.golden)

    print(f"config: {describe_config(settings)}")
    print(f"golden set: {len(golden)} questions\n")

    rows = evaluate_retrieval(pipeline, golden)
    for row in rows:
        marker = "PASS" if row.hit else "MISS"
        print(f"  [{marker}] rr={row.reciprocal_rank:.2f}  {row.question[:70]}")
        if not row.hit:
            print(f"         got: {row.retrieved}")
        if row.evidence_page_hit is not None:
            print(
                f"         evidence page hit={row.evidence_page_hit} "
                f"recall={row.evidence_page_recall:.2f} "
                f"rr={row.evidence_page_reciprocal_rank:.2f}"
            )

    metrics = summarize_retrieval(rows)
    print(
        f"\nfiling hit@{settings.top_k}={metrics['filing_hit_rate']:.2f}  "
        f"filing MRR (unique sources)={metrics['filing_mrr_unique_sources']:.2f}"
    )
    print(
        f"evidence annotations: {metrics['annotated_questions']}/{metrics['questions']} questions"
    )
    if metrics["annotated_questions"]:
        print(
            f"evidence-page hit@{settings.top_k}={metrics['evidence_page_hit_rate']:.2f}  "
            f"recall@{settings.top_k}={metrics['evidence_page_recall']:.2f}  "
            f"MRR (chunk ranks)={metrics['evidence_page_mrr']:.2f}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "config": describe_config(settings),
            "table_boost_enabled": True,
            "golden": str(args.golden),
            "metrics": metrics,
            "rows": [asdict(row) for row in rows],
        }
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Retrieval report saved to {args.output}")

    if args.answers:
        print("\njudging answers (LLM calls)...")
        faith, rel, correct = [], [], []
        for item in golden:
            scores, answer = judge_answer(pipeline, item["question"], item.get("answer"))
            f, r, c = (scores.get(k) for k in ("faithfulness", "relevance", "correctness"))
            if f:
                faith.append(f)
            if r:
                rel.append(r)
            if c:
                correct.append(c)
            print(f"  faith={f} rel={r} correct={c}  {item['question'][:55]}")
        if faith:
            summary = (
                f"\nanswers: faithfulness={statistics.mean(faith):.2f}/5"
                f"  relevance={statistics.mean(rel):.2f}/5"
            )
            if correct:
                summary += f"  correctness={statistics.mean(correct):.2f}/5"
            print(summary)


if __name__ == "__main__":
    main()

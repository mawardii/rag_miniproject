"""Build evals/golden.jsonl from the FinanceBench open-source question set.

Filters the 150 open questions down to those answerable from the curated corpus
(see scripts/download_corpus.py) and converts them to our eval format, keeping
the gold answer so the LLM judge can grade correctness against ground truth.

FinanceBench annotations: (c) Patronus AI, CC BY-NC 4.0
(github.com/patronus-ai/financebench, arXiv:2311.11944).
"""

import json
from pathlib import Path

from download_corpus import CORPUS

EVALS_DIR = Path(__file__).resolve().parents[1] / "evals"
SOURCE = EVALS_DIR / "financebench_open_source.jsonl"
TARGET = EVALS_DIR / "golden.jsonl"


def main() -> None:
    corpus = set(CORPUS)
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line]
    golden = [
        {
            "question": row["question"],
            "expected_sources": [row["doc_name"] + ".pdf"],
            "answer": row["answer"],
            "financebench_id": row["financebench_id"],
            "question_type": row["question_type"],
            "question_reasoning": row["question_reasoning"],
            "evidence": [
                {
                    "source": evidence["doc_name"] + ".pdf",
                    # FinanceBench uses zero-based PDF indexes; golden pages are one-based.
                    "page": evidence["evidence_page_num"] + 1,
                    "text": evidence["evidence_text"],
                }
                for evidence in row["evidence"]
            ],
        }
        for row in rows
        if row["doc_name"] in corpus
    ]
    with TARGET.open("w", encoding="utf-8", newline="\n") as fh:
        for item in golden:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"{len(golden)} questions over {len(corpus)} documents -> {TARGET}")


if __name__ == "__main__":
    main()

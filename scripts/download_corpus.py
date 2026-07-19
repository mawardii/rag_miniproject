"""Download the curated FinanceBench corpus into docs/.

The PDFs are public SEC filings redistributed by the FinanceBench project
(github.com/patronus-ai/financebench). They are NOT committed to this repo —
run this script after cloning, then `rag-ingest`.

Curation rationale: the 18 documents below cover 69 of FinanceBench's 150
open-source questions and deliberately include same-company document pairs
(3M, Amcor, BestBuy, J&J, Pfizer) — near-duplicate filings are the realistic
hard case for retrieval — plus a mix of 10-K, 10-Q, 8-K, and earnings releases.
"""

import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs"
DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"

CORPUS = [
    "3M_2022_10K",
    "3M_2023Q2_10Q",
    "AES_2022_10K",
    "AMCOR_2023_10K",
    "AMCOR_2023Q4_EARNINGS",
    "AMD_2022_10K",
    "AMERICANEXPRESS_2022_10K",
    "BESTBUY_2023_10K",
    "BESTBUY_2024Q2_10Q",
    "BOEING_2022_10K",
    "CVSHEALTH_2022_10K",
    "JOHNSON_JOHNSON_2022_10K",
    "JOHNSON_JOHNSON_2023_8K_dated-2023-08-30",
    "PEPSICO_2022_10K",
    "PFIZER_2021_10K",
    "Pfizer_2023Q2_10Q",
    "ULTABEAUTY_2023Q4_EARNINGS",
    "VERIZON_2022_10K",
]


def main() -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    for name in CORPUS:
        target = DOCS_DIR / f"{name}.pdf"
        if target.exists():
            print(f"exists   {target.name}")
            continue
        url = f"{BASE_URL}/{name}.pdf"
        print(f"fetching {target.name} ...")
        try:
            urllib.request.urlretrieve(url, target)
        except Exception as exc:
            print(f"FAILED   {target.name}: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
    print(f"\n{len(CORPUS)} documents ready in {DOCS_DIR}")


if __name__ == "__main__":
    main()

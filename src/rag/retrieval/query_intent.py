"""Detect whether a question likely needs a specific numeric line item.

Financial-statement lookups ("inventory turnover", "quick ratio") are where
retrieval most often buries the one chunk with the actual number (see
retrievers.py:boost_table_candidates for why). This is a cheap, pre-embedding
signal used to decide whether to widen the search to table-tagged chunks —
false positives just do a bit of extra (cheap) search; false negatives fall
back to normal retrieval, so an imprecise regex is an acceptable trade-off.
"""

import re

_NUMERIC_INTENT_RE = re.compile(
    r"\b("
    r"how (?:many|much)|ratio|turnover|margin|percent(?:age)?|growth|"
    r"increase|decrease|revenue|earnings|eps|debt|liquidity|capex|"
    r"capital expenditure|assets?|liabilit\w*|equity|cash flow|expense|"
    r"cost of (?:sales|goods|revenue)|inventory|dividend|shares? outstanding|"
    r"book value|quick ratio|current ratio"
    r")\b",
    re.IGNORECASE,
)


def is_numeric_query(question: str) -> bool:
    return bool(_NUMERIC_INTENT_RE.search(question)) or "%" in question


def extract_financial_terms(question: str) -> list[str]:
    """The specific line-item words in a question ("inventory", "margin", ...).

    Used to focus a keyword search on the term that actually identifies a table
    row, instead of the full question — generic words shared by every question
    on a filing ("Corporation", "FY2022") otherwise drown out the one term
    (e.g. "inventory") that discriminates it from every other row in the same
    company's tables.
    """
    return [m.group(0).lower() for m in _NUMERIC_INTENT_RE.finditer(question)]

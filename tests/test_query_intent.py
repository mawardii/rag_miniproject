from rag.retrieval.query_intent import is_numeric_query


def test_is_numeric_query_detects_ratio_question() -> None:
    assert is_numeric_query("Does 3M have a healthy quick ratio for Q2 FY2023?")


def test_is_numeric_query_detects_inventory_turnover_question() -> None:
    assert is_numeric_query("Roughly how many times has AES sold its inventory in FY2022?")


def test_is_numeric_query_detects_percent_sign() -> None:
    assert is_numeric_query("What was the change in gross margin %?")


def test_is_numeric_query_false_for_narrative_question() -> None:
    assert not is_numeric_query("What risks does AMD disclose related to its supply chain?")

from rag.retrieval.rerank import _parse_ranking


def test_parse_ranking_accepts_json_object() -> None:
    assert _parse_ranking('{"ranking": [3, 1, 2]}') == [3, 1, 2]


def test_parse_ranking_accepts_json_list() -> None:
    assert _parse_ranking("[2, 1]") == [2, 1]


def test_parse_ranking_falls_back_to_digits() -> None:
    assert _parse_ranking("best passages are 4, then 2") == [4, 2]

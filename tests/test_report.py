"""Report formatting tests."""

from crank.report import format_json, format_table
from crank.types import ClusterIdentity, ClusterScore, ScoringMode


def test_table_includes_rank_header() -> None:
    score = ClusterScore(
        identity=ClusterIdentity(name="prod"),
        rank=1,
        total_score=80.0,
        base_score=70.0,
        scoring_mode=ScoringMode.HEURISTIC,
        keyword_boost=10.0,
        area_contributions=(),
        top_features=(),
        summary="test",
    )
    table = format_table([score])
    assert "RANK" in table
    assert "BASE" in table
    assert "prod" in table


def test_json_output_is_valid() -> None:
    import json

    score = ClusterScore(
        identity=ClusterIdentity(name="x"),
        rank=1,
        total_score=1.0,
        base_score=1.0,
        scoring_mode=ScoringMode.HEURISTIC,
        keyword_boost=0.0,
        area_contributions=(),
        top_features=(),
        summary="s",
    )
    parsed = json.loads(format_json([score]))
    assert parsed[0]["cluster"] == "x"
    assert parsed[0]["base_score"] == 1.0
    assert parsed[0]["scoring_mode"] in ("heuristic", "ml")

"""Human-readable ranking output."""

from __future__ import annotations

import json
from typing import Any

from crank.types import ClusterScore


def format_table(scores: list[ClusterScore]) -> str:
    if not scores:
        return "No clusters scored."
    lines = [
        "RANK  SCORE   BASE   MODE        KW     CLUSTER              SUMMARY",
        "----  -----   ----   ----------  ----   -------------------  -------",
    ]
    for s in scores:
        name = s.identity.name[:19].ljust(19)
        mode = s.scoring_mode.value[:10].ljust(10)
        lines.append(
            f"{s.rank:4}  {s.total_score:5.1f}  {s.base_score:5.1f}  {mode}  "
            f"{s.keyword_boost:5.1f}  {name}  {s.summary}"
        )
    return "\n".join(lines)


def format_json(scores: list[ClusterScore]) -> str:
    payload: list[dict[str, Any]] = []
    for s in scores:
        payload.append(
            {
                "rank": s.rank,
                "cluster": s.identity.name,
                "context": s.identity.context,
                "total_score": s.total_score,
                "base_score": s.base_score,
                "scoring_mode": s.scoring_mode.value,
                "keyword_boost": s.keyword_boost,
                "areas": [
                    {
                        "area": a.area.value,
                        "score": a.score,
                        "keywords": list(a.matched_keywords),
                    }
                    for a in s.area_contributions
                ],
                "top_features": list(s.top_features),
                "summary": s.summary,
            }
        )
    return json.dumps(payload, indent=2)

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from types import SimpleNamespace

import pandas as pd

from qf_guard.data import normalize_transactions
from qf_guard.scoring import RiskScorer


@dataclass(slots=True)
class RealtimePipeline:
    scorer: RiskScorer = field(default_factory=RiskScorer)
    is_fitted: bool = False
    training_metadata: dict[str, str] = field(default_factory=dict)
    config: SimpleNamespace = field(
        default_factory=lambda: SimpleNamespace(
            graph=SimpleNamespace(candidate_rate=1.0)
        )
    )

    def score_transactions(
        self,
        transactions: pd.DataFrame,
        *,
        update_history: bool = True,
    ) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model chưa được nạp")
        return self.scorer.score(
            normalize_transactions(transactions),
            candidate_rate_cap=self.config.graph.candidate_rate,
            update_history=update_history,
        )


@dataclass(slots=True)
class RealtimeService:
    pipeline: RealtimePipeline = field(default_factory=RealtimePipeline)

    def score_transactions(self, transactions: pd.DataFrame) -> list[dict[str, object]]:
        return self.pipeline.score_transactions(transactions).to_dict(orient="records")


@lru_cache(maxsize=1)
def get_service() -> RealtimeService:
    return RealtimeService()

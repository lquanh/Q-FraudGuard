from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pandas as pd

from qf_guard.data import normalize_transactions
from qf_guard.online import OnlineFeatures
from qf_guard.scoring import RiskScorer
from qf_guard.temporal_graph import TemporalNeighborhood


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
    """Stateful scoring service; keep one instance per stream/session.

    The scorer carries the transaction history used for causal features, so
    sharing a service between independent streams would mix their histories.
    """

    pipeline: RealtimePipeline = field(default_factory=RealtimePipeline)

    @classmethod
    def from_scorer(cls, scorer: RiskScorer) -> "RealtimeService":
        return cls(RealtimePipeline(scorer=scorer, is_fitted=True))

    def score_transactions(self, transactions: pd.DataFrame) -> list[dict[str, object]]:
        return self.pipeline.score_transactions(transactions).to_dict(orient="records")


class EventScorer:
    """Score normalized events in source order with an independent incremental state."""

    def __init__(self, scorer: RiskScorer, *, warm_history=True):
        self.model = scorer
        scorer._check_fitted()
        self.features = OnlineFeatures(
            getattr(scorer, "confirmed_minute_", {}), getattr(scorer, "label_delay_minutes_", 1440),
            tuple(getattr(scorer, "feature_groups_", ()) or ()),
        )
        self.temporal_graph = getattr(scorer, "temporal_graph_", None)
        self.neighborhood = TemporalNeighborhood() if self.temporal_graph is not None else None
        if warm_history and scorer.history_transactions_ is not None:
            self.warm(scorer.history_transactions_)

    def warm(self, events: pd.DataFrame) -> None:
        self.features.warm(events)
        if self.neighborhood is not None:
            self.neighborhood.warm(events)

    def score(self, events: pd.DataFrame) -> pd.DataFrame:
        events = events.reset_index(drop=True)
        batch, graph, matrix = self.features.transform(events)
        columns = getattr(self.model, "feature_columns_", None)
        if columns is not None:
            matrix = matrix.loc[:, list(columns)]
        temporal_score = (self.temporal_graph.predict(self.neighborhood.transform(events))
                          if self.neighborhood is not None else None)
        return self.model._decisions(
            self.model._predict_components(batch, graph, matrix, temporal_score=temporal_score)
        )

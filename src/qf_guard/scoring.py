from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from qf_guard.config import ScoringConfig
from qf_guard.data import CANONICAL_COLUMNS, normalize_transactions
from qf_guard.explainability import reason_codes, reason_text, risk_band
from qf_guard.features import engineer_features, graph_risk_features


class RiskScorer:
    def __init__(self, config: ScoringConfig | None = None) -> None:
        self.config = config or ScoringConfig()
        self.supervised_model: Any | None = None
        self.anomaly_model: Any | None = None
        self.calibrator: Any | None = None
        self.amount_threshold_ = 0.0
        self.anomaly_low_ = 0.0
        self.anomaly_high_ = 1.0
        self.candidate_threshold_ = .5
        self.validation_stats_: dict[str, float] = {}
        self.learned_weights_ = self.config.normalized_weights
        self.history_transactions_: pd.DataFrame | None = None
        self.is_fitted_ = False

    def score(
        self,
        frame: pd.DataFrame,
        candidate_rate_cap: float | None = None,
        *,
        update_history: bool = False,
    ) -> pd.DataFrame:
        components = self._score_components(frame)
        threshold = self.candidate_threshold_
        if candidate_rate_cap is not None and 0 < candidate_rate_cap < 1 and len(components) > 1:
            threshold = max(
                threshold,
                float(components["risk_score"].quantile(1 - candidate_rate_cap)),
            )
        components["is_candidate"] = components["risk_score"] >= threshold
        codes = [reason_codes(row) for row in components.to_dict(orient="records")]
        components["reason_codes"] = ["|".join(value) for value in codes]
        components["reason_text"] = [reason_text(value) for value in codes]
        components["risk_band"] = components["risk_score"].map(risk_band)
        if update_history:
            history = (
                (self.history_transactions_, frame)
                if self.history_transactions_ is not None
                else (frame,)
            )
            self.history_transactions_ = self._feature_history(*history)
        return components

    def _score_components(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        batch = engineer_features(frame, history=self.history_transactions_)
        graph = graph_risk_features(batch.transactions, history=self.history_transactions_)
        raw_probability = np.clip(
            self.supervised_model.predict_proba(batch.matrix)[:, 1],
            1e-6,
            1 - 1e-6,
        )
        supervised = (
            self.calibrator.predict_proba(_logit(raw_probability).reshape(-1, 1))[:, 1]
            if self.calibrator is not None
            else raw_probability
        )
        raw_anomaly = -self.anomaly_model.score_samples(batch.matrix)
        anomaly = np.clip(
            (raw_anomaly - self.anomaly_low_) / (self.anomaly_high_ - self.anomaly_low_),
            0,
            1,
        )
        rules = self._rule_components(batch.transactions, batch.matrix)
        rule = (
            .27 * rules["rule_amount_spike"]
            + .25 * rules["rule_balance_drain"]
            + .15 * rules["rule_balance_mismatch"]
            + .16 * rules["rule_shared_device"]
            + .10 * rules["rule_rapid_activity"]
            + .07 * rules["rule_cross_border"]
        ).clip(0, 1)
        weights = self.learned_weights_
        score = (
            weights[0] * supervised
            + weights[1] * anomaly
            + weights[2] * graph["graph_score"].to_numpy()
            + weights[3] * rule.to_numpy()
        )
        result = batch.transactions.copy()
        result["supervised_score"] = np.clip(supervised, 0, 1)
        result["anomaly_score"] = anomaly
        result["graph_score"] = graph["graph_score"].to_numpy()
        result["rule_score"] = rule.to_numpy()
        for column in rules:
            result[column] = rules[column].to_numpy()
        result["risk_score"] = np.clip(score, 0, 1)
        return result

    def _rule_components(
        self,
        transactions: pd.DataFrame,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "rule_amount_spike": (transactions["amount"] >= self.amount_threshold_).astype(float),
                "rule_balance_drain": (features["sender_balance_delta"] >= .93).astype(float),
                "rule_balance_mismatch": (
                    (features["sender_balance_error"] >= .12)
                    | (features["receiver_balance_error"] >= .20)
                ).astype(float),
                "rule_shared_device": (features["device_account_count"] >= np.log1p(3)).astype(float),
                "rule_rapid_activity": (features["velocity_1h"] >= 5).astype(float),
                "rule_cross_border": (features["is_cross_border"] >= .5).astype(float),
            },
            index=transactions.index,
        )

    def _check_fitted(self) -> None:
        if not self.is_fitted_ or self.supervised_model is None or self.anomaly_model is None:
            raise RuntimeError("RiskScorer is not ready")

    @staticmethod
    def _feature_history(*frames: pd.DataFrame) -> pd.DataFrame:
        canonical = [
            normalize_transactions(frame).loc[:, list(CANONICAL_COLUMNS)]
            for frame in frames
        ]
        return normalize_transactions(pd.concat(canonical, ignore_index=True)).loc[
            :, list(CANONICAL_COLUMNS)
        ].copy()

    @classmethod
    def load(cls, path: str | Path) -> "RiskScorer":
        model = joblib.load(Path(path))
        if not isinstance(model, cls):
            raise TypeError("artifact is not a RiskScorer")
        return model


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))

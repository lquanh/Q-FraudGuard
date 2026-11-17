from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from qf_guard.config import ScoringConfig
from qf_guard.data import CANONICAL_COLUMNS, normalize_transactions
from qf_guard.explainability import reason_codes, reason_text, risk_band
from qf_guard.features import engineer_features, graph_risk_features


RULE_WEIGHTS = {
    "rule_amount_spike": .27,
    "rule_balance_drain": .25,
    "rule_balance_mismatch": .15,
    "rule_shared_device": .16,
    "rule_rapid_activity": .10,
    "rule_cross_border": .07,
}
AMOUNT_SPIKE_QUANTILE = .985
ANOMALY_LOW_QUANTILE = .05
ANOMALY_HIGH_QUANTILE = .98
ALERT_HOLD_QUANTILE = .99


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

    # ------------------------------------------------------------------ training
    def fit(self, train: pd.DataFrame, validation: pd.DataFrame) -> "RiskScorer":
        """Fit on ``train``; calibrate, weight and threshold on ``validation``.

        Both frames must be labelled (``is_fraud``/``isFraud``) and
        ``validation`` must be strictly later in time than ``train`` so that
        validation features only see the past.
        """

        from lightgbm import LGBMClassifier
        from sklearn.ensemble import IsolationForest
        from sklearn.linear_model import LogisticRegression
        from sklearn.utils.class_weight import compute_sample_weight

        train = normalize_transactions(train)
        validation = normalize_transactions(validation)
        for name, frame in (("train", train), ("validation", validation)):
            if "is_fraud" not in frame or frame["is_fraud"].nunique() < 2:
                raise ValueError(f"{name} split needs both fraud and normal labels")
        seed = self.config.random_state

        train_batch = engineer_features(train)
        y_train = train_batch.transactions["is_fraud"].to_numpy()
        self.supervised_model = LGBMClassifier(
            n_estimators=240,
            learning_rate=.05,
            deterministic=True,
            force_col_wise=True,
            n_jobs=1,
            random_state=seed,
            verbosity=-1,
        ).fit(
            train_batch.matrix,
            y_train,
            # PaySim fraud is ~0.1%; balanced weights keep the trees from
            # collapsing to the majority class (the calibrator restores scale).
            sample_weight=compute_sample_weight("balanced", y_train),
        )
        self.anomaly_model = IsolationForest(
            n_estimators=160, max_samples=1024, n_jobs=1, random_state=seed
        ).fit(train_batch.matrix)
        raw_anomaly = -self.anomaly_model.score_samples(train_batch.matrix)
        self.anomaly_low_ = float(np.quantile(raw_anomaly, ANOMALY_LOW_QUANTILE))
        self.anomaly_high_ = float(np.quantile(raw_anomaly, ANOMALY_HIGH_QUANTILE))
        self.amount_threshold_ = float(train["amount"].quantile(AMOUNT_SPIKE_QUANTILE))

        self.calibrator = None
        self.history_transactions_ = self._feature_history(train)
        self.is_fitted_ = True
        raw_validation = self._score_components(validation)
        y_validation = raw_validation["is_fraud"].to_numpy()
        self.calibrator = LogisticRegression(max_iter=400, random_state=seed).fit(
            _logit(raw_validation["supervised_score"].to_numpy()).reshape(-1, 1),
            y_validation,
        )
        components = self._score_components(validation)
        self._choose_weights(components, y_validation)

        final = self._weighted(components, self.learned_weights_)
        self.candidate_threshold_ = float(
            np.quantile(final, 1 - self.config.target_candidate_rate)
        )
        candidates = final >= self.candidate_threshold_
        frauds = max(1, int(y_validation.sum()))
        alpha, beta, gamma, delta = self.learned_weights_
        self.validation_stats_ = {
            "candidate_threshold": self.candidate_threshold_,
            "candidate_rate": float(candidates.mean()),
            "reduction_ratio": float(1 - candidates.mean()),
            "fraud_retention": float((candidates & (y_validation == 1)).sum() / frauds),
            "alpha_supervised": alpha,
            "beta_anomaly": beta,
            "gamma_graph": gamma,
            "delta_rules": delta,
            "operational_candidate_threshold": self.candidate_threshold_,
            "alert_hold_threshold": float(
                max(self.candidate_threshold_, np.quantile(final, ALERT_HOLD_QUANTILE))
            ),
        }
        self.history_transactions_ = self._feature_history(train, validation)
        return self

    def _choose_weights(self, components: pd.DataFrame, labels: np.ndarray) -> None:
        """Grid-search fusion weights: maximise fraud retention, then precision ranking."""

        if not self.config.optimize_weights_on_validation:
            self.learned_weights_ = self.config.normalized_weights
            return
        from sklearn.metrics import average_precision_score

        step = self.config.weight_grid_step
        rate = self.config.target_candidate_rate
        levels = np.round(np.arange(0, 1 + 1e-9, step), 6)
        best_key, best_weights = None, self.config.normalized_weights
        for alpha, beta, gamma in product(levels, repeat=3):
            delta = round(1 - alpha - beta - gamma, 6)
            if delta < -1e-9 or alpha <= 0:
                continue
            weights = (float(alpha), float(beta), float(gamma), float(max(0.0, delta)))
            score = self._weighted(components, weights)
            selected = score >= np.quantile(score, 1 - rate)
            if selected.mean() > 1.2 * rate:
                continue  # heavy ties: the threshold would not reduce the workload
            retention = float((selected & (labels == 1)).sum() / max(1, labels.sum()))
            key = (round(retention, 6), float(average_precision_score(labels, score)))
            if best_key is None or key > best_key:
                best_key, best_weights = key, weights
        self.learned_weights_ = best_weights

    # ------------------------------------------------------------------- scoring
    def score(
        self,
        frame: pd.DataFrame,
        candidate_rate_cap: float | None = None,
        *,
        update_history: bool = False,
    ) -> pd.DataFrame:
        components = self._score_components(frame)
        components["risk_score"] = np.clip(
            self._weighted(components, self.learned_weights_), 0, 1
        )
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

    @staticmethod
    def _weighted(
        components: pd.DataFrame, weights: tuple[float, float, float, float]
    ) -> np.ndarray:
        return (
            weights[0] * components["supervised_score"].to_numpy()
            + weights[1] * components["anomaly_score"].to_numpy()
            + weights[2] * components["graph_score"].to_numpy()
            + weights[3] * components["rule_score"].to_numpy()
        )

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
        rule = sum(weight * rules[column] for column, weight in RULE_WEIGHTS.items()).clip(0, 1)
        result = batch.transactions.copy()
        result["supervised_score"] = np.clip(supervised, 0, 1)
        result["anomaly_score"] = anomaly
        result["graph_score"] = graph["graph_score"].to_numpy()
        result["rule_score"] = rule.to_numpy()
        for column in rules:
            result[column] = rules[column].to_numpy()
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

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target, compress=3)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "RiskScorer":
        model = joblib.load(Path(path))
        if not isinstance(model, cls):
            raise TypeError("artifact is not a RiskScorer")
        return model


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))

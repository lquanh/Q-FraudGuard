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
from qf_guard.features import (
    FEATURE_GROUPS,
    _transaction_context,
    attribute_features,
    confirmed_contact_features,
    engineer_features,
    graph_risk_features,
)
from qf_guard.temporal_graph import TemporalGraphModel, TemporalNeighborhood


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
        self.temporal_graph_: TemporalGraphModel | None = None
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
        self.feature_groups_ = tuple(getattr(self.config, "feature_groups", ()) or ())
        unknown = set(self.feature_groups_) - set(FEATURE_GROUPS)
        if unknown:
            raise ValueError(f"unknown feature groups: {sorted(unknown)}")
        self.label_delay_minutes_ = int(getattr(self.config, "label_delay_minutes", 1440))

        train_batch = engineer_features(train)
        y_train = train_batch.transactions["is_fraud"].to_numpy()
        # labels of the training period only; each becomes usable after the delay
        self.confirmed_minute_ = _confirmed_minutes(train)
        train_matrix = self._model_matrix(train_batch, train_batch.transactions, None)
        params = dict(
            n_estimators=240,
            learning_rate=.05,
            deterministic=True,
            force_col_wise=True,
            n_jobs=1,
            random_state=seed,
            verbosity=-1,
        )
        params.update(getattr(self.config, "lightgbm_params", None) or {})
        self.supervised_model = LGBMClassifier(**params).fit(
            train_matrix,
            y_train,
            # PaySim fraud is ~0.1%; balanced weights keep the trees from
            # collapsing to the majority class (the calibrator restores scale).
            sample_weight=compute_sample_weight("balanced", y_train),
        )
        self.feature_columns_ = tuple(train_matrix.columns)
        self.anomaly_model = IsolationForest(
            n_estimators=160, max_samples=1024, n_jobs=1, random_state=seed
        ).fit(train_matrix)
        raw_anomaly = -self.anomaly_model.score_samples(train_matrix)
        self.anomaly_low_ = float(np.quantile(raw_anomaly, ANOMALY_LOW_QUANTILE))
        self.anomaly_high_ = float(np.quantile(raw_anomaly, ANOMALY_HIGH_QUANTILE))
        self.amount_threshold_ = float(train["amount"].quantile(AMOUNT_SPIKE_QUANTILE))

        self.calibrator = None
        epochs = getattr(self.config, "temporal_graph_epochs", 0)
        if epochs < 0:
            raise ValueError("temporal graph epochs must be nonnegative")
        self.temporal_graph_ = (
            TemporalGraphModel.fit(train, seed=self.config.random_state, epochs=epochs)
            if epochs else None
        )
        self.history_transactions_ = self._feature_history(train)
        self.is_fitted_ = True
        raw_validation = self._score_components(validation)
        y_validation = raw_validation["is_fraud"].to_numpy()
        self.calibrator = LogisticRegression(max_iter=400, random_state=seed).fit(
            _logit(raw_validation["supervised_score"].to_numpy()).reshape(-1, 1),
            y_validation,
        )
        self._calibrate_risk(self._score_components(validation))
        self.history_transactions_ = self._feature_history(train, validation)
        # from now on (test / live stream) validation labels are confirmed too
        self.confirmed_minute_ = _confirmed_minutes(train, validation)
        return self

    def _calibrate_risk(self, components: pd.DataFrame) -> None:
        y_validation = components["is_fraud"].to_numpy()
        self._choose_weights(components, y_validation)
        final = self._weighted(components, self.learned_weights_)
        target_recall = getattr(self.config, "target_recall", None)
        if target_recall:
            # lowest score that still keeps target_recall of validation frauds above it
            self.candidate_threshold_ = float(
                np.quantile(final[y_validation == 1], 1 - float(target_recall), method="lower")
            )
        else:
            self.candidate_threshold_ = float(
                np.quantile(final, 1 - self.config.target_candidate_rate)
            )
        candidates = final >= self.candidate_threshold_
        frauds = max(1, int(y_validation.sum()))
        alpha, beta, gamma, delta = self.learned_weights_
        from sklearn.metrics import average_precision_score, roc_auc_score

        manual_rate = getattr(self.config, "review_rate", .02)
        self.validation_stats_ = {
            "pr_auc": float(average_precision_score(y_validation, final)),
            "roc_auc": float(roc_auc_score(y_validation, final)),
            "candidate_threshold": self.candidate_threshold_,
            "candidate_rate": float(candidates.mean()),
            "reduction_ratio": float(1 - candidates.mean()),
            "fraud_retention": float((candidates & (y_validation == 1)).sum() / frauds),
            "alpha_supervised": alpha,
            "beta_anomaly": beta,
            "gamma_graph": gamma,
            "delta_rules": delta,
            "operational_candidate_threshold": self.candidate_threshold_,
            "review_threshold": review_threshold(final, self.candidate_threshold_, manual_rate),
            "review_rate_target": manual_rate,
            "validation_rows": len(components),
            "alert_hold_threshold": float(
                max(self.candidate_threshold_, np.quantile(final, ALERT_HOLD_QUANTILE))
            ),
        }
    # ----------------------------------------------------------------- features
    def _model_matrix(
        self,
        batch: Any,
        frame: pd.DataFrame,
        history: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Base features plus the optional groups this scorer was trained with."""

        groups = tuple(getattr(self, "feature_groups_", ()) or ())
        if not groups:
            return batch.matrix
        parts = [batch.matrix]
        attributes = [g for g in groups if g in ("attributes", "formats")]
        if attributes:
            parts.append(attribute_features(batch.transactions, tuple(attributes)).reset_index(drop=True))
        if "confirmed" in groups:
            context, mask, _ = _transaction_context(frame, history)
            labels = confirmed_contact_features(
                context, getattr(self, "confirmed_minute_", {}), self.label_delay_minutes_
            )
            if mask is not None:
                labels = labels.loc[mask]
            parts.append(labels.reset_index(drop=True))
        matrix = pd.concat(parts, axis=1)
        ordered = list(batch.matrix.columns) + [
            column for group in FEATURE_GROUPS if group in groups for column in FEATURE_GROUPS[group]
        ]
        return matrix.loc[:, ordered].astype(float)

    def _choose_weights(self, components: pd.DataFrame, labels: np.ndarray) -> None:
        """Grid-search fusion weights: maximise fraud retention, then precision ranking."""

        minimum = getattr(self.config, "min_tgnn_weight", 0.0)
        if not 0 <= minimum <= .95 or (minimum and "graph_model" not in components):
            raise ValueError("a minimum TGNN weight of 0–0.95 requires a trained TGNN")
        original = self.config.normalized_weights
        best_weights = original
        if minimum > original[2]:
            scale = (1 - minimum) / (1 - original[2])
            best_weights = (original[0] * scale, original[1] * scale, minimum, original[3] * scale)
        if not self.config.optimize_weights_on_validation:
            self.learned_weights_ = best_weights
            return
        from sklearn.metrics import average_precision_score

        step = self.config.weight_grid_step
        rate = self.config.target_candidate_rate
        levels = np.round(np.arange(0, 1 + 1e-9, step), 6)
        best_key = None
        for alpha, beta, gamma in product(levels, repeat=3):
            delta = round(1 - alpha - beta - gamma, 6)
            if delta < -1e-9 or alpha <= 0 or gamma < minimum:
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
        scored = self._decisions(components, candidate_rate_cap)
        if update_history:
            history = (frame,) if self.history_transactions_ is None else (self.history_transactions_, frame)
            self.history_transactions_ = self._feature_history(*history)
        return scored

    def _decisions(self, components, candidate_rate_cap=None):
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
        manual = self.validation_stats_.get("review_threshold", self.validation_stats_.get(
            "alert_hold_threshold", self.candidate_threshold_))
        components["manual_review"] = components["is_candidate"] & (components["risk_score"] >= manual)
        codes = [reason_codes(row) for row in components.to_dict(orient="records")]
        components["reason_codes"] = ["|".join(value) for value in codes]
        components["reason_text"] = [reason_text(value) for value in codes]
        components["risk_band"] = components["risk_score"].map(risk_band)
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
        matrix = self._model_matrix(batch, frame, self.history_transactions_)
        temporal_score = None
        if model := getattr(self, "temporal_graph_", None):
            transactions, mask, _ = _transaction_context(frame, self.history_transactions_)
            neighborhood = TemporalNeighborhood()
            predictions = []
            for start in range(0, len(transactions), 4096):
                stop = start + 4096
                chunk = transactions.iloc[start:stop]
                select = None if mask is None else mask[start:stop]
                if select is not None and not select.any():
                    neighborhood.warm(chunk)
                    continue
                values = model.predict(neighborhood.transform(chunk))
                predictions.append(values if select is None else values[select])
            temporal_score = np.concatenate(predictions)
        return self._predict_components(batch, graph, matrix, temporal_score=temporal_score)

    def _predict_components(self, batch, graph, matrix, *, temporal_score=None) -> pd.DataFrame:
        raw_probability = np.clip(
            self.supervised_model.predict_proba(matrix)[:, 1],
            1e-6,
            1 - 1e-6,
        )
        supervised = (
            self.calibrator.predict_proba(_logit(raw_probability).reshape(-1, 1))[:, 1]
            if self.calibrator is not None
            else raw_probability
        )
        raw_anomaly = -self.anomaly_model.score_samples(matrix)
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
        result["graph_score"] = graph["graph_score"].to_numpy() if temporal_score is None else temporal_score
        if temporal_score is not None:
            result["graph_model"] = "tgnn"
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
        if isinstance(model.config, ScoringConfig) and not hasattr(model.config, "review_rate"):
            model.config.review_rate = ScoringConfig().review_rate
        if isinstance(model.config, ScoringConfig) and not hasattr(model.config, "temporal_graph_epochs"):
            model.config.temporal_graph_epochs = 0
        if isinstance(model.config, ScoringConfig) and not hasattr(model.config, "min_tgnn_weight"):
            model.config.min_tgnn_weight = 0.0
        return model


def training_matrices(
    train: pd.DataFrame, validation: pd.DataFrame, config: ScoringConfig
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Exactly the supervised design matrices ``RiskScorer.fit`` builds.

    Used for fast hyper-parameter search: validation features see only the
    training history and only training-period labels, as inside ``fit``.
    """

    probe = RiskScorer(config)
    probe.feature_groups_ = tuple(config.feature_groups or ())
    probe.label_delay_minutes_ = int(config.label_delay_minutes)
    train = normalize_transactions(train)
    validation = normalize_transactions(validation)
    train_batch = engineer_features(train)
    probe.confirmed_minute_ = _confirmed_minutes(train)
    X_train = probe._model_matrix(train_batch, train_batch.transactions, None)
    history = RiskScorer._feature_history(train)
    validation_batch = engineer_features(validation, history=history)
    X_validation = probe._model_matrix(validation_batch, validation, history)
    return (
        X_train,
        train_batch.transactions["is_fraud"].to_numpy(),
        X_validation,
        validation_batch.transactions["is_fraud"].to_numpy(),
    )


def _confirmed_minutes(*frames: pd.DataFrame) -> dict[str, int]:
    """First minute each account appears in a labelled fraud/laundering transaction."""

    first: dict[str, int] = {}
    for frame in frames:
        data = normalize_transactions(frame)
        if "is_fraud" not in data:
            continue
        positive = data[data["is_fraud"] == 1]
        minutes = positive["timestamp"].astype("int64").to_numpy() // 60_000_000_000
        for column in ("sender_id", "receiver_id"):
            for account, minute in zip(positive[column].astype(str), minutes.tolist(), strict=True):
                if account not in first or minute < first[account]:
                    first[account] = int(minute)
    return first


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def review_threshold(validation_scores, candidate_threshold: float, rate: float) -> float:
    scores = np.asarray(validation_scores, dtype=float)
    if (not 0 < rate <= 1 or scores.ndim != 1 or not scores.size or not np.isfinite(scores).all()
            or (scores < 0).any() or (scores > 1).any()):
        raise ValueError("review rate must be in (0, 1] and validation scores must be finite")
    return max(candidate_threshold, float(np.quantile(scores, 1 - rate, method="higher")))

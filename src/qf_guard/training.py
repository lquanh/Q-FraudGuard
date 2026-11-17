"""Reproducible offline training for the realtime PaySim risk scorer."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from qf_guard.config import ScoringConfig
from qf_guard.data import normalize_transactions
from qf_guard.scoring import RiskScorer


def temporal_split(
    frame: pd.DataFrame, train_fraction: float = .6, validation_fraction: float = .2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/validation/test split (no shuffling, no look-ahead)."""

    data = normalize_transactions(frame)
    first = int(len(data) * train_fraction)
    second = int(len(data) * (train_fraction + validation_fraction))
    if not 0 < first < second < len(data):
        raise ValueError("split fractions leave an empty partition")
    return data.iloc[:first], data.iloc[first:second], data.iloc[second:]


def train_and_evaluate(
    frame: pd.DataFrame,
    *,
    target_candidate_rate: float = .05,
    random_state: int = 42,
) -> tuple[RiskScorer, dict[str, Any]]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    started = perf_counter()
    train, validation, test = temporal_split(frame)
    config = replace(
        ScoringConfig(), target_candidate_rate=target_candidate_rate, random_state=random_state
    )
    scorer = RiskScorer(config).fit(train, validation)
    scored = scorer.score(test)
    labels = scored["is_fraud"].to_numpy()
    candidates = scored["is_candidate"].to_numpy()
    detected = int((candidates & (labels == 1)).sum())
    metrics: dict[str, Any] = {
        "mode": "normal_temporal_paysim_lightgbm",
        "rows": int(len(train) + len(validation) + len(test)),
        "target_candidate_rate": target_candidate_rate,
        "split_rows": [len(train), len(validation), len(test)],
        "split_fraud": [int(part["is_fraud"].sum()) for part in (train, validation, test)],
        "candidate_threshold": scorer.candidate_threshold_,
        "learned_weights": list(scorer.learned_weights_),
        "validation_stats": scorer.validation_stats_,
        "test_pr_auc": float(average_precision_score(labels, scored["risk_score"])),
        "test_roc_auc": float(roc_auc_score(labels, scored["risk_score"])),
        "test_fraud_capture_rate": float(detected / max(1, labels.sum())),
        "test_precision": float(detected / max(1, candidates.sum())),
        "test_candidates": int(candidates.sum()),
        "test_candidate_rate": float(np.mean(candidates)),
        "test_fraud_detected": detected,
        "test_false_alerts": int(candidates.sum() - detected),
        "runtime_seconds": perf_counter() - started,
    }
    return scorer, metrics

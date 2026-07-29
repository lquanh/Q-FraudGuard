"""Leakage-conscious, explainable transaction and graph feature engineering."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qf_guard.data import CANONICAL_COLUMNS, normalize_transactions


TRANSACTION_TYPES = ("PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT")
FEATURE_COLUMNS = (
    "amount_log",
    "amount_to_balance",
    "sender_balance_delta",
    "receiver_balance_delta",
    "sender_balance_error",
    "receiver_balance_error",
    "velocity_1h",
    "velocity_24h",
    "receiver_novelty",
    "device_account_count",
    "sender_frequency",
    "receiver_frequency",
    "hour_sin",
    "hour_cos",
    "is_cross_border",
    *(f"type_{tx_type.lower()}" for tx_type in TRANSACTION_TYPES),
)


@dataclass(slots=True)
class FeatureBatch:
    transactions: pd.DataFrame
    matrix: pd.DataFrame


def engineer_features(
    frame: pd.DataFrame,
    history: pd.DataFrame | None = None,
) -> FeatureBatch:
    """Build causal numeric features, optionally carrying state from earlier records."""

    transactions, current_mask, current_transactions = _transaction_context(frame, history)
    old_sender = transactions["old_balance_sender"].to_numpy(dtype=float)
    new_sender = transactions["new_balance_sender"].to_numpy(dtype=float)
    old_receiver = transactions["old_balance_receiver"].to_numpy(dtype=float)
    new_receiver = transactions["new_balance_receiver"].to_numpy(dtype=float)
    amount = transactions["amount"].to_numpy(dtype=float)

    features = pd.DataFrame(index=transactions.index)
    features["amount_log"] = np.log1p(amount)
    features["amount_to_balance"] = amount / (old_sender + 1.0)
    features["sender_balance_delta"] = np.maximum(0.0, old_sender - new_sender) / (old_sender + 1.0)
    features["receiver_balance_delta"] = np.maximum(0.0, new_receiver - old_receiver) / (
        old_receiver + amount + 1.0
    )
    features["sender_balance_error"] = np.abs(old_sender - amount - new_sender) / (
        old_sender + amount + 1.0
    )
    features["receiver_balance_error"] = np.abs(old_receiver + amount - new_receiver) / (
        old_receiver + amount + 1.0
    )
    features["velocity_1h"] = _rolling_entity_count(transactions, "sender_id", 60)
    features["velocity_24h"] = _rolling_entity_count(transactions, "sender_id", 24 * 60)
    features["receiver_novelty"] = (
        ~transactions.duplicated(subset=["sender_id", "receiver_id"], keep="first")
    ).astype(float)
    device_accounts = _cumulative_unique_count(transactions, "device_id", "sender_id")
    features["device_account_count"] = np.log1p(device_accounts)
    features["sender_frequency"] = np.log1p(
        transactions.groupby("sender_id", sort=False).cumcount().to_numpy(dtype=float) + 1.0
    )
    features["receiver_frequency"] = np.log1p(
        transactions.groupby("receiver_id", sort=False).cumcount().to_numpy(dtype=float) + 1.0
    )
    hour = transactions["timestamp"].dt.hour + transactions["timestamp"].dt.minute / 60
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["is_cross_border"] = (~transactions["country"].isin(["VN", "UNKNOWN"])).astype(float)
    for tx_type in TRANSACTION_TYPES:
        features[f"type_{tx_type.lower()}"] = (transactions["type"] == tx_type).astype(float)

    features = features.loc[:, list(FEATURE_COLUMNS)].replace([np.inf, -np.inf], 0).fillna(0)
    if current_mask is not None:
        features = features.loc[current_mask].reset_index(drop=True)
        transactions = current_transactions
    return FeatureBatch(
        transactions=transactions.reset_index(drop=True),
        matrix=features.astype(float).reset_index(drop=True),
    )


def graph_risk_features(
    frame: pd.DataFrame,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create causal structure-only risk signals without using fraud labels."""

    transactions, current_mask, _ = _transaction_context(frame, history)
    sender_count = (
        transactions.groupby("sender_id", sort=False).cumcount().to_numpy(dtype=float) + 1.0
    )
    receiver_count = (
        transactions.groupby("receiver_id", sort=False).cumcount().to_numpy(dtype=float) + 1.0
    )
    sender_unique_receivers = _cumulative_unique_count(
        transactions, "sender_id", "receiver_id"
    )
    receiver_unique_senders = _cumulative_unique_count(
        transactions, "receiver_id", "sender_id"
    )
    shared_device_accounts = _cumulative_unique_count(
        transactions, "device_id", "sender_id"
    )
    flow_through = _receiver_seen_as_sender(transactions)
    rapid = _rolling_entity_count(transactions, "sender_id", 60)

    components = pd.DataFrame(
        {
            "fan_out": np.clip((sender_unique_receivers - 1.0) / 5.0, 0, 1),
            "fan_in": np.clip((receiver_unique_senders - 1.0) / 5.0, 0, 1),
            "sender_activity": np.clip((sender_count - 1.0) / 8.0, 0, 1),
            "receiver_activity": np.clip((receiver_count - 1.0) / 8.0, 0, 1),
            "shared_device": np.clip((shared_device_accounts - 1) / 4, 0, 1),
            "flow_through": flow_through,
            "rapid_activity": np.clip((rapid - 1) / 7, 0, 1),
        },
        index=transactions.index,
    )
    components["graph_score"] = (
        0.17 * components["fan_out"]
        + 0.13 * components["fan_in"]
        + 0.08 * components["sender_activity"]
        + 0.07 * components["receiver_activity"]
        + 0.27 * components["shared_device"]
        + 0.15 * components["flow_through"]
        + 0.13 * components["rapid_activity"]
    ).clip(0, 1)
    if current_mask is not None:
        components = components.loc[current_mask]
    return components.reset_index(drop=True)


def _transaction_context(
    frame: pd.DataFrame,
    history: pd.DataFrame | None,
) -> tuple[pd.DataFrame, np.ndarray | None, pd.DataFrame]:
    """Combine canonical history with current rows without mixing labels or metadata."""

    current = normalize_transactions(frame)
    if history is None or len(history) == 0:
        return current, None, current
    past = normalize_transactions(history)
    context_marker = "__qf_guard_current_context__"
    past_context = past.loc[:, list(CANONICAL_COLUMNS)].copy()
    current_context = current.loc[:, list(CANONICAL_COLUMNS)].copy()
    past_context[context_marker] = False
    current_context[context_marker] = True
    context = pd.concat((past_context, current_context), ignore_index=True)
    context = context.sort_values("timestamp", kind="stable").reset_index(drop=True)
    current_mask = context.pop(context_marker).to_numpy(dtype=bool)
    return context, current_mask, current


def _rolling_entity_count(
    frame: pd.DataFrame, entity_column: str, window_minutes: int
) -> np.ndarray:
    histories: dict[str, deque[int]] = defaultdict(deque)
    result = np.ones(len(frame), dtype=float)
    times = frame["timestamp"].astype("int64").to_numpy() // 60_000_000_000
    entities = frame[entity_column].astype(str).to_numpy()
    for position, (entity, minute) in enumerate(zip(entities, times, strict=True)):
        history = histories[entity]
        boundary = int(minute - window_minutes)
        while history and history[0] < boundary:
            history.popleft()
        history.append(int(minute))
        result[position] = len(history)
    return result


def _cumulative_unique_count(
    frame: pd.DataFrame, group_column: str, value_column: str
) -> np.ndarray:
    """Count distinct values seen so far without consulting future records."""

    seen: dict[str, set[str]] = defaultdict(set)
    result = np.empty(len(frame), dtype=float)
    groups = frame[group_column].astype(str).to_numpy()
    values = frame[value_column].astype(str).to_numpy()
    for position, (group, value) in enumerate(zip(groups, values, strict=True)):
        seen[group].add(value)
        result[position] = len(seen[group])
    return result


def _receiver_seen_as_sender(frame: pd.DataFrame) -> np.ndarray:
    """Mark flow-through only when the receiver has appeared as a sender already."""

    seen_senders: set[str] = set()
    result = np.zeros(len(frame), dtype=float)
    senders = frame["sender_id"].astype(str).to_numpy()
    receivers = frame["receiver_id"].astype(str).to_numpy()
    for position, (sender, receiver) in enumerate(zip(senders, receivers, strict=True)):
        seen_senders.add(sender)
        result[position] = float(receiver in seen_senders)
    return result

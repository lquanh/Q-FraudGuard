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


#: AMLworld payment formats (one-hot when the "amlworld" feature set is used).
AMLWORLD_FORMATS = ("ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire")
ATTRIBUTE_COLUMNS = ("is_cross_bank", "is_self_transfer", "currency_mismatch")
FORMAT_COLUMNS = tuple(f"fmt_{name.lower().replace(' ', '_')}" for name in AMLWORLD_FORMATS)
LABEL_COLUMNS = (
    "sender_confirmed",
    "receiver_confirmed",
    "sender_confirmed_contacts",
    "receiver_confirmed_contacts",
)
#: Optional feature groups on top of FEATURE_COLUMNS.
FEATURE_GROUPS = {
    "attributes": ATTRIBUTE_COLUMNS,
    "formats": FORMAT_COLUMNS,
    "confirmed": LABEL_COLUMNS,
}


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
    features = transaction_attributes(transactions)
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
    features = features.loc[:, list(FEATURE_COLUMNS)].replace([np.inf, -np.inf], 0).fillna(0)
    if current_mask is not None:
        features = features.loc[current_mask].reset_index(drop=True)
        transactions = current_transactions
    return FeatureBatch(
        transactions=transactions.reset_index(drop=True),
        matrix=features.astype(float).reset_index(drop=True),
    )


def transaction_attributes(transactions: pd.DataFrame) -> pd.DataFrame:
    """Features of the event itself, shared by offline and incremental scoring."""
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
    hour = transactions["timestamp"].dt.hour + transactions["timestamp"].dt.minute / 60
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["is_cross_border"] = (~transactions["country"].isin(["VN", "UNKNOWN"])).astype(float)
    for tx_type in TRANSACTION_TYPES:
        features[f"type_{tx_type.lower()}"] = (transactions["type"] == tx_type).astype(float)

    return features


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

    components = graph_components(sender_count, receiver_count, sender_unique_receivers,
                                  receiver_unique_senders, shared_device_accounts, flow_through, rapid)
    if current_mask is not None:
        components = components.loc[current_mask]
    return components.reset_index(drop=True)


def graph_components(sender_count, receiver_count, sender_unique_receivers,
                     receiver_unique_senders, shared_device_accounts, flow_through, rapid):
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
    return components


def _bank(account_ids: pd.Series) -> pd.Series:
    """Bank prefix of "bank:account" ids (AMLworld); empty when there is none."""

    parts = account_ids.astype(str).str.split(":", n=1)
    return parts.map(lambda p: p[0] if len(p) == 2 else "")


def attribute_features(transactions: pd.DataFrame, groups: tuple[str, ...]) -> pd.DataFrame:
    """Per-transaction attributes that need no history (AMLworld columns).

    Missing source columns give zeros, so a PaySim frame simply gets a block of
    zeros instead of an error.
    """

    out = pd.DataFrame(index=transactions.index)
    if "attributes" in groups:
        sender_bank, receiver_bank = _bank(transactions["sender_id"]), _bank(transactions["receiver_id"])
        out["is_cross_bank"] = ((sender_bank != receiver_bank) & (sender_bank != "")).astype(float)
        out["is_self_transfer"] = (transactions["sender_id"] == transactions["receiver_id"]).astype(float)
        if {"payment_currency", "receiving_currency"} <= set(transactions.columns):
            paid, received = transactions["payment_currency"], transactions["receiving_currency"]
            out["currency_mismatch"] = (paid.notna() & received.notna() & paid.ne(received)).astype(float)
        else:
            out["currency_mismatch"] = 0.0
    if "formats" in groups:
        fmt = transactions["payment_format"].astype(str) if "payment_format" in transactions else None
        for name, column in zip(AMLWORLD_FORMATS, FORMAT_COLUMNS, strict=True):
            out[column] = (fmt == name).astype(float) if fmt is not None else 0.0
    return out


def confirmed_contact_features(
    context: pd.DataFrame,
    confirmed_minute: dict[str, int],
    delay_minutes: int,
) -> pd.DataFrame:
    """Causal features from a list of *confirmed* laundering accounts (SIMO-style).

    ``confirmed_minute[account]`` is when the account was first involved in a
    laundering transaction; it only becomes known ``delay_minutes`` later.  For
    every row (in time order) this returns, using only information available
    before the row:

    * whether the sender / receiver is already on the confirmed list;
    * how many distinct confirmed accounts the sender / receiver has dealt with.
    """

    n = len(context)
    minutes = context["timestamp"].astype("int64").to_numpy() // 60_000_000_000
    senders = context["sender_id"].astype(str).to_numpy()
    receivers = context["receiver_id"].astype(str).to_numpy()
    activation = sorted((minute + delay_minutes, account) for account, minute in confirmed_minute.items())
    active: set[str] = set()
    neighbours: dict[str, set[str]] = defaultdict(set)
    counted: dict[str, set[str]] = defaultdict(set)
    s_conf = np.zeros(n)
    r_conf = np.zeros(n)
    s_contacts = np.zeros(n)
    r_contacts = np.zeros(n)
    pointer = 0

    def activate(account: str) -> None:
        active.add(account)
        for other in neighbours.get(account, ()):
            counted[other].add(account)

    order = np.argsort(minutes, kind="stable")
    for position in order.tolist():
        now = minutes[position]
        while pointer < len(activation) and activation[pointer][0] <= now:
            activate(activation[pointer][1])
            pointer += 1
        sender, receiver = senders[position], receivers[position]
        s_conf[position] = float(sender in active)
        r_conf[position] = float(receiver in active)
        s_contacts[position] = len(counted.get(sender, ()))
        r_contacts[position] = len(counted.get(receiver, ()))
        if sender != receiver:
            neighbours[sender].add(receiver)
            neighbours[receiver].add(sender)
            if receiver in active:
                counted[sender].add(receiver)
            if sender in active:
                counted[receiver].add(sender)
    return pd.DataFrame(
        {
            "sender_confirmed": s_conf,
            "receiver_confirmed": r_conf,
            "sender_confirmed_contacts": np.log1p(s_contacts),
            "receiver_confirmed_contacts": np.log1p(r_contacts),
        },
        index=context.index,
    )


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

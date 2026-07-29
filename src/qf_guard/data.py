from __future__ import annotations

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = (
    "transaction_id",
    "timestamp",
    "type",
    "amount",
    "sender_id",
    "receiver_id",
    "old_balance_sender",
    "new_balance_sender",
    "old_balance_receiver",
    "new_balance_receiver",
    "device_id",
    "country",
)

PAYSIM_ALIASES = {
    "nameOrig": "sender_id",
    "nameDest": "receiver_id",
    "oldbalanceOrg": "old_balance_sender",
    "newbalanceOrig": "new_balance_sender",
    "oldbalanceDest": "old_balance_receiver",
    "newbalanceDest": "new_balance_receiver",
    "isFraud": "is_fraud",
}


def normalize_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("transaction data must not be empty")
    normalized = frame.rename(columns=PAYSIM_ALIASES).copy()
    count = len(normalized)
    if "transaction_id" not in normalized:
        normalized["transaction_id"] = [f"TX-{index:08d}" for index in range(count)]
    normalized["transaction_id"] = normalized["transaction_id"].astype(str)
    if "timestamp" not in normalized:
        if "step" in normalized:
            normalized["timestamp"] = pd.Timestamp("2026-01-01T00:00:00Z") + pd.to_timedelta(
                normalized["step"], unit="h"
            )
        else:
            normalized["timestamp"] = pd.date_range(
                "2026-01-01", periods=count, freq="min", tz="UTC"
            )
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="coerce")
    if normalized["timestamp"].isna().any():
        raise ValueError("timestamp contains invalid values")
    defaults = {
        "type": "TRANSFER",
        "device_id": "UNKNOWN_DEVICE",
        "country": "UNKNOWN",
        "old_balance_sender": 0.0,
        "new_balance_sender": 0.0,
        "old_balance_receiver": 0.0,
        "new_balance_receiver": 0.0,
    }
    for column, default in defaults.items():
        if column not in normalized:
            normalized[column] = default
    required = ("amount", "sender_id", "receiver_id")
    missing = [column for column in required if column not in normalized]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    numeric = (
        "amount",
        "old_balance_sender",
        "new_balance_sender",
        "old_balance_receiver",
        "new_balance_receiver",
    )
    for column in numeric:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        if normalized[column].isna().any():
            raise ValueError(f"{column} contains invalid values")
        normalized[column] = normalized[column].clip(lower=0).astype(float)
    normalized["type"] = normalized["type"].fillna("UNKNOWN").astype(str).str.upper()
    for column in ("sender_id", "receiver_id", "device_id", "country"):
        normalized[column] = normalized[column].fillna("UNKNOWN").astype(str)
    if "is_fraud" in normalized:
        normalized["is_fraud"] = pd.to_numeric(normalized["is_fraud"], errors="coerce")
        labels = normalized["is_fraud"].to_numpy(dtype=float)
        if normalized["is_fraud"].isna().any() or not np.isin(labels, (0, 1)).all():
            raise ValueError("is_fraud must contain binary values")
        normalized["is_fraud"] = normalized["is_fraud"].astype(int)
    normalized = normalized.sort_values("timestamp", kind="stable").reset_index(drop=True)
    duplicates = normalized["transaction_id"].duplicated(keep=False)
    if duplicates.any():
        normalized.loc[duplicates, "transaction_id"] = [
            f"{value}-{index}"
            for index, value in zip(
                normalized.index[duplicates],
                normalized.loc[duplicates, "transaction_id"],
                strict=True,
            )
        ]
    return normalized

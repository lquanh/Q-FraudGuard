"""Dataset adapters into the canonical transaction schema of :mod:`qf_guard.data`.

IBM AMLworld (Altman et al., NeurIPS 2023 Datasets & Benchmarks) is a
*synthetic* multi-bank transaction dataset with laundering labels, licensed
CDLA-Sharing-1.0.  Unlike PaySim it has repeated accounts, so suspicious
candidates in one window share senders/receivers and the investigation QUBO
gets real pairwise terms.

The adapter keeps only what the scorer uses.  Two approximations are explicit:

* amounts are converted to USD with fixed, approximate 2022 rates (the
  dataset mixes 15 currencies; only relative scale matters to the scorer);
* AMLworld payment formats are mapped onto the five canonical types.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

__all__ = [
    "AMLWORLD_USD_RATES",
    "AMLWORLD_TYPE_MAP",
    "amlworld_to_canonical",
    "read_amlworld_days",
]

#: Approximate USD value of one unit of each AMLworld currency (2022 levels).
AMLWORLD_USD_RATES: dict[str, float] = {
    "US Dollar": 1.0,
    "Euro": 1.0,
    "Swiss Franc": 1.05,
    "Yuan": 0.145,
    "Shekel": 0.29,
    "Rupee": 0.0125,
    "UK Pound": 1.17,
    "Yen": 0.0072,
    "Ruble": 0.016,
    "Bitcoin": 20_000.0,
    "Canadian Dollar": 0.76,
    "Australian Dollar": 0.68,
    "Mexican Peso": 0.050,
    "Saudi Riyal": 0.266,
    "Brazil Real": 0.19,
}

#: AMLworld "Payment Format" -> canonical transaction type.
AMLWORLD_TYPE_MAP: dict[str, str] = {
    "Credit Card": "PAYMENT",
    "Cash": "CASH_OUT",
    "Reinvestment": "CASH_IN",
    "Cheque": "DEBIT",
    "ACH": "TRANSFER",
    "Wire": "TRANSFER",
    "Bitcoin": "TRANSFER",
}


def amlworld_to_canonical(frame: pd.DataFrame, *, id_offset: int = 0) -> pd.DataFrame:
    """Map raw AMLworld ``*_Trans.csv`` rows onto the canonical columns."""

    required = {
        "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
        "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing AMLworld columns: {', '.join(sorted(missing))}")
    unknown = set(frame["Payment Currency"].unique()) - set(AMLWORLD_USD_RATES)
    if unknown:
        raise ValueError(f"no USD rate for currencies: {', '.join(sorted(map(str, unknown)))}")
    rate = frame["Payment Currency"].map(AMLWORLD_USD_RATES).astype(float)
    canonical = pd.DataFrame(
        {
            "transaction_id": [f"AML-{id_offset + i:09d}" for i in range(len(frame))],
            "timestamp": pd.to_datetime(frame["Timestamp"], format="%Y/%m/%d %H:%M", utc=True),
            "type": frame["Payment Format"].map(AMLWORLD_TYPE_MAP).fillna("TRANSFER"),
            "amount": (frame["Amount Paid"].astype(float) * rate).round(2),
            "sender_id": frame["From Bank"].astype(str) + ":" + frame["Account"].astype(str),
            "receiver_id": frame["To Bank"].astype(str) + ":" + frame["Account.1"].astype(str),
            "is_fraud": frame["Is Laundering"].astype(int),
            "payment_format": frame["Payment Format"].astype(str),
            "payment_currency": frame["Payment Currency"].astype(str),
            "receiving_currency": (
                frame["Receiving Currency"].astype(str)
                if "Receiving Currency" in frame
                else frame["Payment Currency"].astype(str)
            ),
        }
    )
    return canonical


def read_amlworld_days(
    archive: str | Path,
    member: str = "HI-Small_Trans.csv",
    *,
    start_day: str = "2022/09/03",
    days: int = 2,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Stream ``member`` out of the zip (never extracted) and keep ``days`` days."""

    first = pd.Timestamp(start_day.replace("/", "-"))
    wanted = {(first + pd.Timedelta(days=k)).strftime("%Y/%m/%d") for k in range(days)}
    parts: list[pd.DataFrame] = []
    offset = 0
    with zipfile.ZipFile(archive) as bundle, bundle.open(member) as handle:
        for chunk in pd.read_csv(handle, chunksize=chunksize):
            keep = chunk[chunk["Timestamp"].str[:10].isin(wanted)]
            if len(keep):
                parts.append(amlworld_to_canonical(keep, id_offset=offset))
                offset += len(keep)
    if not parts:
        raise ValueError(f"no rows for {sorted(wanted)} in {member}")
    return pd.concat(parts, ignore_index=True).sort_values("timestamp", kind="stable").reset_index(drop=True)

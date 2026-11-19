"""Incremental, event-time feature computation for a chronological stream."""

from collections import Counter, defaultdict, deque

import numpy as np
import pandas as pd

from qf_guard.features import (
    FEATURE_COLUMNS, LABEL_COLUMNS, FeatureBatch, attribute_features, graph_components,
    transaction_attributes,
)


class OnlineFeatures:
    def __init__(self, confirmed: dict[str, int], delay: int, groups: tuple[str, ...]):
        self.groups = groups
        self.activation = {key: minute + delay for key, minute in confirmed.items()} if "confirmed" in groups else {}
        self.contacts = defaultdict(set)
        self.sent = Counter()
        self.received = Counter()
        self.payees = defaultdict(set)
        self.fan_in = Counter()
        self.devices = defaultdict(set)
        self.hour = defaultdict(deque)
        self.day = defaultdict(deque)
        self.last_time = None

    def _pair(self, sender, receiver):
        partners = self.payees[sender]
        if receiver in partners:
            return False
        partners.add(receiver)
        self.fan_in[receiver] += 1
        if self.activation and sender != receiver:
            if sender in self.activation:
                self.contacts[receiver].add(sender)
            if receiver in self.activation:
                self.contacts[sender].add(receiver)
        return True

    def _minutes(self, frame):
        times = frame["timestamp"]
        if not times.is_monotonic_increasing or (self.last_time is not None and times.iloc[0] < self.last_time):
            raise ValueError("event timestamp precedes the stream's committed history")
        return times.array.as_unit("ns").asi8 // 60_000_000_000

    @staticmethod
    def _visit(window, minute, width):
        while window and window[0] < minute - width:
            window.popleft()
        window.append(minute)
        return len(window)

    def warm(self, history: pd.DataFrame) -> None:
        """Restore sufficient statistics without predicting or retaining raw transactions."""
        if history.empty:
            return
        minutes = self._minutes(history)
        self.sent.update(history.sender_id.value_counts(sort=False).to_dict())
        self.received.update(history.receiver_id.value_counts(sort=False).to_dict())
        for sender, receiver in history[["sender_id", "receiver_id"]].drop_duplicates().itertuples(index=False, name=None):
            self._pair(sender, receiver)
        for device, sender in history[["device_id", "sender_id"]].drop_duplicates().itertuples(index=False, name=None):
            self.devices[device].add(sender)
        # Earlier visits cannot contribute to any subsequent event's rolling window.
        for windows, width in ((self.hour, 60), (self.day, 1440)):
            start = minutes.searchsorted(minutes[-1] - width)
            for sender, minute in zip(history.sender_id.iloc[start:], minutes[start:].tolist(), strict=True):
                self._visit(windows[sender], minute, width)
        self.last_time = history.timestamp.iloc[-1]

    def transform(self, frame: pd.DataFrame):
        minutes = self._minutes(frame)
        n = len(frame)
        counts = np.empty((n, 9), dtype=float)
        labels = np.zeros((n, len(LABEL_COLUMNS)), dtype=float)
        for i, (sender, receiver, device, minute) in enumerate(zip(
            frame.sender_id, frame.receiver_id, frame.device_id, minutes.tolist(), strict=True
        )):
            if self.activation:
                for j, account in enumerate((sender, receiver)):
                    labels[i, j] = account in self.activation and self.activation[account] <= minute
                    labels[i, j + 2] = np.log1p(sum(
                        self.activation[a] <= minute for a in self.contacts.get(account, ())
                    ))
            novel = self._pair(sender, receiver)
            self.sent[sender] += 1
            self.received[receiver] += 1
            self.devices[device].add(sender)
            counts[i] = (
                self.sent[sender], self.received[receiver], len(self.payees[sender]),
                self.fan_in[receiver], len(self.devices[device]), receiver in self.sent,
                self._visit(self.hour[sender], minute, 60),
                self._visit(self.day[sender], minute, 1440), novel,
            )
        self.last_time = frame.timestamp.iloc[-1]
        sent, received, fan_out, fan_in, devices, flow, hour, day, novelty = counts.T
        base = transaction_attributes(frame).assign(
            velocity_1h=hour, velocity_24h=day, receiver_novelty=novelty,
            device_account_count=np.log1p(devices), sender_frequency=np.log1p(sent),
            receiver_frequency=np.log1p(received),
        ).loc[:, list(FEATURE_COLUMNS)].replace([np.inf, -np.inf], 0).fillna(0)
        graph = graph_components(sent, received, fan_out, fan_in, devices, flow, hour)
        parts = [base]
        if self.groups:
            parts.append(attribute_features(frame, self.groups))
        if "confirmed" in self.groups:
            parts.append(pd.DataFrame(labels, columns=LABEL_COLUMNS))
        return FeatureBatch(frame, base), graph, pd.concat(parts, axis=1)

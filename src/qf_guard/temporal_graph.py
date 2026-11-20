"""Temporal message passing over recent incident edges of both transaction accounts."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

ACCOUNT_COLUMNS = ["sender_id", "receiver_id", "amount"]
NEIGHBORS = 8
MESSAGE_SIZE = 4
MAX_AGE_SECONDS = 7 * 24 * 3600
SCALE = 16.0


class TemporalNeighborhood:
    """The ordered stream prefix, independent of model weights and transaction labels."""

    def __init__(self):
        self.edges: dict[str, deque[tuple[int, str, float, float]]] = {}
        self.last_time: int | None = None

    def _read(self, events: pd.DataFrame, *, collect: bool):
        count = len(events)
        if not count:
            return None
        times = events["timestamp"].array.as_unit("ns").asi8
        if (times[1:] < times[:-1]).any() or (self.last_time is not None and times[0] < self.last_time):
            raise ValueError("event timestamp precedes the temporal graph history")
        rows = events[ACCOUNT_COLUMNS].itertuples(index=False, name=None)
        if collect:
            edge = np.zeros((count, 2, NEIGHBORS, MESSAGE_SIZE), dtype=np.float32)
            present = np.zeros((count, 2, NEIGHBORS), dtype=bool)
            current = np.zeros((count, 2), dtype=np.float32)
        for i, ((sender, receiver, amount), timestamp) in enumerate(zip(rows, times, strict=True)):
            stamp = int(timestamp)
            value = float(np.log1p(amount)) / SCALE
            if collect:
                current[i] = value, float(sender == receiver)
                for endpoint, (account, peer) in enumerate(((sender, receiver), (receiver, sender))):
                    for j, (seen, neighbor, size, direction) in enumerate(reversed(self.edges.get(account, ()))):
                        age = (stamp - seen) / 1e9
                        if age > MAX_AGE_SECONDS:
                            break
                        edge[i, endpoint, j] = size, np.log1p(age) / SCALE, direction, float(neighbor == peer)
                        present[i, endpoint, j] = True
            if sender not in self.edges:
                self.edges[sender] = deque(maxlen=NEIGHBORS)
            self.edges[sender].append((stamp, receiver, value, 1.0))
            if receiver != sender:
                if receiver not in self.edges:
                    self.edges[receiver] = deque(maxlen=NEIGHBORS)
                self.edges[receiver].append((stamp, sender, value, -1.0))
        self.last_time = int(times[-1])
        return (edge, present, current) if collect else None

    def warm(self, events: pd.DataFrame) -> None:
        self._read(events, collect=False)

    def transform(self, events: pd.DataFrame):
        if events.empty:
            raise ValueError("temporal graph requires at least one transaction")
        return self._read(events, collect=True)


@dataclass(slots=True)
class TemporalGraphModel:
    """A shared edge MLP, masked mean aggregation, and a transaction classifier.

    Store only NumPy weights so a trained artifact does not require PyTorch in serving.
    """

    edge_weight: np.ndarray
    edge_bias: np.ndarray
    hidden_weight: np.ndarray
    hidden_bias: np.ndarray
    output_weight: np.ndarray
    output_bias: np.ndarray

    def predict(self, features) -> np.ndarray:
        edge, present, current = features
        messages = np.maximum(edge @ self.edge_weight + self.edge_bias, 0)
        mean = (messages * present[..., None]).sum(axis=2)
        mean /= np.maximum(present.sum(axis=2, keepdims=True), 1)
        joined = np.concatenate((current, mean.reshape(len(current), -1)), axis=1)
        hidden = np.maximum(joined @ self.hidden_weight + self.hidden_bias, 0)
        logits = (hidden @ self.output_weight + self.output_bias).ravel()
        return 1 / (1 + np.exp(-np.clip(logits, -30, 30)))

    @classmethod
    def fit(cls, transactions: pd.DataFrame, *, seed: int, epochs: int):
        labels = transactions["is_fraud"].to_numpy(dtype=np.float32)
        if epochs < 1 or set(np.unique(labels)) != {0, 1}:
            raise ValueError("TGNN training requires positive epochs and both binary classes")
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError("Install qf-guard[tgnn] to train the TGNN.") from exc

        edge_layer = nn.Linear(MESSAGE_SIZE, 8)
        hidden_layer = nn.Linear(2 + 2 * edge_layer.out_features, 16)
        output_layer = nn.Linear(hidden_layer.out_features, 1)
        layers = nn.ModuleList((edge_layer, hidden_layer, output_layer))
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for layer in layers:
                bound = layer.in_features ** -0.5
                layer.weight.uniform_(-bound, bound, generator=generator)
                layer.bias.uniform_(-bound, bound, generator=generator)
        optimizer = torch.optim.Adam(layers.parameters(), lr=.005)
        rate = float(labels.mean())
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - rate) / rate))
        for _ in range(epochs):
            history = TemporalNeighborhood()
            for start in range(0, len(transactions), 2048):
                stop = start + 2048
                edge, present, current = map(torch.from_numpy, history.transform(transactions.iloc[start:stop]))
                messages = edge_layer(edge).relu() * present.unsqueeze(-1)
                mean = messages.sum(dim=2) / present.sum(dim=2, keepdim=True).clamp(min=1)
                joined = torch.cat((current, mean.flatten(start_dim=1)), dim=1)
                logits = output_layer(hidden_layer(joined).relu()).flatten()
                loss = criterion(logits, torch.from_numpy(labels[start:stop]))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return cls(*(tensor.detach().numpy().copy()
                     for layer in layers for tensor in (layer.weight.T, layer.bias)))

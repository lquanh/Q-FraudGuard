"""Measure CPU selection on prepared AMLworld QUBOs and check embeddings on ideal Z12."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from time import perf_counter_ns

import networkx as nx
import numpy as np
import pandas as pd
from dwave.embedding.diagnostic import is_valid_embedding
from dwave.graphs import zephyr_graph
from minorminer.busclique import find_clique_embedding
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qf_guard.postprocess import swap_local_search  # noqa: E402
from qf_guard.prioritization import PriorityConfig, build_priority_qubo  # noqa: E402
from qf_guard.qubo import QUBOModel  # noqa: E402

SIZES = (40, 48, 64, 80, 96, 112, 128, 144, 160, 176, 184)
POOL_SIZE, WINDOWS, REPEATS = 1920, 5, 7


def select_by_swap_gain(model: QUBOModel) -> np.ndarray:
    """Start at top-K and take the best improving swap for the investigation reward."""
    risks = np.asarray(model.metadata["risks"])
    selected = np.zeros(model.n_variables, dtype=bool)
    selected[np.argsort(-risks, kind="stable")[:model.exact_k]] = True
    weights = np.zeros((model.n_variables, model.n_variables))
    edges = model.metadata["suspicious_edges"]
    if edges:
        left, right, weight = np.asarray(edges).T
        left, right = left.astype(int), right.astype(int)
        weights[left, right] = weights[right, left] = model.metadata["connectivity_strength"] * weight

    marginal = risks + weights[:, selected].sum(axis=1)
    for _ in range(1000):
        inside, outside = np.flatnonzero(selected), np.flatnonzero(~selected)
        # A newly selected alert cannot earn the link to the one it replaces.
        gains = marginal[outside][None, :] - marginal[inside][:, None] - weights[np.ix_(inside, outside)]
        best = int(np.argmax(gains))
        if gains.flat[best] <= 1e-12:
            return selected.astype(np.int8)
        row, col = np.unravel_index(best, gains.shape)
        leave, enter = inside[row], outside[col]
        selected[leave], selected[enter] = False, True
        marginal += weights[:, enter] - weights[:, leave]
    raise RuntimeError("Swap search reached its iteration limit")


def benchmark() -> list[dict]:
    training = ROOT / "artifacts" / "training"
    settings = json.loads((training / "amlworld_full_operating_points.json").read_text())
    scores = pd.read_pickle(training / "full_scores_test.pkl")
    flagged = scores.loc[scores.risk_score >= settings["0.96"]["threshold"]]
    pool = flagged.sort_values("timestamp", kind="stable").head(POOL_SIZE)
    if len(pool) != POOL_SIZE or not pool.transaction_id.is_unique:
        raise ValueError("Expected 1,920 distinct screened transactions")
    records = pool[["transaction_id", "sender_id", "receiver_id", "risk_score"]].to_dict("records")
    config = PriorityConfig(risk_normalization="max", connectivity_strength=0.01)
    graph = zephyr_graph(12)
    rows, models, controls = [], [], []
    for n in SIZES:
        embedding = find_clique_embedding(n, graph, seed=42, use_cache=False)
        if set(embedding) != set(range(n)) or not is_valid_embedding(embedding, nx.complete_graph(n), graph):
            raise ValueError(f"No verified embedding for K{n}")
        physical_qubits = sum(map(len, embedding.values()))
        full_windows = len(records) // n
        for j in range(WINDOWS):
            window = ((2 * j + 1) * full_windows) // (2 * WINDOWS)
            group = records[window * n : (window + 1) * n]
            model = build_priority_qubo(group, config)
            models.append(model)
            rows.append(dict(n=n, k=model.exact_k, physical_qubits=physical_qubits,
                             window_index=window, first_transaction_id=group[0]["transaction_id"]))
            if j == 0 and n in (40, 80, 128, 184):
                controls.append(model)

    order = [(i, repeat) for i in range(len(rows)) for repeat in range(1, REPEATS + 1)]
    random.Random(184160).shuffle(order)
    with threadpool_limits(limits=1, user_api="blas"):
        for model in models:
            select_by_swap_gain(model)
        for index, repeat in order:
            model = models[index]
            start = perf_counter_ns()
            selected = select_by_swap_gain(model)
            elapsed = perf_counter_ns() - start
            if not model.is_feasible(selected):
                raise ValueError("CPU selection violates exact K")
            rows[index][f"gain_swap_{repeat}_ns"] = elapsed
        for model in controls:
            initial = np.zeros(model.n_variables, dtype=np.int8)
            initial[np.argsort(-np.asarray(model.metadata["risks"]), kind="stable")[:model.exact_k]] = 1
            expected = model.evaluate(swap_local_search(model, initial))
            actual = model.evaluate(select_by_swap_gain(model))
            if not np.isclose(expected, actual, atol=1e-7, rtol=0):
                raise ValueError(f"CPU control disagrees at N={model.n_variables}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="CSV for per-window measurements")
    args = parser.parse_args()
    result = benchmark()
    with args.output.open("w", encoding="utf-8", newline="") as output:
        fields = ["n", "k", "physical_qubits", "window_index", "first_transaction_id"]
        fields += [f"gain_swap_{i}_ns" for i in range(1, REPEATS + 1)]
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)

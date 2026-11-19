"""Time one CPU solve and estimate one D-Wave submission locally, from a prepared QUBO."""

from __future__ import annotations

import argparse
import random
import sys
from math import comb, isclose
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from dwave.cloud.solver import StructuredSolver
from dwave.cloud.testing.mocks import qpu_pegasus_solver_data, qpu_problem_timing_data
from dwave.graphs import pegasus_graph
from minorminer.busclique import find_clique_embedding

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qpu_check import candidate_windows  # noqa: E402
from qf_guard.postprocess import swap_local_search  # noqa: E402
from qf_guard.prioritization import PriorityConfig, build_priority_qubo, exact_optimum  # noqa: E402
from qf_guard.solvers import ExactSolver  # noqa: E402

SIZES = (*range(8, 25), 32, 48, 64, 128)
WINDOWS, REPEATS, SEED = 10, 3, 20261120


def time_cpu(model, enumerator):
    """Each call uses the same prepared model. Keep validation outside the clocks."""
    start = perf_counter_ns()
    initial = np.zeros(model.n_variables, dtype=np.int8)
    order = np.argsort(-np.asarray(model.metadata["risks"]), kind="stable")
    initial[order[:model.exact_k]] = 1
    swap_local_search(model, initial)
    swap_ms = (perf_counter_ns() - start) / 1e6

    start = perf_counter_ns()
    _, certificate = exact_optimum(model)
    milp_ms = (perf_counter_ns() - start) / 1e6
    if certificate["status"] != "optimal":
        raise ValueError(f"MILP failed: {certificate['status']}")

    enumeration_ms = None
    if comb(model.n_variables, model.exact_k) <= enumerator.max_states:
        start = perf_counter_ns()
        result = enumerator.solve(model)
        enumeration_ms = (perf_counter_ns() - start) / 1e6
        if not isclose(result.objective, certificate["energy"], abs_tol=1e-9):
            raise ValueError("Enumeration and MILP disagree")
    return {"cpu_enumeration_ms": enumeration_ms, "cpu_milp_ms": milp_ms, "cpu_swap_ms": swap_ms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=ROOT / "artifacts/training/full_scores_test.pkl")
    parser.add_argument("--output", type=Path, help="write CSV here (defaults to standard output)")
    parser.add_argument("--order", choices=("both", "by-size", "interleaved"), default="both",
                        help="order in which the CPU calls are timed")
    args = parser.parse_args()
    scores = pd.read_pickle(args.scores)
    config = PriorityConfig(risk_normalization="max", connectivity_strength=0.01)
    enumerator = ExactSolver()
    graph = pegasus_graph(16)
    profile = qpu_problem_timing_data("advantage")  # Advantage_system4.1, August 2022.
    spec = qpu_pegasus_solver_data(16)
    spec["properties"]["problem_timing_data"] = profile
    solver = StructuredSolver(client=None, data=spec)
    inputs = {}
    for n in SIZES:
        embedding = find_clique_embedding(n, graph, seed=42, use_cache=False)
        if len(embedding) != n:
            raise ValueError(f"Could not embed all {n} logical qubits")
        physical = len({qubit for chain in embedding.values() for qubit in chain})
        qpu_ms = solver.estimate_qpu_access_time(physical, num_reads=1, annealing_time=20) / 1000
        windows = candidate_windows(scores, 0.00016, n, WINDOWS, SEED)
        if len(windows) != WINDOWS:
            raise ValueError(f"Expected {WINDOWS} linked windows for N={n}")
        for window_id, window in enumerate(windows):
            model = build_priority_qubo(window, config)
            inputs[n, window_id] = model, window[0]["transaction_id"], physical, qpu_ms

    rows = []
    orders = ("by-size", "interleaved") if args.order == "both" else (args.order,)
    for order in orders:
        schedule = [(n, w, r) for n, w in inputs for r in range(1, REPEATS + 1)]
        if order == "interleaved":
            random.Random(SEED).shuffle(schedule)
        time_cpu(inputs[SIZES[0], 0][0], enumerator)  # warm up all CPU solvers
        for step, (n, window_id, repeat) in enumerate(schedule, start=1):
            model, first_id, physical, qpu_ms = inputs[n, window_id]
            rows.append({
                "order": order, "step": step, "n": n, "k": model.exact_k,
                "window": window_id, "repeat": repeat, "first_transaction_id": first_id,
                "physical_qubits": physical, "qpu_access_estimate_ms": qpu_ms,
                **time_cpu(model, enumerator),
            })
            if step % 70 == 0:
                print(f"{order}: {step}/{len(schedule)} calls", file=sys.stderr, flush=True)
    pd.DataFrame(rows).to_csv(args.output or sys.stdout, index=False,
                              float_format="%.7f", lineterminator="\n")


if __name__ == "__main__":
    main()

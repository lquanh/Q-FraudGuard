"""Sample selection shared by every QUBO backend (CPU, SQA, QPU)."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Mapping

import numpy as np

from qf_guard.qubo import QUBOModel
from qf_guard.results import SolverResult


def repair_exact_k(model: QUBOModel, sample: np.ndarray) -> np.ndarray:
    """Greedily move an infeasible sample onto the exact-K hyperplane.

    Each step flips the single variable that lowers the QUBO energy the most
    among the flips that move the selection count towards ``k``.  Hardware
    reads (for example with broken chains) can return ``k±1`` selections; this
    keeps those reads usable while reporting that a repair happened.
    """

    current = model.validate_sample(sample).astype(np.int8)
    if model.exact_k is None:
        return current
    k = model.exact_k
    while int(current.sum()) != k:
        adding = int(current.sum()) < k
        flippable = np.flatnonzero(current == (0 if adding else 1))
        trials = np.repeat(current[None, :], len(flippable), axis=0)
        trials[np.arange(len(flippable)), flippable] = 1 if adding else 0
        energies = model.evaluate_many(trials)
        current = trials[int(np.argmin(energies))]
    return current


def swap_local_search(
    model: QUBOModel, sample: np.ndarray, max_iterations: int = 1_000
) -> np.ndarray:
    """Steepest-descent over single swaps (one 1→0 and one 0→1), keeping exact-K.

    Feasible exact-K states are two bit flips apart, so single-flip annealers
    (SQA, SA and the QPU alike) must cross a penalty barrier between them.
    This classical polish only ever lowers the energy of a feasible sample.
    """

    current = model.validate_sample(sample).astype(np.int8)
    if model.exact_k is None or not model.is_feasible(current):
        return current
    energy = model.evaluate(current)
    for _ in range(max_iterations):
        ones, zeros = np.flatnonzero(current == 1), np.flatnonzero(current == 0)
        if len(ones) == 0 or len(zeros) == 0:
            break
        pairs = np.array(np.meshgrid(ones, zeros, indexing="ij")).reshape(2, -1).T
        trials = np.repeat(current[None, :], len(pairs), axis=0)
        rows = np.arange(len(pairs))
        trials[rows, pairs[:, 0]] = 0
        trials[rows, pairs[:, 1]] = 1
        energies = model.evaluate_many(trials)
        best = int(np.argmin(energies))
        if energies[best] >= energy - 1e-12:
            break
        current, energy = trials[best], float(energies[best])
    return current


def best_result(
    model: QUBOModel,
    samples: np.ndarray,
    *,
    started: float,
    finished: float,
    metadata: Mapping[str, Any],
    local_search: bool = True,
) -> SolverResult:
    """Pick the lowest-energy feasible sample (repairing only if none is feasible).

    ``metadata["raw_objective"]`` is the energy the backend itself reached, so
    hardware quality can be judged separately from the optional swap polish.
    """

    samples = np.asarray(samples, dtype=np.int8)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    if samples.shape[1] != model.n_variables or len(samples) == 0:
        raise ValueError("solver returned samples of the wrong shape")
    feasible = np.asarray([model.is_feasible(sample) for sample in samples], dtype=bool)
    repaired = False
    if feasible.any():
        pool = samples[feasible]
        chosen = pool[int(np.argmin(model.evaluate_many(pool)))]
    else:
        lowest = samples[int(np.argmin(model.evaluate_many(samples)))]
        chosen = repair_exact_k(model, lowest)
        repaired = True
    raw_objective = model.evaluate(chosen)
    if local_search:
        chosen = swap_local_search(model, chosen)
    finished = max(finished, perf_counter()) if local_search else finished
    return SolverResult(
        objective=model.evaluate(chosen),
        runtime=max(0.0, finished - started),
        feasible=model.is_feasible(chosen),
        selected_indices=model.selected_indices(chosen),
        sample=chosen,
        metadata={
            **metadata,
            "num_samples": int(len(samples)),
            "num_feasible_samples": int(feasible.sum()),
            "repaired": repaired,
            "raw_objective": raw_objective,
            "local_search": local_search,
            "local_search_gain": raw_objective - model.evaluate(chosen),
        },
    )

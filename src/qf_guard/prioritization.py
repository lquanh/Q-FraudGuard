"""Turn a window of fraud candidates into an investigation QUBO and solve it."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import count
from time import perf_counter
from typing import Any

import numpy as np

from qf_guard.qubo import QUBOModel, build_investigation_qubo, exact_k_penalty_bound
from qf_guard.results import SolverResult


IGNORED_ACCOUNTS = frozenset({"", "UNKNOWN"})


RISK_NORMALIZATIONS = ("none", "max", "rank")


@dataclass(frozen=True, slots=True)
class PriorityConfig:
    selection_fraction: float = 0.25
    shared_account_weight: float = 0.5
    connectivity_strength: float = 0.25
    penalty_margin: float = 1.10
    #: How risk scores are scaled inside each window before building the QUBO.
    #: "none" keeps raw scores (the link bonus then depends on the scorer's
    #: scale); "max" divides by the window's highest score, so a link is worth a
    #: fixed share of the top risk; "rank" uses (rank / N).
    risk_normalization: str = "none"


def normalized_risks(risks: Sequence[float], method: str = "none") -> list[float]:
    values = np.asarray(risks, dtype=float)
    if method == "none" or len(values) == 0:
        return values.tolist()
    if method == "max":
        top = float(values.max())
        return (values / top).tolist() if top > 0 else values.tolist()
    if method == "rank":
        ranks = values.argsort(kind="stable").argsort(kind="stable")
        return ((ranks + 1) / len(values)).tolist()
    raise ValueError(f"risk_normalization must be one of {RISK_NORMALIZATIONS}")


@dataclass(frozen=True, slots=True)
class PrioritySelection:
    transaction_ids: tuple[str, ...]
    selected_ids: frozenset[str]
    k: int
    edges: tuple[tuple[int, int, float], ...]
    result: SolverResult | None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    certificate: Mapping[str, Any] | None = None


def _accounts(candidate: Mapping[str, Any]) -> set[str]:
    return {str(candidate.get("sender_id", "")), str(candidate.get("receiver_id", ""))} - set(
        IGNORED_ACCOUNTS
    )


def candidate_edges(
    candidates: Sequence[Mapping[str, Any]], weight: float = 0.5
) -> list[tuple[int, int, float]]:
    """Connect candidates that share a sender or receiver account."""

    accounts = [_accounts(candidate) for candidate in candidates]
    return [
        (left, right, weight)
        for left in range(len(candidates))
        for right in range(left + 1, len(candidates))
        if accounts[left] & accounts[right]
    ]


def selection_size(n_candidates: int, fraction: float) -> int:
    return 0 if n_candidates == 0 else min(n_candidates, max(1, round(n_candidates * fraction)))


def build_priority_qubo(
    candidates: Sequence[Mapping[str, Any]],
    config: PriorityConfig = PriorityConfig(),
    *,
    name: str = "priority_window",
) -> QUBOModel:
    risks = normalized_risks(
        [float(candidate["risk_score"]) for candidate in candidates],
        getattr(config, "risk_normalization", "none"),
    )
    edges = candidate_edges(candidates, config.shared_account_weight)
    bound = exact_k_penalty_bound(
        risks, edges, connectivity_strength=config.connectivity_strength
    )
    return build_investigation_qubo(
        risks,
        edges,
        k=selection_size(len(risks), config.selection_fraction),
        connectivity_strength=config.connectivity_strength,
        penalty_strength=max(bound * config.penalty_margin, bound + 0.05),
        name=name,
    )


def exact_optimum(model: QUBOModel) -> tuple[np.ndarray, dict[str, Any]]:
    """Certified optimum of an investigation QUBO (HiGHS MILP, local CPU).

    On the feasible set ``sum(x) = K`` the penalty term is zero, so the MILP
    maximises the business objective ``sum r_i x_i + c sum w_ij x_i x_j`` with
    the cardinality as a hard constraint and ``y_ij <= x_i, y_ij <= x_j``
    (valid because ``c, w >= 0``).  It stays sparse even though the QUBO is
    dense, and it is exact for any window size the dashboard produces.
    """

    from scipy import sparse
    from scipy.optimize import Bounds, LinearConstraint, milp

    started = perf_counter()
    meta = model.metadata
    risks = np.asarray(meta["risks"], dtype=float)
    edges = list(meta["suspicious_edges"])
    k = int(meta["k"])
    strength = float(meta["connectivity_strength"])
    n, m = len(risks), len(edges)
    cost = np.concatenate([-risks, -strength * np.asarray([w for _, _, w in edges], dtype=float)])
    rows, cols, vals = [0] * n, list(range(n)), [1.0] * n
    lower, upper = [float(k)], [float(k)]
    for e, (i, j, _) in enumerate(edges):
        for endpoint in (i, j):
            rows += [len(lower), len(lower)]
            cols += [n + e, int(endpoint)]
            vals += [1.0, -1.0]
            lower.append(-np.inf)
            upper.append(0.0)
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(len(lower), n + m))
    result = milp(
        cost,
        constraints=[LinearConstraint(A, np.asarray(lower), np.asarray(upper))],
        integrality=np.concatenate([np.ones(n), np.zeros(m)]),
        bounds=Bounds(0, 1),
        options={"mip_rel_gap": 0.0},
    )
    if result.x is None:
        raise RuntimeError(f"MILP certificate failed: {result.message}")
    x = np.round(result.x[:n]).astype(np.int8)
    return x, {
        "status": "optimal" if result.status == 0 else str(result.message),
        "seconds": perf_counter() - started,
        "energy": model.evaluate(x),
    }


def certify(model: QUBOModel, result: SolverResult, *, reference=None) -> dict[str, Any]:
    """Compare a solver's answer with the certified optimum of the same window.

    Gaps are relative shortfalls of the investigation reward (``-energy`` on
    the feasible set): ``raw_gap`` is what the backend itself returned (QPU,
    SQA, ...), ``final_gap`` is after the shared classical post-processing.
    """

    x_opt, info = exact_optimum(model) if reference is None else reference
    optimum = float(info["energy"])
    reward_opt = -optimum
    scale = max(1.0, abs(optimum))

    def gap(energy: float | None) -> float | None:
        if energy is None or reward_opt <= 0:
            return None
        return max(0.0, (reward_opt - (-float(energy))) / reward_opt)

    raw = result.metadata.get("raw_objective")
    certificate = {
        "optimum_energy": optimum,
        "solver_energy": float(result.objective),
        "raw_energy": None if raw is None else float(raw),
        "final_gap": gap(result.objective),
        "raw_gap": gap(raw),
        "matches_optimum": abs(float(result.objective) - optimum) <= 1e-9 * scale,
        "raw_matches_optimum": raw is not None and abs(float(raw) - optimum) <= 1e-9 * scale,
        "certified": info["status"] == "optimal",
        "certify_ms": info["seconds"] * 1000,
        "optimal_indices": tuple(int(i) for i in np.flatnonzero(x_opt)),
    }
    counts = result.metadata.get("measured_counts")
    if counts and certificate["certified"]:
        samples = np.frombuffer("".join(counts).encode("ascii"), dtype=np.uint8).reshape(-1, model.n_variables) - 48
        weights = np.fromiter(counts.values(), dtype=np.int64)
        feasible = samples.sum(axis=1) == model.exact_k
        energies = model.evaluate_many(samples[feasible]) if np.any(feasible) else np.array([])
        optimal = np.abs(energies - optimum) <= 1e-9 * scale
        hits = int(weights[feasible][optimal].sum())
        certificate.update(measured_reads=int(weights.sum()), feasible_reads=int(weights[feasible].sum()),
                           optimal_reads=hits, measured_hit=hits > 0)
    return certificate


def _certify_fast_jobs(model, solver, first, started, first_ready, limit, on_measurement, attempt_offset,
                       accept_fallback):
    """Stop at the first returned histogram with a measured optimum. Keep the reference off the QPU."""
    reference = exact_optimum(model)
    best = first
    best_cert = None
    attempts = []
    for index in count() if limit is None else range(limit):
        if index == 0:
            result, ready = first, first_ready
        else:
            try:
                retry = getattr(solver, "solve_measurement", None)
                result = retry(model, attempt_offset + index) if retry else solver.solve(model)
            except Exception as exc:
                if limit is None:
                    raise
                attempts.append({"error": type(exc).__name__, "error_detail": str(exc)[:300],
                                 "time_to_response_s": perf_counter() - started})
                break
            ready = perf_counter()
        meta = result.metadata
        if (meta.get("fallback_reason") or meta.get("backend") != first.metadata.get("backend")
                or meta.get("solver") not in ("quafu_fast_xy", "quafu_fast_local")):
            if accept_fallback and meta.get("fallback_reason"):
                best, best_cert = result, certify(model, result, reference=reference)
            elif best_cert is None:
                best_cert = certify(model, first, reference=reference)
            attempts.append({"backend": meta.get("backend"), "error": "fallback", "time_to_response_s": ready - started})
            break
        cert = certify(model, result, reference=reference)
        attempt = {key: meta.get(key) for key in (
            "backend", "chip", "task_id", "shots", "physical_qubits", "readout_qubits",
            "circuit_candidate_order", "reverse_provider_bits", "measured_counts", "logical_2q", "transpiled_2q",
            "client_setup_s", "circuit_build_s", "submit_to_counts_s", "server_time_s", "variant", "attempt_index",
            "gamma", "beta", "leader_bias", "leader_probability",
        )}
        attempt.update({key: cert.get(key) for key in ("measured_reads", "feasible_reads", "optimal_reads", "measured_hit", "certified")})
        attempt.update(time_to_response_s=ready - started, time_to_verified_s=perf_counter() - started,
                       selected_indices=list(result.selected_indices), optimum_energy=cert["optimum_energy"])
        attempts.append(attempt)
        if on_measurement:
            on_measurement(attempt)
        # Prefer a selection actually measured on the chip over a repaired one.
        if best_cert is None or (bool(meta["repaired"]), result.objective) < (bool(best.metadata["repaired"]), best.objective):
            best, best_cert = result, cert
        if cert.get("measured_hit"):
            best, best_cert = result, cert
            break
        if not cert["certified"]:
            break
    winner = next((a for a in attempts if a.get("measured_hit")), None)
    best_cert.update(
        measurement_attempts=attempts, measured_hit=bool(winner) if best_cert["certified"] else None,
        optimal_reads=sum(a.get("optimal_reads") or 0 for a in attempts),
        feasible_reads=sum(a.get("feasible_reads") or 0 for a in attempts),
        measured_reads=sum(a.get("measured_reads") or 0 for a in attempts),
        time_to_first_hit_s=winner["time_to_response_s"] if winner else None,
        time_to_verified_hit_s=winner["time_to_verified_s"] if winner else None,
        elapsed_s=perf_counter() - started,
    )
    return best, best_cert


def select_priority(
    candidates: Sequence[Mapping[str, Any]],
    solver: Any,
    config: PriorityConfig = PriorityConfig(),
    *,
    name: str = "priority_window",
    certify_optimum: bool = False,
    max_measurement_jobs: int | None = 3,
    on_measurement: Callable[[dict], None] | None = None,
    attempt_offset: int = 0,
    accept_fallback: bool = False,
) -> PrioritySelection:
    """Choose the K candidates (and their connections) to investigate first.

    With ``certify_optimum`` the window is also solved exactly (MILP) so the
    caller can show how far the chosen backend is from the optimum.
    """

    if (max_measurement_jobs is not None and max_measurement_jobs < 1) or attempt_offset < 0:
        raise ValueError("max_measurement_jobs must be positive")
    started = perf_counter()
    ids = tuple(str(candidate["transaction_id"]) for candidate in candidates)
    if not ids:
        return PrioritySelection((), frozenset(), 0, (), None, {"solver": "none"})
    k = selection_size(len(ids), config.selection_fraction)
    if k == len(ids):
        return PrioritySelection(
            ids, frozenset(ids), k, (), None, {"solver": "trivial", "backend": "none"}
        )
    model = build_priority_qubo(candidates, config, name=name)
    retry = getattr(solver, "solve_measurement", None)
    result = retry(model, attempt_offset) if retry and attempt_offset else solver.solve(model)
    certificate = None
    if certify_optimum:
        if result.metadata.get("solver") in ("quafu_fast_xy", "quafu_fast_local"):
            result, certificate = _certify_fast_jobs(model, solver, result, started, perf_counter(), max_measurement_jobs,
                                                    on_measurement, attempt_offset, accept_fallback)
        else:
            certificate = certify(model, result)
    return PrioritySelection(
        transaction_ids=ids,
        selected_ids=frozenset(ids[index] for index in result.selected_indices),
        k=k,
        edges=tuple(model.metadata["suspicious_edges"]),
        result=result,
        metadata=dict(result.metadata),
        certificate=certificate,
    )

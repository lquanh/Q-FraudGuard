"""End-to-end ring inference: evidence scores -> Ising MAP -> flagged accounts."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from qf_guard.rings.core_solvers import AutoCoreSolver
from qf_guard.rings.evidence import EvidenceScores
from qf_guard.rings.model import RingQUBO, build_ring_qubo
from qf_guard.rings.reduce import Core, CoreSolver, ReductionStats, solve_with_reductions

__all__ = ["RingInferenceResult", "infer_ring", "explain_account"]


@dataclass(slots=True)
class RingInferenceResult:
    x: np.ndarray
    energy: float
    model: RingQUBO
    stats: ReductionStats
    cores: list[Core]
    runtime: float
    solver_log: list[dict] = field(default_factory=list)

    @property
    def flagged(self) -> np.ndarray:
        return np.flatnonzero(self.x == 1)


def infer_ring(
    scores: EvidenceScores,
    kappa: float = 1.0,
    beta0: float = 0.0,
    core_solver: CoreSolver | None = None,
    use_lp: bool = True,
) -> RingInferenceResult:
    started = perf_counter()
    model = build_ring_qubo(scores, kappa=kappa, beta0=beta0)
    log: list[dict] = []
    solver = core_solver if core_solver is not None else AutoCoreSolver(log=log)
    x, stats, cores = solve_with_reductions(model, solver, use_lp=use_lp)
    return RingInferenceResult(
        x=x,
        energy=model.energy(x),
        model=model,
        stats=stats,
        cores=cores,
        runtime=perf_counter() - started,
        solver_log=log,
    )


def explain_account(result: RingInferenceResult, account: int, top: int = 5) -> list[dict]:
    """Strongest pieces of evidence acting on ``account`` in the final assignment."""

    model, x = result.model, result.x
    s = 2 * x.astype(int) - 1
    mask = (model.i == account) | (model.j == account)
    rows = []
    for i, j, J in zip(model.i[mask].tolist(), model.j[mask].tolist(), model.J[mask].tolist()):
        other = j if i == account else i
        support = J * s[account] * s[other]
        rows.append({"other": int(other), "J": float(J), "other_flagged": bool(x[other]),
                     "supports_decision": bool(support > 0), "weight": float(abs(J))})
    rows.sort(key=lambda r: -r["weight"])
    return rows[:top]

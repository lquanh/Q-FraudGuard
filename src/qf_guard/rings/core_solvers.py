"""Solvers for frustrated cores: exact (brute force, MILP), SA, SQA, local descent.

Every solver maps a :class:`~qf_guard.rings.reduce.Core` to a binary vector.
Quantum circuits (QAOA) live in :mod:`qf_guard.rings.qaoa`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from qf_guard.rings.reduce import Core

__all__ = [
    "brute_force",
    "brute_force_all",
    "local_descent",
    "simulated_annealing",
    "milp_exact",
    "openjij_sqa",
    "AutoCoreSolver",
]


def _states(start: int, stop: int, n: int) -> np.ndarray:
    codes = np.arange(start, stop, dtype=np.int64)
    return ((codes[:, None] >> np.arange(n)) & 1).astype(np.float32)


def brute_force_all(core: Core, chunk: int = 1 << 16) -> np.ndarray:
    """Energies of all ``2**n`` states, index ``k`` = bits of ``k`` (variable q = bit q)."""

    n = core.size
    if n > 24:
        raise ValueError("brute force limited to 24 variables")
    a = core.a.astype(np.float64)
    B = core.B.astype(np.float64)
    out = np.empty(1 << n)
    for start in range(0, 1 << n, chunk):
        stop = min(1 << n, start + chunk)
        X = _states(start, stop, n).astype(np.float64)
        out[start:stop] = X @ a + 0.5 * np.einsum("bi,ij,bj->b", X, B, X)
    return out


def brute_force(core: Core) -> np.ndarray:
    energies = brute_force_all(core)
    best = int(np.argmin(energies))
    return ((best >> np.arange(core.size)) & 1).astype(np.int8)


def local_descent(core: Core, x: np.ndarray, max_passes: int = 100) -> np.ndarray:
    """Steepest single-flip descent (classical polish shared by every solver)."""

    x = np.asarray(x, dtype=np.int8).copy()
    field = core.a + core.B @ x
    for _ in range(max_passes * core.size):
        delta = (1 - 2 * x) * field
        k = int(np.argmin(delta))
        if delta[k] >= -1e-12:
            break
        step = 1 - 2 * int(x[k])
        x[k] += step
        field += core.B[:, k] * step
    return x


def simulated_annealing(
    core: Core, sweeps: int = 400, restarts: int = 16, t_hot: float | None = None,
    t_cold: float = 0.02, seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = core.size
    scale = max(np.abs(core.a).max(initial=0.0), np.abs(core.B).max(initial=0.0), 1e-9)
    t_hot = t_hot if t_hot is not None else 2.0 * scale
    X = (rng.random((restarts, n)) < 0.5).astype(np.int8)
    F = core.a[None, :] + X @ core.B
    temperatures = np.geomspace(t_hot, t_cold * scale, sweeps)
    rows = np.arange(restarts)
    for T in temperatures:
        for k in rng.permutation(n):
            delta = (1 - 2 * X[:, k]) * F[:, k]
            accept = (delta <= 0) | (rng.random(restarts) < np.exp(-np.maximum(delta, 0) / T))
            if not accept.any():
                continue
            step = np.where(accept, 1 - 2 * X[:, k], 0).astype(np.int8)
            X[rows, k] += step
            F += step[:, None].astype(float) * core.B[k][None, :]
    energies = X @ core.a + 0.5 * np.einsum("bi,ij,bj->b", X.astype(float), core.B, X.astype(float))
    best = X[int(np.argmin(energies))]
    return local_descent(core, best)


@dataclass(frozen=True, slots=True)
class MilpResult:
    x: np.ndarray
    optimal: bool
    objective: float
    bound: float | None


def milp_exact(core: Core, time_limit: float = 30.0) -> MilpResult:
    """Exact optimum (or best-found with a bound) via the standard linearisation."""

    n = core.size
    iu, ju = np.nonzero(np.triu(core.B, 1))
    b = core.B[iu, ju]
    m = len(b)
    c = np.concatenate([core.a, b])
    rows, cols, vals, ub = [], [], [], []
    r = 0
    for e in range(m):
        y = n + e
        if b[e] > 0:
            rows += [r, r, r]
            cols += [int(iu[e]), int(ju[e]), y]
            vals += [1.0, 1.0, -1.0]
            ub.append(1.0)
            r += 1
        else:
            rows += [r, r, r + 1, r + 1]
            cols += [y, int(iu[e]), y, int(ju[e])]
            vals += [1.0, -1.0, 1.0, -1.0]
            ub += [0.0, 0.0]
            r += 2
    constraints = []
    if r:
        A = sparse.csr_matrix((vals, (rows, cols)), shape=(r, n + m))
        constraints = [LinearConstraint(A, -np.inf, np.asarray(ub))]
    integrality = np.concatenate([np.ones(n), np.zeros(m)])
    result = milp(
        c,
        constraints=constraints,
        integrality=integrality,
        bounds=Bounds(0, 1),
        options={"time_limit": time_limit, "disp": False},
    )
    if result.x is None:
        x = local_descent(core, np.zeros(n, dtype=np.int8))
        return MilpResult(x=x, optimal=False, objective=core.energy(x), bound=None)
    x = np.round(result.x[:n]).astype(np.int8)
    bound = getattr(result, "mip_dual_bound", None)
    return MilpResult(x=x, optimal=result.status == 0, objective=core.energy(x), bound=bound)


def openjij_sqa(core: Core, num_reads: int = 32, num_sweeps: int = 200, seed: int = 0) -> np.ndarray:
    """Simulated quantum annealing (OpenJij, local CPU) on the core QUBO."""

    import openjij

    Q = {}
    for i in range(core.size):
        if core.a[i] != 0:
            Q[(i, i)] = float(core.a[i])
        for j in range(i + 1, core.size):
            if core.B[i, j] != 0:
                Q[(i, j)] = float(core.B[i, j])
    response = openjij.SQASampler().sample_qubo(
        Q, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed, sparse=True
    )
    states = np.asarray(response.states)
    if states.ndim == 1:
        states = states.reshape(1, -1)
    X = np.zeros((len(states), core.size), dtype=np.int8)
    X[:, [int(v) for v in response.indices]] = states
    energies = [core.energy(x) for x in X]
    return local_descent(core, X[int(np.argmin(energies))])


@dataclass(slots=True)
class AutoCoreSolver:
    """Exact brute force up to ``exact_limit`` variables, SA (or MILP) above."""

    exact_limit: int = 20
    large: str = "sa"
    milp_time_limit: float = 30.0
    seed: int = 0
    log: list[dict] | None = None

    def __call__(self, core: Core) -> np.ndarray:
        if core.size <= self.exact_limit:
            x, method, optimal = brute_force(core), "exact", True
        elif self.large == "milp":
            res = milp_exact(core, self.milp_time_limit)
            x, method, optimal = res.x, "milp", res.optimal
        else:
            x, method, optimal = simulated_annealing(core, seed=self.seed), "sa", False
        if self.log is not None:
            self.log.append({"size": core.size, "method": method, "optimal": optimal,
                             "energy": core.energy(x)})
        return x

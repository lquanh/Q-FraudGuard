"""Exact classical preprocessing that isolates the frustrated core.

The pipeline only ever applies reductions that preserve at least one global
optimum, so the core it leaves is the part classical shortcuts cannot settle:

1. fix seeds;
2. roof duality: solve the LP relaxation of the standard linearisation (HiGHS
   dual simplex).  Coordinates the LP sets to 0 or 1 are *persistent*: some
   global optimum agrees with them (Hammer, Hansen & Simeone, 1984).  Balanced
   (unfrustrated) parts are solved completely by this step;
3. dominance fixing and exact elimination of degree-0/1/2 variables;
4. repeat 2-3 until nothing changes.

What remains, split into connected components, is the *frustrated core*: the
only part that needs a combinatorial solver (exact, SA, SQA or QAOA).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from qf_guard.rings.model import RingQUBO

__all__ = ["Core", "ReductionStats", "Reducer", "solve_with_reductions"]

_TOL = 1e-9


@dataclass(frozen=True, slots=True)
class Core:
    """A connected frustrated component: ``E = const + a.x + sum_{i<j} B_ij x_i x_j``."""

    variables: np.ndarray
    a: np.ndarray
    B: np.ndarray

    @property
    def size(self) -> int:
        return int(len(self.variables))

    def energy(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(self.a @ x + 0.5 * x @ self.B @ x)

    def n_edges(self) -> int:
        return int(np.count_nonzero(np.triu(self.B, 1)))


@dataclass(slots=True)
class ReductionStats:
    n_total: int = 0
    n_fixed_seeds: int = 0
    n_after_seeds: int = 0
    n_fixed_lp: int = 0
    n_eliminated: int = 0
    n_fixed_dominance: int = 0
    lp_rounds: int = 0
    core_sizes: list[int] = field(default_factory=list)
    core_edges: list[int] = field(default_factory=list)
    lp_seconds: float = 0.0
    core_seconds: float = 0.0

    @property
    def n_core(self) -> int:
        return int(sum(self.core_sizes))

    @property
    def max_core(self) -> int:
        return int(max(self.core_sizes, default=0))


class Reducer:
    def __init__(self, model: RingQUBO) -> None:
        self.n = model.n
        self.a = np.array(model.a, dtype=float)
        self.adj: list[dict[int, float]] = [dict() for _ in range(self.n)]
        for i, j, b in zip(model.i.tolist(), model.j.tolist(), model.b.tolist()):
            if i == j or b == 0:
                continue
            self.adj[i][j] = self.adj[i].get(j, 0.0) + b
            self.adj[j][i] = self.adj[j].get(i, 0.0) + b
        self.const = float(model.const)
        self.free = np.ones(self.n, dtype=bool)
        self.x = np.full(self.n, -1, dtype=np.int8)
        self.stack: list[tuple] = []
        self.stats = ReductionStats(n_total=self.n)

    # -- primitive operations -------------------------------------------------
    def fix(self, i: int, value: int) -> None:
        if not self.free[i]:
            return
        for j, b in self.adj[i].items():
            if value:
                self.a[j] += b
            del self.adj[j][i]
        self.const += self.a[i] * value
        self.adj[i] = {}
        self.free[i] = False
        self.x[i] = value

    def _remove(self, i: int) -> None:
        for j in list(self.adj[i]):
            del self.adj[j][i]
        self.adj[i] = {}
        self.free[i] = False

    def _add_edge(self, j: int, k: int, value: float) -> None:
        if abs(value) <= _TOL:
            return
        new = self.adj[j].get(k, 0.0) + value
        if abs(new) <= _TOL:
            self.adj[j].pop(k, None)
            self.adj[k].pop(j, None)
        else:
            self.adj[j][k] = new
            self.adj[k][j] = new

    # -- reductions -------------------------------------------------------------
    def roof_duality(self) -> int:
        free = np.flatnonzero(self.free)
        if len(free) == 0:
            return 0
        position = -np.ones(self.n, dtype=np.int64)
        position[free] = np.arange(len(free))
        ei, ej, eb = [], [], []
        for i in free.tolist():
            for j, b in self.adj[i].items():
                if i < j:
                    ei.append(position[i]), ej.append(position[j]), eb.append(b)
        ei, ej, eb = np.asarray(ei, dtype=np.int64), np.asarray(ej, dtype=np.int64), np.asarray(eb)
        nf, m = len(free), len(eb)
        c = np.concatenate([self.a[free], eb])
        pos, neg = eb > 0, eb < 0
        y = nf + np.arange(m)
        # b > 0:  x_i + x_j - y <= 1
        p_idx = np.flatnonzero(pos)
        rows_p = np.repeat(np.arange(len(p_idx)), 3)
        cols_p = np.column_stack([ei[p_idx], ej[p_idx], y[p_idx]]).ravel()
        vals_p = np.tile([1.0, 1.0, -1.0], len(p_idx))
        # b < 0:  y - x_i <= 0,  y - x_j <= 0
        n_idx = np.flatnonzero(neg)
        base = len(p_idx)
        rows_n = base + np.repeat(np.arange(2 * len(n_idx)), 2)
        cols_n = np.column_stack(
            [y[n_idx], ei[n_idx], y[n_idx], ej[n_idx]]
        ).ravel()
        vals_n = np.tile([1.0, -1.0, 1.0, -1.0], len(n_idx))
        n_rows = base + 2 * len(n_idx)
        started = perf_counter()
        if n_rows:
            A = sparse.csr_matrix(
                (np.concatenate([vals_p, vals_n]), (np.concatenate([rows_p, rows_n]), np.concatenate([cols_p, cols_n]))),
                shape=(n_rows, nf + m),
            )
            rhs = np.concatenate([np.ones(len(p_idx)), np.zeros(2 * len(n_idx))])
            result = linprog(c, A_ub=A, b_ub=rhs, bounds=(0.0, 1.0), method="highs-ds")
        else:
            result = linprog(c, bounds=(0.0, 1.0), method="highs-ds")
        self.stats.lp_seconds += perf_counter() - started
        self.stats.lp_rounds += 1
        if result.status != 0 or result.x is None:
            return 0
        values = result.x[:nf]
        fixed = 0
        for local, value in enumerate(values.tolist()):
            if value <= 1e-7:
                self.fix(int(free[local]), 0)
                fixed += 1
            elif value >= 1 - 1e-7:
                self.fix(int(free[local]), 1)
                fixed += 1
        self.stats.n_fixed_lp += fixed
        return fixed

    def eliminate(self) -> int:
        removed = 0
        queue = [int(i) for i in np.flatnonzero(self.free)]
        while queue:
            i = queue.pop()
            if not self.free[i]:
                continue
            neighbours = self.adj[i]
            lo = self.a[i] + sum(min(0.0, b) for b in neighbours.values())
            hi = self.a[i] + sum(max(0.0, b) for b in neighbours.values())
            touched = list(neighbours)
            if lo >= -_TOL:
                self.fix(i, 0)
                self.stats.n_fixed_dominance += 1
            elif hi <= _TOL:
                self.fix(i, 1)
                self.stats.n_fixed_dominance += 1
            elif len(neighbours) == 1:
                (j, b), = neighbours.items()
                ai = self.a[i]
                g0, g1 = min(0.0, ai), min(0.0, ai + b)
                self.a[j] += g1 - g0
                self.const += g0
                self.stack.append(("deg1", i, j, ai, b))
                self._remove(i)
                self.stats.n_eliminated += 1
            elif len(neighbours) == 2:
                (j, bij), (k, bik) = neighbours.items()
                ai = self.a[i]
                g00, g10 = min(0.0, ai), min(0.0, ai + bij)
                g01, g11 = min(0.0, ai + bik), min(0.0, ai + bij + bik)
                self.a[j] += g10 - g00
                self.a[k] += g01 - g00
                self.const += g00
                self.stack.append(("deg2", i, j, k, ai, bij, bik))
                self._remove(i)
                self._add_edge(j, k, g11 - g10 - g01 + g00)
                self.stats.n_eliminated += 1
            else:
                continue
            removed += 1
            queue.extend(t for t in touched if self.free[t])
        return removed

    def reduce(self, use_lp: bool = True, max_rounds: int = 6) -> None:
        for _ in range(max_rounds):
            changed = 0
            if use_lp:
                changed += self.roof_duality()
            changed += self.eliminate()
            if changed == 0:
                break

    # -- cores and reconstruction -------------------------------------------------
    def cores(self) -> list[Core]:
        free = np.flatnonzero(self.free)
        seen = np.zeros(self.n, dtype=bool)
        out: list[Core] = []
        for start in free.tolist():
            if seen[start]:
                continue
            component, frontier = [], [start]
            seen[start] = True
            while frontier:
                v = frontier.pop()
                component.append(v)
                for w in self.adj[v]:
                    if not seen[w]:
                        seen[w] = True
                        frontier.append(w)
            variables = np.asarray(sorted(component), dtype=np.int64)
            index = {int(v): k for k, v in enumerate(variables.tolist())}
            B = np.zeros((len(variables), len(variables)))
            for v in variables.tolist():
                for w, b in self.adj[v].items():
                    B[index[v], index[w]] = b
            out.append(Core(variables=variables, a=self.a[variables].copy(), B=B))
        return out

    def assign_core(self, core: Core, solution: np.ndarray) -> None:
        for v, value in zip(core.variables.tolist(), np.asarray(solution, dtype=np.int8).tolist()):
            self.x[v] = value
            self.free[v] = False

    def back_substitute(self) -> np.ndarray:
        x = self.x
        for record in reversed(self.stack):
            if record[0] == "deg1":
                _, i, j, ai, b = record
                x[i] = 1 if ai + b * x[j] < 0 else 0
            else:
                _, i, j, k, ai, bij, bik = record
                x[i] = 1 if ai + bij * x[j] + bik * x[k] < 0 else 0
        if np.any(x < 0):
            raise RuntimeError("unassigned variables after back-substitution")
        return x.astype(np.int8)


CoreSolver = Callable[[Core], np.ndarray]


def solve_with_reductions(
    model: RingQUBO, core_solver: CoreSolver, use_lp: bool = True
) -> tuple[np.ndarray, ReductionStats, list[Core]]:
    reducer = Reducer(model)
    for k, v in model.fixed.items():
        reducer.fix(int(k), int(v))
    reducer.stats.n_fixed_seeds = len(model.fixed)
    reducer.stats.n_after_seeds = int(reducer.free.sum())
    reducer.reduce(use_lp=use_lp)
    cores = reducer.cores()
    started = perf_counter()
    for core in cores:
        reducer.assign_core(core, core_solver(core))
        reducer.stats.core_sizes.append(core.size)
        reducer.stats.core_edges.append(core.n_edges())
    reducer.stats.core_seconds = perf_counter() - started
    x = reducer.back_substitute()
    return x, reducer.stats, cores

"""QAOA for the investigation QUBO: Ising mapping, exact simulation, scoring.

Hardware-independent part of the gate-based (Quafu / IBM) path.  The circuit is

    |+>^n  ->  prod_{l=1..p} [ exp(-i gamma_l H_C) exp(-i beta_l sum_i X_i) ]  ->  measure

with ``H_C = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j`` from the QUBO through
``x = (1 - z) / 2`` (|0> means "not selected", |1> means "selected").  Angles
are chosen on an exact noiseless simulation (n <= ~14), then the same circuit
is run on hardware; nothing is tuned on the hardware itself.

Bitstrings use the convention "character i = qubit i = variable i".
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize

from qf_guard.qubo import QUBOModel

__all__ = [
    "IsingForm",
    "qubo_to_ising",
    "objective_ising",
    "all_energies",
    "qaoa_probabilities",
    "optimise_angles",
    "line_order",
    "xy_blocks",
    "xy_initial_state",
    "xy_qaoa_probabilities",
    "optimise_xy_angles",
]


@dataclass(frozen=True, slots=True)
class IsingForm:
    h: np.ndarray
    J: dict[tuple[int, int], float]
    const: float
    scale: float

    @property
    def n(self) -> int:
        return int(len(self.h))


def qubo_to_ising(model: QUBOModel) -> IsingForm:
    """Exact Ising form of ``x @ Q @ x + offset`` with ``x = (1 - z) / 2``.

    Coefficients are divided by their largest magnitude (``scale``) so the
    same angle ranges work for every window.
    """

    Q = np.asarray(model.Q, dtype=float)
    n = Q.shape[0]
    h = np.array([-Q[i, i] / 2 - (Q[i].sum() - Q[i, i]) / 2 for i in range(n)])
    J = {(i, j): Q[i, j] / 2 for i in range(n) for j in range(i + 1, n) if Q[i, j] != 0}
    const = float(model.offset + np.trace(Q) / 2 + sum(Q[i, j] for i in range(n) for j in range(i + 1, n)) / 2)
    scale = float(max(np.abs(h).max(initial=0.0), max((abs(v) for v in J.values()), default=0.0), 1e-12))
    return IsingForm(h=h / scale, J={k: v / scale for k, v in J.items()}, const=const, scale=scale)


def _bits(n: int) -> np.ndarray:
    """Row k = bits of basis state k with qubit i at bit position i (little endian)."""

    codes = np.arange(2**n, dtype=np.int64)
    return ((codes[:, None] >> np.arange(n)) & 1).astype(np.int8)


def all_energies(model: QUBOModel) -> np.ndarray:
    """QUBO energy of every basis state (index = little-endian integer of x)."""

    return model.evaluate_many(_bits(model.n_variables))


def _cost_diagonal(ising: IsingForm) -> np.ndarray:
    z = 1 - 2 * _bits(ising.n).astype(float)
    diag = z @ ising.h
    for (i, j), w in ising.J.items():
        diag += w * z[:, i] * z[:, j]
    return diag


def qaoa_probabilities(ising: IsingForm, gammas, betas) -> np.ndarray:
    """Exact output distribution of the ideal circuit (index = little-endian integer)."""

    n = ising.n
    diag = _cost_diagonal(ising)
    state = np.full(2**n, 2 ** (-n / 2), dtype=complex)
    for gamma, beta in zip(gammas, betas, strict=True):
        state *= np.exp(-1j * gamma * diag)
        c, s = np.cos(beta), -1j * np.sin(beta)
        for q in range(n):
            shaped = state.reshape(2 ** (n - q - 1), 2, 2**q)
            a, b = shaped[:, 0, :].copy(), shaped[:, 1, :].copy()
            shaped[:, 0, :] = c * a + s * b
            shaped[:, 1, :] = s * a + c * b
            state = shaped.reshape(-1)
    return np.abs(state) ** 2


def optimise_angles(ising: IsingForm, energies: np.ndarray, p: int = 1, seed: int = 0,
                    restarts: int = 12) -> tuple[np.ndarray, np.ndarray, float]:
    """Minimise the ideal expected QUBO energy over (gamma, beta); returns best angles."""

    rng = np.random.default_rng(seed)

    def objective(theta: np.ndarray) -> float:
        probs = qaoa_probabilities(ising, theta[:p], theta[p:])
        return float(probs @ energies)

    best_theta, best_value = None, np.inf
    for _ in range(restarts):
        start = np.concatenate([rng.uniform(0, np.pi, p), rng.uniform(0, np.pi / 2, p)])
        result = minimize(objective, start, method="Nelder-Mead",
                          options={"maxiter": 400 * p, "xatol": 1e-4, "fatol": 1e-9})
        if result.fun < best_value:
            best_theta, best_value = result.x, float(result.fun)
    return best_theta[:p], best_theta[p:], best_value


# ---------------------------------------------------------------------------------------
# Constraint-preserving QAOA (Hadfield et al., 2019): XY mixer + weight-K initial state.
# The exact-K penalty is gone: only the business objective -sum r x - c sum w x x is
# phased, and the mixer exp(-i beta (XX+YY)/2) on neighbouring qubits never changes
# the number of selected candidates.
# ---------------------------------------------------------------------------------------
def objective_ising(model: QUBOModel) -> IsingForm:
    """Ising form of the investigation objective only (no penalty), scaled to max |coef| = 1.

    Terms that are constant on the weight-K subspace are dropped, since the XY mixer
    never leaves it and a constant only adds a global phase. When K = 1, every
    x_i x_j vanishes, so links need no two-qubit gates at all.
    """

    risks = np.asarray(model.metadata["risks"], dtype=float)
    strength = float(model.metadata["connectivity_strength"])
    n = len(risks)
    edges = list(model.metadata["suspicious_edges"])
    complete_uniform = len(edges) == n * (n - 1) // 2 and len({w for _, _, w in edges}) <= 1
    if model.exact_k <= 1 or complete_uniform:  # sum over pairs is fixed by K
        edges = []
    h = risks / 2.0
    J: dict[tuple[int, int], float] = {}
    for i, j, w in edges:
        h[i] += strength * w / 4
        h[j] += strength * w / 4
        J[(min(i, j), max(i, j))] = J.get((min(i, j), max(i, j)), 0.0) - strength * w / 4
    scale = float(max(np.abs(h).max(initial=0.0), max((abs(v) for v in J.values()), default=0.0), 1e-12))
    return IsingForm(h=h / scale, J={key: value / scale for key, value in J.items()}, const=0.0, scale=scale)


def xy_blocks(n: int, k: int) -> list[list[int]]:
    """Qubit blocks that each start with one excitation (W state per block)."""

    if k < 1 or k > n:
        raise ValueError("need 1 <= k <= n")
    sizes = [n // k + (1 if b < n % k else 0) for b in range(k)]
    blocks, start = [], 0
    for size in sizes:
        blocks.append(list(range(start, start + size)))
        start += size
    return blocks


def xy_initial_state(n: int, k: int) -> np.ndarray:
    """Product of W states, one per block: Hamming weight exactly k (little-endian index)."""

    state = np.zeros(2**n, dtype=complex)
    blocks = xy_blocks(n, k)
    amplitude = 1.0 / np.sqrt(np.prod([len(b) for b in blocks]))
    from itertools import product as _product

    for choice in _product(*blocks):
        state[sum(1 << q for q in choice)] = amplitude
    return state


@lru_cache(maxsize=64)
def _xy_tables(n: int, k: int):
    """Weight-k basis, the line-mixer swap pairs inside it, and the initial state (cached per n, k)."""

    basis = np.array([code for code in range(2**n) if bin(code).count("1") == k], dtype=np.int64)
    position = {int(code): pos for pos, code in enumerate(basis)}
    pairs = []
    for a in range(n - 1):
        flip = (1 << a) | (1 << (a + 1))
        first = [pos for pos, code in enumerate(basis) if (code >> a) & 1 and not (code >> (a + 1)) & 1]
        second = [position[int(basis[pos]) ^ flip] for pos in first]  # |..1_a 0_a+1..> <-> |..0_a 1_a+1..>
        pairs.append((np.array(first, dtype=np.int64), np.array(second, dtype=np.int64)))
    return basis, tuple(pairs), xy_initial_state(n, k)[basis]


def _xy_state(diag: np.ndarray, tables, gammas, betas) -> np.ndarray:
    _, pairs, initial = tables
    state = initial.astype(complex)
    for gamma, beta in zip(gammas, betas, strict=True):
        state = state * np.exp(-1j * gamma * diag)
        c, s = np.cos(beta), -1j * np.sin(beta)
        for first, second in pairs:
            x, y = state[first], state[second]
            state[first] = c * x + s * y
            state[second] = s * x + c * y
    return state


def xy_qaoa_probabilities(ising: IsingForm, k: int, gammas, betas) -> np.ndarray:
    """Exact output distribution of the XY-mixer circuit (line mixer 0-1, 1-2, ..., n-2 - n-1).

    Simulated inside the weight-k subspace (C(n, k) amplitudes), returned over all 2**n states.
    """

    tables = _xy_tables(ising.n, k)
    state = _xy_state(_cost_diagonal(ising)[tables[0]], tables, gammas, betas)
    probs = np.zeros(2**ising.n)
    probs[tables[0]] = np.abs(state) ** 2
    return probs


def _interp(angles: np.ndarray) -> np.ndarray:
    """Depth q -> q+1 schedule by linear interpolation (Zhou et al., PRX 10, 021067)."""

    q = len(angles)
    padded = np.concatenate([[0.0], angles, [0.0]])
    return np.array([(j / q) * padded[j] + ((q - j) / q) * padded[j + 1] for j in range(q + 1)])


def line_order(n: int, edges) -> list[int]:
    """Order of candidates on a qubit line that puts as many linked pairs side by side as possible.

    Maximum-weight Hamiltonian path by dynamic programming over subsets (n <= ~14).
    The XY mixer acts on line neighbours, so a linked pair that ends up adjacent
    needs no SWAP gates on a chain of physical qubits.
    """

    if n <= 2 or not edges:
        return list(range(n))
    if n > 14:  # exact DP is O(2^n n^2): walk the link graph greedily instead
        neighbours = {v: set() for v in range(n)}
        for i, j, _ in edges:
            neighbours[i].add(j)
            neighbours[j].add(i)
        order: list[int] = []
        seen: set[int] = set()
        while len(order) < n:
            v = min((u for u in range(n) if u not in seen), key=lambda u: (len(neighbours[u] - seen), u))
            while v is not None:
                order.append(v)
                seen.add(v)
                options = [u for u in neighbours[v] if u not in seen]
                v = min(options, key=lambda u: (len(neighbours[u] - seen), u)) if options else None
        return order
    linked = np.zeros((n, n), dtype=int)
    for i, j, _ in edges:
        linked[i, j] = linked[j, i] = 1
    full = 1 << n
    best = np.full((full, n), -1, dtype=int)
    parent = np.full((full, n), -1, dtype=int)
    for v in range(n):
        best[1 << v, v] = 0
    for mask in range(full):
        for v in range(n):
            score = best[mask, v]
            if score < 0:
                continue
            for u in range(n):
                if mask >> u & 1:
                    continue
                nxt = mask | 1 << u
                if score + linked[v, u] > best[nxt, u]:
                    best[nxt, u] = score + linked[v, u]
                    parent[nxt, u] = v
    mask, v = full - 1, int(np.argmax(best[full - 1]))
    order = []
    while v >= 0:
        order.append(v)
        mask, v = mask ^ (1 << v), int(parent[mask, v])
    return order[::-1]


def optimise_xy_angles(ising: IsingForm, energies: np.ndarray, k: int, p: int = 1, seed: int = 0,
                       restarts: int = 12, warm_start=None) -> tuple[np.ndarray, np.ndarray, float]:
    """Minimise the ideal expected objective of the XY circuit (all amplitude stays feasible).

    ``warm_start = (gammas, betas)`` from depth p-1 is padded with an identity layer,
    so the depth-p result is never worse than depth p-1 (layer-by-layer growth).
    """

    rng = np.random.default_rng(seed)
    tables = _xy_tables(ising.n, k)
    diag = _cost_diagonal(ising)[tables[0]]
    target = np.asarray(energies, dtype=float)[tables[0]]

    def objective(theta: np.ndarray) -> float:
        return float(np.abs(_xy_state(diag, tables, theta[:p], theta[p:])) ** 2 @ target)

    starts =[np.concatenate([rng.uniform(0, np.pi, p), rng.uniform(0, np.pi, p)]) for _ in range(restarts)]
    if warm_start is not None and len(warm_start[0]) == p - 1 >= 1:
        previous_g, previous_b = (np.asarray(a, dtype=float) for a in warm_start)
        starts.insert(0, np.concatenate([np.append(previous_g, 0.0), np.append(previous_b, 0.0)]))
        starts.insert(1, np.concatenate([_interp(previous_g), _interp(previous_b)]))  # INTERP (Zhou et al., 2020)
    best_theta, best_value = None, np.inf
    for start in starts:
        result = minimize(objective, start, method="Nelder-Mead",
                          options={"maxiter": 400 * p, "xatol": 1e-4, "fatol": 1e-9})
        if result.fun < best_value:
            best_theta, best_value = result.x, float(result.fun)
    return best_theta[:p], best_theta[p:], best_value

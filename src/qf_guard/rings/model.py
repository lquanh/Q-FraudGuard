"""Ising / QUBO model of "which accounts belong to the reported ring".

Spin ``s_i = +1`` means account ``i`` is operated by a ring.  With node field
``h`` and pairwise dependence ``J`` from :mod:`qf_guard.rings.evidence`::

    E(s) = - sum_(i,j) J_ij s_i s_j - sum_i h_i s_i

``J > 0`` is evidence for "same state", ``J < 0`` for "one in, one out".  When
both signs meet on a cycle the model is *frustrated*: no assignment satisfies
every piece of evidence and the minimum-energy assignment is the most
consistent story.  Reported accounts (seeds) are fixed to ``+1``.

For solvers the model is stored in QUBO form with ``x = (1 + s) / 2``::

    E(x) = const + sum_i a_i x_i + sum_(i<j) b_ij x_i x_j
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qf_guard.rings.evidence import EvidenceScores

__all__ = ["RingQUBO", "build_ring_qubo"]


@dataclass(frozen=True, slots=True)
class RingQUBO:
    n: int
    a: np.ndarray
    i: np.ndarray
    j: np.ndarray
    b: np.ndarray
    const: float
    fixed: dict[int, int] = field(default_factory=dict)
    h: np.ndarray | None = None
    J: np.ndarray | None = None

    def energy(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(self.const + self.a @ x + np.sum(self.b * x[self.i] * x[self.j]))

    def energy_many(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self.const + X @ self.a + (X[:, self.i] * X[:, self.j]) @ self.b

    def respects_fixed(self, x: np.ndarray) -> bool:
        return all(int(x[k]) == v for k, v in self.fixed.items())


def build_ring_qubo(scores: EvidenceScores, kappa: float = 1.0, beta0: float = 0.0,
                    seeds: np.ndarray | None = None) -> RingQUBO:
    """Turn evidence scores into the ring QUBO (energy = negative log posterior + const).

    Per account: ``-(logit_i + beta0) x_i``.  Per evidence pair, exactly the
    2x2 log-likelihood table: ``-kappa (L1 - L0)`` on each endpoint and
    ``-kappa (L2 - 2 L1 + L0)`` on ``x_i x_j``.  ``kappa`` discounts pair
    evidence (pairs sharing one dorm Wi-Fi are not independent) and ``beta0``
    moves the operating point; both are tuned on training worlds only.
    """

    n = scores.n
    a = -(scores.node_logit + beta0).astype(float)
    i, j = scores.i, scores.j
    unary = -kappa * (scores.L1 - scores.L0)
    np.add.at(a, i, unary)
    np.add.at(a, j, unary)
    b = -kappa * (scores.L2 - 2.0 * scores.L1 + scores.L0)
    keep = b != 0
    i, j, b = i[keep], j[keep], b[keep]
    J = -b / 4.0
    h = -a / 2.0
    np.add.at(h, i, J)
    np.add.at(h, j, J)
    seed_ids = scores.seeds if seeds is None else seeds
    fixed = {int(k): 1 for k in np.asarray(seed_ids, dtype=np.int64)}
    return RingQUBO(n=n, a=a, i=i, j=j, b=b, const=0.0, fixed=fixed, h=h, J=J)

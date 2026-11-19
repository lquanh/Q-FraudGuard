"""Classical baselines that the Ising inference must beat.

All of them consume exactly the same learned evidence scores as the Ising
model.  Each exposes a single operating-point knob so that full recall /
false-positive curves can be traced.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from qf_guard.rings.evidence import EvidenceScores

__all__ = ["node_threshold", "link_boost", "union_find", "personalized_pagerank", "greedy_growth"]


def _with_seeds(flags: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    flags = flags.copy()
    flags[seeds] = True
    return flags


def node_threshold(scores: EvidenceScores, tau: float) -> np.ndarray:
    """Per-account score only (no graph)."""

    return _with_seeds(scores.node_prob >= tau, scores.seeds)


def link_boost(scores: EvidenceScores, tau: float, low_ratio: float = 0.25, tau_link: float = 0.3) -> np.ndarray:
    """Common risk-propagation rule: flag accounts scoring >= tau, plus accounts
    scoring >= low_ratio * tau that have a strong link (P(both) >= tau_link) to
    an already flagged account; repeat until stable."""

    flags = _with_seeds(scores.node_prob >= tau, scores.seeds)
    eligible = scores.node_prob >= low_ratio * tau
    strong = scores.p_both >= tau_link
    si, sj = scores.i[strong], scores.j[strong]
    while True:
        add = np.zeros_like(flags)
        add[si[flags[sj] & eligible[si] & ~flags[si]]] = True
        add[sj[flags[si] & eligible[sj] & ~flags[sj]]] = True
        if not add.any():
            return flags
        flags |= add


def union_find(scores: EvidenceScores, tau: float) -> np.ndarray:
    """Industry default: link pairs whose P(both in ring) >= tau, flag the seeds' components.

    Hub attributes (devices > 80 accounts, IPs > 60) were already pruned when
    the candidate pairs were built.
    """

    n = scores.n
    parent = np.arange(n)

    def find(v: int) -> int:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for i, j in zip(scores.i[scores.p_both >= tau].tolist(), scores.j[scores.p_both >= tau].tolist()):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    roots = np.array([find(v) for v in range(n)])
    seed_roots = set(roots[scores.seeds].tolist())
    return _with_seeds(np.isin(roots, list(seed_roots)), scores.seeds)


def personalized_pagerank(scores: EvidenceScores, alpha: float = 0.2, iterations: int = 60) -> np.ndarray:
    """Guilt-by-association score from the seeds over P(both)-weighted edges."""

    n = scores.n
    w = scores.p_both
    W = sparse.coo_matrix(
        (np.concatenate([w, w]), (np.concatenate([scores.i, scores.j]), np.concatenate([scores.j, scores.i]))),
        shape=(n, n),
    ).tocsr()
    degree = np.asarray(W.sum(axis=1)).ravel()
    inv = np.where(degree > 0, 1.0 / np.maximum(degree, 1e-12), 0.0)
    P = sparse.diags(inv) @ W
    restart = np.zeros(n)
    restart[scores.seeds] = 1.0 / max(len(scores.seeds), 1)
    r = restart.copy()
    for _ in range(iterations):
        r = alpha * restart + (1 - alpha) * (P.T @ r)
    return r


def greedy_growth(scores: EvidenceScores, tau: float, cannot_link: float = 0.5) -> np.ndarray:
    """Grow from the seeds, adding the strongest P(both) link first, refusing any
    account with a strong negative-evidence link (J <= -cannot_link) to the set.

    This is the natural greedy version of the same inference: it uses the
    negative evidence the Ising model uses, but decides one account at a time.
    """

    n = scores.n
    flagged = np.zeros(n, dtype=bool)
    flagged[scores.seeds] = True
    neighbours: list[list[tuple[int, float, float]]] = [[] for _ in range(n)]
    for i, j, p, J in zip(scores.i.tolist(), scores.j.tolist(), scores.p_both.tolist(), scores.J.tolist()):
        neighbours[i].append((j, p, J))
        neighbours[j].append((i, p, J))
    import heapq

    heap: list[tuple[float, int]] = []
    for s in scores.seeds.tolist():
        for v, p, _ in neighbours[s]:
            if not flagged[v] and p >= tau:
                heapq.heappush(heap, (-p, v))
    while heap:
        _, v = heapq.heappop(heap)
        if flagged[v]:
            continue
        if any(flagged[u] and J <= -cannot_link for u, _, J in neighbours[v]):
            continue
        flagged[v] = True
        for u, p, _ in neighbours[v]:
            if not flagged[u] and p >= tau:
                heapq.heappush(heap, (-p, u))
    return flagged

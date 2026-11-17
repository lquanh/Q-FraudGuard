"""Turn a window of fraud candidates into an investigation QUBO and solve it."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from qf_guard.qubo import QUBOModel, build_investigation_qubo, exact_k_penalty_bound
from qf_guard.results import SolverResult


IGNORED_ACCOUNTS = frozenset({"", "UNKNOWN"})


@dataclass(frozen=True, slots=True)
class PriorityConfig:
    selection_fraction: float = 0.25
    shared_account_weight: float = 0.5
    connectivity_strength: float = 0.25
    penalty_margin: float = 1.10


@dataclass(frozen=True, slots=True)
class PrioritySelection:
    transaction_ids: tuple[str, ...]
    selected_ids: frozenset[str]
    k: int
    edges: tuple[tuple[int, int, float], ...]
    result: SolverResult | None
    metadata: Mapping[str, Any] = field(default_factory=dict)


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
    risks = [float(candidate["risk_score"]) for candidate in candidates]
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


def select_priority(
    candidates: Sequence[Mapping[str, Any]],
    solver: Any,
    config: PriorityConfig = PriorityConfig(),
    *,
    name: str = "priority_window",
) -> PrioritySelection:
    """Choose the K candidates (and their connections) to investigate first."""

    ids = tuple(str(candidate["transaction_id"]) for candidate in candidates)
    if not ids:
        return PrioritySelection((), frozenset(), 0, (), None, {"solver": "none"})
    k = selection_size(len(ids), config.selection_fraction)
    if k == len(ids):
        return PrioritySelection(
            ids, frozenset(ids), k, (), None, {"solver": "trivial", "backend": "none"}
        )
    model = build_priority_qubo(candidates, config, name=name)
    result = solver.solve(model)
    return PrioritySelection(
        transaction_ids=ids,
        selected_ids=frozenset(ids[index] for index in result.selected_indices),
        k=k,
        edges=tuple(model.metadata["suspicious_edges"]),
        result=result,
        metadata=dict(result.metadata),
    )

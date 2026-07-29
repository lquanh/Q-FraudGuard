"""QUBO models used by QF-Guard.

The builders in this module deliberately keep one unambiguous matrix
convention: for a binary vector ``x`` the energy is exactly

``x @ model.Q @ x + model.offset``.

``Q`` is stored as a symmetric matrix.  Consequently, an interaction whose
usual QUBO coefficient is ``b_ij * x_i * x_j`` is represented by
``Q[i, j] == Q[j, i] == b_ij / 2``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

import numpy as np


__all__ = [
    "QUBOModel",
    "build_structured_fraud_qubo",
    "build_investigation_qubo",
    "build_fraud_qubo",
    "build_selection_qubo",
]


Edge = tuple[int, int, float]
EdgeInput = Iterable[Sequence[Real]] | Mapping[tuple[int, int], Real] | np.ndarray | None


def _as_float_vector(values: Iterable[Real], *, name: str) -> np.ndarray:
    """Convert an iterable to a non-empty, finite, one-dimensional vector."""

    if isinstance(values, np.ndarray):
        raw = values
    else:
        try:
            raw = list(values)
        except TypeError as exc:
            raise TypeError(f"{name} must be a one-dimensional iterable") from exc

    try:
        vector = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only real numbers") from exc

    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _edge_index(value: Real, *, n_variables: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("edge indices must be integers")
    index = int(value)
    if float(value) != index:
        raise ValueError("edge indices must be integers")
    if not 0 <= index < n_variables:
        raise ValueError(f"edge index {index} is outside [0, {n_variables - 1}]")
    return index


def _normalise_edges(edges: EdgeInput, n_variables: int) -> tuple[Edge, ...]:
    """Validate, canonicalise, and aggregate an undirected weighted edge set.

    Accepted formats are ``(i, j)`` / ``(i, j, weight)`` records, a mapping
    from ``(i, j)`` to weight, or a symmetric adjacency matrix.  Duplicate
    records are summed because they represent independent suspicious signals.
    """

    if edges is None:
        return ()

    records: Iterable[Sequence[Real]]
    if isinstance(edges, Mapping):
        mapped_records: list[tuple[Real, Real, Real]] = []
        for key, weight in edges.items():
            if not isinstance(key, Sequence) or len(key) != 2:
                raise ValueError("edge mapping keys must be (i, j) pairs")
            mapped_records.append((key[0], key[1], weight))
        records = mapped_records
    elif isinstance(edges, np.ndarray):
        array = np.asarray(edges)
        if array.ndim != 2:
            raise ValueError("edges array must be two-dimensional")

        if array.shape == (n_variables, n_variables) and np.allclose(
            array, array.T, rtol=1e-10, atol=1e-12, equal_nan=False
        ):
            try:
                adjacency = np.asarray(array, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError("adjacency matrix must be numeric") from exc
            if not np.all(np.isfinite(adjacency)):
                raise ValueError("adjacency matrix must contain only finite values")
            if np.any(adjacency < 0):
                raise ValueError("suspicious-edge weights must be non-negative")
            if not np.allclose(np.diag(adjacency), 0.0, atol=1e-12):
                raise ValueError("self-loops are not supported")
            records = [
                (i, j, adjacency[i, j])
                for i in range(n_variables)
                for j in range(i + 1, n_variables)
                if adjacency[i, j] != 0.0
            ]
        elif array.shape[1] in (2, 3):
            records = array.tolist()
        else:
            raise ValueError(
                "edges array must be a symmetric adjacency matrix or an N-by-2/3 edge list"
            )
    else:
        try:
            records = list(edges)
        except TypeError as exc:
            raise TypeError("edges must be an iterable, mapping, or adjacency matrix") from exc

    aggregated: dict[tuple[int, int], float] = {}
    for record in records:
        if not isinstance(record, Sequence) or isinstance(record, (str, bytes)):
            raise ValueError("each edge must be a sequence (i, j) or (i, j, weight)")
        if len(record) not in (2, 3):
            raise ValueError("each edge must contain two indices and an optional weight")
        i = _edge_index(record[0], n_variables=n_variables)
        j = _edge_index(record[1], n_variables=n_variables)
        if i == j:
            raise ValueError("self-loops are not supported")
        weight_value = 1.0 if len(record) == 2 else record[2]
        if isinstance(weight_value, (bool, np.bool_)) or not isinstance(weight_value, Real):
            raise ValueError("edge weights must be real numbers")
        weight = float(weight_value)
        if not np.isfinite(weight):
            raise ValueError("edge weights must be finite")
        if weight < 0:
            raise ValueError("suspicious-edge weights must be non-negative")
        if weight == 0:
            continue
        key = (i, j) if i < j else (j, i)
        aggregated[key] = aggregated.get(key, 0.0) + weight

    return tuple((i, j, weight) for (i, j), weight in sorted(aggregated.items()))


@dataclass(frozen=True, slots=True)
class QUBOModel:
    """A validated binary quadratic model.

    Parameters
    ----------
    Q:
        Square matrix in the convention ``energy = x @ Q @ x + offset``.
        An asymmetric input is replaced with its energy-equivalent symmetric
        part, ``(Q + Q.T) / 2``.
    offset:
        Constant energy term.  Keeping it makes reported energies comparable
        with the original negative-log-likelihood or constrained objective.
    exact_k:
        Optional semantic constraint used for feasibility reporting and by
        constraint-aware classical solvers.  The investigation builder also
        encodes this constraint into ``Q`` as a penalty.
    metadata:
        Builder and experiment information; it does not affect the energy.
    """

    Q: np.ndarray
    offset: float = 0.0
    exact_k: int | None = None
    name: str = "qubo"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            matrix = np.asarray(self.Q, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Q must be a numeric square matrix") from exc
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Q must be a square matrix")
        if matrix.shape[0] == 0:
            raise ValueError("Q must contain at least one variable")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Q must contain only finite values")

        symmetric = np.array((matrix + matrix.T) / 2.0, dtype=float, copy=True)
        symmetric.setflags(write=False)

        if isinstance(self.offset, (bool, np.bool_)) or not isinstance(self.offset, Real):
            raise ValueError("offset must be a finite real number")
        offset = float(self.offset)
        if not np.isfinite(offset):
            raise ValueError("offset must be a finite real number")

        exact_k = self.exact_k
        if exact_k is not None:
            if isinstance(exact_k, (bool, np.bool_)) or not isinstance(exact_k, Integral):
                raise ValueError("exact_k must be an integer or None")
            exact_k = int(exact_k)
            if not 0 <= exact_k <= symmetric.shape[0]:
                raise ValueError("exact_k must be between 0 and the number of variables")

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        object.__setattr__(self, "Q", symmetric)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "exact_k", exact_k)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_variables(self) -> int:
        """Number of binary decision variables."""

        return int(self.Q.shape[0])

    def validate_sample(self, sample: Sequence[int] | np.ndarray) -> np.ndarray:
        """Return a copied int8 sample or raise a descriptive validation error."""

        candidate = np.asarray(sample)
        if candidate.ndim != 1 or candidate.shape[0] != self.n_variables:
            raise ValueError(f"sample must have shape ({self.n_variables},)")
        if not np.all(np.isin(candidate, (0, 1))):
            raise ValueError("sample must contain only binary values 0 and 1")
        return np.asarray(candidate, dtype=np.int8).copy()

    def evaluate(self, sample: Sequence[int] | np.ndarray) -> float:
        """Evaluate one binary sample, including the constant offset."""

        candidate = self.validate_sample(sample).astype(float, copy=False)
        return float(candidate @ self.Q @ candidate + self.offset)

    energy = evaluate

    def evaluate_many(self, samples: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
        """Vectorised evaluation of a two-dimensional binary sample array."""

        candidates = np.asarray(samples)
        if candidates.ndim != 2 or candidates.shape[1] != self.n_variables:
            raise ValueError(f"samples must have shape (n_samples, {self.n_variables})")
        if not np.all(np.isin(candidates, (0, 1))):
            raise ValueError("samples must contain only binary values 0 and 1")
        values = np.asarray(candidates, dtype=float)
        return np.einsum("bi,ij,bj->b", values, self.Q, values) + self.offset

    def constraint_violation(self, sample: Sequence[int] | np.ndarray) -> int:
        """Return the absolute exact-K violation (zero when unconstrained)."""

        candidate = self.validate_sample(sample)
        if self.exact_k is None:
            return 0
        return abs(int(candidate.sum()) - self.exact_k)

    def is_feasible(self, sample: Sequence[int] | np.ndarray) -> bool:
        """Whether a sample satisfies the model's semantic constraints."""

        return self.constraint_violation(sample) == 0

    def selected_indices(self, sample: Sequence[int] | np.ndarray) -> tuple[int, ...]:
        """Return indices whose binary value is one."""

        candidate = self.validate_sample(sample)
        return tuple(int(i) for i in np.flatnonzero(candidate))

    def to_qubo_dict(self, *, atol: float = 0.0) -> dict[tuple[int, int], float]:
        """Return conventional one-entry-per-interaction QUBO coefficients.

        The constant offset is intentionally not included.  Diagonal entries
        are linear coefficients, while an off-diagonal entry ``(i, j)`` is
        ``2 * Q[i, j]`` so external BQM/QUBO libraries preserve the energy.
        """

        if atol < 0:
            raise ValueError("atol must be non-negative")
        coefficients: dict[tuple[int, int], float] = {}
        for i in range(self.n_variables):
            linear = float(self.Q[i, i])
            if abs(linear) > atol:
                coefficients[(i, i)] = linear
            for j in range(i + 1, self.n_variables):
                quadratic = float(2.0 * self.Q[i, j])
                if abs(quadratic) > atol:
                    coefficients[(i, j)] = quadratic
        return coefficients


def build_structured_fraud_qubo(
    probabilities: Iterable[Real],
    suspicious_edges: EdgeInput = None,
    *,
    pairwise_strength: float = 1.0,
    epsilon: float = 1e-9,
    name: str = "structured_fraud_classification",
) -> QUBOModel:
    """Build the graph-aware fraud-classification energy from the proposal.

    For calibrated fraud probabilities ``p_i`` and suspicious-edge weights
    ``w_ij``, the represented energy is

    ``sum_i[-x_i log(p_i) - (1-x_i) log(1-p_i)]``
    ``+ pairwise_strength * sum_(i,j) w_ij (x_i-x_j)^2``.

    ``epsilon`` follows the proposal literally by being added inside each
    logarithm, keeping probabilities at exactly zero or one finite.
    """

    probability = _as_float_vector(probabilities, name="probabilities")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("probabilities must lie in the closed interval [0, 1]")
    if isinstance(pairwise_strength, (bool, np.bool_)) or not isinstance(pairwise_strength, Real):
        raise ValueError("pairwise_strength must be a non-negative real number")
    pairwise_strength = float(pairwise_strength)
    if not np.isfinite(pairwise_strength) or pairwise_strength < 0:
        raise ValueError("pairwise_strength must be a non-negative finite number")
    if isinstance(epsilon, (bool, np.bool_)) or not isinstance(epsilon, Real):
        raise ValueError("epsilon must be a real number in (0, 0.5)")
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be a finite real number in (0, 0.5)")

    n_variables = int(probability.size)
    edges = _normalise_edges(suspicious_edges, n_variables)
    cost_if_zero = -np.log(1.0 - probability + epsilon)
    cost_if_one = -np.log(probability + epsilon)

    matrix = np.zeros((n_variables, n_variables), dtype=float)
    np.fill_diagonal(matrix, cost_if_one - cost_if_zero)
    offset = float(cost_if_zero.sum())

    for i, j, weight in edges:
        scaled_weight = pairwise_strength * weight
        matrix[i, i] += scaled_weight
        matrix[j, j] += scaled_weight
        matrix[i, j] -= scaled_weight
        matrix[j, i] -= scaled_weight

    return QUBOModel(
        Q=matrix,
        offset=offset,
        name=name,
        metadata={
            "problem_type": "structured_fraud_classification",
            "probabilities": tuple(float(value) for value in probability),
            "suspicious_edges": edges,
            "pairwise_strength": pairwise_strength,
            "epsilon": epsilon,
            "matrix_convention": "energy = x @ Q @ x + offset; Q is symmetric",
        },
    )


def build_investigation_qubo(
    risks: Iterable[Real],
    suspicious_edges: EdgeInput = None,
    *,
    k: int,
    connectivity_strength: float = 1.0,
    penalty_strength: float | None = None,
    name: str = "investigation_selection",
) -> QUBOModel:
    """Build an exact-K investigation-selection QUBO.

    The business objective maximised for feasible samples is

    ``sum_i risks[i]*z_i + connectivity_strength*sum_(i,j) w_ij*z_i*z_j``.

    The returned minimisation QUBO negates that score and adds
    ``penalty_strength * (sum_i z_i - k)^2``.  If no penalty is supplied, a
    conservative value greater than the full non-negative reward bound is
    chosen, which guarantees that every global QUBO optimum is feasible.
    """

    risk = _as_float_vector(risks, name="risks")
    if np.any(risk < 0):
        raise ValueError("risks must be non-negative")
    n_variables = int(risk.size)
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral):
        raise ValueError("k must be an integer")
    k = int(k)
    if not 0 <= k <= n_variables:
        raise ValueError("k must be between 0 and the number of risks")
    if isinstance(connectivity_strength, (bool, np.bool_)) or not isinstance(
        connectivity_strength, Real
    ):
        raise ValueError("connectivity_strength must be a non-negative real number")
    connectivity_strength = float(connectivity_strength)
    if not np.isfinite(connectivity_strength) or connectivity_strength < 0:
        raise ValueError("connectivity_strength must be a non-negative finite number")

    edges = _normalise_edges(suspicious_edges, n_variables)
    reward_bound = float(risk.sum() + connectivity_strength * sum(weight for _, _, weight in edges))
    if penalty_strength is None:
        penalty = max(1.0, reward_bound + 1.0)
    else:
        if isinstance(penalty_strength, (bool, np.bool_)) or not isinstance(penalty_strength, Real):
            raise ValueError("penalty_strength must be a positive real number")
        penalty = float(penalty_strength)
        if not np.isfinite(penalty) or penalty <= 0:
            raise ValueError("penalty_strength must be a positive finite number")

    matrix = np.zeros((n_variables, n_variables), dtype=float)
    np.fill_diagonal(matrix, -risk)
    for i, j, weight in edges:
        half_pair_coefficient = -0.5 * connectivity_strength * weight
        matrix[i, j] += half_pair_coefficient
        matrix[j, i] += half_pair_coefficient

    diagonal_penalty = penalty * (1 - 2 * k)
    matrix[np.diag_indices(n_variables)] += diagonal_penalty
    for i in range(n_variables):
        for j in range(i + 1, n_variables):
            matrix[i, j] += penalty
            matrix[j, i] += penalty
    offset = penalty * (k**2)

    return QUBOModel(
        Q=matrix,
        offset=offset,
        exact_k=k,
        name=name,
        metadata={
            "problem_type": "investigation_selection",
            "risks": tuple(float(value) for value in risk),
            "suspicious_edges": edges,
            "k": k,
            "connectivity_strength": connectivity_strength,
            "penalty_strength": penalty,
            "penalty_was_automatic": penalty_strength is None,
            "reward_upper_bound": reward_bound,
            "objective_sense": "minimise negative investigation reward",
            "matrix_convention": "energy = x @ Q @ x + offset; Q is symmetric",
        },
    )


build_fraud_qubo = build_structured_fraud_qubo
build_selection_qubo = build_investigation_qubo

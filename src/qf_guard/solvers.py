"""QUBO solver backends: exact CPU, OpenJij SQA, D-Wave QPU / hybrid and Quafu-SQC QPU.

Every backend returns a :class:`~qf_guard.results.SolverResult` whose
``metadata`` always contains ``solver``, ``backend`` and ``hardware`` so the
UI and benchmark reports state honestly where a selection was computed.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from time import perf_counter
from typing import Any, Mapping, Protocol

import numpy as np

from qf_guard.postprocess import best_result
from qf_guard.qubo import QUBOModel
from qf_guard.results import SolverResult
from qf_guard.sqa import OpenJijSQASolver


__all__ = [
    "Solver",
    "ExactSolver",
    "OpenJijSQASolver",
    "DWaveQPUSolver",
    "DWaveHybridCQMSolver",
    "FallbackSolver",
    "SOLVER_NAMES",
    "build_solver",
    "solver_from_env",
]


class Solver(Protocol):
    def solve(self, model: QUBOModel) -> SolverResult: ...


@dataclass(slots=True)
class ExactSolver:
    """Brute-force ground truth for small models (used to measure optimality gaps)."""

    max_states: int = 2_000_000
    chunk_size: int = 65_536

    def solve(self, model: QUBOModel) -> SolverResult:
        started = perf_counter()
        n = model.n_variables
        if model.exact_k is not None:
            states = comb(n, model.exact_k)
            if states > self.max_states:
                raise ValueError(f"exact search needs {states} states (> {self.max_states})")
            best_energy, best_sample = np.inf, None
            iterator = combinations(range(n), model.exact_k)
            while True:
                block = [indices for _, indices in zip(range(self.chunk_size), iterator)]
                if not block:
                    break
                samples = np.zeros((len(block), n), dtype=np.int8)
                for row, indices in enumerate(block):
                    samples[row, list(indices)] = 1
                energies = model.evaluate_many(samples)
                position = int(np.argmin(energies))
                if energies[position] < best_energy:
                    best_energy, best_sample = float(energies[position]), samples[position]
        else:
            if 2**n > self.max_states:
                raise ValueError(f"exact search needs 2**{n} states (> {self.max_states})")
            codes = np.arange(2**n, dtype=np.int64)
            samples = ((codes[:, None] >> np.arange(n)) & 1).astype(np.int8)
            best_sample = samples[int(np.argmin(model.evaluate_many(samples)))]
        return best_result(
            model,
            np.asarray(best_sample).reshape(1, -1),
            started=started,
            finished=perf_counter(),
            metadata={"solver": "exact", "backend": "cpu-exhaustive", "hardware": "local CPU"},
            local_search=False,
        )


def _ordered_samples(sampleset: Any, n_variables: int) -> np.ndarray:
    """Return ``sampleset`` rows as an ``(reads, n)`` array ordered by variable 0..n-1."""

    variables = list(sampleset.variables)
    order = [variables.index(index) for index in range(n_variables)]
    return np.asarray(sampleset.record.sample, dtype=np.int8)[:, order]


def _supported(sampler: Any, params: Mapping[str, Any]) -> dict[str, Any]:
    accepted = getattr(sampler, "parameters", None)
    return {
        key: value
        for key, value in params.items()
        if value is not None and (accepted is None or key in accepted)
    }


def _solver_name(sampler: Any) -> str:
    candidates = (sampler, getattr(sampler, "child", None))
    for candidate in candidates:
        if type(candidate).__name__ == "MockDWaveSampler":
            return "MockDWaveSampler (offline simulator, not quantum hardware)"
        name = getattr(getattr(candidate, "solver", None), "name", None)
        if name:
            return str(name)
    for candidate in candidates:
        properties = getattr(candidate, "properties", None) or {}
        if properties.get("chip_id"):
            return str(properties["chip_id"])
    return type(sampler).__name__


@dataclass(slots=True)
class DWaveQPUSolver:
    """Run the QUBO directly on a D-Wave annealing QPU through Leap.

    The exact-K penalty couples every pair of candidates, so the problem is a
    complete graph; :class:`dwave.system.DWaveCliqueSampler` uses cached clique
    embeddings, which is the most qubit-efficient mapping for that structure.
    Credentials come from ``DWAVE_API_TOKEN`` (or ``dwave config``) unless
    ``token`` is given.  ``sampler`` may be injected for offline testing, for
    example ``EmbeddingComposite(MockDWaveSampler())``.
    """

    num_reads: int = 100
    annealing_time: float | None = None
    chain_strength: float | None = None
    solver: str | None = None
    token: str | None = None
    label_prefix: str = "qf-guard"
    local_search: bool = True
    sampler: Any = None
    backend_name: str = "dwave-qpu"
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _get_sampler(self) -> Any:
        with self._lock:
            if self.sampler is None:
                from dwave.system import DWaveCliqueSampler

                config: dict[str, Any] = {}
                if self.solver:
                    config["solver"] = self.solver
                if self.token:
                    config["token"] = self.token
                self.sampler = DWaveCliqueSampler(**config)
            return self.sampler

    def solve(self, model: QUBOModel) -> SolverResult:
        sampler = self._get_sampler()
        largest = getattr(sampler, "largest_clique_size", None)
        if largest is not None and model.n_variables > int(largest):
            raise ValueError(
                f"{model.n_variables} variables exceed the QPU clique limit of {largest}"
            )
        params = _supported(
            sampler,
            {
                "num_reads": self.num_reads,
                "annealing_time": self.annealing_time,
                "chain_strength": self.chain_strength,
                "label": f"{self.label_prefix}:{model.name}"[:64],
            },
        )
        started = perf_counter()
        sampleset = sampler.sample(model.to_bqm(), **params)
        samples = _ordered_samples(sampleset, model.n_variables)
        finished = perf_counter()
        info = dict(sampleset.info)
        timing = dict(info.get("timing", {}) or {})
        record = sampleset.record
        chain_breaks = (
            float(np.mean(record.chain_break_fraction))
            if "chain_break_fraction" in (record.dtype.names or ())
            else None
        )
        return best_result(
            model,
            samples,
            started=started,
            finished=finished,
            local_search=self.local_search,
            metadata={
                "solver": "dwave_qpu",
                "backend": self.backend_name,
                "hardware": _solver_name(sampler),
                "num_reads": self.num_reads,
                "annealing_time_us": self.annealing_time,
                "qpu_access_time_us": float(timing.get("qpu_access_time", 0.0)),
                "qpu_programming_time_us": float(timing.get("qpu_programming_time", 0.0)),
                "qpu_sampling_time_us": float(timing.get("qpu_sampling_time", 0.0)),
                "chain_break_fraction": chain_breaks,
                "problem_id": info.get("problem_id"),
            },
        )


@dataclass(slots=True)
class DWaveHybridCQMSolver:
    """Leap hybrid CQM solver with the exact-K rule as a native constraint.

    Instead of a penalty, the investigation reward is the objective and
    ``sum(x) == k`` is a hard constraint.  Hybrid jobs have a minimum time
    limit of a few seconds, so this backend suits batch review, not ticks.
    """

    time_limit: float | None = None
    token: str | None = None
    label_prefix: str = "qf-guard"
    local_search: bool = True
    sampler: Any = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _get_sampler(self) -> Any:
        with self._lock:
            if self.sampler is None:
                from dwave.system import LeapHybridCQMSampler

                config = {"token": self.token} if self.token else {}
                self.sampler = LeapHybridCQMSampler(**config)
            return self.sampler

    @staticmethod
    def to_cqm(model: QUBOModel) -> Any:
        import dimod

        meta = model.metadata
        if meta.get("problem_type") != "investigation_selection" or model.exact_k is None:
            raise ValueError("hybrid CQM backend requires an investigation-selection model")
        n = model.n_variables
        objective = dimod.BinaryQuadraticModel("BINARY")
        objective.add_linear_from((i, -float(risk)) for i, risk in enumerate(meta["risks"]))
        strength = float(meta["connectivity_strength"])
        objective.add_quadratic_from(
            (i, j, -strength * float(weight)) for i, j, weight in meta["suspicious_edges"]
        )
        cqm = dimod.ConstrainedQuadraticModel()
        cqm.set_objective(objective)
        cqm.add_constraint_from_iterable(
            [(i, 1.0) for i in range(n)], "==", rhs=model.exact_k, label="exact_k"
        )
        return cqm

    def solve(self, model: QUBOModel) -> SolverResult:
        sampler = self._get_sampler()
        cqm = self.to_cqm(model)
        params = _supported(
            sampler,
            {"time_limit": self.time_limit, "label": f"{self.label_prefix}:{model.name}"[:64]},
        )
        started = perf_counter()
        sampleset = sampler.sample_cqm(cqm, **params)
        samples = _ordered_samples(sampleset, model.n_variables)
        finished = perf_counter()
        info = dict(sampleset.info)
        return best_result(
            model,
            samples,
            started=started,
            finished=finished,
            local_search=self.local_search,
            metadata={
                "solver": "dwave_hybrid_cqm",
                "backend": "dwave-hybrid",
                "hardware": "Leap hybrid CQM (classical + QPU)",
                "qpu_access_time_us": float(info.get("qpu_access_time", 0.0)),
                "run_time_us": float(info.get("run_time", 0.0)),
                "problem_id": info.get("problem_id"),
            },
        )


@dataclass(slots=True)
class FallbackSolver:
    """Use ``primary`` (usually a QPU) and fall back on errors or budget exhaustion.

    ``qpu_budget_seconds`` caps cumulative QPU access time for this process so a
    long-running demo cannot consume a whole Leap trial allocation.
    """

    primary: Solver
    fallback: Solver
    qpu_budget_seconds: float | None = None
    qpu_seconds_used: float = 0.0
    primary_calls: int = 0
    fallback_calls: int = 0
    last_error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def budget_exhausted(self) -> bool:
        return (
            self.qpu_budget_seconds is not None
            and self.qpu_seconds_used >= self.qpu_budget_seconds
        )

    def solve(self, model: QUBOModel) -> SolverResult:
        return self.solve_measurement(model, 0)

    def solve_measurement(self, model: QUBOModel, attempt_index: int) -> SolverResult:
        with self._lock:
            reason = "qpu_budget_exhausted" if self.budget_exhausted else None
            if reason is None:
                try:
                    if attempt_index and hasattr(self.primary, "solve_measurement"):
                        result = self.primary.solve_measurement(model, attempt_index)
                    else:
                        result = self.primary.solve(model)
                except Exception as exc:  # network, auth, embedding, quota ...
                    reason = f"{type(exc).__name__}: {exc}"[:300]
                else:
                    self.primary_calls += 1
                    meta = result.metadata
                    self.qpu_seconds_used += (
                        float(meta.get("qpu_access_time_us") or 0.0) / 1e6
                        or float(meta.get("server_time_s") or 0.0)
                    )
                    self.last_error = None
                    return result
            self.fallback_calls += 1
            self.last_error = reason
            result = self.fallback.solve(model)
            return SolverResult(
                objective=result.objective,
                runtime=result.runtime,
                feasible=result.feasible,
                selected_indices=result.selected_indices,
                sample=result.sample,
                metadata={**result.metadata, "fallback_reason": reason},
            )

    def stats(self) -> dict[str, Any]:
        return {
            "primary_calls": self.primary_calls,
            "fallback_calls": self.fallback_calls,
            "qpu_seconds_used": self.qpu_seconds_used,
            "qpu_budget_seconds": self.qpu_budget_seconds,
            "last_error": self.last_error,
        }


def mock_qpu_sampler(pegasus_shape: int = 16) -> Any:
    """``DWaveCliqueSampler`` whose child is an offline ``MockDWaveSampler``.

    It exercises the exact production code path (clique embedding, chains,
    unembedding, timing info) with a classical simulator standing in for the
    chip.  Results from it are *not* quantum-hardware results.
    """

    import functools
    import warnings
    from unittest import mock

    from dwave.system.samplers import clique
    from dwave.system.testing import MockDWaveSampler

    child = functools.partial(
        MockDWaveSampler,
        topology_type="pegasus",
        topology_shape=[pegasus_shape],
        parameter_warnings=False,
    )
    with warnings.catch_warnings(), mock.patch.object(clique, "DWaveSampler", child):
        warnings.simplefilter("ignore")
        return clique.DWaveCliqueSampler()


SOLVER_NAMES = (
    "sqa", "exact", "dwave-qpu", "dwave-hybrid", "dwave-mock",
    "quafu-qpu", "quafu-mock", "quafu-anneal-qpu", "quafu-anneal-mock", "quafu-fast-qpu", "quafu-fast-mock",
)
QPU_FAMILIES = ("dwave", "quafu")


def _float_env(env: Mapping[str, str], key: str, default: float | None) -> float | None:
    value = env.get(key, "").strip()
    return float(value) if value else default


def build_solver(name: str, env: Mapping[str, str] | None = None) -> Solver:
    """Construct one backend (without fallback) from its name and environment."""

    env = os.environ if env is None else env
    key = name.strip().lower()
    if key == "sqa":
        return OpenJijSQASolver(num_reads=64, num_sweeps=200, trotter=8)
    if key == "exact":
        return ExactSolver()
    token = env.get("DWAVE_API_TOKEN") or None
    if key == "dwave-qpu":
        return DWaveQPUSolver(
            num_reads=int(_float_env(env, "QFG_QPU_NUM_READS", 100) or 100),
            annealing_time=_float_env(env, "QFG_QPU_ANNEALING_TIME_US", None),
            chain_strength=_float_env(env, "QFG_QPU_CHAIN_STRENGTH", None),
            solver=env.get("DWAVE_SOLVER") or None,
            token=token,
        )
    if key == "dwave-mock":
        return DWaveQPUSolver(
            num_reads=int(_float_env(env, "QFG_QPU_NUM_READS", 100) or 100),
            sampler=mock_qpu_sampler(),
            label_prefix="qf-guard-mock",
            backend_name="dwave-mock",
        )
    if key == "dwave-hybrid":
        return DWaveHybridCQMSolver(
            time_limit=_float_env(env, "QFG_HYBRID_TIME_LIMIT_S", None), token=token
        )
    if key in SOLVER_NAMES and key.startswith("quafu-"):
        from qf_guard.quafu import LocalSimulatorClient, QuafuCircuitSolver, read_token

        mock = key.endswith("-mock")
        anneal = key.startswith("quafu-anneal-")
        fast = key.startswith("quafu-fast-")
        variant = env.get("QFG_QUAFU_FAST_CIRCUIT", "local").strip().lower()
        if fast and variant not in ("full", "paired", "local"):
            raise ValueError("QFG_QUAFU_FAST_CIRCUIT must be full, paired or local")
        if anneal:
            steps = int(env.get("QFG_QUAFU_ANNEAL_STEPS", "").strip() or "3")
            duration = float(env.get("QFG_QUAFU_ANNEAL_DURATION", "").strip() or "2.0")
            if steps < 1 or not np.isfinite(duration) or duration <= 0:
                raise ValueError("invalid digital annealing schedule")
        else:
            steps = int(_float_env(env, "QFG_QAOA_LAYERS", 3) or 3)
            duration = 2.0
        return QuafuCircuitSolver(
            chip=env.get("QFG_QUAFU_CHIP", "").strip() or "Haituo",
            backup_chip=env.get("QFG_QUAFU_BACKUP_CHIP", "Baihua").strip() or None,
            shots=1024 if fast else int(_float_env(env, "QFG_QUAFU_SHOTS", 2048) or 2048),
            layers=1 if fast else steps,
            max_qubits=(int(env.get("QFG_QUAFU_FAST_MAX_QUBITS", "") or 18) if fast
                        else int(_float_env(env, "QFG_QUAFU_MAX_QUBITS", 12) or 12)),
            token=None if mock else read_token(env),
            backend_name=key,
            digital_annealing=anneal,
            anneal_duration=duration,
            fast_xy=fast,
            fast_variant=variant,
            readout_correction=not fast,
            poll_seconds=0.2 if fast else 2.0,
            client=LocalSimulatorClient() if mock else None,
        )
    raise ValueError(f"unknown solver {name!r}; choose one of {', '.join(SOLVER_NAMES)}")


def solver_from_env(env: Mapping[str, str] | None = None) -> Solver:
    """Solver selected by ``QFG_SOLVER``; QPU backends fall back to SQA by default.

    The QPU budget counts D-Wave QPU access time, or Quafu-SQC server time per task
    (default 10 s for D-Wave, 600 s for Quafu, whose tasks take about 1 s each).
    """

    env = os.environ if env is None else env
    name = env.get("QFG_SOLVER", "sqa").strip().lower() or "sqa"
    solver = build_solver(name, env)
    if not name.startswith(QPU_FAMILIES) or env.get("QFG_SOLVER_FALLBACK", "sqa") == "none":
        return solver
    return FallbackSolver(
        primary=solver,
        fallback=build_solver(env.get("QFG_SOLVER_FALLBACK", "sqa"), env),
        qpu_budget_seconds=_float_env(
            env, "QFG_QPU_BUDGET_SECONDS", 600.0 if name.startswith("quafu") else 10.0
        ),
    )

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from qf_guard.qubo import QUBOModel
from qf_guard.results import SolverResult


@dataclass(slots=True)
class OpenJijSQASolver:
    num_reads: int = 64
    num_sweeps: int = 100
    trotter: int = 4
    beta: float = 100.0
    gamma: float = 1.0
    seed: int = 42

    def solve(self, model: QUBOModel) -> SolverResult:
        import openjij

        started = perf_counter()
        response = openjij.SQASampler().sample_qubo(
            model.to_qubo_dict(),
            num_reads=self.num_reads,
            num_sweeps=self.num_sweeps,
            trotter=self.trotter,
            beta=self.beta,
            gamma=self.gamma,
            seed=self.seed,
            sparse=True,
            reinitialize_state=True,
        )
        states = np.asarray(response.states)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        indices = [int(value) for value in response.indices]
        samples = np.zeros((len(states), model.n_variables), dtype=np.int8)
        samples[:, indices] = states
        feasible = np.asarray([model.is_feasible(sample) for sample in samples])
        pool = np.flatnonzero(feasible) if feasible.any() else np.arange(len(samples))
        objectives = model.evaluate_many(samples[pool])
        sample = samples[int(pool[int(np.argmin(objectives))])]
        return SolverResult(
            objective=model.evaluate(sample),
            runtime=perf_counter() - started,
            feasible=model.is_feasible(sample),
            selected_indices=model.selected_indices(sample),
            sample=sample,
            metadata={
                "solver": "openjij_sqa",
                "num_reads": self.num_reads,
                "num_sweeps": self.num_sweeps,
                "trotter": self.trotter,
            },
        )

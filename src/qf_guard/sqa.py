from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from qf_guard.postprocess import best_result
from qf_guard.qubo import QUBOModel
from qf_guard.results import SolverResult


@dataclass(slots=True)
class OpenJijSQASolver:
    """Simulated Quantum Annealing on the local CPU (no quantum hardware)."""

    num_reads: int = 64
    num_sweeps: int = 100
    trotter: int = 4
    beta: float = 100.0
    gamma: float = 1.0
    seed: int = 42
    local_search: bool = True

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
        return best_result(
            model,
            samples,
            started=started,
            finished=perf_counter(),
            local_search=self.local_search,
            metadata={
                "solver": "openjij_sqa",
                "backend": "cpu-simulated-quantum-annealing",
                "hardware": "local CPU",
                "num_reads": self.num_reads,
                "num_sweeps": self.num_sweeps,
                "trotter": self.trotter,
            },
        )

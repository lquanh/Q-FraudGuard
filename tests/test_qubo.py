from itertools import product

import numpy as np
import pytest

from qf_guard.qubo import build_investigation_qubo, exact_k_penalty_bound


def random_instance(seed: int, n: int = 8):
    rng = np.random.default_rng(seed)
    risks = rng.uniform(0, 1, n)
    edges = [(i, j, float(rng.uniform(0, 1))) for i in range(n) for j in range(i + 1, n)
             if rng.random() < .3]
    return risks, edges


def all_states(n: int) -> np.ndarray:
    return np.array(list(product((0, 1), repeat=n)), dtype=np.int8)


@pytest.mark.parametrize("seed", range(5))
def test_energy_matches_business_objective(seed):
    risks, edges = random_instance(seed)
    model = build_investigation_qubo(risks, edges, k=3, connectivity_strength=.4,
                                     penalty_strength=2.5)
    for x in all_states(len(risks))[::7]:
        reward = risks @ x + .4 * sum(w * x[i] * x[j] for i, j, w in edges)
        expected = -reward + 2.5 * (x.sum() - 3) ** 2
        assert model.evaluate(x) == pytest.approx(expected)


@pytest.mark.parametrize("seed", range(25))
def test_tight_penalty_keeps_every_global_optimum_feasible(seed):
    risks, edges = random_instance(seed)
    bound = exact_k_penalty_bound(risks, edges, connectivity_strength=.6)
    model = build_investigation_qubo(risks, edges, k=2 + seed % 4, connectivity_strength=.6,
                                     penalty_strength=bound * 1.001 + 1e-9)
    states = all_states(model.n_variables)
    energies = model.evaluate_many(states)
    optima = states[np.isclose(energies, energies.min())]
    assert all(model.is_feasible(state) for state in optima)


def test_tight_penalty_is_much_smaller_than_default():
    risks, edges = random_instance(1, n=30)
    default = build_investigation_qubo(risks, edges, k=5).metadata["penalty_strength"]
    assert exact_k_penalty_bound(risks, edges) < default / 3


def test_to_bqm_preserves_energy_and_variables():
    risks, edges = random_instance(3)
    risks[0] = 0.0
    model = build_investigation_qubo(risks, edges, k=2, penalty_strength=3.0)
    bqm = model.to_bqm()
    assert set(bqm.variables) == set(range(model.n_variables))
    for x in all_states(model.n_variables)[::11]:
        assert bqm.energy(dict(enumerate(x))) == pytest.approx(model.evaluate(x))

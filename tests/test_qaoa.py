import numpy as np
import pytest

from qf_guard.prioritization import PriorityConfig, build_priority_qubo
from qf_guard.qaoa import _cost_diagonal, all_energies, optimise_angles, qaoa_probabilities, qubo_to_ising


def window(seed, n=6):
    rng = np.random.default_rng(seed)
    return [{"transaction_id": f"T{i}", "risk_score": float(rng.uniform(.001, .02)),
             "sender_id": f"A{rng.integers(0, 3)}", "receiver_id": f"B{rng.integers(0, 3)}"} for i in range(n)]


@pytest.mark.parametrize("seed", range(5))
def test_ising_form_reproduces_every_qubo_energy(seed):
    model = build_priority_qubo(window(seed), PriorityConfig(risk_normalization="max", connectivity_strength=0.25))
    ising = qubo_to_ising(model)
    assert np.allclose(_cost_diagonal(ising) * ising.scale + ising.const, all_energies(model))


def test_ideal_qaoa_is_a_distribution_and_beats_uniform_guessing():
    model = build_priority_qubo(window(1, n=5), PriorityConfig(risk_normalization="max"))
    energies = all_energies(model)
    ising = qubo_to_ising(model)
    gammas, betas, value = optimise_angles(ising, energies, p=1, restarts=4)
    probs = qaoa_probabilities(ising, gammas, betas)
    assert probs.sum() == pytest.approx(1.0)
    assert value < energies.mean()  # better than measuring |+>^n


def test_xy_mixer_keeps_exactly_k_selected_and_finds_small_optima():
    from qf_guard.qaoa import objective_ising, optimise_xy_angles, xy_initial_state, xy_qaoa_probabilities

    model = build_priority_qubo(window(4, n=6), PriorityConfig(risk_normalization="max", connectivity_strength=0.25))
    k, energies = model.exact_k, all_energies(model)
    weights = np.array([bin(i).count("1") for i in range(2**6)])
    assert np.all(np.abs(xy_initial_state(6, k))[weights != k] == 0)
    ising = objective_ising(model)
    gammas, betas, value = optimise_xy_angles(ising, energies, k, p=2, restarts=6)
    probs = xy_qaoa_probabilities(ising, k, gammas, betas)
    assert probs[weights != k].sum() == pytest.approx(0.0, abs=1e-12)  # never leaves the K-subspace
    safe = np.where(weights == k, energies, 0.0)
    start_value = xy_qaoa_probabilities(ising, k, [0.0, 0.0], [0.0, 0.0]) @ safe  # the initial state itself
    assert value <= start_value + 1e-9


@pytest.mark.parametrize(("seed", "n"), [(0, 4), (1, 5), (2, 6), (3, 8)])
def test_objective_ising_matches_qubo_on_the_k_subspace(seed, n):
    from qf_guard.qaoa import objective_ising

    model = build_priority_qubo(window(seed, n=n), PriorityConfig(risk_normalization="max", connectivity_strength=0.25))
    ising = objective_ising(model)
    feasible = np.array([bin(i).count("1") == model.exact_k for i in range(2**n)])
    gap = (_cost_diagonal(ising) * ising.scale - all_energies(model))[feasible]
    assert np.ptp(gap) == pytest.approx(0.0, abs=1e-9)  # equal up to a constant on every K-selection
    if model.exact_k == 1:
        assert ising.J == {}  # links cannot matter when only one candidate is chosen


def test_line_order_puts_linked_candidates_side_by_side():
    from qf_guard.qaoa import line_order

    edges = [(0, 3, 0.5), (3, 5, 0.5), (1, 4, 0.5), (2, 4, 0.5)]  # two paths: 0-3-5 and 1-4-2
    order = line_order(6, edges)
    assert sorted(order) == list(range(6))
    adjacent = {frozenset(pair) for pair in zip(order, order[1:])}
    assert all(frozenset((i, j)) in adjacent for i, j, _ in edges)
    pairs = [(i, i + 1, 0.5) for i in range(0, 20, 2)]  # large window: greedy walk, still exact here
    order = line_order(20, pairs)
    assert sorted(order) == list(range(20))
    adjacent = {frozenset(pair) for pair in zip(order, order[1:])}
    assert all(frozenset((i, j)) in adjacent for i, j, _ in pairs)


def test_small_window_xy_qaoa_is_almost_always_optimal():
    from qf_guard.qaoa import objective_ising, optimise_xy_angles, xy_qaoa_probabilities

    model = build_priority_qubo(window(8, n=4), PriorityConfig(risk_normalization="max", connectivity_strength=0.01))
    energies = all_energies(model)
    ising = objective_ising(model)
    gammas, betas, _ = optimise_xy_angles(ising, energies, model.exact_k, p=3, restarts=6)
    probs = xy_qaoa_probabilities(ising, model.exact_k, gammas, betas)
    feasible = np.array([bin(i).count("1") == model.exact_k for i in range(16)])
    optimum = energies[feasible].min()
    assert probs[np.isclose(energies, optimum) & feasible].sum() > 0.9

import dimod
import numpy as np
import pytest

from qf_guard.postprocess import repair_exact_k, swap_local_search
from qf_guard.prioritization import build_priority_qubo, select_priority
from qf_guard.solvers import (
    DWaveHybridCQMSolver,
    DWaveQPUSolver,
    ExactSolver,
    FallbackSolver,
    OpenJijSQASolver,
    build_solver,
    mock_qpu_sampler,
    solver_from_env,
)


def candidates(seed: int, n: int = 12):
    rng = np.random.default_rng(seed)
    return [
        {
            "transaction_id": f"T{i}",
            "risk_score": float(rng.uniform(.2, 1)),
            "sender_id": f"C{rng.integers(0, n)}",
            "receiver_id": f"C{rng.integers(0, n)}",
        }
        for i in range(n)
    ]


@pytest.fixture(scope="module")
def mock_qpu():
    return DWaveQPUSolver(sampler=mock_qpu_sampler(), num_reads=100)


def test_exact_solver_matches_brute_force():
    model = build_priority_qubo(candidates(0, 10))
    result = ExactSolver().solve(model)
    states = np.array(np.meshgrid(*[[0, 1]] * 10)).reshape(10, -1).T
    feasible = states[states.sum(axis=1) == model.exact_k]
    assert result.objective == pytest.approx(model.evaluate_many(feasible).min())


@pytest.mark.parametrize("seed", range(4))
def test_sqa_returns_feasible_and_reports_raw_energy(seed):
    model = build_priority_qubo(candidates(seed))
    result = OpenJijSQASolver(num_reads=64, num_sweeps=200, trotter=8).solve(model)
    assert result.feasible
    assert result.metadata["raw_objective"] >= result.objective - 1e-12
    assert result.metadata["hardware"] == "local CPU"


@pytest.mark.parametrize("seed", range(4))
def test_mock_qpu_path_is_feasible_with_timing(mock_qpu, seed):
    model = build_priority_qubo(candidates(seed, 20))
    result = mock_qpu.solve(model)
    assert result.feasible
    assert result.metadata["backend"] == "dwave-qpu"
    assert "not quantum hardware" in result.metadata["hardware"]
    assert result.metadata["qpu_access_time_us"] > 0
    assert 0 <= result.metadata["chain_break_fraction"] <= 1


def test_qpu_rejects_models_larger_than_clique(mock_qpu):
    too_big = mock_qpu.sampler.largest_clique_size + 1
    with pytest.raises(ValueError, match="clique limit"):
        mock_qpu.solve(build_priority_qubo(candidates(0, too_big)))


def test_hybrid_cqm_uses_native_constraint():
    model = build_priority_qubo(candidates(1, 10))
    cqm = DWaveHybridCQMSolver.to_cqm(model)
    assert "exact_k" in cqm.constraints
    result = DWaveHybridCQMSolver(sampler=dimod.ExactCQMSolver()).solve(model)
    assert result.objective == pytest.approx(ExactSolver().solve(model).objective)


def test_repair_and_swap_search_reach_feasible_optimum():
    model = build_priority_qubo(candidates(2, 10))
    start = np.ones(model.n_variables, dtype=np.int8)
    repaired = repair_exact_k(model, start)
    assert model.is_feasible(repaired)
    polished = swap_local_search(model, repaired)
    assert model.evaluate(polished) <= model.evaluate(repaired)
    assert model.is_feasible(polished)


class Broken:
    def solve(self, model):
        raise ConnectionError("no route to Leap")


def test_fallback_on_error_and_budget():
    model = build_priority_qubo(candidates(3))
    fallback = FallbackSolver(Broken(), ExactSolver())
    result = fallback.solve(model)
    assert "ConnectionError" in result.metadata["fallback_reason"]
    assert fallback.stats()["fallback_calls"] == 1

    budgeted = FallbackSolver(
        DWaveQPUSolver(sampler=mock_qpu_sampler(), num_reads=20),
        ExactSolver(),
        qpu_budget_seconds=1e-9,
    )
    assert "fallback_reason" not in budgeted.solve(model).metadata
    assert budgeted.solve(model).metadata["fallback_reason"] == "qpu_budget_exhausted"


def test_solver_from_env():
    assert isinstance(solver_from_env({}), OpenJijSQASolver)
    wrapped = solver_from_env({"QFG_SOLVER": "dwave-qpu", "QFG_QPU_BUDGET_SECONDS": "3"})
    assert isinstance(wrapped, FallbackSolver) and wrapped.qpu_budget_seconds == 3
    bare = solver_from_env({"QFG_SOLVER": "dwave-qpu", "QFG_SOLVER_FALLBACK": "none"})
    assert isinstance(bare, DWaveQPUSolver)
    with pytest.raises(ValueError):
        build_solver("ibm-qpu")


def test_select_priority_trivial_and_edges():
    single = candidates(0, 1)
    assert select_priority(single, Broken()).selected_ids == {"T0"}
    linked = [
        {"transaction_id": "A", "risk_score": .5, "sender_id": "X", "receiver_id": "Y"},
        {"transaction_id": "B", "risk_score": .5, "sender_id": "Y", "receiver_id": "Z"},
        {"transaction_id": "C", "risk_score": .55, "sender_id": "P", "receiver_id": "Q"},
        {"transaction_id": "D", "risk_score": .1, "sender_id": "R", "receiver_id": "S"},
        {"transaction_id": "E", "risk_score": .1, "sender_id": "T", "receiver_id": "U"},
        {"transaction_id": "F", "risk_score": .1, "sender_id": "V", "receiver_id": "W"},
        {"transaction_id": "G", "risk_score": .1, "sender_id": "UNKNOWN", "receiver_id": "UNKNOWN"},
        {"transaction_id": "H", "risk_score": .1, "sender_id": "UNKNOWN", "receiver_id": "UNKNOWN"},
    ]
    selection = select_priority(linked, ExactSolver())
    assert selection.k == 2 and len(selection.edges) == 1
    # connected A+B (0.5+0.5+0.25*0.5) beat the single highest risk C (0.55)
    assert selection.selected_ids == {"A", "B"}

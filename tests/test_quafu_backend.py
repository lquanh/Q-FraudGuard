import numpy as np
import pytest

from qf_guard.prioritization import PriorityConfig, build_priority_qubo, select_priority
from qf_guard.quafu import (
    LocalSimulatorClient,
    QuafuQAOASolver,
    _permuted_energies,
    _samples_in_model_order,
    sampling_time_estimate,
)
from qf_guard.solvers import ExactSolver, FallbackSolver, solver_from_env

CONFIG = PriorityConfig(risk_normalization="max", connectivity_strength=0.25)


def window(seed, n=8):
    rng = np.random.default_rng(seed)
    return [
        {
            "transaction_id": f"T{i}",
            "risk_score": float(rng.uniform(0.001, 0.02)),
            "sender_id": f"A{rng.integers(0, 3)}",
            "receiver_id": f"B{rng.integers(0, 3)}",
        }
        for i in range(n)
    ]


class FailingClient(LocalSimulatorClient):
    def submit(self, qasm, name, shots, n):
        raise RuntimeError("network down")


@pytest.fixture(scope="module")
def mock_solver():
    return QuafuQAOASolver(client=LocalSimulatorClient(), backend_name="quafu-mock", restarts=3)


@pytest.mark.parametrize("seed", range(3))
def test_quafu_path_returns_the_certified_optimum(mock_solver, seed):
    model = build_priority_qubo(window(seed), CONFIG)
    result = mock_solver.solve(model)
    assert result.feasible
    assert result.objective == pytest.approx(ExactSolver().solve(model).objective)
    assert result.metadata["backend"] == "quafu-mock"
    assert "not quantum hardware" in result.metadata["hardware"]
    assert result.metadata["local_search"] is False  # the answer is a measured selection


def test_circuit_order_maps_back_to_candidates():
    model = build_priority_qubo(window(4), CONFIG)
    order = [3, 0, 7, 1, 6, 2, 5, 4]
    energies = _permuted_energies(model, order)
    for code in (0b00000011, 0b10100000, 0b01000100):
        bitstring = "".join(str((code >> qubit) & 1) for qubit in range(8))  # char j = circuit qubit j
        sample = _samples_in_model_order({bitstring: 1}, order)[0]
        assert model.evaluate(sample) == pytest.approx(energies[code])


def test_windows_above_the_qubit_limit_are_refused():
    solver = QuafuQAOASolver(client=LocalSimulatorClient(), max_qubits=6)
    with pytest.raises(ValueError, match="qubit limit"):
        solver.solve(build_priority_qubo(window(0), CONFIG))


def test_digital_annealing_implements_transverse_field_schedule():
    from qiskit import qasm2
    from qiskit.quantum_info import Statevector
    from qf_guard.qaoa import qaoa_probabilities, qubo_to_ising
    from qf_guard.quafu import digital_annealing_qasm

    model = build_priority_qubo(window(11, n=5), CONFIG)
    ising = qubo_to_ising(model)  # includes the exact-K penalty
    steps, duration = 3, 2.0
    s = (np.arange(steps) + 0.5) / steps
    dt = duration / steps
    expected = qaoa_probabilities(ising, dt * s, -dt * (1 - s))
    circuit = qasm2.loads(digital_annealing_qasm(ising, steps, duration))
    circuit.remove_final_measurements()
    assert np.allclose(Statevector(circuit).probabilities(), expected, atol=1e-10)


def test_anneal_backend_reports_raw_counts_in_candidate_order():
    solver = solver_from_env({"QFG_SOLVER": "quafu-anneal-mock", "QFG_SOLVER_FALLBACK": "none"})
    candidates = window(6, n=6)
    batch = select_priority(candidates, solver, CONFIG, certify_optimum=True)
    counts = batch.metadata["measured_counts"]
    returned = "".join(str(bit) for bit in batch.result.sample)
    assert sum(counts.values()) == batch.metadata["shots"]
    assert counts[returned] > 0
    assert returned.count("1") == batch.k
    assert batch.metadata["solver"] == "quafu_digital_annealing"
    assert not batch.metadata["repaired"] and not batch.metadata["local_search"]
    assert batch.certificate["matches_optimum"]


def test_quafu_token_alias_is_scoped_to_project_file(tmp_path, monkeypatch):
    from qf_guard import quafu

    monkeypatch.setattr(quafu, "PROJECT_ROOT", tmp_path)
    assert quafu.read_token({"API_TOKEN": "another-service"}) is None
    (tmp_path / ".env").write_text("API_TOKEN=local-alias\n", encoding="utf-8")
    assert quafu.read_token({}) == "local-alias"
    (tmp_path / ".env").write_text("API_TOKEN=local-alias\nQUAFU_API_TOKEN=named\n", encoding="utf-8")
    assert quafu.read_token({}) == "named"
    assert quafu.read_token({"QPU_API_TOKEN": "override"}) == "override"


@pytest.mark.parametrize("n,k", [(7, 3), (5, 4)])
def test_paired_cost_phases_match_full_hamiltonian(n, k):
    from qiskit import qasm2
    from qiskit.quantum_info import Statevector
    from qf_guard.qaoa import IsingForm, _cost_diagonal, _xy_state, _xy_tables, xy_blocks
    from qf_guard.quafu import paired_xy_qasm

    rng = np.random.default_rng(12)
    ising = IsingForm(rng.normal(size=n), {(a, b): rng.normal() for a in range(n) for b in range(a)}, 0, 1)
    basis, pairs, _ = _xy_tables(n, k)
    initial = np.zeros(len(basis))
    for i, code in enumerate(basis):
        weight = 1.0
        for block in xy_blocks(n, k):
            occupied = [q for q in block if (code >> q) & 1]
            if occupied == [block[0]]:
                weight *= 1 if len(block) == 1 else 0.8
            elif len(block) > 1 and occupied == [block[1]]:
                weight *= 0.2
            else:
                weight = 0
        initial[i] = np.sqrt(weight)
    gamma, beta = 1.2, -0.3
    expected = np.zeros(2**n, dtype=complex)
    expected[basis] = _xy_state(_cost_diagonal(ising)[basis], (basis, pairs, initial), [gamma], [beta])
    circuit = qasm2.loads(paired_xy_qasm(ising, k, gamma, beta, 0.8))
    circuit.remove_final_measurements()
    assert Statevector(circuit).equiv(Statevector(expected))


def test_native_line_has_dense_wires_before_provider_layout():
    from qiskit import qasm2
    from qf_guard.qaoa import objective_ising
    from qf_guard.quafu import _native_chain_qasm, paired_xy_qasm

    model = build_priority_qubo(window(9, n=8), CONFIG)
    qasm = paired_xy_qasm(objective_ising(model), model.exact_k, 1.2, -0.3, 0.8)
    native = qasm2.loads(_native_chain_qasm(qasm), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    assert native.num_qubits == model.n_variables
    assert native.count_ops()["measure"] == model.n_variables
    for gate in native.data:
        if gate.operation.name == "cz":
            a, b = (native.find_bit(q).index for q in gate.qubits)
            assert abs(a - b) == 1


def test_local_warm_circuit_matches_conditional_pair_energies():
    from qiskit import QuantumCircuit, qasm2
    from qiskit.quantum_info import Statevector
    from qf_guard.qaoa import xy_blocks
    from qf_guard.quafu import local_warm_qasm

    model = build_priority_qubo(window(10, n=7), CONFIG)
    order = [4, 0, 3, 6, 5, 2, 1]
    n = len(order)
    seed = np.zeros(n, dtype=np.int8)
    seed[[block[0] for block in xy_blocks(n, model.exact_k)]] = 1
    states = (np.arange(2**n)[:, None] >> np.arange(n)) & 1
    energies = np.zeros(2**n)
    coefficients = []
    for i in range(0, n, 2):
        block = list(range(i, min(i + 2, n)))
        values = []
        for mask in range(1 << len(block)):
            local = seed.copy()
            local[block] = (mask >> np.arange(len(block))) & 1
            original = np.zeros(n, dtype=np.int8)
            original[order] = local
            values.append(model.evaluate(original))
        labels = np.arange(len(values))
        for parity in range(1, len(values)):
            signs = np.array([(-1) ** int(x & parity).bit_count() for x in labels])
            coefficients.append(np.mean(signs * values))
        codes = states[:, block] @ (1 << np.arange(len(block)))
        energies += np.array(values)[codes]
    gamma, beta, p = np.pi / 2, 0.06 * np.pi, 0.9
    p1 = np.where(seed, p, model.exact_k * (1 - p) / (n - model.exact_k))
    initial = np.sqrt(np.where(states, p1, 1 - p1).prod(axis=1))
    expected = Statevector(initial * np.exp(-1j * gamma * energies / max(abs(x) for x in coefficients)))
    mixer = QuantumCircuit(n)
    mixer.rx(2 * beta, range(n))
    out = qasm2.loads(local_warm_qasm(model, order, gamma, beta, p))
    out.remove_final_measurements()
    assert Statevector(out).equiv(expected.evolve(mixer))


def test_quafu_error_without_status_ends_polling_immediately():
    from types import SimpleNamespace
    from qf_guard.quafu import QuafuClient

    client = QuafuClient.__new__(QuafuClient)
    client.client = SimpleNamespace(result=lambda tid: {"error": "Transpiler rejected layout"})
    assert client.wait(1, poll=0, max_minutes=0)["status"] == "Failed"


@pytest.mark.parametrize("hits,expected_jobs,success", [
    ([False, True, True], 2, True), ([True, False, False], 1, True), ([False, False, False], 3, False),
])
def test_first_hit_ignores_classical_repair_and_stops(hits, expected_jobs, success):
    from qf_guard.postprocess import best_result

    class MeasuredSolver:
        def __init__(self):
            self.jobs = 0

        def solve(self, model):
            # An all-zero read is repaired to the optimum by best_result. It
            # must still count as a miss. Both of the first two alerts tie.
            bits = "0100" if hits[self.jobs] else "0000"
            self.jobs += 1
            sample = np.array([int(c) for c in bits], dtype=np.int8)
            return best_result(model, sample, started=0, finished=1, local_search=False,
                               metadata={"solver": "quafu_fast_xy", "backend": "quafu-fast-mock",
                                         "measured_counts": {bits: 1024}, "task_id": self.jobs})

    solver = MeasuredSolver()
    candidates = [{"transaction_id": str(i), "risk_score": risk} for i, risk in enumerate([1, 1, 0.3, 0.1])]
    result = select_priority(candidates, solver, CONFIG, certify_optimum=True, max_measurement_jobs=3)
    cert = result.certificate
    assert solver.jobs == expected_jobs
    assert cert["measured_hit"] is success
    assert cert["measured_reads"] == 1024 * expected_jobs
    assert cert["optimal_reads"] == (1024 if success else 0)
    assert (cert["time_to_first_hit_s"] is not None) is success


def test_env_wraps_quafu_in_fallback_with_its_own_budget():
    wrapped = solver_from_env({"QFG_SOLVER": "quafu-mock"})
    assert isinstance(wrapped, FallbackSolver)
    assert isinstance(wrapped.primary, QuafuQAOASolver)
    assert wrapped.qpu_budget_seconds == 600.0
    bare = solver_from_env({"QFG_SOLVER": "quafu-mock", "QFG_SOLVER_FALLBACK": "none"})
    assert isinstance(bare, QuafuQAOASolver)


def test_sampling_estimate_uses_every_shot_in_every_batch(monkeypatch):
    monkeypatch.delenv("QFG_QUAFU_ASSUMED_SHOT_PERIOD_US", raising=False)
    assert sampling_time_estimate(8192) == {"assumed_shot_period_us": 100, "estimated_sampling_ms": 819.2}
    monkeypatch.setenv("QFG_QUAFU_ASSUMED_SHOT_PERIOD_US", "25")
    assert sampling_time_estimate(2048)["estimated_sampling_ms"] == pytest.approx(51.2)
    assert not sampling_time_estimate(None)
    monkeypatch.setenv("QFG_QUAFU_ASSUMED_SHOT_PERIOD_US", "nan")
    assert not sampling_time_estimate(1024)


def test_fast_mode_does_not_count_simulator_fallback_as_a_chip_measurement():
    fallback = solver_from_env({"QFG_SOLVER": "quafu-fast-mock", "QFG_SOLVER_FALLBACK": "none"})
    solver = FallbackSolver(primary=QuafuQAOASolver(client=FailingClient()), fallback=fallback)
    selection = select_priority(window(1), solver, CONFIG, certify_optimum=True, max_measurement_jobs=None)
    cert = selection.certificate
    assert solver.fallback_calls == 1
    assert cert["measured_hit"] is False
    assert cert["measured_reads"] == cert["optimal_reads"] == 0
    assert cert["measurement_attempts"][0]["error"] == "fallback"


def test_failed_qpu_task_falls_back_and_says_why():
    solver = FallbackSolver(primary=QuafuQAOASolver(client=FailingClient()), fallback=ExactSolver())
    result = solver.solve(build_priority_qubo(window(1), CONFIG))
    assert result.feasible
    assert result.metadata["backend"] == "cpu-exhaustive"
    assert "network down" in result.metadata["fallback_reason"]


def test_dashboard_path_certifies_the_qpu_answer(mock_solver):
    selection = select_priority(window(2), mock_solver, CONFIG, certify_optimum=True)
    assert selection.certificate["matches_optimum"] is True
    assert len(selection.selected_ids) == selection.k

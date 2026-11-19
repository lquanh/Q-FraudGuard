import numpy as np
import pandas as pd
import pytest

from qf_guard.datasets import AMLWORLD_TYPE_MAP, amlworld_to_canonical
from qf_guard.data import normalize_transactions
from qf_guard.prioritization import build_priority_qubo, certify, exact_optimum, select_priority
from qf_guard.solvers import ExactSolver, OpenJijSQASolver


def window(seed: int, n: int, accounts: int):
    rng = np.random.default_rng(seed)
    return [
        {
            "transaction_id": f"T{i}",
            "risk_score": float(rng.uniform(0.2, 1.0)),
            "sender_id": f"A{rng.integers(0, accounts)}",
            "receiver_id": f"A{rng.integers(0, accounts)}",
        }
        for i in range(n)
    ]


@pytest.mark.parametrize("seed", range(12))
def test_milp_certificate_matches_brute_force(seed):
    candidates = window(seed, n=int(8 + seed % 7), accounts=6)
    model = build_priority_qubo(candidates)
    exact = ExactSolver().solve(model)
    x, info = exact_optimum(model)
    assert info["status"] == "optimal"
    assert int(x.sum()) == model.exact_k
    assert info["energy"] == pytest.approx(exact.objective, abs=1e-9)


def test_certificate_reports_zero_gap_for_exact_solver():
    model = build_priority_qubo(window(3, n=12, accounts=5))
    cert = certify(model, ExactSolver().solve(model))
    assert cert["matches_optimum"] and cert["certified"]
    assert cert["final_gap"] == pytest.approx(0.0, abs=1e-12)


def test_certificate_handles_windows_too_large_for_brute_force():
    candidates = window(9, n=40, accounts=12)  # C(40, 10) states: exact enumeration refuses
    model = build_priority_qubo(candidates)
    with pytest.raises(ValueError):
        ExactSolver().solve(model)
    selection = select_priority(candidates, OpenJijSQASolver(num_reads=16), certify_optimum=True)
    cert = selection.certificate
    assert cert["certified"]
    assert cert["final_gap"] >= 0.0
    assert cert["solver_energy"] >= cert["optimum_energy"] - 1e-9


def test_risk_normalization_scales_inside_the_window():
    from qf_guard.prioritization import PriorityConfig, normalized_risks

    assert normalized_risks([0.002, 0.004, 0.001], "max") == pytest.approx([0.5, 1.0, 0.25])
    assert normalized_risks([0.002, 0.004, 0.001], "rank") == pytest.approx([2 / 3, 1.0, 1 / 3])
    assert normalized_risks([0.2, 0.1], "none") == pytest.approx([0.2, 0.1])
    with pytest.raises(ValueError):
        normalized_risks([0.1], "zscore")
    candidates = window(4, n=12, accounts=4)
    model = build_priority_qubo(candidates, PriorityConfig(risk_normalization="max"))
    assert max(model.metadata["risks"]) == pytest.approx(1.0)
    # normalisation never changes the order, so with c = 0 the choice is still top-K
    plain = select_priority(candidates, ExactSolver(), PriorityConfig(connectivity_strength=0.0))
    scaled = select_priority(candidates, ExactSolver(),
                             PriorityConfig(connectivity_strength=0.0, risk_normalization="max"))
    assert plain.selected_ids == scaled.selected_ids


def test_amlworld_adapter_produces_canonical_rows():
    raw = pd.DataFrame(
        {
            "Timestamp": ["2022/09/03 00:10", "2022/09/03 00:12"],
            "From Bank": [10, 70],
            "Account": ["800A", "800B"],
            "To Bank": [70, 10],
            "Account.1": ["800B", "800C"],
            "Amount Received": [100.0, 5.0],
            "Receiving Currency": ["US Dollar", "Euro"],
            "Amount Paid": [100.0, 2.0],
            "Payment Currency": ["US Dollar", "Bitcoin"],
            "Payment Format": ["ACH", "Bitcoin"],
            "Is Laundering": [0, 1],
        }
    )
    canonical = normalize_transactions(amlworld_to_canonical(raw))
    assert list(canonical["sender_id"]) == ["10:800A", "70:800B"]
    assert canonical.loc[1, "amount"] == pytest.approx(40_000.0)
    assert set(canonical["type"]) <= set(AMLWORLD_TYPE_MAP.values())
    assert canonical["is_fraud"].tolist() == [0, 1]
    # the second row's sender is the first row's receiver: AMLworld keeps account identity
    assert canonical.loc[0, "receiver_id"] == canonical.loc[1, "sender_id"]

"""Check D-Wave Leap access and run one tiny QUBO on the real QPU.

Usage (token from DWAVE_API_TOKEN or `dwave config create`):
    python scripts/quantum_check.py            # list solvers + 1 tiny QPU job
    python scripts/quantum_check.py --no-run   # only list solvers (no QPU time)

The tiny job uses ~10-30 ms of QPU access time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qf_guard.prioritization import build_priority_qubo  # noqa: E402
from qf_guard.solvers import DWaveQPUSolver, ExactSolver  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-run", action="store_true", help="do not submit a QPU job")
    parser.add_argument("--solver", help="specific QPU name, e.g. Advantage_system4.1")
    args = parser.parse_args()

    from dwave.cloud import Client

    try:
        with Client.from_config() as client:
            solvers = client.get_solvers()
            print("Solvers available to this token:")
            for solver in solvers:
                kind = "QPU" if solver.qpu else ("hybrid" if solver.hybrid else "other")
                print(f"  - {solver.name:<32} {kind}")
    except Exception as exc:
        print(f"Cannot reach D-Wave Leap: {type(exc).__name__}: {exc}")
        print("Set DWAVE_API_TOKEN (see README, 'Triển khai lên D-Wave').")
        return 1
    if args.no_run:
        return 0

    candidates = [
        {"transaction_id": f"T{i}", "risk_score": risk, "sender_id": s, "receiver_id": r}
        for i, (risk, s, r) in enumerate(
            [(.91, "A", "B"), (.42, "C", "D"), (.77, "B", "E"), (.35, "F", "G"),
             (.66, "H", "A"), (.29, "I", "J"), (.58, "K", "L"), (.31, "M", "N")]
        )
    ]
    model = build_priority_qubo(candidates, name="quantum_check")
    exact = ExactSolver().solve(model)
    result = DWaveQPUSolver(num_reads=50, solver=args.solver).solve(model)
    report = {
        "hardware": result.metadata["hardware"],
        "problem_id": result.metadata["problem_id"],
        "selected": result.selected_indices,
        "exact_optimum": exact.selected_indices,
        "raw_qpu_energy": result.metadata["raw_objective"],
        "final_energy": result.objective,
        "optimal_energy": exact.objective,
        "feasible_reads": result.metadata["num_feasible_samples"],
        "chain_break_fraction": result.metadata["chain_break_fraction"],
        "qpu_access_time_ms": result.metadata["qpu_access_time_us"] / 1000,
        "wall_time_s": result.runtime,
    }
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

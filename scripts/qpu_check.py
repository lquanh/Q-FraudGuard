"""Deployment check: send real AMLworld candidate windows through a QUBO backend.

The same code path as the dashboard (``select_priority`` with ``solver_from_env``):
every window is solved by the chosen backend and certified against the MILP
optimum on the CPU.

    python scripts/qpu_check.py --solver dwave-mock --windows 3                 # D-Wave path, offline
    python scripts/qpu_check.py --solver dwave-qpu --windows 5 --window-size 16 # D-Wave Leap QPU
    python scripts/qpu_check.py --solver dwave-hybrid --windows 3               # Leap hybrid CQM
    python scripts/qpu_check.py --solver quafu-mock --windows 3                 # Quafu path, offline
    .venv-quafu\\Scripts\\python scripts/qpu_check.py --solver quafu-qpu --windows 5   # Quafu-SQC chip

Credentials come from the environment or the git-ignored .env (DWAVE_API_TOKEN,
QPU_API_TOKEN) and are never printed.  A JSON report is written to
artifacts/deploy/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qf_guard.prioritization import PriorityConfig, candidate_edges, select_priority  # noqa: E402
from qf_guard.quafu import sampling_time_estimate  # noqa: E402
from qf_guard.solvers import SOLVER_NAMES, FallbackSolver, solver_from_env  # noqa: E402


def candidate_windows(scores: pd.DataFrame, threshold: float, size: int, count: int, seed: int) -> list[list[dict]]:
    """Consecutive flagged transactions (time order) in windows of ``size`` that contain a link."""

    flagged = scores[scores.risk_score >= threshold].sort_values("timestamp", kind="stable")
    columns = [c for c in ("transaction_id", "risk_score", "sender_id", "receiver_id", "is_fraud") if c in flagged]
    records = flagged[columns].to_dict("records")
    windows = [records[i : i + size] for i in range(0, len(records) - size + 1, size)]
    linked = [w for w in windows if candidate_edges(w)]
    order = np.random.default_rng(seed).permutation(len(linked))
    return [linked[i] for i in order[:count]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solver", default=os.environ.get("QFG_SOLVER", "sqa"), choices=SOLVER_NAMES)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--window-size", type=int, help="default: 18 for fast Quafu, 8 for other Quafu modes, 16 otherwise")
    parser.add_argument("--fallback", default=os.environ.get("QFG_SOLVER_FALLBACK", "sqa"),
                        help="backend used when the QPU fails, or 'none'")
    parser.add_argument("--scores", type=Path, default=ROOT / "artifacts" / "training" / "full_scores_test.pkl")
    parser.add_argument("--threshold", type=float, default=0.00016, help="candidate threshold of the full model")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "deploy")
    args = parser.parse_args()

    fast_quafu = args.solver.startswith("quafu-fast-")
    quafu_default = 18 if fast_quafu else 8
    size = args.window_size or (quafu_default if args.solver.startswith("quafu") else 16)
    if args.windows < 1 or (args.window_size is not None and args.window_size < 2):
        parser.error("a deployment check needs at least one window and two candidates per window")
    env = {**os.environ, "QFG_SOLVER": args.solver, "QFG_SOLVER_FALLBACK": args.fallback}
    config = PriorityConfig(
        risk_normalization=env.get("QFG_RISK_NORMALIZATION", "max").strip().lower() or "max",
        connectivity_strength=float(env.get("QFG_CONNECTIVITY_STRENGTH", "0.01")),
    )
    if args.solver.startswith("quafu-") and args.solver.endswith("-qpu"):
        try:  # informational: queue length per chip, or "Calibrating" / "Offline"
            from quark import Task

            from qf_guard.quafu import read_token

            token = read_token(env)
            if token:
                os.environ["QPU_API_TOKEN"] = token
            print(f"Quafu-SQC queue per chip: {Task().status()}", flush=True)
        except Exception as exc:
            print(f"Quafu-SQC status unavailable ({type(exc).__name__})", flush=True)
    solver = solver_from_env(env)
    windows = candidate_windows(pd.read_pickle(args.scores), args.threshold, size, args.windows, args.seed)
    print(f"backend {args.solver}: {len(windows)} windows of {size} candidates", flush=True)

    rows = []
    for index, window in enumerate(windows):
        started = time.perf_counter()
        try:
            selection = select_priority(window, solver, config, name=f"check_{index:02d}", certify_optimum=True,
                                        max_measurement_jobs=None if fast_quafu else 3)
        except Exception as exc:
            rows.append(dict(window=index, n=len(window), backend=None, certified=False,
                             matches_optimum=False, primary_only=False, error=type(exc).__name__))
            print(f"  window {index}: {args.solver} failed ({type(exc).__name__})", flush=True)
            break
        meta, cert = dict(selection.metadata), dict(selection.certificate or {})
        row = {
            "window": index,
            "n": len(window),
            "k": selection.k,
            "links": len(selection.edges),
            "backend": meta.get("backend"),
            "hardware": meta.get("hardware"),
            "job_id": meta.get("task_id") or meta.get("problem_id"),
            "chip": meta.get("chip"),
            "seconds": round(time.perf_counter() - started, 2),
            "server_time_s": meta.get("server_time_s"),
            "qpu_access_time_us": meta.get("qpu_access_time_us"),
            "matches_optimum": cert.get("matches_optimum"),
            "certified": bool(cert.get("certified")),
            "final_gap": cert.get("final_gap"),
            "fallback_reason": meta.get("fallback_reason"),
            "primary_only": not meta.get("fallback_reason") and not any(
                attempt.get("error") for attempt in cert.get("measurement_attempts", ())
            ),
            "selected": sorted(selection.selected_ids),
        }
        if "measured_counts" in meta:
            counts = Counter()
            attempts = cert.get("measurement_attempts")
            for attempt in attempts or [meta]:
                counts.update(attempt.get("measured_counts", {}))
            row.update({key: meta[key] for key in ("schedule", "trotter_steps", "dimensionless_duration",
                       "qaoa_layers", "gamma", "beta", "leader_bias") if key in meta})
            row.update(
                algorithm=meta["solver"], physical_qubits=meta["physical_qubits"],
                readout_qubits=meta["readout_qubits"],
                logical_2q=meta["logical_2q"], transpiled_2q=meta["transpiled_2q"],
                candidate_ids=list(selection.transaction_ids), measured_counts=counts,
                measured_reads=cert.get("measured_reads"), feasible_reads=cert.get("feasible_reads"),
                optimal_reads=cert.get("optimal_reads"), measured_hit=cert.get("measured_hit"),
                optimum_energy=cert["optimum_energy"], returned_energy=cert["solver_energy"],
                repaired=meta["repaired"], local_search=meta["local_search"],
            )
            if attempts:
                row.update(attempts=attempts, time_to_first_hit_s=cert["time_to_first_hit_s"])
        if fast_quafu:
            row["primary_only"] = row["primary_only"] and row["backend"] == args.solver
            row.update(sampling_time_estimate(row.get("measured_reads")))
            verdict = "measured optimum" if row["primary_only"] and row.get("measured_hit") else "no confirmed primary hit"
            detail = f"{row.get('optimal_reads') or 0}/{row.get('measured_reads') or 0} optimal shots"
            if row.get("time_to_first_hit_s") is not None:
                detail += f", client received winning batch in {row['time_to_first_hit_s']:.3f} s"
            if "estimated_sampling_ms" in row:
                detail += f", estimated sampling {row['estimated_sampling_ms']:.1f} ms at {row['assumed_shot_period_us']:g} us/shot"
        else:
            verdict = "optimal" if row["matches_optimum"] else f"gap {row['final_gap']}"
            detail = f"{row['seconds']} s"
        rows.append(row)
        where = f"fallback ({row['fallback_reason']})" if row["fallback_reason"] else row["hardware"]
        print(f"  window {index}: K={row['k']}/{row['n']} -> {verdict}; {row['backend']} on {where}; "
              f"job {row['job_id']}; {detail}", flush=True)

    optimal = sum(r["certified"] and bool(r["matches_optimum"]) for r in rows)
    on_primary = sum(r["primary_only"] for r in rows)
    if fast_quafu:
        hits = sum(bool(r.get("measured_hit") and r["primary_only"] and r["certified"]) for r in rows)
        print(f"Measured certified optimum on {args.solver} in {hits}/{args.windows} requested windows.")
    else:
        print(f"optimal (MILP-certified): {optimal}/{len(rows)}; answered by {args.solver}: {on_primary}/{len(rows)}")
    if isinstance(solver, FallbackSolver) and not fast_quafu:
        print(f"solver stats: {solver.stats()}")
    passed = len(rows) == args.windows and optimal == on_primary == args.windows
    if fast_quafu:
        passed = passed and hits == args.windows
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"qpu_check_{args.solver}_{stamp}.json"
    path.write_text(json.dumps({"solver": args.solver, "window_size": size, "passed": passed,
        "requested_windows": args.windows, "primary_windows": on_primary, "config": {
        "risk_normalization": config.risk_normalization, "connectivity_strength": config.connectivity_strength},
        "rows": rows}, indent=2, default=str), encoding="utf-8")
    print(f"report -> {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

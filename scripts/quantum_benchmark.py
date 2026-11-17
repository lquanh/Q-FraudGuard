"""Benchmark QUBO backends on real PaySim candidate windows.

Every window is solved by each requested backend and compared with the exact
optimum and with a purely classical "top-K by risk" baseline.

Usage:
    python scripts/quantum_benchmark.py --solvers exact,sqa,dwave-mock
    python scripts/quantum_benchmark.py --solvers exact,sqa,dwave-qpu --windows 10

``dwave-qpu`` / ``dwave-hybrid`` use real Leap resources (DWAVE_API_TOKEN) and
never fall back silently: a failure is reported as an error for that window.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qf_guard.prioritization import PriorityConfig, build_priority_qubo  # noqa: E402
from qf_guard.scoring import RiskScorer  # noqa: E402
from qf_guard.solvers import SOLVER_NAMES, build_solver  # noqa: E402
from qf_guard.training import temporal_split  # noqa: E402


def candidate_windows(model_path: Path, data_path: Path, size: int) -> list[list[dict]]:
    scorer = RiskScorer.load(model_path)
    train, validation, test = temporal_split(pd.read_csv(data_path))
    scorer.history_transactions_ = scorer._feature_history(train, validation)
    scored = scorer.score(test)
    candidates = scored.loc[scored["is_candidate"]]
    records = [
        {
            "transaction_id": str(row.transaction_id),
            "risk_score": float(row.risk_score),
            "sender_id": str(row.sender_id),
            "receiver_id": str(row.receiver_id),
            "is_fraud": int(row.is_fraud),
        }
        for row in candidates.itertuples(index=False)
    ]
    return [records[i : i + size] for i in range(0, len(records) - size + 1, size)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solvers", default="exact,sqa,dwave-mock")
    parser.add_argument("--windows", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--model", type=Path,
                        default=ROOT / "artifacts" / "models" / "paysim_lightgbm_scorer.joblib")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "paysim.csv")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    names = [name.strip() for name in args.solvers.split(",") if name.strip()]
    unknown = [name for name in names if name not in SOLVER_NAMES]
    if unknown:
        parser.error(f"unknown solvers {unknown}; choose from {SOLVER_NAMES}")
    if "exact" not in names:
        names.insert(0, "exact")
    solvers = {name: build_solver(name) for name in names}
    windows = candidate_windows(args.model, args.data, args.window_size)[: args.windows]
    if not windows:
        print("not enough candidates for one window")
        return 1
    print(f"{len(windows)} windows x {args.window_size} candidates, solvers: {', '.join(names)}")

    config = PriorityConfig()
    rows: list[dict] = []
    for index, window in enumerate(windows):
        model = build_priority_qubo(window, config, name=f"bench_{index:03d}")
        labels = np.array([candidate["is_fraud"] for candidate in window])
        top_k = np.argsort([-c["risk_score"] for c in window], kind="stable")[: model.exact_k]
        greedy = np.zeros(model.n_variables, dtype=np.int8)
        greedy[top_k] = 1
        optimum = None
        for name, solver in solvers.items():
            row = {"window": index, "solver": name, "n": model.n_variables, "k": model.exact_k,
                   "edges": len(model.metadata["suspicious_edges"]),
                   "frauds_in_window": int(labels.sum()),
                   "greedy_topk_energy": model.evaluate(greedy),
                   "greedy_topk_frauds": int(labels[top_k].sum())}
            try:
                result = solver.solve(model)
            except Exception as exc:
                rows.append({**row, "error": f"{type(exc).__name__}: {exc}"[:300]})
                continue
            if name == "exact":
                optimum = result.objective
            selected = list(result.selected_indices)
            rows.append({
                **row,
                "energy": result.objective,
                "raw_energy": result.metadata.get("raw_objective", result.objective),
                "optimal": bool(optimum is not None and abs(result.objective - optimum) < 1e-9),
                "raw_optimal": bool(
                    optimum is not None
                    and abs(result.metadata.get("raw_objective", result.objective) - optimum) < 1e-9
                ),
                "gap": None if optimum is None else result.objective - optimum,
                "feasible": result.feasible,
                "repaired": result.metadata.get("repaired"),
                "runtime_s": result.runtime,
                "qpu_access_time_us": result.metadata.get("qpu_access_time_us", 0.0),
                "chain_break_fraction": result.metadata.get("chain_break_fraction"),
                "problem_id": result.metadata.get("problem_id"),
                "hardware": result.metadata.get("hardware"),
                "selected_frauds": int(labels[selected].sum()),
                "same_as_greedy_topk": sorted(selected) == sorted(top_k.tolist()),
            })

    frame = pd.DataFrame(rows)
    ok = frame[frame.get("error").isna()] if "error" in frame else frame
    summary = (
        ok.groupby("solver", sort=False)
        .agg(windows=("window", "count"), optimal=("optimal", "mean"),
             raw_optimal=("raw_optimal", "mean"), mean_gap=("gap", "mean"),
             feasible=("feasible", "mean"), mean_runtime_s=("runtime_s", "mean"),
             qpu_ms_total=("qpu_access_time_us", lambda s: float(np.nansum(s)) / 1000),
             selected_frauds=("selected_frauds", "sum"),
             same_as_greedy=("same_as_greedy_topk", "mean"))
    )
    print(summary.to_string(float_format=lambda value: f"{value:.4g}"))
    errors = frame[frame["error"].notna()] if "error" in frame else frame.iloc[0:0]
    if len(errors):
        print(f"\n{len(errors)} failed solves, first: {errors.iloc[0]['solver']}: "
              f"{errors.iloc[0]['error']}")
    first = frame.drop_duplicates("window")
    print(f"\nclassical top-K baseline: {int(first['greedy_topk_frauds'].sum())} frauds selected; "
          f"windows contain {int(first['frauds_in_window'].sum())} frauds in total")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or ROOT / "artifacts" / "benchmarks" / f"quantum_benchmark_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "created_utc": stamp,
        "solvers": names,
        "window_size": args.window_size,
        "summary": json.loads(summary.to_json(orient="index")),
        "rows": json.loads(frame.to_json(orient="records")),
    }, indent=2), encoding="utf-8")
    print(f"report -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Benchmark QUBO backends on real candidate windows (PaySim or AMLworld).

Every window is solved by each requested backend and compared with

* the certified optimum (HiGHS MILP, works for any window size),
* a purely classical "top-K by risk" baseline (does the QUBO change the decision?),
* a "random K + swap" control (does the backend add anything beyond the shared
  classical post-processing?).

Usage:
    python scripts/quantum_benchmark.py --dataset amlworld --solvers sqa --window-size 16
    python scripts/quantum_benchmark.py --solvers exact,sqa,dwave-mock
    python scripts/quantum_benchmark.py --solvers sqa,dwave-qpu --windows 10

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

from qf_guard.postprocess import swap_local_search  # noqa: E402
from qf_guard.prioritization import PriorityConfig, build_priority_qubo, exact_optimum  # noqa: E402
from qf_guard.scoring import RiskScorer  # noqa: E402
from qf_guard.solvers import SOLVER_NAMES, build_solver  # noqa: E402
from qf_guard.training import temporal_split  # noqa: E402

DATASETS = {
    "paysim": ("paysim_lightgbm_scorer.joblib", "paysim.csv"),
    "amlworld": ("amlworld_lightgbm_scorer.joblib", "amlworld_hi_small.csv"),
}


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
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="paysim")
    parser.add_argument("--solvers", default="sqa")
    parser.add_argument("--windows", type=int, default=50)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--model", type=Path, help="default: artifacts/models/<dataset> scorer")
    parser.add_argument("--data", type=Path, help="default: data/<dataset> csv")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random-K + swap control")
    parser.add_argument("--connectivity-strength", type=float, default=PriorityConfig().connectivity_strength,
                        help="QUBO reward for candidates sharing an account (c)")
    parser.add_argument("--risk-normalization", choices=("none", "max", "rank"), default="none",
                        help="scale risk scores inside each window before building the QUBO")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model_name, data_name = DATASETS[args.dataset]
    model_path = args.model or ROOT / "artifacts" / "models" / model_name
    data_path = args.data or ROOT / "data" / data_name

    names = [name.strip() for name in args.solvers.split(",") if name.strip()]
    unknown = [name for name in names if name not in SOLVER_NAMES]
    if unknown:
        parser.error(f"unknown solvers {unknown}; choose from {SOLVER_NAMES}")
    solvers = {name: build_solver(name) for name in names}
    windows = candidate_windows(model_path, data_path, args.window_size)[: args.windows]
    if not windows:
        print("not enough candidates for one window")
        return 1
    print(f"[{args.dataset}] {len(windows)} windows x {args.window_size} candidates, "
          f"solvers: {', '.join(names)} (reference: MILP certificate)")

    rng = np.random.default_rng(args.seed)
    config = PriorityConfig(connectivity_strength=args.connectivity_strength,
                            risk_normalization=args.risk_normalization)
    rows: list[dict] = []
    window_rows: list[dict] = []
    for index, window in enumerate(windows):
        model = build_priority_qubo(window, config, name=f"bench_{index:03d}")
        labels = np.array([candidate["is_fraud"] for candidate in window])
        top_k = np.argsort([-c["risk_score"] for c in window], kind="stable")[: model.exact_k]
        greedy = np.zeros(model.n_variables, dtype=np.int8)
        greedy[top_k] = 1
        x_opt, info = exact_optimum(model)
        optimum = float(info["energy"])
        random_k = np.zeros(model.n_variables, dtype=np.int8)
        random_k[rng.choice(model.n_variables, model.exact_k, replace=False)] = 1
        control = swap_local_search(model, random_k)
        window_rows.append({
            "window": index, "n": model.n_variables, "k": model.exact_k,
            "edges": len(model.metadata["suspicious_edges"]),
            "optimum_differs_from_topk": bool(model.evaluate(greedy) > optimum + 1e-9),
            "random_swap_optimal": bool(abs(model.evaluate(control) - optimum) < 1e-9),
            "milp_ms": info["seconds"] * 1000,
            "frauds_in_window": int(labels.sum()),
            "optimum_frauds": int(labels[x_opt == 1].sum()),
            "topk_frauds": int(labels[top_k].sum()),
        })
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
    windows_frame = pd.DataFrame(window_rows)
    ok = frame[frame.get("error").isna()] if "error" in frame else frame
    summary = (
        ok.groupby("solver", sort=False)
        .agg(windows=("window", "count"), optimal=("optimal", "mean"),
             raw_optimal=("raw_optimal", "mean"), mean_gap=("gap", "mean"),
             feasible=("feasible", "mean"), mean_runtime_s=("runtime_s", "mean"),
             qpu_ms_total=("qpu_access_time_us", lambda s: float(np.nansum(s)) / 1000),
             selected_frauds=("selected_frauds", "sum"),
             same_as_greedy=("same_as_greedy_topk", "mean"))
    ) if len(ok) else pd.DataFrame()
    print(summary.to_string(float_format=lambda value: f"{value:.4g}"))
    errors = frame[frame["error"].notna()] if "error" in frame else frame.iloc[0:0]
    if len(errors):
        print(f"\n{len(errors)} failed solves, first: {errors.iloc[0]['solver']}: "
              f"{errors.iloc[0]['error']}")
    structure = {
        "windows": int(len(windows_frame)),
        "windows_with_links": int((windows_frame["edges"] > 0).sum()),
        "mean_links": float(windows_frame["edges"].mean()),
        "optimum_differs_from_topk": int(windows_frame["optimum_differs_from_topk"].sum()),
        "random_swap_control_optimal": float(windows_frame["random_swap_optimal"].mean()),
        "milp_ms_mean": float(windows_frame["milp_ms"].mean()),
        "frauds_in_windows": int(windows_frame["frauds_in_window"].sum()),
        "frauds_selected_by_optimum": int(windows_frame["optimum_frauds"].sum()),
        "frauds_selected_by_topk": int(windows_frame["topk_frauds"].sum()),
    }
    print("\nwindow structure: " + json.dumps(structure))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or ROOT / "artifacts" / "benchmarks" / f"quantum_benchmark_{args.dataset}_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "created_utc": stamp,
        "dataset": args.dataset,
        "solvers": names,
        "window_size": args.window_size,
        "connectivity_strength": args.connectivity_strength,
        "structure": structure,
        "summary": json.loads(summary.to_json(orient="index")) if len(summary) else {},
        "windows": json.loads(windows_frame.to_json(orient="records")),
        "rows": json.loads(frame.to_json(orient="records")),
    }, indent=2), encoding="utf-8")
    print(f"report -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

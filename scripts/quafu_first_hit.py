"""Check whether each AMLworld window yields a raw sample at the certified optimum.

Save the measured bit strings and the quality certificate. Estimate sampling time
from the shot count and an assumed shot period.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qpu_check import candidate_windows, measurement_evidence  # noqa: E402
from qf_guard.postprocess import best_result, swap_local_search  # noqa: E402
from qf_guard.prioritization import PriorityConfig, build_priority_qubo, exact_optimum, select_priority, selection_size  # noqa: E402
from qf_guard.quafu import read_token, sampling_time_estimate  # noqa: E402
from qf_guard.solvers import ExactSolver, build_solver  # noqa: E402


def cpu_controls(window, config):
    """Evaluate the three reference selections on the same scored window."""
    model = build_priority_qubo(window, config)
    k = selection_size(len(window), config.selection_fraction)
    ranked = sorted(range(len(window)), key=lambda i: -window[i]["risk_score"])[:k]
    top_k = np.zeros(len(window), dtype=np.int8)
    top_k[ranked] = 1
    initial = np.zeros(len(window), dtype=np.int8)
    initial[np.argsort(-np.asarray(model.metadata["risks"]), kind="stable")[:k]] = 1
    improved = swap_local_search(model, initial)
    return {
        "top_k": {"energy": model.evaluate(top_k)},
        "top_k_swap": {"energy": model.evaluate(improved)},
        "exact": {"energy": float(ExactSolver().solve(model).objective)},
    }


def save(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def finish_window(row, window, config):
    """Finalize from saved histograms, including after an interruption following the winning job."""
    attempts = row["attempts"]
    winner = next((a for a in attempts if a["measured_hit"]), None)
    last = winner or attempts[-1]
    model = build_priority_qubo(window, config)
    if "certified" in last:
        certified, optimum = last["certified"], last["optimum_energy"]
    else:
        # Older checkpoints stored the counts but not the reference.
        _, reference = exact_optimum(model)
        certified, optimum = reference["status"] == "optimal", reference["energy"]
    if not certified:
        raise RuntimeError("the reference solver did not certify optimality")
    tol = 1e-9 * max(1, abs(optimum))
    for control in row["cpu"].values():
        control["hit"] = abs(control["energy"] - optimum) <= tol
    if not row["cpu"]["exact"]["hit"]:
        raise RuntimeError("MILP optimum disagrees with exhaustive enumeration")

    records = [winner] if winner else attempts
    strings = dict.fromkeys(bits for a in records for bits, count in a["measured_counts"].items() if count > 0)
    samples = np.frombuffer("".join(strings).encode("ascii"), dtype=np.uint8).reshape(-1, len(window)) - 48
    best = best_result(model, samples, started=0, finished=0, metadata={}, local_search=False)
    has_raw_optimum = not best.metadata["repaired"] and abs(best.objective - optimum) <= tol
    if bool(winner) != has_raw_optimum:
        raise RuntimeError("saved measurements disagree with the hit certificate")
    row.update(status="hit" if winner else "miss", k=model.exact_k,
               selected=sorted(row["candidate_ids"][idx] for idx in best.selected_indices),
               optimum_energy=optimum, certified=True, optimal_reads=sum(a["optimal_reads"] for a in attempts),
               measured_reads=sum(a["measured_reads"] for a in attempts),
               **{key: last.get(key) for key in ("gamma", "beta", "leader_bias", "leader_probability")})
    row.pop("error", None)
    row.update(sampling_time_estimate(row["measured_reads"]))


def stress_windows(windows, config, count):
    """Select a separate diagnostic set where plain top-K is suboptimal. Do not pass the solution to the QPU."""
    selected = []
    exact = ExactSolver()
    for window in windows:
        model = build_priority_qubo(window, config)
        ranked = np.zeros(model.n_variables, dtype=np.int8)
        ranked[np.argsort(-np.asarray(model.metadata["risks"]), kind="stable")[:model.exact_k]] = 1
        optimum = exact.solve(model).objective
        if model.evaluate(ranked) > optimum + 1e-9 * max(1.0, abs(optimum)):
            selected.append(window)
            if len(selected) == count:
                break
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=["quafu-fast-qpu", "quafu-fast-mock"], default="quafu-fast-qpu")
    parser.add_argument("--sizes", nargs="+", type=int, default=[8, 12, 15, 18])
    parser.add_argument("--windows", type=int, default=10, help="number of linked windows per size")
    parser.add_argument("--max-jobs", type=int, default=3)
    parser.add_argument("--until-hit", action="store_true", help="keep measuring each window until a certified raw hit")
    parser.add_argument("--where-ranking-misses", action="store_true", help="separate challenge set where top-K is suboptimal")
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--scores", type=Path, default=ROOT / "artifacts/training/full_scores_test.pkl")
    parser.add_argument("--report", type=Path, help="checkpoint file; resume the same plan if it exists")
    args = parser.parse_args()
    if min(args.windows, args.max_jobs) < 1 or any(n < 2 for n in args.sizes) or len(set(args.sizes)) != len(args.sizes):
        parser.error("provide positive windows and jobs, and distinct sizes of at least two")
    config = PriorityConfig(risk_normalization="max", connectivity_strength=0.01)
    solver = build_solver(args.solver)  # no CPU fallback
    scores = pd.read_pickle(args.scores)
    pool_size = len(scores) if args.where_ranking_misses else args.windows
    groups = {n: candidate_windows(scores, 0.00016, n, pool_size, args.seed) for n in args.sizes}
    if args.where_ranking_misses:
        groups = {n: stress_windows(windows, config, args.windows) for n, windows in groups.items()}
    if any(len(ws) != args.windows for ws in groups.values()):
        parser.error("not enough linked windows to run the full plan")
    plan = [(n, i, groups[n][i]) for i in range(args.windows) for n in args.sizes]
    rows = [{"n": n, "window": i, "candidate_ids": [w["transaction_id"] for w in win], "status": "pending"}
            for n, i, win in plan]
    settings = {"solver": args.solver, "seed": args.seed, "sizes": args.sizes, "windows_per_size": args.windows,
                "max_jobs": None if args.until_hit else args.max_jobs, "shots_per_job": solver.shots, "scores": args.scores.name,
                "risk_normalization": config.risk_normalization, "connectivity_strength": config.connectivity_strength,
                "variant": solver.fast_variant,
                "selection": "linked disjoint windows within each size, sampled before solving"}
    if args.where_ranking_misses:
        settings["selection"] += ", restricted to top-K suboptimal under exhaustive enumeration"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = args.report or ROOT / "artifacts/deploy" / f"quafu_first_hit_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"settings": settings, "rows": rows}
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
        report["settings"].pop("cpu_repeats", None)
        if settings != report["settings"] or [r["candidate_ids"] for r in rows] != [r["candidate_ids"] for r in report["rows"]]:
            parser.error("existing checkpoint refers to a different experiment")
        for saved_row in report["rows"]:
            for key in ("time_to_first_hit_s", "time_to_verified_hit_s", "elapsed_s"):
                saved_row.pop(key, None)
            for control in saved_row.get("cpu", {}).values():
                control.pop("seconds", None)
            if "attempts" in saved_row:
                saved_row["attempts"] = [measurement_evidence(attempt) for attempt in saved_row["attempts"]]
    save(path, report)
    print(f"report -> {path}", flush=True)
    for row, (n, i, window) in zip(report["rows"], plan, strict=True):
        if row["status"] != "pending" and not (args.until_hit and row["status"] in ("miss", "error")):
            continue
        if not row.get("cpu"):
            row["cpu"] = cpu_controls(window, config)
        previous = list(row.get("attempts", []))

        def checkpoint(attempt):
            row.setdefault("attempts", []).append(measurement_evidence(attempt))
            save(path, report)

        try:
            remaining = None if args.until_hit else args.max_jobs - len(previous)
            if not any(a.get("measured_hit") for a in previous) and (remaining is None or remaining > 0):
                selection = select_priority(window, solver, config, name=f"hit_{n}_{i:02d}", certify_optimum=True,
                                            max_measurement_jobs=remaining, on_measurement=checkpoint,
                                            attempt_offset=len(previous))
                certified_jobs = (selection.certificate or {}).get("measurement_attempts", [])
                if not certified_jobs or any(a.get("error") for a in certified_jobs):
                    raise RuntimeError("solver did not return the requested measurements")
            finish_window(row, window, config)
        except Exception as exc:
            message = str(exc)
            token = read_token()
            if token:
                message = message.replace(token, "[redacted]")
            row.update(status="error", error=f"{type(exc).__name__}: {message}"[:400])
        save(path, report)
        ids = [a.get("task_id") for a in row.get("attempts", [])]
        details = []
        if "estimated_sampling_ms" in row:
            details.append(f"estimated sampling {row['estimated_sampling_ms']:.1f} ms "
                           f"at {row['assumed_shot_period_us']:g} us/shot")
        print(f"N={n} window={i} {row['status']} optimal shots={row.get('optimal_reads')}/{row.get('measured_reads')} "
              f"jobs={ids} {', '.join(details)} {row.get('error', '')}".rstrip(), flush=True)
    hits = sum(row["status"] == "hit" for row in report["rows"])
    print(f"Measured certified optimum in {hits}/{len(report['rows'])} windows.", flush=True)
    return int(any(row["status"] == "error" or (args.until_hit and row["status"] != "hit")
                   for row in report["rows"]))


if __name__ == "__main__":
    raise SystemExit(main())

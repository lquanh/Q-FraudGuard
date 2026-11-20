"""Execute and record investigation windows independently of a browser session."""
from __future__ import annotations

import json
from importlib.util import find_spec
import os
from threading import Lock
from time import perf_counter

import pandas as pd

from qf_guard.prioritization import PriorityConfig, select_priority
from qf_guard.quafu import sampling_time_estimate
from qf_guard.solvers import FallbackSolver, solver_from_env


RECORDING = "quafu_dashboard_18_20261118T220101Z"
_SOLVERS = {}
_SOLVER_LOCK = Lock()
SWITCHABLE_BACKENDS = {"exact": "CPU · Giải chính xác", "quafu-fast-mock": "Mô phỏng Quafu trên CPU",
                       "quafu-fast-qpu": "QPU Quafu", "dwave-qpu": "QPU D-Wave"}


def backend_requirement(backend):
    """Check local prerequisites without contacting either hardware provider."""
    if backend.startswith("quafu-"):
        if find_spec("qiskit") is None:
            return "Cần cài Qiskit trong môi trường chạy dashboard và tiến trình chọn hồ sơ."
        if backend.endswith("-qpu"):
            from qf_guard.quafu import read_token
            if find_spec("quark") is None or not read_token():
                return "Cần SDK Quafu và QPU_API_TOKEN trên máy chủ."
    if backend == "dwave-qpu":
        if find_spec("dwave") is None or find_spec("dwave.system") is None:
            return "Cần cài D-Wave Ocean trên máy chủ."
        if not os.environ.get("DWAVE_API_TOKEN"):
            from dwave.cloud.config import load_config
            if not load_config().get("token"):
                return "Cần DWAVE_API_TOKEN hoặc cấu hình D-Wave Leap trên máy chủ."
    return None


def configured_priority():
    return PriorityConfig(
        connectivity_strength=float(os.environ.get("QFG_CONNECTIVITY_STRENGTH", "0.01")),
        risk_normalization=os.environ.get("QFG_RISK_NORMALIZATION", "max").strip().lower() or "max",
    )


def load_haituo_recording(store, root):
    evidence = json.loads((root / "artifacts/deploy" / (RECORDING + ".json")).read_bytes())
    with (root / "data/quafu_dashboard_replay.csv").open(encoding="utf-8") as source:
        source.readline()  # dataset attribution
        policy = json.loads(source.readline().removeprefix("# "))
        scored = pd.read_csv(source, keep_default_na=False, float_precision="round_trip")
    store.load_recording(scored, evidence, recording_id=RECORDING, policy=policy)


def dashboard_solver(backend, timeout):
    # Concurrent first submissions must share the same process-wide QPU budget.
    key = (backend, timeout)
    with _SOLVER_LOCK:
        if key not in _SOLVERS:
            solver = solver_from_env({**os.environ, "QFG_SOLVER": backend})
            primary = solver.primary if isinstance(solver, FallbackSolver) else solver
            if hasattr(primary, "max_wait_minutes"):
                primary.max_wait_minutes = min(primary.max_wait_minutes, timeout / 60)
            _SOLVERS[key] = solver
        return _SOLVERS[key]


def is_qpu_backend(backend):
    return backend.endswith("-qpu") or backend == "dwave-hybrid"


def _summary(selection):
    meta, cert, result = selection.metadata, selection.certificate or {}, selection.result
    info = {k: meta.get(k) for k in ("backend", "hardware", "fallback_reason", "problem_id",
                                     "repaired", "local_search_gain")}
    info.update({k: cert.get(k) for k in ("certified", "matches_optimum", "raw_matches_optimum", "optimum_energy",
                                         "raw_gap", "final_gap", "measured_hit", "optimal_reads", "measured_reads")})
    info.update(n=len(selection.transaction_ids), k=selection.k, edges=len(selection.edges),
                selected_energy=float(result.objective) if result is not None else None,
                task_id=str(meta["task_id"]) if meta.get("task_id") is not None else None)
    backend = info["backend"] or ""
    if is_qpu_backend(backend):
        if backend.startswith("quafu-") and not info["fallback_reason"]:
            info.update(sampling_time_estimate(info["measured_reads"]))
    else:
        info.update(solver_ms=float(result.runtime) * 1000 if result is not None else None,
                    certify_ms=cert.get("certify_ms"))
    return info


def solve_review_window(store, lease, batch, *, backend, config, max_jobs, timeout,
                        solver_factory=dashboard_solver):
    started = perf_counter()
    summary = {}
    number = 0

    def measured(attempt):
        nonlocal number
        number += 1
        detail = {key: attempt.get(key) for key in ("backend", "chip", "shots", "measured_reads", "optimal_reads",
                                                    "measured_hit")}
        if attempt.get("task_id") is not None:
            detail["task_id"] = str(attempt["task_id"])
        if not store.record_measurement(lease, number, detail):
            raise TimeoutError("Cửa sổ đã hết hạn hoặc đã chuyển sang lần thử khác.")

    try:
        if not store.active_run(lease):
            store.expire_windows()
            return
        solver = solver_factory(backend, timeout)
        if not store.active_run(lease):
            store.expire_windows()
            return
        selection = select_priority(batch, solver, config, name="window_" + lease, certify_optimum=True,
                                    max_measurement_jobs=max_jobs, on_measurement=measured, accept_fallback=True)
        summary = _summary(selection)
        misses = (selection.certificate or {}).get("measurement_attempts", [])
        last_error = None
        for attempt in misses:
            if attempt.get("error"):
                last_error = attempt.get("error_detail") or attempt["error"]
                store.record_measurement(lease, number + 1, {"error": last_error, "backend": attempt.get("backend")})
        if (backend.startswith("quafu-fast-") and selection.result is not None
                and not summary["measured_hit"] and not summary["fallback_reason"]):
            reason = last_error or f"Chưa có mẫu tối ưu sau {number} lô."
            if not isinstance(solver, FallbackSolver) or not store.active_run(lease):
                raise RuntimeError(reason)
            selection = select_priority(batch, solver.fallback, config, name="fallback_" + lease, certify_optimum=True,
                                        max_measurement_jobs=max_jobs)
            summary = _summary(selection)
            summary["fallback_reason"] = reason
        if selection.result is not None and not selection.result.feasible:
            raise ValueError("Bộ giải trả về danh sách không thỏa ràng buộc.")
        if not is_qpu_backend(backend):
            summary["client_elapsed_s"] = perf_counter() - started
        store.finish_window(lease, selection.selected_ids, summary=summary)
    except Exception as exc:
        store.finish_window(lease, None, summary=summary, error=f"{type(exc).__name__}: {exc}"[:500])

"""Quafu-SQC (BAQIS) superconducting QPU backend for the investigation QUBO.

By default each window becomes a constraint-preserving QAOA circuit: one W state per block
of qubits, the investigation objective as phases, and an XY mixer on
neighbouring qubits, so every basis state of the ideal circuit selects exactly
K candidates.  Candidates are ordered so linked pairs sit side by side on the
chip's chain of qubits.  Angles are trained on the exact simulator before the
job is sent (windows of up to ~14 qubits); the answer is the best measured
selection. The digital annealing backend instead uses a fixed transverse-field
schedule and the full D-Wave QUBO, including its exact-K penalty.

The Quafu SDK (``quarkstudio``) lives in the isolated ``.venv-quafu``
environment, so the real backend runs under that interpreter.  ``quafu-mock``
sends the identical circuit to a local noiseless simulator instead.  The token
is read from ``QPU_API_TOKEN`` / ``QUAFU_API_TOKEN`` (or the git-ignored .env)
and is never printed or stored.
"""

from __future__ import annotations

import contextlib
import heapq
import io
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from qf_guard.postprocess import best_result
from qf_guard.qaoa import IsingForm, all_energies, line_order, objective_ising, optimise_xy_angles, qubo_to_ising, xy_blocks
from qf_guard.qubo import QUBOModel
from qf_guard.results import SolverResult

__all__ = [
    "read_token",
    "sampling_time_estimate",
    "xy_qasm_circuit",
    "digital_annealing_qasm",
    "sample_qasm",
    "two_qubit_gates",
    "QuafuClient",
    "LocalSimulatorClient",
    "QuafuQAOASolver",
    "QuafuCircuitSolver",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sampling_time_estimate(measured_reads: int | None) -> dict[str, float]:
    """Estimate sampling for complete batches at an assumed full shot period, including reset and readout."""
    if measured_reads is None or measured_reads <= 0:
        return {}
    try:
        period_us = float(os.environ.get("QFG_QUAFU_ASSUMED_SHOT_PERIOD_US", "100"))
    except ValueError:
        return {}
    milliseconds = measured_reads * (period_us / 1000)
    if not math.isfinite(milliseconds) or milliseconds <= 0:
        return {}
    return {"assumed_shot_period_us": period_us, "estimated_sampling_ms": milliseconds}


def read_token(env: Mapping[str, str] | None = None) -> str | None:
    """Quafu-SQC token from the environment or the git-ignored .env (never logged)."""

    env = os.environ if env is None else env
    keys = ("QPU_API_TOKEN", "QUAFU_API_TOKEN")
    for key in keys:
        if token := (env.get(key) or "").strip():
            return token
    dotenv = PROJECT_ROOT / ".env"
    if not dotenv.is_file():
        return None
    # API_TOKEN is the legacy key in this project's .env. A generic token
    # in the process environment may belong to an unrelated service.
    local_keys = (*keys, "API_TOKEN")
    values = {}
    for line in dotenv.read_text(encoding="utf-8-sig").splitlines():
        name, separator, value = line.strip().removeprefix("export ").partition("=")
        if separator and name.strip() in local_keys:
            values[name.strip()] = value.strip().strip('"').strip("'")
    return next((values[k] for k in local_keys if values.get(k)), None)


def _measured_qasm(circuit, optimization_level: int = 3) -> str:
    from qiskit import qasm2, transpile

    circuit.measure(range(circuit.num_qubits), range(circuit.num_qubits))
    compiled = transpile(circuit, basis_gates=["cx", "rz", "ry", "rx", "h", "x"], optimization_level=optimization_level)
    return qasm2.dumps(compiled)


def xy_qasm_circuit(ising: IsingForm, k: int, gammas, betas, *, optimization_level: int = 3,
                    leader_bias: float = 0.0) -> str:
    """OpenQASM 2.0 of the XY-mixer QAOA: W state per block, objective phases, line XY mixer.

    Built with Qiskit and compiled to {cx, rz, ry, rx, h, x} without a layout, so the
    measurement ``q[i] -> c[i]`` keeps "character i = qubit i".
    """

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXPlusYYGate

    n = ising.n
    qc = QuantumCircuit(n, n)
    for block in xy_blocks(n, k):
        qc.x(block[0])
        size = len(block)
        for t in range(size - 1):
            a, b = block[t], block[t + 1]
            stay = 1 / (size - t)
            if t == 0:
                stay += leader_bias * (1 - stay)
            qc.cry(float(2 * np.arccos(np.sqrt(stay))), a, b)
            qc.cx(b, a)
    for gamma, beta in zip(gammas, betas, strict=True):
        for (i, j), weight in ising.J.items():
            qc.rzz(float(2 * gamma * weight), i, j)
        for i in range(n):
            if ising.h[i] != 0:
                qc.rz(float(2 * gamma * ising.h[i]), i)
        for a in range(n - 1):
            qc.append(XXPlusYYGate(float(2 * beta), 0.0), [a, a + 1])
    return _measured_qasm(qc, optimization_level)


def paired_xy_qasm(ising: IsingForm, k: int, gamma: float, beta: float, leader_probability: float) -> str:
    """Single XY layer, starting each block in a biased superposition of its first two ranks.

    Before the cost layer, a pair has Z_leader = -Z_challenger and every other
    qubit is |0>. Substitute these identities into the full Ising cost. The
    resulting phases are exact on this initial state. The subsequent line mixer
    acts on every candidate, including candidates outside the initial pairs.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXPlusYYGate

    n = ising.n
    blocks = xy_blocks(n, k)
    qc = QuantumCircuit(n, n)
    # owner = -1 denotes a fixed Z value. A singleton is occupied (Z = -1).
    owner = np.full(n, -1, dtype=int)
    sign = np.ones(n, dtype=int)
    for block in blocks:
        leader = block[0]
        qc.x(leader)
        sign[leader] = -1
        if len(block) > 1:
            challenger = block[1]
            owner[[leader, challenger]] = challenger
            qc.ry(2 * np.arccos(np.sqrt(leader_probability)), challenger)
            qc.cx(challenger, leader)
    fields = np.zeros(n, dtype=float)
    couplings = {}
    for i in np.flatnonzero(owner >= 0):
        fields[owner[i]] += sign[i] * ising.h[i]
    for (i, j), weight in ising.J.items():
        a, b = sorted((owner[i], owner[j]))
        weight *= sign[i] * sign[j]
        if a == b:
            continue  # constant on the seed
        if a < 0:
            fields[b] += weight
        else:
            couplings[a, b] = couplings.get((a, b), 0.0) + weight
    for i in np.flatnonzero(fields):
        qc.rz(2 * gamma * fields[i], i)
    for (i, j), weight in couplings.items():
        if weight:
            qc.rzz(2 * gamma * weight, int(i), int(j))
    for a in range(n - 1):
        qc.append(XXPlusYYGate(2 * beta), [a, a + 1])
    return _measured_qasm(qc, optimization_level=1)


def local_warm_qasm(model: QUBOModel, order: list[int], gamma: float, beta: float, confidence: float) -> str:
    """One layer with an independent biased seed and disjoint nearest-neighbour cost gates.

    Keep each paired QUBO interaction. Linearize all other pair interactions
    at the seed. For an individual pair this matches the full objective with
    other pairs fixed at the seed, up to a constant. Score every measured
    feasible bitstring against the original QUBO.
    """
    from qiskit import QuantumCircuit, qasm2

    n, k = len(order), model.exact_k
    leaders = [block[0] for block in xy_blocks(n, k)]
    anchor = np.zeros(n)
    anchor[leaders] = 1
    matrix = model.Q[np.ix_(order, order)]
    cross = 2 * matrix.copy()
    np.fill_diagonal(cross, 0)
    linear = matrix.diagonal() + cross @ anchor
    couplings = []
    for a in range(0, n - 1, 2):
        b = a + 1
        pair = cross[a, b]
        linear[a] -= pair * anchor[b]
        linear[b] -= pair * anchor[a]
        couplings.append(pair / 4)
    fields = -linear / 2
    for i, jzz in enumerate(couplings):
        fields[2*i:2*i+2] -= jzz
    scale = max(np.abs(fields).max(), max(map(abs, couplings), default=0.0), 1e-12)
    # A high K/N can make the equal expected-weight choice exceed probability one.
    other_p = min(1.0, k * (1 - confidence) / (n - k))
    probabilities = np.where(anchor, confidence, other_p)
    circuit = QuantumCircuit(n, n)
    for q, p in enumerate(probabilities):
        circuit.ry(2 * np.arcsin(np.sqrt(p)), q)
        circuit.rz(2 * gamma * fields[q] / scale, q)
    for i, jzz in enumerate(couplings):
        a, b = 2*i, 2*i+1
        if jzz:
            circuit.cx(a, b)
            circuit.rz(2 * gamma * jzz / scale, b)
            circuit.cx(a, b)
    circuit.rx(2 * beta, range(n))
    circuit.measure(range(n), range(n))
    return qasm2.dumps(circuit)


def digital_annealing_qasm(ising: IsingForm, steps: int, duration: float) -> str:
    """First-order midpoint Trotter circuit for H(s) = -(1-s) sum(X) + s H_Ising.

    Time is dimensionless in units of the normalized Ising coefficients. The
    ideal linear interpolation is independent of any D-Wave chip calibration.
    """
    from qiskit import QuantumCircuit

    if steps < 1 or not np.isfinite(duration) or duration <= 0:
        raise ValueError("digital annealing needs positive steps and a finite positive duration")
    qc = QuantumCircuit(ising.n, ising.n)
    qc.h(range(ising.n))
    dt = duration / steps
    for step in range(steps):
        s = (step + 0.5) / steps
        for (i, j), coupling in ising.J.items():
            qc.rzz(2 * dt * s * coupling, i, j)
        for i, field_z in enumerate(ising.h):
            if field_z:
                qc.rz(2 * dt * s * field_z, i)
        qc.rx(-2 * dt * (1 - s), range(ising.n))
    return _measured_qasm(qc)


def sample_qasm(qasm: str, shots: int, seed: int) -> dict[str, int]:
    """Noiseless local sampling of a QASM circuit; keys as "char i = qubit i"."""

    from qiskit import qasm2
    from qiskit.quantum_info import Statevector

    circuit = qasm2.loads(qasm)
    circuit.remove_final_measurements()
    state = Statevector(circuit)
    state.seed(seed)
    return {key[::-1]: int(value) for key, value in state.sample_counts(shots).items()}


def two_qubit_gates(openqasm: str) -> int:
    return len(re.findall(r"^\s*(cz|cx|cnot|iswap|ecr)\b", openqasm or "", flags=re.MULTILINE | re.IGNORECASE))


def _parse_time(value: Any) -> datetime | None:
    text = str(value or "")[:26]
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _calibrated_chain(graph, n: int) -> list[int]:
    """A connected layout for the line mixer when the chip publishes no preset of size n."""
    paths = [(0.0, (q,)) for q, props in graph.nodes(data=True) if props.get("fidelity", 0) > 0]
    for _ in range(1, n):
        longer = []
        for score, path in paths:
            for q, edge in graph[path[-1]].items():
                fidelity = edge.get("fidelity", 0)
                if q not in path and graph.nodes[q].get("fidelity", 0) > 0 and 0 < fidelity <= 1:
                    longer.append((score + math.log(fidelity), (*path, q)))
        paths = heapq.nlargest(128, longer)
        if not paths:
            return []
    return list(max(paths)[1]) if paths else []


def _native_chain_qasm(qasm: str) -> str:
    """Route on a line in the U/CZ basis. Quafu maps dense wire i to target_qubits[i]."""
    from qiskit import qasm2, transpile

    circuit = qasm2.loads(qasm)
    n = circuit.num_qubits
    links = [[a, b] for i in range(n - 1) for a, b in ((i, i + 1), (i + 1, i))]
    routed = transpile(circuit, basis_gates=["u", "cz"], coupling_map=links,
                       initial_layout=list(range(n)), optimization_level=2, seed_transpiler=7)
    return qasm2.dumps(routed)


class QuafuClient:
    """Thin wrapper over quarkstudio's Task client for one chip (submit, poll, read counts)."""

    def __init__(self, chip: str, token: str | None = None, readout_correction: bool = True,
                 dd: str | None = "XY4", extended_layout: bool = False, native_layout: bool = False):
        if token:
            os.environ["QPU_API_TOKEN"] = token  # quarkstudio reads it from the process environment
        try:
            from quark import Task
            from quark.circuit import Backend
        except ImportError as exc:  # pragma: no cover - depends on the isolated environment
            raise RuntimeError(
                "Quafu SDK (quarkstudio) is not installed here; run with .venv-quafu\\Scripts\\python"
            ) from exc
        self.client = Task()
        self.chip = chip
        self.readout_correction = readout_correction
        self.dd = dd
        self.reverse_keys: bool | None = None
        self.hardware = f"Quafu-SQC {chip} (superconducting QPU)"
        with contextlib.redirect_stdout(io.StringIO()):
            backend = Backend(chip)
            self.priority = {len(group): list(group) for group in backend.priority_qubits}
            self.layout_graph = backend.graph if extended_layout else None
            self.native_layout = native_layout and backend.two_qubit_gate_basis.lower() == "cz"

    def status(self) -> dict[str, Any]:
        return dict(self.client.status())

    def submit(self, qasm: str, name: str, shots: int, n: int) -> tuple[int, list[int]]:
        if n not in self.priority and self.layout_graph is not None:
            self.priority[n] = _calibrated_chain(self.layout_graph, n)
        target = list(self.priority.get(n, []))
        native = self.native_layout and bool(target) and name != "qfg_bit_order_check"
        if native:
            qasm = _native_chain_qasm(qasm)
        options = {"compiler": None if native else "quarkcircuit", "correct": self.readout_correction,
                   "open_dd": self.dd, "target_qubits": target}
        task = {"chip": self.chip, "name": name, "circuit": qasm, "shots": shots, "options": options,
                "compile": not native}
        tid = self.client.run(task)
        if not isinstance(tid, int):
            raise RuntimeError(f"submission refused: {tid}")
        return tid, options["target_qubits"]

    def wait(self, tid: int, poll: float, max_minutes: float) -> dict:
        deadline = time.time() + max_minutes * 60
        while True:
            res = self.client.result(tid)
            status = str(res.get("status", "")) if isinstance(res, dict) else ""
            if isinstance(res, dict) and not status and res.get("error"):
                return {**res, "status": "Failed"}
            if status.lower() in ("finished", "failed", "canceled", "cancelled"):
                return res
            if time.time() > deadline:
                return {"status": f"timeout (last: {status or 'no result yet'})"}
            time.sleep(poll)

    def calibrate_bit_order(self, poll: float, max_minutes: float) -> bool:
        """Run X on qubit 0 of a 3-qubit circuit; True if keys must be reversed to get char i = qubit i."""

        qasm = "\n".join(['OPENQASM 2.0;', 'include "qelib1.inc";', 'qreg q[3];', 'creg c[3];', 'x q[0];',
                          'measure q[0] -> c[0];', 'measure q[1] -> c[1];', 'measure q[2] -> c[2];'])
        tid, _ = self.submit(qasm, "qfg_bit_order_check", 1024, 3)
        res = self.wait(tid, poll, max_minutes)
        counts = res.get("count") or {}
        if not counts:
            raise RuntimeError(f"bit-order check failed: status {res.get('status')}")
        top = max(counts, key=counts.get)
        if top.startswith("1") and top.count("1") == 1:
            self.reverse_keys = False
        elif top.endswith("1") and top.count("1") == 1:
            self.reverse_keys = True
        else:
            raise RuntimeError(f"cannot infer bit order from outcome {top}")
        return self.reverse_keys

    def counts(self, res: dict, field: str = "count") -> dict[str, int]:
        """Counts keyed "char i = qubit i".  For the readout-corrected field (which may hold
        quasi-probabilities) negative values are clipped and the total rescaled to the shots."""

        raw = res.get(field) or {}
        if not raw:
            return {}
        values = {(key[::-1] if self.reverse_keys else key): float(value) for key, value in raw.items()}
        if field == "count":
            return {key: int(value) for key, value in values.items()}
        clipped = {key: max(0.0, value) for key, value in values.items()}
        total = sum(clipped.values())
        shots = sum((res.get("count") or {}).values()) or 1
        return {key: int(round(value / total * shots)) for key, value in clipped.items() if value > 0} if total else {}

    @staticmethod
    def server_seconds(res: dict) -> float | None:
        created, finished = _parse_time(res.get("created")), _parse_time(res.get("finished"))
        return (finished - created).total_seconds() if created and finished else None


class LocalSimulatorClient:
    """Stand-in for :class:`QuafuClient`: the same circuit on a local noiseless simulator."""

    hardware = "local noiseless simulator (not quantum hardware)"
    chip = None

    def __init__(self, seed: int = 7):
        self.seed = seed
        self.reverse_keys: bool | None = None
        self._jobs: dict[int, tuple[str, int]] = {}
        self._next = 1

    def status(self) -> dict[str, Any]:
        return {"local": 0}

    def submit(self, qasm: str, name: str, shots: int, n: int) -> tuple[int, list[int]]:
        tid, self._next = self._next, self._next + 1
        self._jobs[tid] = (qasm, shots)
        return tid, list(range(n))

    def wait(self, tid: int, poll: float, max_minutes: float) -> dict:
        qasm, shots = self._jobs.pop(tid)
        return {"status": "Finished", "count": sample_qasm(qasm, shots, self.seed + tid)}

    def calibrate_bit_order(self, poll: float, max_minutes: float) -> bool:
        self.reverse_keys = False
        return False

    def counts(self, res: dict, field: str = "count") -> dict[str, int]:
        return dict(res.get(field) or {})

    @staticmethod
    def server_seconds(res: dict) -> float | None:
        return None


def _permuted_ising(ising: IsingForm, order: list[int]) -> IsingForm:
    """Circuit qubit j carries candidate ``order[j]``."""

    position = {candidate: qubit for qubit, candidate in enumerate(order)}
    J: dict[tuple[int, int], float] = {}
    for (a, b), weight in ising.J.items():
        key = tuple(sorted((position[a], position[b])))
        J[key] = J.get(key, 0.0) + weight
    return IsingForm(h=np.asarray(ising.h, dtype=float)[list(order)], J=J, const=ising.const, scale=ising.scale)


def _permuted_energies(model: QUBOModel, order: list[int]) -> np.ndarray:
    """QUBO energy of every circuit basis state (index = little-endian integer in circuit order)."""

    n = len(order)
    codes = np.arange(2**n, dtype=np.int64)
    bits = (codes[:, None] >> np.arange(n)) & 1
    original = (bits << np.asarray(order, dtype=np.int64)[None, :]).sum(axis=1)
    return all_energies(model)[original]


def _samples_in_model_order(counts: Mapping[str, int], order: list[int]) -> np.ndarray:
    rows = np.zeros((len(counts), len(order)), dtype=np.int8)
    for row, bitstring in enumerate(counts):
        if len(bitstring) != len(order) or set(bitstring) - {"0", "1"}:
            raise ValueError("Quafu returned a measurement with unexpected width or non-binary digits")
        for qubit, char in enumerate(bitstring):
            if char == "1":
                rows[row, order[qubit]] = 1
    return rows


@dataclass(slots=True)
class QuafuCircuitSolver:
    """Solve one investigation window with XY-QAOA or digitized annealing.

    ``chip`` is used while it is online; otherwise ``backup_chip``.  ``client`` may be
    injected (``LocalSimulatorClient`` for ``quafu-mock`` and tests).
    """

    chip: str = "Haituo"
    backup_chip: str | None = "Baihua"
    shots: int = 2048
    layers: int = 3
    max_qubits: int = 12
    readout_correction: bool = True
    dd: str | None = "XY4"
    token: str | None = None
    poll_seconds: float = 2.0
    max_wait_minutes: float = 10.0
    restarts: int = 8
    seed: int = 7
    backend_name: str = "quafu-qpu"
    label_prefix: str = "qfg"
    client: Any = None
    digital_annealing: bool = False
    anneal_duration: float = 2.0
    fast_xy: bool = False
    fast_variant: str = "local"
    _clients: dict = field(default_factory=dict, repr=False)
    _status: tuple = field(default=(0.0, {}), repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _chip_client(self, chip: str) -> Any:
        if chip not in self._clients:
            self._clients[chip] = QuafuClient(chip, token=self.token, readout_correction=self.readout_correction,
                                              dd=self.dd, extended_layout=self.fast_xy,
                                              native_layout=self.fast_xy and self.fast_variant != "full")
        return self._clients[chip]

    def _get_client(self) -> Any:
        if self.client is not None:
            client = self.client
        else:
            primary = self._chip_client(self.chip)
            stamp, status = self._status
            if time.time() - stamp > 60:
                status = primary.status()
                self._status = (time.time(), status)
            chips = [self.chip] + ([self.backup_chip] if self.backup_chip else [])
            online = [chip for chip in chips if isinstance(status.get(chip), int)]
            if not online:
                raise RuntimeError(f"no Quafu-SQC chip available: {', '.join(f'{c}={status.get(c)}' for c in chips)}")
            client = self._chip_client(online[0])
        if client.reverse_keys is None:
            client.calibrate_bit_order(self.poll_seconds, self.max_wait_minutes)
        return client

    def _build_circuit(self, model: QUBOModel, order: list[int]) -> tuple[str, dict]:
        if self.digital_annealing:
            ising = _permuted_ising(qubo_to_ising(model), order)
            circuit = digital_annealing_qasm(ising, self.layers, self.anneal_duration)
            return circuit, {
                "solver": "quafu_digital_annealing",
                "schedule": "linear_transverse_field",
                "trotter_steps": self.layers,
                "dimensionless_duration": self.anneal_duration,
            }

        if self.fast_xy and self.fast_variant == "local":
            gamma, beta, confidence = np.pi / 2, 0.06 * np.pi, 0.9
            return local_warm_qasm(model, order, gamma, beta, confidence), {
                "solver": "quafu_fast_local", "variant": "local", "qaoa_layers": 1, "mixer": "x",
                "gamma": gamma, "beta": beta, "leader_probability": confidence,
            }
        ising = _permuted_ising(objective_ising(model), order)
        if self.fast_xy:
            # Fixed on held-out AMLworld validation windows before testing. The
            # circuit still encodes this window's own risks and account links.
            gamma = 3 * np.pi / 8
            if self.fast_variant == "paired":
                beta, probability = -np.pi / 10, 0.8
                circuit = paired_xy_qasm(ising, model.exact_k, gamma, beta, probability)
                return circuit, {"solver": "quafu_fast_xy", "variant": "paired", "qaoa_layers": 1, "mixer": "xy",
                                 "gamma": gamma, "beta": beta, "leader_probability": probability}
            beta, bias = (-np.pi / 8, 0.5) if ising.n > 12 else (-np.pi / 6, 0.0)
            circuit = xy_qasm_circuit(ising, model.exact_k, [gamma], [beta], optimization_level=1,
                                      leader_bias=bias)
            return circuit, {"solver": "quafu_fast_xy", "variant": "full", "qaoa_layers": 1, "mixer": "xy",
                             "gamma": gamma, "beta": beta, "leader_bias": bias}
        energies = _permuted_energies(model, order)
        angles = None
        for p in range(1, self.layers + 1):
            gammas, betas, _ = optimise_xy_angles(ising, energies, model.exact_k, p=p, seed=self.seed,
                                                  restarts=self.restarts, warm_start=angles)
            angles = (gammas, betas)
        return xy_qasm_circuit(ising, model.exact_k, *angles), {
            "solver": "quafu_qaoa", "qaoa_layers": self.layers, "mixer": "xy",
        }

    def solve(self, model: QUBOModel) -> SolverResult:
        return self.solve_measurement(model, 0)

    def solve_measurement(self, model: QUBOModel, attempt_index: int) -> SolverResult:
        n, k = model.n_variables, model.exact_k
        if k is None or not 0 < k < n:
            raise ValueError("the Quafu backend needs an exact-K window with 0 < K < N")
        if n > self.max_qubits:
            raise ValueError(f"{n} candidates exceed the {self.max_qubits}-qubit limit of the Quafu backend")
        acquired = perf_counter()
        with self._lock:
            client = self._get_client()
            started = perf_counter()
            if self.fast_xy:
                # Deal candidates by risk rank across the K initial W blocks.
                # Each block starts in a superposition over all its candidates.
                ranks = np.argsort(-np.asarray(model.metadata["risks"]), kind="stable")
                if self.fast_variant == "local" and attempt_index % 2:
                    # Alternate the original risk ranking with a boundary swap.
                    # The only feedback is that the previous job missed.
                    risks = np.asarray(model.metadata["risks"])[ranks]
                    swaps = sorted((risks[i] - risks[j], i, j) for i in range(k) for j in range(k, n))
                    _, a, b = swaps[(attempt_index // 2) % len(swaps)]
                    ranks[a], ranks[b] = ranks[b], ranks[a]
                order = np.concatenate([ranks[b::k] for b in range(k)]).tolist()
            else:
                order = line_order(n, model.metadata["suspicious_edges"])
            qasm, method = self._build_circuit(model, order)
            compiled = perf_counter()
            tid, qubits = client.submit(qasm, f"{self.label_prefix}_{model.name}"[:60], self.shots, n)
            result = client.wait(tid, self.poll_seconds, self.max_wait_minutes)
            status = str(result.get("status", ""))
            counts = client.counts(result) if status.lower() == "finished" else {}
            counts = {bits: count for bits, count in counts.items() if count > 0}
            if not counts:
                detail = str(result.get("error") or "")[:240]
                if self.token:
                    detail = detail.replace(self.token, "[redacted]")
                raise RuntimeError(f"Quafu task {tid} ended with status {status or 'unknown'}. {detail}")
            finished = perf_counter()
        server = client.server_seconds(result)
        transpiled = str(result.get("transpiled") or result.get("circuit") or "")
        samples = _samples_in_model_order(counts, order)
        if self.digital_annealing or self.fast_xy:
            # Keep the hardware readings in the original candidate order so
            # a certificate can audit individual shots and any later repair.
            method["measured_counts"] = {
                "".join(str(bit) for bit in sample): count
                for sample, count in zip(samples, counts.values(), strict=True)
            }
            method.update(circuit_candidate_order=order, reverse_provider_bits=client.reverse_keys, attempt_index=attempt_index,
                          client_setup_s=started - acquired, circuit_build_s=compiled - started,
                          submit_to_counts_s=finished - compiled)
            readout = re.findall(r"measure\s+\w+\[(\d+)\]\s*->\s*\w+\[(\d+)\]", transpiled)
            method["readout_qubits"] = [int(q) for q, c in sorted(readout, key=lambda pair: int(pair[1]))]
        return best_result(
            model,
            samples,
            started=started,
            finished=finished,
            local_search=False,
            metadata={
                **method,
                "backend": self.backend_name,
                "hardware": client.hardware,
                "chip": client.chip,
                "task_id": tid,
                "shots": self.shots,
                "physical_qubits": qubits,
                "logical_2q": two_qubit_gates(qasm),
                "transpiled_2q": two_qubit_gates(transpiled) or None,
                "server_time_s": server,
            },
        )


# Public name used by earlier callers. Its default circuit remains XY-QAOA.
QuafuQAOASolver = QuafuCircuitSolver

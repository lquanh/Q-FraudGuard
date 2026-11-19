"""Replay the published Haituo benchmarks, then audit AML labels and CPU controls offline."""

import csv
import json
from collections import Counter
from itertools import combinations
from math import isclose
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = {
    "quafu_verified_hits_8_12_15_18_20261118T212255Z.json": 10,
    "quafu_ranking_misses_8_12_15_18_20261118T212601Z.json": 1,
}


def feasible_energies(candidates, k, strength, *, normalize_by_max=True):
    """Evaluate the published objective directly. Its penalty vanishes at exact K."""
    risks = [float(candidate["risk_score"]) for candidate in candidates]
    scale = (max(risks) or 1.0) if normalize_by_max else 1.0
    accounts = [{c["sender_id"], c["receiver_id"]} - {"", "UNKNOWN"} for c in candidates]
    # Histograms put candidate 0 at the left of the bit string.
    bits = [1 << i for i in reversed(range(len(risks)))]
    pairs = [bits[i] | bits[j] for i, j in combinations(range(len(risks)), 2)
             if accounts[i] & accounts[j]]
    energies = {}
    for indices in combinations(range(len(risks)), k):
        mask = sum(bits[i] for i in indices)
        joint = sum((mask & pair) == pair for pair in pairs)
        energies[mask] = -sum(risks[i] / scale for i in indices) - 0.5 * strength * joint
    ranking = sorted(range(len(risks)), key=risks.__getitem__, reverse=True)
    return energies, sum(bits[i] for i in ranking[:k])


def verify_window(row, inputs, settings, task_ids):
    ids = row["candidate_ids"]
    n, k = row["n"], row["k"]
    if len(ids) != n or len(set(ids)) != n or k != max(1, round(0.25 * n)):
        raise ValueError("invalid candidate list or K")
    candidates = [inputs[i] for i in ids]
    energies, ranked = feasible_energies(candidates, k, settings["connectivity_strength"])
    optimum = min(energies.values())
    winners = {mask for mask, energy in energies.items() if isclose(energy, optimum, abs_tol=1e-9)}
    improved = ranked
    while True:
        neighbors = (mask for mask in energies if (mask ^ improved).bit_count() == 2)
        best = min(neighbors, key=energies.__getitem__, default=improved)
        if energies[best] >= energies[improved] - 1e-12:
            break
        improved = best
    if not row["certified"] or not isclose(row["optimum_energy"], optimum, abs_tol=1e-9):
        raise ValueError("reported certificate differs from exhaustive enumeration")
    for name, energy in (("exact", optimum), ("top_k", energies[ranked]), ("top_k_swap", energies[improved])):
        control = row["cpu"][name]
        if (not isclose(control["energy"], energy, abs_tol=1e-9)
                or control["hit"] != isclose(energy, optimum, abs_tol=1e-9)):
            raise ValueError(f"CPU control disagrees: {name}")

    totals = Counter(windows=1, top_k=int(ranked in winners), top_k_swap=int(improved in winners))
    selected = set(row["selected"])
    selected_bits = "".join("1" if ident in selected else "0" for ident in ids)
    if len(selected) != k or int(selected_bits, 2) not in winners:
        raise ValueError("reported selection is not optimal")
    for index, attempt in enumerate(row["attempts"]):
        task = str(attempt["task_id"])
        if (attempt["backend"] != "quafu-fast-qpu" or attempt["chip"] != "Haituo"
                or not task.isdigit() or task in task_ids):
            raise ValueError(f"unexpected backend or task ID in job {index + 1}")
        task_ids.add(task)
        counts = attempt["measured_counts"]
        if any(len(bits) != n or set(bits) - {"0", "1"} or type(count) is not int or count <= 0
               for bits, count in counts.items()):
            raise ValueError(f"invalid histogram in task {task}")
        reads = sum(counts.values())
        feasible = sum(count for bits, count in counts.items() if int(bits, 2) in energies)
        optimal = sum(count for bits, count in counts.items() if int(bits, 2) in winners)
        if (reads != settings["shots_per_job"] or reads != attempt["shots"]
                or (reads, feasible, optimal, bool(optimal)) !=
                (attempt["measured_reads"], attempt["feasible_reads"], attempt["optimal_reads"], attempt["measured_hit"])):
            raise ValueError(f"shot totals disagree in task {task}")
        if optimal and (index != len(row["attempts"]) - 1 or selected_bits not in counts):
            raise ValueError(f"winning batch or selected bit string disagrees in task {task}")
        totals.update(jobs=1, shots=reads, optimal=optimal, hits=int(optimal > 0))
    if (row["status"] != "hit" or totals["hits"] != 1 or totals["optimal"] != row["optimal_reads"]
            or totals["shots"] != row["measured_reads"]):
        raise ValueError("window result disagrees with the measured histograms")

    # Score the selections already made, in the same bit order as the histogram.
    labels = [candidate["is_fraud"] for candidate in candidates]
    if any(label not in ("0", "1") for label in labels):
        raise ValueError("retrospective evaluation requires a binary AMLworld label for every candidate")
    positives = int("".join(labels), 2)
    totals.update(slots=k, positive_candidates=positives.bit_count(),
                  qpu_frauds=(positives & int(selected_bits, 2)).bit_count(),
                  top_k_frauds=(positives & ranked).bit_count())

    # CPU-only sensitivity on the identical windows. The QPU used the archived settings above.
    for mode, normalize in (("max_c025", True), ("raw_c025", False)):
        alternate, top_k = feasible_energies(candidates, k, 0.25, normalize_by_max=normalize)
        totals[mode] = int(isclose(alternate[top_k], min(alternate.values()), abs_tol=1e-9))
    return totals


def main():
    with (ROOT / "data/quafu_benchmark_candidates.csv").open(encoding="utf-8", newline="") as stream:
        records = csv.DictReader(line for line in stream if not line.startswith("#"))
        inputs = {row["transaction_id"]: row for row in records}
    task_ids = set()
    for filename, per_size in REPORTS.items():
        report = json.loads((ROOT / "artifacts/deploy" / filename).read_bytes())
        settings = report["settings"]
        rows = report["rows"]
        expected = {(n, i) for n in (8, 12, 15, 18) for i in range(per_size)}
        if len(rows) != len(expected) or {(r["n"], r["window"]) for r in rows} != expected:
            raise ValueError(f"{filename} has an incomplete or duplicate window")
        if (settings["solver"], settings["risk_normalization"], settings["connectivity_strength"],
                settings["shots_per_job"]) != ("quafu-fast-qpu", "max", 0.01, 1024):
            raise ValueError(f"{filename} uses unexpected benchmark settings")
        totals = Counter()
        for row in rows:
            try:
                totals.update(verify_window(row, inputs, settings, task_ids))
            except (ValueError, KeyError) as exc:
                raise ValueError(f"{filename}, N={row['n']}, window={row['window']}: {exc}") from exc
        print(f"{filename}: QPU {totals['hits']}/{len(rows)}, CPU top-K {totals['top_k']}/{len(rows)}, "
              f"CPU top-K + swap {totals['top_k_swap']}/{len(rows)}, "
              f"{totals['jobs']} jobs, {totals['optimal']}/{totals['shots']} optimal shots")
        print(f"  AMLworld positives among selected slots: QPU {totals['qpu_frauds']}/{totals['slots']}, "
              f"top-K {totals['top_k_frauds']}/{totals['slots']} "
              f"({totals['positive_candidates']} positive candidate occurrences)")
        print(f"  CPU-only top-K optimum at c=0.25: max normalization {totals['max_c025']}/{len(rows)}, "
              f"raw scores {totals['raw_c025']}/{len(rows)}")


if __name__ == "__main__":
    main()

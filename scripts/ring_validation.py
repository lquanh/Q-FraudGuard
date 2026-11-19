"""Kill-criteria test for the ring-inference idea (synthetic worlds only).

Question 1: after exact classical preprocessing (roof duality + eliminations),
            is there a frustrated core big enough to need a combinatorial /
            quantum solver?  Kill if it is almost always <= 10 accounts.
Question 2: does the globally consistent (Ising MAP) decision flag fewer
            innocent accounts than simpler methods at the same recall?

Protocol per noise preset: evidence models are fitted on 3 worlds, every
method's operating point is tuned (best F1) on 2 other worlds, and everything
is reported on 4 held-out test worlds.  All methods use the same model outputs.

    python scripts/ring_validation.py            # writes artifacts/ring_validation/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qf_guard.rings import baselines as B  # noqa: E402
from qf_guard.rings.evaluate import Counts, best_f1, count, fp_at_recall  # noqa: E402
from qf_guard.rings.evidence import build_evidence, fit_models, score_evidence  # noqa: E402
from qf_guard.rings.inference import infer_ring  # noqa: E402
from qf_guard.rings.synth import NOISE_PRESETS, WorldConfig, generate_world  # noqa: E402

KAPPAS = (0.02, 0.05, 0.1, 0.25, 0.5)
BETAS = (-2.0, -1.0, 0.0, 1.0, 2.0)
TAUS = (0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05)
LINK_TAUS = (0.9, 0.7, 0.5, 0.3, 0.2, 0.1)
RECALL_TARGETS = (0.75, 0.8, 0.85, 0.9)
METHODS = ("ising", "node", "link_boost", "union_find", "greedy", "pagerank")


def curves_for(world, evidence, scores, core_log: dict | None = None) -> dict[str, dict]:
    out: dict[str, dict] = {m: {} for m in METHODS}
    for kappa in KAPPAS:
        for beta0 in BETAS:
            result = infer_ring(scores, kappa=kappa, beta0=beta0)
            out["ising"][(kappa, beta0)] = count(result.x.astype(bool), world, evidence)
            if core_log is not None:
                core_log.setdefault((kappa, beta0), []).append(
                    {"max_core": result.stats.max_core, "n_core": result.stats.n_core,
                     "n_free": result.stats.n_after_seeds, "lp_fixed": result.stats.n_fixed_lp,
                     "cores": len(result.stats.core_sizes), "seconds": round(result.runtime, 3)}
                )
    for tau in TAUS:
        out["node"][tau] = count(B.node_threshold(scores, tau), world, evidence)
        out["link_boost"][tau] = count(B.link_boost(scores, tau), world, evidence)
    for tau in LINK_TAUS:
        out["union_find"][tau] = count(B.union_find(scores, tau), world, evidence)
        out["greedy"][tau] = count(B.greedy_growth(scores, tau), world, evidence)
    ppr = B.personalized_pagerank(scores)
    decision = np.ones(scores.n, dtype=bool)
    decision[scores.seeds] = False
    for q in (0.995, 0.99, 0.98, 0.97, 0.95, 0.9):
        threshold = np.quantile(ppr[decision], q)
        flags = ppr >= threshold
        flags[scores.seeds] = True
        out["pagerank"][q] = count(flags, world, evidence)
    return out


def frontier_beats(frontier: dict, a: str, b: str, margin: float = 0.2) -> bool:
    """``a`` beats ``b`` if it reaches every recall target ``b`` reaches, never flags
    more innocents there, and flags at least ``margin`` fewer at one target."""

    def fp(method: str, r: float):
        return frontier[method][str(r)]

    reach_a = [r for r in RECALL_TARGETS if fp(a, r) is not None]
    reach_b = [r for r in RECALL_TARGETS if fp(b, r) is not None]
    if not reach_b or not set(reach_b) <= set(reach_a):
        return False
    pairs = [(fp(a, r), fp(b, r)) for r in reach_b]
    return all(x <= y for x, y in pairs) and any(y > 0 and x <= (1 - margin) * y for x, y in pairs)


def verdict(frontier: dict, cores_at_tuned: list[dict]) -> dict:
    small = float(np.mean([c["max_core"] <= 10 for c in cores_at_tuned]))
    return {
        "core_criterion_small_share": small,
        "core_criterion_kill": bool(small >= 0.9),
        "beats_union_find": frontier_beats(frontier, "ising", "union_find"),
        "beats_node_only": frontier_beats(frontier, "ising", "node"),
        "beats_link_boost": frontier_beats(frontier, "ising", "link_boost"),
    }


def add_curves(total: dict, part: dict) -> None:
    for method, curve in part.items():
        bucket = total.setdefault(method, {})
        for key, c in curve.items():
            bucket.setdefault(key, Counts())
            bucket[key] += c


def run_preset(name: str, base_seed: int, n_fit: int, n_tune: int, n_test: int, n_persons: int) -> dict:
    cfg = WorldConfig(n_persons=n_persons).with_overrides(**NOISE_PRESETS[name])
    started = time.time()
    fit_worlds = [generate_world(cfg, seed=base_seed + k) for k in range(n_fit)]
    tune_worlds = [generate_world(cfg, seed=base_seed + 100 + k) for k in range(n_tune)]
    test_worlds = [generate_world(cfg, seed=base_seed + 200 + k) for k in range(n_test)]
    models = fit_models([build_evidence(w) for w in fit_worlds], seed=base_seed)

    tune_total: dict = {}
    for w in tune_worlds:
        e = build_evidence(w)
        add_curves(tune_total, curves_for(w, e, score_evidence(e, models)))
    tuned = {m: best_f1(curve) for m, curve in tune_total.items()}

    test_total: dict = {}
    core_log: dict = {}
    world_stats = []
    for w in test_worlds:
        e = build_evidence(w)
        sc = score_evidence(e, models)
        add_curves(test_total, curves_for(w, e, sc, core_log))
        world_stats.append({
            "seed": w.seed, "accounts": w.n_accounts, "active_ring": int(e.node_label.sum()),
            "seeds": int(len(e.seeds)), "false_seeds": int(e.seed_is_false.sum()),
            "pairs": int(len(e.pairs)), "negative_J_frac": float(np.mean(sc.J < 0)) if len(sc.J) else 0.0,
        })

    def summary(c: Counts) -> dict:
        return {"recall": round(c.recall, 4), "precision": round(c.precision, 4), "fp": c.fp, "tp": c.tp,
                "positives": c.positives, "sleeper_recall": round(c.sleeper_recall, 4),
                "sleepers": c.sleepers}

    tuned_test = {m: {"key": str(tuned[m]), **summary(test_total[m][tuned[m]])} for m in METHODS}
    frontier = {m: {str(r): fp_at_recall(test_total[m], r) for r in RECALL_TARGETS} for m in METHODS}
    ising_key = tuned["ising"]
    cores_at_tuned = core_log[ising_key]
    cores_grid = {
        f"kappa={k},beta0={b}": {
            "max_core_per_world": [c["max_core"] for c in core_log[(k, b)]],
            "f1_test": round(test_total["ising"][(k, b)].f1, 4),
        }
        for (k, b) in core_log
    }
    return {
        "preset": name,
        "config": asdict(cfg),
        "seconds": round(time.time() - started, 1),
        "worlds": world_stats,
        "tuned_on_validation_worlds": tuned_test,
        "fp_at_recall_test": frontier,
        "ising_cores_at_tuned_point": cores_at_tuned,
        "ising_cores_grid": cores_grid,
        "verdict": verdict(frontier, cores_at_tuned),
    }


def markdown(results: list[dict]) -> str:
    lines = [
        "# Ring-inference validation (synthetic worlds)",
        "",
        "All data is synthetic (`qf_guard.rings.synth`). Results test the idea under the",
        "generator's assumptions; they are not evidence about real banks.",
        "",
    ]
    names = {"ising": "Ising MAP (global)", "node": "Per-account score", "link_boost": "Score + strong-link boost",
             "union_find": "Union-find (hub-pruned)", "greedy": "Greedy growth + cannot-link",
             "pagerank": "Personalized PageRank"}
    for r in results:
        v = r["verdict"]
        lines += [f"## Preset `{r['preset']}`", "",
                  "Operating point tuned on validation worlds, measured on held-out test worlds:", "",
                  "| Method | Tuned key | Recall | Innocents flagged (FP) | Precision | Recall on not-yet-used accounts |",
                  "|---|---|---|---|---|---|"]
        for m in METHODS:
            t = r["tuned_on_validation_worlds"][m]
            lines.append(f"| {names[m]} | `{t['key']}` | {t['recall']:.3f} | {t['fp']} | {t['precision']:.3f} | "
                         f"{t['sleeper_recall']:.3f} ({t['sleepers']}) |")
        lines += ["", "Fewest innocents flagged at a given recall (test worlds, any operating point):", "",
                  "| Method | " + " | ".join(f"recall ≥ {x}" for x in RECALL_TARGETS) + " |",
                  "|---|" + "---|" * len(RECALL_TARGETS)]
        for m in METHODS:
            vals = r["fp_at_recall_test"][m]
            lines.append(f"| {names[m]} | " + " | ".join("—" if vals[str(x)] is None else str(vals[str(x)])
                                                        for x in RECALL_TARGETS) + " |")
        cores = r["ising_cores_at_tuned_point"]
        lines += ["", f"Frustrated core at the tuned Ising point (per test world): max size "
                  f"{[c['max_core'] for c in cores]}, accounts left to the solver {[c['n_core'] for c in cores]} "
                  f"of {[c['n_free'] for c in cores]} free.", "",
                  f"Verdict: small-core share {v['core_criterion_small_share']:.2f} "
                  f"(kill if ≥ 0.9 → {'KILL' if v['core_criterion_kill'] else 'pass'}). "
                  "On the recall/false-positive frontier (reaches every recall the other method reaches, "
                  "never more innocents, ≥ 20% fewer somewhere), Ising beats union-find: "
                  f"{v['beats_union_find']}; beats per-account score: {v['beats_node_only']}; "
                  f"beats score + link boost: {v['beats_link_boost']}.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--presets", nargs="+", default=["clean", "realistic", "hard"])
    parser.add_argument("--fit", type=int, default=3)
    parser.add_argument("--tune", type=int, default=2)
    parser.add_argument("--test", type=int, default=4)
    parser.add_argument("--persons", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("artifacts/ring_validation"))
    parser.add_argument("--report-only", action="store_true",
                        help="rebuild report.md (and verdicts) from an existing results.json")
    args = parser.parse_args()

    if args.report_only:
        results = json.loads((args.out / "results.json").read_text(encoding="utf-8"))
        for res in results:
            res["verdict"] = verdict(res["fp_at_recall_test"], res["ising_cores_at_tuned_point"])
        (args.out / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        (args.out / "report.md").write_text(markdown(results), encoding="utf-8")
        print(f"rebuilt {args.out / 'report.md'}")
        return

    results = []
    for k, name in enumerate(args.presets):
        print(f"[{name}] running...", flush=True)
        res = run_preset(name, args.seed + 10_000 * k, args.fit, args.tune, args.test, args.persons)
        print(f"[{name}] done in {res['seconds']} s: verdict {res['verdict']}", flush=True)
        results.append(res)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    (args.out / "report.md").write_text(markdown(results), encoding="utf-8")
    print(f"wrote {args.out / 'report.md'}")


if __name__ == "__main__":
    main()

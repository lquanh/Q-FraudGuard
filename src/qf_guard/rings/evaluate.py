"""Metrics for ring inference on synthetic worlds (ground truth is synthetic)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qf_guard.rings.evidence import Evidence
from qf_guard.rings.synth import World

__all__ = ["Counts", "count", "sleeper_mask", "best_f1", "fp_at_recall"]


@dataclass(slots=True)
class Counts:
    tp: int = 0
    fp: int = 0
    positives: int = 0
    sleeper_tp: int = 0
    sleepers: int = 0

    def __iadd__(self, other: "Counts") -> "Counts":
        self.tp += other.tp
        self.fp += other.fp
        self.positives += other.positives
        self.sleeper_tp += other.sleeper_tp
        self.sleepers += other.sleepers
        return self

    @property
    def recall(self) -> float:
        return self.tp / self.positives if self.positives else 0.0

    @property
    def precision(self) -> float:
        flagged = self.tp + self.fp
        return self.tp / flagged if flagged else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def sleeper_recall(self) -> float:
        return self.sleeper_tp / self.sleepers if self.sleepers else 0.0


def sleeper_mask(world: World, evidence: Evidence) -> np.ndarray:
    """Active ring accounts that have not moved scam money yet at decision time."""

    first_use = world.meta["first_use_minute"]
    return evidence.node_label.astype(bool) & (first_use >= evidence.minute)


def count(flags: np.ndarray, world: World, evidence: Evidence) -> Counts:
    """Score flags on non-seed accounts (seeds are reported, not decisions)."""

    truth = evidence.node_label.astype(bool)
    decision = np.ones(evidence.n, dtype=bool)
    decision[evidence.seeds] = False
    f = np.asarray(flags, dtype=bool) & decision
    sleepers = sleeper_mask(world, evidence) & decision
    return Counts(
        tp=int((f & truth).sum()),
        fp=int((f & ~truth).sum()),
        positives=int((truth & decision).sum()),
        sleeper_tp=int((f & sleepers).sum()),
        sleepers=int(sleepers.sum()),
    )


def best_f1(curve: dict) -> tuple:
    """Parameter key with the best F1 in ``{key: Counts}``."""

    return max(curve, key=lambda key: (curve[key].f1, -curve[key].fp))


def fp_at_recall(curve: dict, target: float) -> int | None:
    """Fewest false positives among operating points reaching ``target`` recall."""

    hits = [c.fp for c in curve.values() if c.recall >= target]
    return min(hits) if hits else None

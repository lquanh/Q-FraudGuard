from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ScoringConfig:
    supervised_model: str = "lightgbm"
    supervised_weight: float = .55
    anomaly_weight: float = .15
    graph_weight: float = .20
    rule_weight: float = .10
    optimize_weights_on_validation: bool = True
    weight_grid_step: float = .05
    target_fraud_retention: float = .95
    target_candidate_rate: float = .15
    random_state: int = 42
    #: Optional feature groups on top of the base set ("attributes", "formats",
    #: "confirmed"); empty keeps the original PaySim feature set.
    feature_groups: tuple[str, ...] = ()
    #: Delay before a confirmed laundering label may be used as a feature.
    label_delay_minutes: int = 1440
    #: Overrides for the LightGBM classifier (None keeps the original settings).
    lightgbm_params: dict | None = None
    #: If set, the alert threshold is chosen on validation so that at least this
    #: share of validation frauds is flagged (instead of a fixed flag rate).
    target_recall: float | None = None
    # Fraction of validation transactions that the manual review team can handle.
    review_rate: float = .02

    @property
    def normalized_weights(self) -> tuple[float, float, float, float]:
        values = (
            self.supervised_weight,
            self.anomaly_weight,
            self.graph_weight,
            self.rule_weight,
        )
        total = sum(values)
        return (
            values[0] / total,
            values[1] / total,
            values[2] / total,
            values[3] / total,
        )

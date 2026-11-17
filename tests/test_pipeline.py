from pathlib import Path

import pandas as pd
import pytest

from qf_guard.explainability import recommended_treatment
from qf_guard.features import engineer_features
from qf_guard.runtime import RealtimeService
from qf_guard.scoring import RiskScorer
from qf_guard.training import temporal_split, train_and_evaluate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "paysim.csv"
MODEL = ROOT / "artifacts" / "models" / "paysim_lightgbm_scorer.joblib"


@pytest.fixture(scope="module")
def paysim():
    return pd.read_csv(DATA)


def test_features_are_causal(paysim):
    head = paysim.iloc[:3000]
    longer = paysim.iloc[:4000]
    left = engineer_features(head).matrix
    right = engineer_features(longer).matrix.iloc[: len(head)]
    pd.testing.assert_frame_equal(left, right)


def test_temporal_split_has_no_overlap(paysim):
    train, validation, test = temporal_split(paysim)
    assert train["timestamp"].max() <= validation["timestamp"].min()
    assert validation["timestamp"].max() <= test["timestamp"].min()


def test_shipped_artifact_scores_stream(paysim):
    scorer = RiskScorer.load(MODEL)
    train, validation, test = temporal_split(paysim)
    scorer.history_transactions_ = scorer._feature_history(validation.iloc[-3000:])
    service = RealtimeService.from_scorer(scorer)
    records = service.score_transactions(test.iloc[:200])
    frame = pd.DataFrame(records)
    assert len(frame) == 200
    assert frame["risk_score"].between(0, 1).all()
    assert {"is_candidate", "reason_codes", "risk_band"} <= set(frame.columns)


def test_training_reaches_usable_quality(paysim):
    scorer, metrics = train_and_evaluate(paysim.iloc[:60_000])
    assert scorer.is_fitted_
    assert metrics["test_roc_auc"] > .9
    assert metrics["test_candidate_rate"] < .1
    assert {"operational_candidate_threshold", "alert_hold_threshold"} <= set(
        scorer.validation_stats_
    )


def test_treatment_uses_supplied_operating_points():
    hold = recommended_treatment({"risk_score": .4}, hold_threshold=.35, verify_threshold=.2)
    verify = recommended_treatment({"risk_score": .25}, hold_threshold=.35, verify_threshold=.2)
    monitor = recommended_treatment({"risk_score": .1}, hold_threshold=.35, verify_threshold=.2)
    assert [item["recommended_action"] for item in (hold, verify, monitor)] == [
        "hold_and_escalate", "step_up_verification", "monitor_and_review"
    ]

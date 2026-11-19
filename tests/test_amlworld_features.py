from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qf_guard.config import ScoringConfig
from qf_guard.features import FEATURE_COLUMNS, FEATURE_GROUPS, attribute_features, confirmed_contact_features
from qf_guard.scoring import RiskScorer, training_matrices
from qf_guard.training import temporal_split

ROOT = Path(__file__).resolve().parents[1]


def frame(rows):
    out = pd.DataFrame(rows, columns=["timestamp", "sender_id", "receiver_id"])
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out


def test_confirmed_list_is_used_only_after_the_delay():
    context = frame(
        [
            ("2022-09-01 00:00", "1:A", "2:X"),   # A laundering here (confirmed minute 0)
            ("2022-09-01 00:05", "3:B", "1:A"),   # B deals with A before A is known
            ("2022-09-01 00:10", "1:A", "2:Y"),   # A not yet known (delay 1 day)
            ("2022-09-02 00:01", "1:A", "2:Z"),   # now known
            ("2022-09-02 00:02", "3:B", "4:C"),   # B has one confirmed contact
        ]
    )
    start = int(context["timestamp"].iloc[0].value // 60_000_000_000)
    feats = confirmed_contact_features(context, {"1:A": start}, delay_minutes=1440)
    assert feats["sender_confirmed"].tolist() == [0, 0, 0, 1, 0]
    assert feats["receiver_confirmed"].tolist() == [0, 0, 0, 0, 0]
    assert feats["sender_confirmed_contacts"].iloc[4] == pytest.approx(np.log1p(1))
    assert feats["sender_confirmed_contacts"].iloc[1] == 0


def test_attribute_features_fall_back_to_zero_without_amlworld_columns():
    paysim_like = frame([("2022-09-01 00:00", "C1", "M2")])
    out = attribute_features(paysim_like, ("attributes", "formats"))
    assert set(out.columns) == set(FEATURE_GROUPS["attributes"]) | set(FEATURE_GROUPS["formats"])
    assert float(out.to_numpy().sum()) == 0.0


def synthetic_amlworld(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    accounts = [f"{rng.integers(1, 6)}:{k:05X}" for k in range(400)]
    mules = set(accounts[:12])
    rows = []
    t0 = pd.Timestamp("2022-09-01", tz="UTC")
    for k in range(n):
        s, r = rng.choice(accounts, 2)
        bad = (s in mules) and rng.random() < 0.6
        rows.append({
            "timestamp": t0 + pd.Timedelta(minutes=int(k * 2)),
            "type": "TRANSFER",
            "amount": float(rng.lognormal(8 if not bad else 9, 1)),
            "sender_id": s,
            "receiver_id": r,
            "is_fraud": int(bad),
            "payment_format": "ACH" if bad else str(rng.choice(["ACH", "Cheque", "Credit Card"])),
            "payment_currency": "US Dollar",
            "receiving_currency": "Euro" if bad and rng.random() < .5 else "US Dollar",
        })
    return pd.DataFrame(rows)


def test_extended_scorer_fits_scores_and_keeps_column_order():
    data = synthetic_amlworld()
    train, validation, test = temporal_split(data)
    config = replace(ScoringConfig(), feature_groups=("attributes", "formats", "confirmed"),
                     target_candidate_rate=.05, weight_grid_step=.25)
    scorer = RiskScorer(config).fit(train, validation)
    expected = list(FEATURE_COLUMNS) + [c for g in ("attributes", "formats", "confirmed") for c in FEATURE_GROUPS[g]]
    assert list(scorer.feature_columns_) == expected
    scored = scorer.score(test)
    assert len(scored) == len(test) and scored["risk_score"].between(0, 1).all()
    X_train, y_train, X_val, y_val = training_matrices(train, validation, config)
    assert list(X_train.columns) == expected and len(X_val) == len(validation) and y_val.sum() > 0


def test_target_recall_threshold_reaches_the_target_on_validation():
    data = synthetic_amlworld(seed=3)
    train, validation, _ = temporal_split(data)
    config = replace(ScoringConfig(), feature_groups=("attributes",), target_recall=0.9, weight_grid_step=.25)
    scorer = RiskScorer(config).fit(train, validation)
    assert scorer.validation_stats_["fraud_retention"] >= 0.9
    # a recall target never flags fewer validation frauds than a stricter flag-rate threshold
    assert scorer.validation_stats_["candidate_rate"] > 0


@pytest.mark.skipif(not (ROOT / "artifacts/models/paysim_lightgbm_scorer.joblib").is_file(),
                    reason="trained PaySim model not present")
def test_previously_saved_paysim_scorer_still_scores():
    scorer = RiskScorer.load(ROOT / "artifacts/models/paysim_lightgbm_scorer.joblib")
    rows = pd.read_csv(ROOT / "data/paysim.csv", nrows=300)
    scored = scorer.score(rows)
    assert len(scored) == 300 and scored["risk_score"].between(0, 1).all()

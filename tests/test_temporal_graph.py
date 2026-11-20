from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qf_guard.data import normalize_transactions
from qf_guard.operations import CaseStore, EventProcessor
from qf_guard.scoring import RiskScorer
from qf_guard.temporal_graph import TemporalGraphModel, TemporalNeighborhood

MODEL = Path(__file__).parents[1] / "artifacts/models/paysim_lightgbm_scorer.joblib"


def sample():
    return normalize_transactions(pd.DataFrame({
        "transaction_id": ["first", "second", "third", "fourth"],
        "timestamp": pd.to_datetime(["2030-01-01T00:00Z", "2030-01-01T00:00Z",
                                     "2030-01-01T00:01Z", "2030-01-10T00:00Z"]),
        "sender_id": ["A", "A", "B", "A"], "receiver_id": ["B", "C", "A", "B"],
        "amount": [800, 10, 20, 30],
    }))


def test_temporal_messages_use_stream_order_and_horizon():
    data = sample()
    edge, present, current = TemporalNeighborhood().transform(data)
    assert not present[0].any()
    assert present[1].sum() == 1 and edge[1, 0, 0, 1] == 0
    assert present[2].sum() == 3
    assert not present[3].any()
    assert current[0, 0] > current[1, 0]
    warm = TemporalNeighborhood()
    warm.warm(data.iloc[:2])
    for expected, recovered in zip((edge, present, current), warm.transform(data.iloc[2:]), strict=True):
        np.testing.assert_allclose(expected[2:], recovered)
    with pytest.raises(ValueError, match="precedes"):
        warm.warm(data.iloc[:1])


def test_graph_scores_match_offline_batches_retries_and_restart(tmp_path):
    rng = np.random.default_rng(5)
    model = RiskScorer.load(MODEL)
    model.temporal_graph_ = TemporalGraphModel(*(rng.random(shape) * .2 for shape in (
        (4, 8), (8,), (18, 16), (16,), (16, 1), (1,),
    )))
    model.history_transactions_ = None
    model_path = model.save(tmp_path / "tgnn.joblib")
    data = sample()
    offline = model.score(data)
    store = CaseStore(tmp_path / "ops.sqlite3", "tgnn-test")
    processor = EventProcessor.load(store, model_path)
    assert processor.ingest(data.iloc[:2]) == 2
    assert processor.ingest(data.iloc[:2]) == 0
    with pytest.raises(ValueError, match="conflicting payload"):
        processor.ingest(data.iloc[1:3].assign(amount=456))
    restored = EventProcessor.load(CaseStore(store.path, store.stream), model_path)
    assert restored.ingest(data.iloc[2:]) == 2
    rows = [restored.store.case(tx) for tx in data.transaction_id]
    assert {row["graph_model"] for row in rows} == {"tgnn"}
    np.testing.assert_allclose([r["graph_score"] for r in rows], offline.graph_score, atol=1e-7)
    np.testing.assert_allclose([r["risk_score"] for r in rows], offline.risk_score, atol=1e-7)


def test_controlled_pilot_fuses_tgnn_and_recalibrates_threshold():
    scorer = RiskScorer()
    scorer.config.min_tgnn_weight = .05
    rng = np.random.default_rng(9)
    components = pd.DataFrame({
        column: rng.random(40) for column in ("supervised_score", "anomaly_score", "graph_score", "rule_score")
    })
    components["is_fraud"] = (components["supervised_score"] > .75).astype(int)
    with pytest.raises(ValueError, match="requires a trained TGNN"):
        scorer._calibrate_risk(components)
    scorer._calibrate_risk(components.assign(graph_model="tgnn"))
    assert scorer.learned_weights_[2] >= .05
    assert np.isclose(sum(scorer.learned_weights_), 1)
    assert scorer.candidate_threshold_ == scorer.validation_stats_["operational_candidate_threshold"]

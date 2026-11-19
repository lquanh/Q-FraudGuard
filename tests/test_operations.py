from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qf_guard.data import normalize_transactions
from qf_guard.features import attribute_features, confirmed_contact_features, engineer_features, graph_risk_features
from qf_guard.online import OnlineFeatures
from qf_guard.operations import CaseStore, EventProcessor, normalize_events
from qf_guard.scoring import RiskScorer, review_threshold


MODEL = Path(__file__).resolve().parents[1] / "artifacts/models/paysim_lightgbm_scorer.joblib"


def events(n, offset=0):
    return pd.DataFrame([dict(
        transaction_id=f"T{i:04d}", timestamp=pd.Timestamp("2027-01-01", tz="UTC") + pd.Timedelta(minutes=i),
        amount=200. + i, sender_id=f"A{i % 6}", receiver_id=f"A{i % 5}",
        device_id=f"D{i % 2}", type="TRANSFER",
    ) for i in range(offset, offset + n)])


def test_incremental_features_match_batch_with_long_history_equal_times_and_delayed_labels():
    data = events(1600)
    data.loc[1500:, "timestamp"] = data.loc[1500, "timestamp"]
    # Account A4 first activates on the first test event, including its earlier contacts.
    first = data.timestamp.iloc[0].value // 60_000_000_000
    confirmed = {"A4": first + 1490, "A1": first + 1499}
    state = OnlineFeatures(confirmed, 10, ("confirmed",))
    history, current = data.iloc[:1499], data.iloc[1499:]
    state.warm(normalize_transactions(history.iloc[:53]))
    state.warm(normalize_transactions(history.iloc[53:]))
    base, graph, matrix = [], [], []
    for start in range(0, len(current), 7):
        part, g, m = state.transform(normalize_transactions(current.iloc[start:start + 7]))
        base.append(part.matrix)
        graph.append(g)
        matrix.append(m)
    full = normalize_transactions(data)
    expected_base = engineer_features(full).matrix.iloc[1499:].reset_index(drop=True)
    expected_graph = graph_risk_features(full).iloc[1499:].reset_index(drop=True)
    expected_labels = confirmed_contact_features(full, confirmed, 10).iloc[1499:].reset_index(drop=True)
    pd.testing.assert_frame_equal(pd.concat(base, ignore_index=True), expected_base)
    pd.testing.assert_frame_equal(pd.concat(graph, ignore_index=True), expected_graph)
    pd.testing.assert_frame_equal(pd.concat(matrix, ignore_index=True)[expected_labels.columns], expected_labels)


def test_review_rate_uses_only_validation_scores_and_keeps_screening_floor():
    assert review_threshold(np.arange(100) / 100, .1, .02) == pytest.approx(.98)
    assert review_threshold([.01, .1, .2], .5, .02) == .5
    with pytest.raises(ValueError):
        review_threshold([.1, np.nan], 0, .02)


def test_durable_ingest_restores_scores_across_restart_and_failure(tmp_path, monkeypatch):
    store = CaseStore(tmp_path / "cases.db", "test")
    model = RiskScorer.load(MODEL)
    first = EventProcessor(store, model, "v1", .3)
    frame = events(20)
    frame["timestamp"] = frame["timestamp"].dt.as_unit("us")
    assert first.ingest(frame.iloc[:10]) == 10
    assert first.ingest(frame.iloc[:10]) == 0
    assert first.ingest(frame.iloc[10:12], expected_offset=0) == 0
    other = EventProcessor(store, model, "v1", .3)
    assert other.ingest(frame.iloc[10:15]) == 5
    original = first._save
    def fail_after_write(*args):
        original(*args)
        raise RuntimeError("simulated failure before commit")
    monkeypatch.setattr(first, "_save", fail_after_write)
    with pytest.raises(RuntimeError, match="before commit"):
        first.ingest(frame.iloc[15:])
    assert store.stats()["processed"] == 15
    monkeypatch.setattr(first, "_save", original)
    assert first.ingest(frame.iloc[15:]) == 5
    rows = store.recent()
    assert [r["transaction_id"] for r in rows] == frame.transaction_id.tolist()
    assert np.allclose([r["risk_score"] for r in rows], model.score(frame).risk_score, rtol=0, atol=1e-12)
    assert store.stats()["labelled"] == 0
    assert store.recent(3, through="T0008")[-1]["transaction_id"] == "T0008"
    with pytest.raises(ValueError, match="conflicting payload"):
        other.ingest(frame.iloc[:1].assign(amount=13))
    with pytest.raises(ValueError, match="precedes"):
        other.ingest(frame.iloc[:1].assign(transaction_id="late"))
    with pytest.raises(ValueError, match="new QFG_STREAM"):
        EventProcessor(store, model, "a different model", .3)
    assert store.stats()["processed"] == 20


def test_no_cases_dropped_and_only_one_operator_decision_is_committed(tmp_path):
    store = CaseStore(tmp_path / "operations.db", "live")
    model = RiskScorer.load(MODEL)
    model.candidate_threshold_ = 0
    processor = EventProcessor(store, model, "model", 0)
    assert processor.ingest(events(530)) == 530
    count, page = store.queue(limit=5, offset=510)
    assert count == 530 and len(page) == 5
    assert store.queue(limit=1)[1][0]["risk_score"] >= page[0]["risk_score"]
    case_id = page[0]["transaction_id"]
    with ThreadPoolExecutor(2) as pool:
        saved = list(pool.map(lambda actor: store.decide(case_id, "request_verification", actor), ("a", "b")))
    assert sorted(saved) == [False, True]
    reopened = CaseStore(store.path, "live")
    assert reopened.queue()[0] == 529
    log = reopened.decisions()
    assert len(log) == 1 and log[0]["actor"] in {"a", "b"} and log[0]["transaction_id"] == case_id
    assert reopened.decisions(after=log[0]["id"]) == []
    assert processor.ingest(events(530)) == 0
    lease, batch = reopened.claim_window(8, 8, 10)
    assert len(batch) == 8
    another, later = store.claim_window(8, 8, 10)
    assert not ({r["transaction_id"] for r in later} & {r["transaction_id"] for r in batch})
    reopened.finish_window(lease, {batch[0]["transaction_id"]})
    store.finish_window(another, None)
    assert reopened.queue(use_qubo=True)[1][0]["transaction_id"] == batch[0]["transaction_id"]


def test_ingest_accepts_missing_labels_and_refuses_ambiguous_ids_and_order():
    rows = events(3).assign(is_fraud=[None, 1, 0])
    assert normalize_events(pd.concat([rows.iloc[:1], rows])).transaction_id.tolist() == rows.transaction_id.tolist()
    for data, message in (
        (rows.drop(columns="transaction_id"), "requires transaction_id"),
        (rows.iloc[::-1], "chronological"),
        (pd.concat([rows.iloc[:1], rows.iloc[:1].assign(amount=0)]), "conflicting"),
    ):
        with pytest.raises(ValueError, match=message):
            normalize_events(data)


def test_heterogeneous_jsonl_producers_get_the_same_features_and_idempotency(tmp_path):
    store = CaseStore(tmp_path / "events.db", "mixed")
    processor = EventProcessor.load(store, MODEL)
    first, second = events(2).to_dict("records")
    first.pop("device_id")
    first.pop("type")
    first["payment_currency"] = "Euro"
    second.update(old_balance_sender=42., receiving_currency="USD", is_fraud=1)
    assert processor.ingest(pd.DataFrame([first, second])) == 2
    assert processor.ingest(pd.DataFrame([first])) == 0
    assert processor.ingest(pd.DataFrame([second])) == 0
    assert store.stats()["labelled"] == 1
    attrs = attribute_features(normalize_events(pd.DataFrame([first, second])), ("attributes",))
    assert attrs.currency_mismatch.tolist() == [0., 0.]

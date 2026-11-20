from pathlib import Path

import pytest

from qf_guard.operations import CaseStore, EventProcessor
from qf_guard.scoring import RiskScorer

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from qf_guard.core_banking import MAX_BODY_BYTES, create_app  # noqa: E402

MODEL = Path(__file__).parents[1] / "artifacts/models/paysim_lightgbm_scorer.joblib"
TOKENS = {key: key * 32 for key in ("ingest", "receipts", "read")}


def headers(role):
    return {"Authorization": f"Bearer {TOKENS[role]}"}


@pytest.fixture
def api(tmp_path):
    model = RiskScorer.load(MODEL)
    model.candidate_threshold_ = 0
    model.validation_stats_["alert_hold_threshold"] = 1
    store = CaseStore(tmp_path / "core.sqlite3", "core-test")
    processor = EventProcessor(store, model, "fixture", threshold=1)
    return TestClient(create_app(processor=processor, tokens=TOKENS))


def transaction(identifier="transfer-1", minute=1):
    return {
        "transaction_id": identifier, "timestamp": f"2030-01-01T12:{minute:02}:00Z",
        "sender_id": "tok-send", "receiver_id": "tok-receive", "amount": 500000,
        "amount_currency": "VND",
    }


def send(api, *events, role="ingest"):
    return api.post("/v1/transactions", headers=headers(role), json={"transactions": events})


def test_stream_validation_retries_and_atomic_conflicts(api):
    assert api.get("/healthz").json() == {"status": "ready"}
    assert api.get("/v1/cases", params={"transaction_id": "transfer-1"}).status_code == 401
    assert send(api, transaction(), role="read").status_code == 401
    assert send(api, dict(transaction(), amount=-10)).status_code == 422
    assert send(api, dict(transaction(), timestamp="2030-01-01T12:01:00")).status_code == 422
    assert send(api, dict(transaction(), timestamp="9999-01-01T12:01:00Z")).status_code == 422
    assert send(api, dict(transaction(), is_fraud=True)).status_code == 422
    assert send(api, transaction("future", 3), transaction("earlier", 2)).status_code == 422
    assert send(api, transaction()).json() == {"received": 1, "inserted": 1, "duplicates": 0}
    assert send(api, transaction(), transaction()).json() == {"received": 2, "inserted": 0, "duplicates": 2}
    assert send(api, dict(transaction(), amount=1), transaction("other", 2)).status_code == 409
    assert send(api, transaction("late", 0)).status_code == 409
    assert send(api, transaction("other", 2)).json()["inserted"] == 1
    assert api.post(
        "/v1/transactions", headers=headers("ingest"), content=b"x" * (MAX_BODY_BYTES + 1),
    ).status_code == 413
    assert api.post(
        "/v1/transactions", headers=headers("ingest"),
        content=iter([b"{", b"x" * MAX_BODY_BYTES, b"}"]),
    ).status_code == 413


def test_verification_receipt_is_recorded_once_and_cursor_advances(api):
    assert send(api, transaction()).is_success
    case = api.get(
        "/v1/cases", headers=headers("read"), params={"transaction_id": "transfer-1"},
    ).json()
    assert case["state"] == "awaiting_info" and case["graph_model"] == "heuristic"
    receipt = {
        "receipt_id": "response-id", "transaction_id": "transfer-1", "account_id": "tok-send",
        "expected_version": case["version"], "action": "info_received", "reason": "Bank verified receipt",
    }

    def report(**changes):
        return api.post("/v1/verification-receipts", headers=headers("receipts"), json=receipt | changes)

    assert report(account_id="wrong-account").status_code == 409
    assert report(expected_version=case["version"] + 1).status_code == 409
    assert report().json() == report().json() == {"recorded": True}
    assert report(reason="Receipt unexpectedly modified").status_code == 409
    page = api.get("/v1/case-events", headers=headers("read"), params={"limit": 1}).json()
    next_page = api.get(
        "/v1/case-events", headers=headers("read"), params={"after": page["next_after"]},
    ).json()
    assert [event["action"] for event in next_page["events"]] == ["info_received"]
    assert next_page["events"][0]["current_state"] == "response_recorded"
    assert next_page["next_after"] > page["next_after"]
    assert api.get(
        "/v1/case-events", headers=headers("read"), params={"after": next_page["next_after"]},
    ).json()["events"] == []


def test_adapter_requires_three_distinct_keys():
    with pytest.raises(RuntimeError, match="three distinct"):
        create_app(tokens={role: "A" * 40 for role in TOKENS})

"""Chạy một phiên recorded và một ca tích hợp HTTP trong hai sổ tạm độc lập."""
from pathlib import Path
from tempfile import TemporaryDirectory
import datetime
import json
import secrets
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from fastapi.testclient import TestClient
from qf_guard.core_banking import create_app
from qf_guard.operations import CaseStore, EventProcessor
from qf_guard.review_runs import load_haituo_recording
from qf_guard.scoring import RiskScorer

def replay(path):
    store = CaseStore(path, "minh-chung-recorded")
    load_haituo_recording(store, root)
    case = "AML-004060379"
    actor = "chuyenvien1"
    def counts():
        return {"review": store.queue()[0], "proposals": store.queue(proposals=True)[0]}
    before = counts()
    assert store.admit(case, actor, "Đối chiếu ba hồ sơ có cùng tài khoản gửi trong phiên Haituo.")
    after = counts()
    assert store.decide(case, "request_verification", actor, reason="Yêu cầu đối chiếu chứng từ của giao dịch 170,49 USD.", require_review=True)
    reopened = CaseStore(path, "minh-chung-recorded")
    assert before == {"review":1,"proposals":3} and after == {"review":2,"proposals":2}
    assert reopened.case(case)["decision_reason"] == "Yêu cầu đối chiếu chứng từ của giao dịch 170,49 USD."
    return {"case_id":case, "protocol":"recorded, luồng tổng hợp, sổ độc lập",
            "before_admit":before, "after_admit":after,
            "journal_after_reopen":[{k:r[k] for k in ("kind","actor","at","detail") if k in r} for r in reopened.journal(case)]}

def sandbox(path):
    model = RiskScorer.load(root / "artifacts/models/paysim_lightgbm_scorer.joblib")
    model.candidate_threshold_ = 0
    model.validation_stats_["alert_hold_threshold"] = 1
    store = CaseStore(path, "minh-chung-sandbox")
    processor = EventProcessor(store, model, "fixture-paysim", threshold=1)
    tokens = {role:secrets.token_hex(32) for role in ("ingest","receipts","read")}
    def headers(role):
        return {"Authorization":"Bearer " + tokens[role]}
    with TestClient(create_app(processor=processor,tokens=tokens)) as api:
        txid = "MC-CORE-001"
        tx = {"transaction_id":txid,"timestamp":"2026-11-18T09:00:00Z",
              "sender_id":"acct-demo-A","receiver_id":"acct-demo-B","amount":500000,"amount_currency":"VND"}
        body = {"transactions":[tx]}
        def send():
            response = api.post("/v1/transactions",headers=headers("ingest"),json=body)
            response.raise_for_status()
            return response.json()
        first, retry = send(), send()
        case = api.get("/v1/cases",params={"transaction_id":txid},headers=headers("read")).json()
        receipt = {"receipt_id":"MC-RECEIPT-001","transaction_id":txid,"account_id":"acct-demo-A",
                   "expected_version":case["version"],"action":"info_received","reason":"Đã đối soát hồ sơ minh họa."}
        def accept():
            r = api.post("/v1/verification-receipts",json=receipt,headers=headers("receipts"))
            r.raise_for_status()
            return r.json()
        recorded, receipt_retry = accept(), accept()
        events = api.get("/v1/case-events",params={"transaction_id":txid},headers=headers("read")).json()
        next_page = api.get("/v1/case-events",params={"after":events["next_after"]},headers=headers("read")).json()
        assert first == {"received":1,"inserted":1,"duplicates":0}
        assert retry == {"received":1,"inserted":0,"duplicates":1}
        assert recorded == receipt_retry == {"recorded":True}
        assert sum(e["action"]=="info_received" for e in events["events"]) == 1 and not next_page["events"]
        return {"case_id":txid, "protocol":"HTTP v1 sandbox, ngưỡng cố định để thử hợp đồng",
                "first_request":first,"same_request_retried":retry,"receipt":recorded,"receipt_retried":receipt_retry,
                "events":events["events"],"cursor":events["next_after"],"next_page_events":next_page["events"]}

with TemporaryDirectory() as folder:
    output={"time":datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "workspace":replay(Path(folder)/"recorded.db"),"core_banking":sandbox(Path(folder)/"api.db")}
    print(json.dumps(output,ensure_ascii=False,indent=2))

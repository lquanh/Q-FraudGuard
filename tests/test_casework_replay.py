from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qf_guard.operations import CaseStore, _write_scores
from qf_guard.explainability import review_signals
from qf_guard.replay import FullReplay, prepare_replay


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts/models/paysim_lightgbm_scorer.joblib"


def store_with_cases(path, stream="test"):
    store = CaseStore(path, stream)
    with store.connection(write=True) as db:
        db.execute("INSERT INTO streams(name,model,review_threshold,auto_routing) VALUES(?,?,?,1)",
                   (stream, "model", .5))
    add_sample(store, ["tx-info","tx-verify","tx-high"], [.2,.6,.95], ["alice","alice","bob"])
    return store


def add_sample(store, ids, scores, senders):
    import json
    from types import SimpleNamespace

    model = SimpleNamespace(candidate_threshold_=.1, validation_stats_={"alert_hold_threshold":.85})
    payloads = [json.dumps(dict(transaction_id=i, sender_id=a, receiver_id="dest", type="TRANSFER",
                                timestamp="2022-09-01T00:00:00+00:00", amount=10)) for i,a in zip(ids,senders)]
    df = pd.DataFrame(dict(transaction_id=ids,risk_score=scores,is_candidate=np.array(scores)>=.1,
                            supervised_score=scores,graph_score=0.,anomaly_score=0.,rule_score=0.,reason_codes=""))
    with store.connection(write=True) as db:
        _write_scores(db,store.stream,df,payloads,.5,model)


@pytest.mark.parametrize(("txid", "action", "source"), [
    ("tx-info", "info_received", "Khách hàng"),
    ("tx-verify", "verify_ok", "Kênh xác minh"),
    ("tx-verify", "verify_failed", "Kênh xác minh"),
])
def test_customer_response_logs_once_without_a_transaction_decision(tmp_path, txid, action, source):
    store = store_with_cases(tmp_path/"cases.db")
    previous = store.case(txid)["work_state"]
    initial = store.work_history(txid)[0]["id"]
    notice = store.latest_alert()
    wrong_action = "verify_ok" if action == "info_received" else "info_received"
    assert not store.customer_response(txid, wrong_action, "wrong kind", 1, account="alice")
    assert not store.customer_response(txid, action, "wrong account", 1, account="other")
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: store.customer_response(txid, action, "evidence", 1, account="alice"), range(2)))
    assert sorted(results) == [False, True]
    assert not store.customer_response(txid, action, "repeat", 2, account="alice")
    assert not store.operator_action(txid, "allow", "analyst", "release", 2)
    assert not store.decide(txid, "continue_monitoring", "analyst")
    change, = store.case_changes(after=initial, txid=txid)
    assert (change["action"], change["actor"], change["reason"], change["previous"]) == (
        action, source + " · alice", "evidence", previous)
    assert change["state"] == change["current_state"] == "response_recorded"
    assert change["current_version"] == 2 and pd.Timestamp(change["at"]).utcoffset().total_seconds() == 0
    assert store.journal(txid)[0]["detail"] == action + " · evidence"
    assert len(store.work_history(txid)) == len(store.journal(txid)) == 2
    assert store.work_queue((previous,))[0] == 0 and store.decisions() == []
    assert txid not in [row["transaction_id"] for row in store.queue(all_candidates=True)[1]]
    assert store.latest_alert() == notice and store.work_queue(("pending_review",))[0] == 1


def test_requested_verification_closes_only_the_response_request(tmp_path):
    store = store_with_cases(tmp_path/"cases.db")
    assert store.operator_action("tx-high", "request_verification", "analyst", "check documents", 1)
    assert store.customer_response("tx-high", "verify_failed", "inconclusive", 2, account="bob")
    assert [e["action"] for e in store.work_history("tx-high")] == [
        "escalate", "request_verification", "verify_failed"]
    assert store.case("tx-high")["work_state"] == "response_recorded"
    assert store.latest_alert() is None and store.work_queue(("pending_review",))[0] == 0


def test_legacy_automatic_dispositions_are_archived_once_and_human_actions_survive(tmp_path):
    store = store_with_cases(tmp_path / "old.db")
    add_sample(store, ["manual", "legacy-operator"], [.6, .2], ["carol", "dan"])
    old_time = "2024-01-12T08:00:00+00:00"
    with store.connection(write=True) as db:
        for txid, previous, reply, disposition, state in (
            ("tx-info", "awaiting_info", "info_received", "resume", "released"),
            ("tx-verify", "awaiting_verification", "verify_failed", "escalate", "pending_review"),
            ("manual", "awaiting_verification", "verify_failed", "escalate", "pending_review"),
            ("legacy-operator", "awaiting_info", "info_received", "resume", "released"),
        ):
            event = db.execute("SELECT seq FROM events WHERE txid=?", (txid,)).fetchone()[0]
            stmt = "INSERT INTO case_history(event,actor,action,previous,state,reason,at) VALUES(?,?,?,?,?,?,?)"
            db.execute(stmt, (event, "Khách hàng", reply, previous, previous, "old response", old_time))
            notice = db.execute(stmt, (event, "Hệ thống", disposition, previous, state, "old rule", old_time)).lastrowid
            db.execute("UPDATE casework SET state=?,notice=?,version=2 WHERE event=?", (state, notice, event))
        db.execute("PRAGMA user_version=0")
    assert store.operator_action("tx-info", "note", "analyst", "saved note", 2)
    assert store.operator_action("manual", "suspend", "analyst", "independent restriction", 2)
    assert store.decide("legacy-operator", "continue_monitoring", "analyst")
    old = {txid: store.work_history(txid) for txid in ("tx-info", "tx-verify", "manual", "tx-high", "legacy-operator")}
    for _ in range(2):
        store = CaseStore(store.path, store.stream)
        for txid in ("tx-info", "tx-verify"):
            history = store.work_history(txid)
            assert history[:-1] == old[txid] and history[-1]["action"] == "archive_response"
            assert store.case(txid)["work_state"] == "response_recorded"
        assert store.work_history("manual") == old["manual"] and store.case("manual")["work_state"] == "suspended"
        assert store.case("manual")["hold_origin"] == store.case("manual")["work_event"]
        assert store.work_history("legacy-operator") == old["legacy-operator"]
        assert store.case("legacy-operator")["work_state"] == "released"
        assert [d["transaction_id"] for d in store.decisions()] == ["legacy-operator"]
        assert store.work_history("tx-high") == old["tx-high"] and store.latest_alert()["txid"] == "tx-high"


def test_holds_are_scoped_persist_and_guard_continuation(tmp_path):
    path = tmp_path/"cases.db"
    store = store_with_cases(path)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda a: store.operator_action("tx-high","claim",a,"review",1), ("kim","minh")))
    assert sorted(results) == [False, True]
    actor = store.case("tx-high")["work_owner"]
    assert not store.operator_action("tx-high","freeze","another","review",2)
    add_sample(store, ["bob-open"], [.2], ["bob"])
    assert store.operator_action("tx-high","suspend",actor,"confirmed risk",2)
    assert not store.customer_response("tx-high","verify_ok","ok",3,account="bob")
    assert store.customer_response("bob-open","info_received","invoice",1,account="bob")
    assert store.case("bob-open")["work_state"] == "response_recorded"
    assert store.case("bob-open")["hold_origin"] == store.case("tx-high")["work_event"]
    assert store.case("tx-high")["work_state"] == "suspended"
    add_sample(store, ["new-bob","new-alice"], [.01,.01], ["bob","alice"])
    assert store.case("new-bob")["work_state"] == "pending_review"
    assert not store.case("new-alice")["work_state"]
    assert not store.operator_action("new-bob","allow","operator","release",1)
    assert store.operator_action("new-bob","freeze","operator","specific transaction",1)
    assert store.operator_action("new-bob","unfreeze","operator","review pending",2)
    assert store.case("new-bob")["work_state"] == "pending_review"
    store = CaseStore(path,"test")
    assert store.operator_action("tx-high","restore_account",actor,"cleared",3)
    assert store.operator_action("new-bob","allow","operator","cleared",3)
    assert store.case("new-bob")["work_state"] == "released"
    assert store.case("new-bob")["hold_origin"] is None
    assert CaseStore(path,"other").case("new-bob") is None
    with store.connection() as db:
        assert not db.execute("PRAGMA foreign_key_check").fetchall()


def test_loop_full_source_restart_and_rollover(tmp_path):
    raw = pd.read_csv(ROOT/"data/paysim.csv", nrows=7).iloc[::-1].reset_index(drop=True)
    raw["step"] = [7,6,5,4,3,2,1]
    raw["split"] = ["history"]*3 + ["stream"]*4
    path, index = tmp_path/"data.csv", tmp_path/"source.sqlite"
    raw.to_csv(path,index=False)
    prepare_replay(path,index,chunk_size=3)
    dbpath = tmp_path/"cases.db"
    store = CaseStore(dbpath,"loop")
    replay = FullReplay(store, MODEL, index)
    assert replay.total == 7 and replay.tick(3) == 3
    assert replay.tick(3) == 3
    restarted = FullReplay(CaseStore(dbpath,"loop"),MODEL,index)
    assert restarted.tick(3) == 1
    assert restarted.progress()["position"] == 7
    assert restarted.tick(3) == 3
    assert replay.tick(3) == 3
    assert restarted.tick(3) == 1
    assert replay.tick(2,interval=60) == 2
    assert restarted.tick(2,interval=60) == 0
    rows = store.recent(20)
    first, second = rows[:7], rows[7:14]
    assert [r["source_transaction_id"] for r in first] == [r["source_transaction_id"] for r in second]
    assert np.allclose([r["risk_score"] for r in first],[r["risk_score"] for r in second])
    assert len({r["transaction_id"] for r in rows}) == 16
    assert replay.progress()["cycle"] == 3
    assert [r["timestamp"] for r in first] == sorted(r["timestamp"] for r in first)
    raw.iloc[:2].to_csv(path,index=False)
    with pytest.raises(ValueError,match="Nguồn"):
        prepare_replay(path,index)


def test_operator_and_customer_ui(tmp_path, monkeypatch, open_dashboard):
    path = tmp_path/"ui.db"
    store = store_with_cases(path)
    monkeypatch.setenv("QFG_SOURCE","loop")
    monkeypatch.setenv("QFG_REPLAY_EXTERNAL","1")
    monkeypatch.setenv("QFG_PRIORITY_MODE","risk")
    monkeypatch.setenv("QFG_CASE_DB",str(path))
    monkeypatch.setenv("QFG_STREAM","test")
    app = open_dashboard()
    assert not app.exception and app.subheader[0].value == "tx-high"
    app.selectbox(key="work_tx-high_1_action").set_value("freeze").run()
    app.text_area[0].set_value("verified by analyst")
    next(b for b in app.button if b.label=="Ghi thao tác").click().run()
    assert store.case("tx-high")["work_state"] == "pending_review"
    app.checkbox[0].check()
    next(b for b in app.button if b.label=="Ghi thao tác").click().run()
    assert store.case("tx-high")["work_state"] == "frozen"
    assert store.work_history("tx-high")[-1]["actor"] == "chuyenvien1"
    app.button_group(key="area").set_value("Phản hồi khách hàng").run()
    app.button(key="customer_pick_tx-info").click().run()
    app.text_area[0].set_value("invoice supplied")
    next(b for b in app.button if b.label=="Ghi phản hồi vào nhật ký").click().run()
    assert not app.exception and store.case("tx-info")["work_state"]=="response_recorded"
    assert any(item.value == "invoice supplied" for item in app.text)
    app.button(key="customer_log_tx-info").click().run()
    assert app.text_input(key="journal_query").value == "tx-info"
    assert not app.exception and any("invoice supplied" in item.value for item in app.markdown)
    app.button_group(key="area").set_value("Phản hồi khách hàng").run()
    app.button(key="customer_pick_tx-verify").click().run()
    submit = "Ghi phản hồi vào nhật ký"
    next(b for b in app.button if b.label == submit).click().run()
    assert store.case("tx-verify")["work_state"] == "awaiting_verification"
    app.radio(key="response_tx-verify_1_action").set_value("verify_failed")
    next(b for b in app.button if b.label == submit).click().run()
    assert not app.exception and store.case("tx-verify")["work_state"] == "response_recorded"
    app.button(key="customer_log_tx-verify").click().run()
    assert not app.exception and app.text_input(key="journal_query").value == "tx-verify"
    entry = store.work_history("tx-verify")[-1]["id"]
    app.button(key=f"log_case_casework_tx-verify_{entry}").click().run()
    assert not app.exception and "Nhật ký hồ sơ" in [t.label for t in app.tabs]
    assert not any(item.label == "Thao tác chuyên viên" for item in app.selectbox)


def test_backend_switch_applies_at_window_boundaries(tmp_path):
    store = store_with_cases(tmp_path / "cases.db")
    first_choice = store.backend_choice("exact")
    first, _ = store.claim_window(1, 1, 0, backend="exact", expected_backend_id=first_choice["id"])
    assert store.change_backend("quafu-fast-mock", "ops", default="exact", expected_id=0)
    assert not store.change_backend("dwave-qpu", "stale", default="exact", expected_id=0)
    assert not store.claim_window(1, 1, 0, backend="exact", expected_backend_id=0)
    current = CaseStore(store.path, store.stream).backend_choice("exact")
    assert (current["backend"], current["actor"]) == ("quafu-fast-mock", "ops")
    next_id, _ = store.claim_window(1, 1, 0, backend=current["backend"], expected_backend_id=current["id"])
    assert store.window(first)["backend"] == "exact"
    assert store.window(next_id)["backend"] == "quafu-fast-mock"
    assert store.change_backend("exact", "ops", default="exact", expected_id=current["id"])
    assert store.window(next_id)["backend"] == "quafu-fast-mock"


def test_operator_views_keep_technical_details_in_server_records(tmp_path, monkeypatch, open_dashboard):
    store = store_with_cases(tmp_path / "ui.db")
    lease, _ = store.claim_window(1, 1, 0, backend="quafu-fast-qpu")
    assert store.finish_window(lease, None, error="Quafu Haituo QPU failed; CPU fallback unavailable")
    for key, value in {"QFG_SOURCE": "loop", "QFG_REPLAY_EXTERNAL": "1", "QFG_EXTERNAL_PRIORITY": "1",
                       "QFG_CASE_DB": str(store.path), "QFG_STREAM": store.stream, "QFG_SOLVER": "exact"}.items():
        monkeypatch.setenv(key, value)
    app = open_dashboard()
    for area in ("Bàn điều tra", "Phản hồi khách hàng", "Phiên ưu tiên", "Nhật ký"):
        app.button_group(key="area").set_value(area).run()
        assert not app.exception
        content = " ".join(item.value for item in [*app.markdown, *app.caption, *app.text, *app.warning, *app.error])
        content += " ".join(b.label for b in app.button)
        assert all(name not in content.lower() for name in ("quafu", "haituo", "qpu", "cpu", "bộ giải", "solver"))
    assert "Quafu" in store.window(lease)["runs"][0]["error"]


def test_account_evidence_is_scoped_to_the_source_time_and_pass(tmp_path):
    store = CaseStore(tmp_path/"links.db", "desk")
    with store.connection(write=True) as db:
        db.executemany("INSERT INTO streams(name,model,review_threshold) VALUES(?,'test',0.5)", [("desk",), ("other",)])

        def insert(txid, source, target, time, amount=10, currency="USD", *, cycle=2, stream="desk", alert=False):
            payload = dict(transaction_id=txid, sender_id=source, receiver_id=target, timestamp=time,
                           replay_cycle=cycle, amount=amount, amount_currency=currency)
            db.execute("INSERT INTO events(stream,txid,payload,result,risk,candidate,review,created) VALUES(?,?,?,?,0,?,0,0)",
                       (stream, txid, json.dumps(payload), json.dumps({"is_candidate": alert}), alert))

        stamp = "2022-09-02T12:00:00+00:00"
        insert("old", "A", "B", "2022-08-30T00:00:00Z")
        insert("other-pass", "A", "B", stamp, cycle=1)
        insert("other-stream", "A", "B", stamp, stream="other")
        insert("first", "A", "B", "2022-09-02T08:00:00Z", alert=True)
        insert("incoming", "C", "B", stamp, alert=True)
        insert("reverse", "B", "A", stamp)
        insert("branch", "A", "D", stamp, 30)
        insert("euro", "A", "E", stamp, 100, "EUR")
        insert("self", "A", "A", stamp, 20)
        insert("same-minute", "A", "B", stamp, 40, "US Dollar")
        insert("other-account", "X", "Y", stamp)
        insert("anchor", "A", "B", stamp, 500)
        insert("later-same-minute", "A", "B", stamp)
    stats, recent = store.account_activity("anchor", hours=24, limit=2)
    assert stats.pop("recorded_since") == "2022-08-30T00:00:00Z"
    assert stats == dict(total=7, alerts=2, sent=5, payees=4, received=3, payers=2, same_pair=2,
                         reverse_pair=1, comparable=4, mean_amount=25)
    assert [r["transaction_id"] for r in recent] == ["same-minute", "self"]
    assert [r["transaction_id"] for r in store.account_activity("anchor", relation="same_pair")[1]] == ["same-minute", "first"]
    assert [r["transaction_id"] for r in store.account_activity("anchor", relation="reverse_pair")[1]] == ["reverse"]
    assert store.account_activity("anchor", hours=168)[0]["same_pair"] == 3
    assert store.account_activity("anchor", hours=1)[0]["same_pair"] == 1
    assert store.account_activity("self")[0]["total"] == 4


def test_unavailable_fields_are_requests_for_evidence():
    row = {"reason_codes": "SHARED_DEVICE|AMOUNT_SPIKE|BALANCE_MISMATCH|BALANCE_DRAIN", "device_id": "UNKNOWN_DEVICE"}
    signals, gaps = review_signals(row)
    assert len(signals) == 1 and "Số tiền" in signals[0]
    assert len(gaps) == 2
    signals, gaps = review_signals(dict(row, device_id="known", old_balance_sender=12))
    assert len(signals) == 4 and not gaps


def test_open_graph_related_history_and_return_to_case(tmp_path, monkeypatch, open_dashboard):
    from xml.etree import ElementTree

    store = store_with_cases(tmp_path/"dossier.db")
    for key, value in {"QFG_SOURCE": "loop", "QFG_REPLAY_EXTERNAL": "1", "QFG_PRIORITY_MODE": "risk",
                       "QFG_CASE_DB": str(store.path), "QFG_STREAM": store.stream}.items():
        monkeypatch.setenv(key, value)
    app = open_dashboard()
    graph = next(item.value for item in app.markdown if 'class="fraud-graph"' in item.value)
    diagram = ElementTree.fromstring(graph)
    stage = diagram.find("./div[@class='flow-stage']")
    svg = stage.find("svg")
    assert svg is not None and svg.findall(".//circle") and svg.findall(".//line[@marker-end]")
    trace = diagram.find(".//details[@class='qfg-disclosure qfg-trace']")
    assert trace is not None and trace.get("open") is None
    active = store.case("tx-high")
    endpoints = {active["sender_id"], active["receiver_id"]}
    assert {b.text for b in trace.findall("./p/bdi")} == endpoints
    assert {g.findtext("title").split(" · ")[0] for g in svg.findall(".//g[@class='flow-account'][@data-trace='current']")} == endpoints
    for node in svg.findall(".//g[@aria-describedby]"):
        tip = stage.find(f"div[@id='{node.get('aria-describedby')}']")
        tid = tip.findtext("strong")
        assert node.get("aria-label") == f"Giao dịch {tid}"
        assert node.get("tabindex") == "0" and tip.get("role") == "tooltip"
        route_tag = node.get("class").split()[-1]
        route = svg.find(f".//g[@class='flow-target flow-route {route_tag}']")
        assert route is not None
        assert route.get("data-trace") == node.get("data-trace") == ("current" if tid == "tx-high" else "context")
        details = {k.text: v.text for k, v in zip(tip.findall("dl/dt"), tip.findall("dl/dd"), strict=True)}
        event = store.case(tid)
        assert details["Tài khoản gửi"] == event["sender_id"] and details["Tài khoản nhận"] == event["receiver_id"]
        assert details["Số tiền"] == "10,00 (chưa rõ loại tiền)" and details["Thời điểm"].endswith("UTC")
    assert len(stage.findall("div[@role='tooltip']")) == len(svg.findall(".//g[@aria-describedby]")) >= 2
    assert "\n\n" not in graph
    app.selectbox(key="history_filter_tx-high").set_value("received").run()
    app.selectbox(key="history_tx_tx-high").set_value("tx-info").run()
    app.button(key="history_open_tx-high").click().run()
    assert not app.exception and app.subheader[0].value == "tx-info"
    app.button(key="case_back_tx-info").click().run()
    assert not app.exception and app.subheader[0].value == "tx-high"

from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
import sqlite3
from time import monotonic, sleep

import numpy as np
import pandas as pd

from qf_guard.operations import CaseStore, EventProcessor
from qf_guard.postprocess import best_result
from qf_guard.prioritization import PriorityConfig
from qf_guard.review_runs import load_haituo_recording, solve_review_window
from qf_guard.scoring import RiskScorer
from qf_guard.solvers import ExactSolver, FallbackSolver


ROOT = Path(__file__).resolve().parents[1]
TARGET = "AML-004060379"


def test_recorded_review_policy_survives_retries_and_restart(tmp_path):
    path = tmp_path / "cases.sqlite"
    store = CaseStore(path, "recorded")
    load_haituo_recording(store, ROOT)
    window_id = store.window_list()[0]["id"]
    assert (store.stats()["processed"], store.stats()["candidates"], store.stats()["reviews"]) == (64, 18, 1)
    assert (store.queue()[0], store.queue(proposals=True)[0]) == (1, 3)
    assert not store.decide(TARGET, "monitor", "alice", require_review=True)
    with ThreadPoolExecutor(2) as pool:
        saved = list(pool.map(lambda actor: store.admit(TARGET, actor, "shared sender"), ("alice", "bob")))
    assert sorted(saved) == [False, True]
    assert (store.queue()[0], store.queue(proposals=True)[0]) == (2, 2)
    assert store.decide(TARGET, "verify", "alice", reason="linked transfers", require_review=True)
    assert not store.decide(TARGET, "verify", "bob", require_review=True)
    with store.connection(write=True) as db:
        db.execute("UPDATE window_runs SET summary=json_set(summary, '$.time_to_first_hit_s', 12345)")
    store = CaseStore(path, "recorded")
    load_haituo_recording(store, ROOT)
    assert (store.queue()[0], store.queue(proposals=True)[0], store.stats()["reviews"]) == (1, 2, 1)
    run = store.window(window_id)
    assert sum(r["selected"] for r in run["members"]) == 4
    assert run["runs"][0]["summary"]["task_id"] == "2611190600591042404"
    assert run["runs"][0]["summary"]["qpu_access_ms"] is None
    assert "time_to_first_hit_s" not in run["runs"][0]["summary"]
    assert store.case(TARGET)["decision_reason"] == "linked transfers"
    for query in (TARGET, window_id):
        assert {entry["kind"] for entry in store.journal(query)} == {"admit", "decide", "run"}
    other = CaseStore(path, "other")
    assert other.case(TARGET) is None and other.window(window_id) is None and other.journal(TARGET) == []


def pending_store(path):
    store = CaseStore(path, "live")
    model = RiskScorer.load(ROOT / "artifacts/models/paysim_lightgbm_scorer.joblib")
    model.candidate_threshold_ = 0
    events = pd.DataFrame([dict(transaction_id=f"T{i}", timestamp="2027-01-01T00:00Z",
                                sender_id="A", receiver_id=f"B{i}", amount=100. + i) for i in range(8)])
    EventProcessor(store, model, "test", 1.).ingest(events)
    return store


def test_expired_worker_cannot_overwrite_retry_and_fallback_has_own_source(tmp_path, monkeypatch):
    now = 100.
    monkeypatch.setattr("qf_guard.operations.time", lambda: now)
    store = pending_store(tmp_path / "cases.db")
    first, batch = store.claim_window(8, 8, 0, backend="quafu-fast-qpu", timeout=2)
    assert store.record_measurement(first, 1, {"task_id": "123"})
    now += 3
    assert not store.finish_window(first, set())
    assert not store.claim_window(8, 8, 0, flush=True)
    with ThreadPoolExecutor(2) as pool:
        attempts = list(pool.map(lambda _: store.retry_window(first, max_attempts=2, timeout=5), range(2)))
    second, retry_batch = next(result for result in attempts if result)
    assert sum(result is None for result in attempts) == 1
    assert [e["transaction_id"] for e in retry_batch] == [e["transaction_id"] for e in batch]
    assert not store.finish_window(first, {batch[0]["transaction_id"]})
    assert not store.record_measurement(first, 2, {})

    class Unavailable:
        def solve(self, model):
            raise RuntimeError("offline provider")

    solver = FallbackSolver(Unavailable(), ExactSolver())
    solve_review_window(store, second, retry_batch, backend="quafu-fast-qpu", config=PriorityConfig(),
                        max_jobs=2, timeout=5, solver_factory=lambda *_: solver)
    window = store.window(first)
    assert window["status"] == "completed" and len(window["runs"]) == 2
    assert window["runs"][0]["error"] and window["runs"][0]["measurements"][0]["task_id"] == "123"
    summary = window["runs"][1]["summary"]
    assert summary["backend"] == "cpu-exhaustive" and "offline provider" in summary["fallback_reason"]
    assert store.queue(proposals=True)[0] == 2
    assert not store.retry_window(first, max_attempts=3, timeout=60)


def test_migrates_old_decisions_and_enforces_attempt_limit(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE decisions(id INTEGER PRIMARY KEY,event INTEGER NOT NULL UNIQUE REFERENCES events(seq),
                      actor TEXT NOT NULL,action TEXT NOT NULL,decided_at TEXT NOT NULL)""")
    store = pending_store(path)
    first, batch = store.claim_window(8, 8, 0)
    assert store.finish_window(first, None, error="first")
    second, _ = store.retry_window(first, max_attempts=2, timeout=60)
    assert store.finish_window(second, None, error="second")
    assert not store.retry_window(first, max_attempts=2, timeout=60)
    assert store.decide(batch[0]["transaction_id"], "verify", "alice", reason="audit")
    assert store.decisions()[0]["reason"] == "audit"


def test_fast_measurements_stop_at_limit_and_mid_run_fallback_is_identified(tmp_path):
    store = pending_store(tmp_path / "bounded.db")

    class Miss:
        calls = 0

        def solve(self, model):
            self.calls += 1
            samples = np.zeros((28, 8), dtype=int)
            for index, pair in enumerate(combinations(range(8), 2)):
                samples[index, list(pair)] = 1
            worst = samples[np.argmax(model.evaluate_many(samples))]
            return best_result(model, worst, started=0, finished=.1, local_search=False,
                               metadata=dict(solver="quafu_fast_local", backend="quafu-fast-mock",
                                             task_id=self.calls, measured_counts={"".join(map(str, worst)): 1}))

    first, batch = store.claim_window(8, 8, 0, backend="quafu-fast-mock")
    miss = Miss()
    options = dict(backend="quafu-fast-mock", config=PriorityConfig(), max_jobs=2, timeout=60)
    solve_review_window(store, first, batch, **options, solver_factory=lambda *_: miss)
    window = store.window(first)
    assert window["status"] == "error" and miss.calls == 2
    assert len(window["runs"][0]["measurements"]) == 2

    class FailAfterMiss(Miss):
        def solve_measurement(self, model, attempt_index):
            raise RuntimeError("second job failed")

    retry, batch = store.retry_window(first, max_attempts=2, timeout=60)
    solver = FallbackSolver(FailAfterMiss(), ExactSolver())
    solve_review_window(store, retry, batch, **options, solver_factory=lambda *_: solver)
    result = store.window(first)
    assert result["status"] == "completed"
    assert result["runs"][1]["summary"]["backend"] == "cpu-exhaustive"
    assert "second job failed" in result["runs"][1]["summary"]["fallback_reason"]
    assert len(result["runs"][1]["measurements"]) == 2


def test_live_dashboard_dispatches_ingested_window(tmp_path, monkeypatch, open_dashboard):
    path = tmp_path / "live.db"
    store = pending_store(path)
    for key, value in {"QFG_SOURCE":"live", "QFG_STREAM":"live", "QFG_CASE_DB":str(path),
                       "QFG_PRIORITY_MODE":"qubo", "QFG_SOLVER":"exact", "QFG_QUANTUM_MIN_BATCH":"8",
                       "QFG_QUANTUM_MAX_BATCH":"8"}.items():
        monkeypatch.setenv(key, value)
    app = open_dashboard()
    assert not app.exception
    deadline = monotonic() + 10
    while store.window_list()[0]["status"] == "running" and monotonic() < deadline:
        sleep(.05)
    assert store.window_list()[0]["actual_backend"] == "cpu-exhaustive"
    assert store.queue(proposals=True)[0] == 2


def test_streamlit_desk_case_to_window_to_journal(tmp_path, monkeypatch, open_dashboard):
    path = tmp_path / "ui.db"
    monkeypatch.setenv("QFG_SOURCE", "recorded")
    monkeypatch.setenv("QFG_STREAM", "ui")
    monkeypatch.setenv("QFG_CASE_DB", str(path))
    monkeypatch.setenv("QFG_OPERATOR", "operator-test")
    app = open_dashboard()
    assert not app.exception
    app.button(key="desk_row_proposals_" + TARGET).click().run()
    assert not app.exception and app.subheader[0].value == TARGET
    assert len([b for b in app.button if (b.key or "").startswith("related_")]) == 2
    next(i for i in app.text_input if i.label == "Lý do tiếp nhận").set_value("shared sender")
    next(b for b in app.button if b.label == "Tiếp nhận điều tra").click().run()
    assert not app.exception and app.button(key="desk_row_review_" + TARGET)
    next(i for i in app.text_input if i.label == "Lý do quyết định").set_value("check beneficiaries")
    next(b for b in app.button if b.label == "Ghi quyết định").click().run()
    assert not app.exception and app.success
    app.button(key="case_window_" + TARGET).click().run()
    assert not app.exception and app.subheader[0].value == "Chọn 4 trong 18 ứng viên"
    assert any("2 trùng" in line.value and "2 chỉ có trong danh sách ưu tiên" in line.value for line in app.markdown)
    app.button_group(key="area").set_value("Nhật ký").run()
    assert not app.exception
    app.text_input(key="journal_query").set_value(TARGET).run()
    assert any("check beneficiaries" in line.value and "chuyenvien1" in line.value for line in app.markdown)
    app.button(key="log_case_decide_" + TARGET).click().run()
    assert app.subheader[0].value == TARGET and not app.exception


def test_dashboard_login_and_logout(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from streamlit.util import calc_hash

    database = tmp_path / "login.db"
    monkeypatch.setenv("QFG_SOURCE", "recorded")
    monkeypatch.setenv("QFG_CASE_DB", str(database))
    monkeypatch.delenv("QFG_DASHBOARD_ADMIN_PASSWORD", raising=False)
    app = AppTest.from_file(str(ROOT / "dashboard/app.py"), default_timeout=40)
    app._page_hash = calc_hash("dashboard")
    app.run()

    assert not app.exception and not app.button_group and not database.exists()
    assert app.text_input(key="login_password").proto.type == 1
    app._page_hash = calc_hash("login")
    app.text_input(key="login_username").set_value("admin")
    app.text_input(key="login_password").set_value("sai")
    next(button for button in app.button if button.label == "Đăng nhập").click().run()
    assert app.error[0].value == "Tên đăng nhập hoặc mật khẩu không đúng."
    assert not app.button_group and not database.exists()

    app.text_input(key="login_username").set_value("admin")
    app.text_input(key="login_password").set_value("admin")
    next(button for button in app.button if button.label == "Đăng nhập").click().run()
    assert not app.exception and app.button_group(key="area").value == "Bàn điều tra"
    assert any("Đang đăng nhập · admin" == item.value for item in app.caption)

    app._page_hash = calc_hash("dashboard")
    app.button(key="operator_logout").click().run()
    assert not app.exception and not app.button_group
    assert app.text_input(key="login_password").value == ""
    assert "dashboard_operator" not in app.session_state

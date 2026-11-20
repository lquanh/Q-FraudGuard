"""Customer response receipts and analyst-controlled case restrictions."""
from datetime import datetime, timezone
from hashlib import sha256
import json


SCHEMA = """
CREATE TABLE IF NOT EXISTS casework (
    event INTEGER PRIMARY KEY REFERENCES events(seq), stream TEXT NOT NULL, account TEXT NOT NULL,
    state TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 1, owner TEXT, updated TEXT NOT NULL, notice INTEGER);
CREATE INDEX IF NOT EXISTS work_inbox ON casework(stream,state,event DESC);
CREATE TABLE IF NOT EXISTS case_history (
    id INTEGER PRIMARY KEY, event INTEGER NOT NULL REFERENCES events(seq), actor TEXT NOT NULL,
    action TEXT NOT NULL, previous TEXT, state TEXT NOT NULL, reason TEXT NOT NULL, at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS case_timeline ON case_history(event,id);
CREATE TABLE IF NOT EXISTS account_holds (
    stream TEXT NOT NULL, account TEXT NOT NULL, origin INTEGER NOT NULL REFERENCES events(seq),
    PRIMARY KEY(stream,account));
CREATE TABLE IF NOT EXISTS customer_receipts (
    stream TEXT NOT NULL, receipt_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
    PRIMARY KEY(stream,receipt_id));
"""

STATES = {
    "awaiting_info": "Chờ bổ sung thông tin", "awaiting_verification": "Chờ xác minh bổ sung",
    "response_recorded": "Đã ghi nhận phản hồi",
    "pending_review": "Chờ chuyên viên quyết định", "released": "Được tiếp tục",
    "frozen": "Đã đóng băng giao dịch", "suspended": "Đã tạm ngưng tài khoản gửi",
}
OPERATIONS = {
    "claim": "Nhận xử lý", "unassign": "Trả lại hàng chờ", "note": "Ghi chú xử lý", "request_info": "Yêu cầu bổ sung thông tin",
    "request_verification": "Yêu cầu xác minh thêm một lớp", "allow": "Cho tiếp tục giao dịch",
    "freeze": "Đóng băng giao dịch này", "suspend": "Tạm ngưng tài khoản gửi",
    "unfreeze": "Gỡ đóng băng giao dịch", "restore_account": "Khôi phục tài khoản gửi",
    "info_received": "Khách hàng bổ sung thông tin", "verify_ok": "Xác minh thành công",
    "verify_failed": "Xác minh thất bại", "route": "Phân luồng tự động",
    "resume": "Hệ thống cho tiếp tục", "escalate": "Thông báo chuyên viên",
    "archive_response": "Cập nhật hồ sơ theo quy trình ghi nhận phản hồi",
}
RESTRICTIONS = {"freeze", "suspend"}
RESPONSE_REQUEST = {
    "info_received": "awaiting_info",
    "verify_ok": "awaiting_verification",
    "verify_failed": "awaiting_verification",
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _log(db, event, actor, action, previous, state, reason, at):
    item = db.execute("INSERT INTO case_history(event,actor,action,previous,state,reason,at) VALUES (?,?,?,?,?,?,?)",
                      (event, actor, action, previous, state, reason, at))
    if state == "pending_review" and (previous != state or action == "unassign"):
        db.execute("UPDATE casework SET notice=? WHERE event=?", (item.lastrowid, event))


def migrate_response_receipts(db):
    """Retire only automatic dispositions paired with a legacy customer receipt."""
    rows = db.execute("""
        SELECT c.event, c.state FROM casework c JOIN case_history a ON a.id=(
            SELECT id FROM case_history WHERE event=c.event AND action!='note' ORDER BY id DESC LIMIT 1)
        WHERE c.owner IS NULL AND c.state=a.state AND a.actor='Hệ thống'
          AND ((a.action='resume' AND c.state='released') OR
               (a.action='escalate' AND c.state='pending_review'))
          AND NOT EXISTS (SELECT 1 FROM decisions d WHERE d.event=c.event)
          AND EXISTS (SELECT 1 FROM case_history r WHERE r.event=c.event
              AND r.at=a.at AND r.id<a.id AND r.state=a.previous
              AND ((r.action='info_received' AND r.previous='awaiting_info') OR
                   (r.action IN ('verify_ok','verify_failed') AND r.previous='awaiting_verification')))
    """).fetchall()
    at = _now()
    for row in rows:
        _log(db, row["event"], "Hệ thống", "archive_response", row["state"], "response_recorded",
             "Đã có phản hồi trong nhật ký. Việc duyệt giao dịch thuộc ngân hàng.", at)
        db.execute("""UPDATE casework SET state='response_recorded', notice=NULL, updated=?, version=version+1
                      WHERE event=?""", (at, row["event"]))


def initial_state(result, review, held=False):
    if held:
        return "pending_review", "Tài khoản gửi đang tạm ngưng. Cần chuyên viên xem xét."
    if result.get("recommended_action") == "hold_and_escalate":
        return "pending_review", "Tín hiệu rủi ro cao. Cần quyết định của chuyên viên."
    if review:
        return "awaiting_verification", "Giao dịch qua ngưỡng kiểm tra tăng cường."
    return "awaiting_info", "Ứng viên qua sàng lọc. Cần bổ sung thông tin giao dịch."


def open_case(db, event, stream, account, state, reason):
    at = _now()
    db.execute("INSERT INTO casework(event,stream,account,state,updated) VALUES (?,?,?,?,?)",
               (event, stream, account, state, at))
    action = "escalate" if state == "pending_review" else "route"
    _log(db, event, "Hệ thống", action, None, state, reason, at)


def operator_choices(state, held=False):
    if state == "frozen":
        return ["unfreeze"]
    if state == "suspended":
        return ["restore_account"]
    if state == "pending_review":
        return ["request_info", "request_verification", "freeze"] + ([] if held else ["allow", "suspend"])
    return []


def _current(db, stream, txid):
    return db.execute("""
        SELECT c.*,e.txid, h.origin AS hold_origin FROM casework c JOIN events e ON e.seq=c.event
        LEFT JOIN account_holds h ON h.stream=c.stream AND h.account=c.account
        WHERE e.stream=? AND e.txid=?
    """, (stream, txid)).fetchone()


def transition(store, txid, action, actor, reason, version, *, customer_account=None, receipt_id=None):
    if not actor.strip() or not reason.strip():
        raise ValueError("Cần danh tính và lý do cho thao tác.")
    if len(reason) > 2000:
        raise ValueError("Nội dung tối đa 2.000 ký tự.")
    fingerprint = None
    if receipt_id is not None:
        if customer_account is None:
            raise ValueError("a receipt ID requires an authenticated customer response")
        payload = json.dumps([txid, action, reason.strip(), version, customer_account], ensure_ascii=False)
        fingerprint = sha256(payload.encode("utf-8")).hexdigest()
    with store.connection(write=True) as db:
        if fingerprint is not None:
            saved = db.execute("SELECT fingerprint FROM customer_receipts WHERE stream=? AND receipt_id=?",
                               (store.stream, receipt_id)).fetchone()
            if saved is not None:
                if saved["fingerprint"] != fingerprint:
                    raise ValueError("receipt ID reused with a different payload")
                return True
        row = _current(db, store.stream, txid)
        if row is None or row["version"] != version or row["state"] == "response_recorded":
            return False
        event, previous, owner, origin = row["event"], row["state"], row["owner"], row["hold_origin"]
        actor, reason, at = actor.strip(), reason.strip(), _now()
        if customer_account is not None:
            if row["account"] != customer_account or RESPONSE_REQUEST.get(action) != previous:
                return False
            new, owner = "response_recorded", None
        else:
            if owner is not None and owner != actor and previous == "pending_review":
                return False
            if action == "claim":
                if previous != "pending_review" or owner is not None:
                    return False
                owner, new = actor, previous
            elif action == "unassign":
                if previous != "pending_review" or owner != actor:
                    return False
                owner, new = None, previous
            elif action == "note":
                new = previous
            elif action in operator_choices(previous, held=origin is not None):
                new = {"allow":"released", "freeze":"frozen", "suspend":"suspended", "request_info":"awaiting_info",
                       "request_verification":"awaiting_verification", "unfreeze":"released",
                       "restore_account":"released"}[action]
                if action == "suspend":
                    db.execute("INSERT INTO account_holds VALUES (?,?,?)", (store.stream, row["account"], event))
                if action == "restore_account":
                    if origin != event:
                        return False
                    db.execute("DELETE FROM account_holds WHERE stream=? AND account=? AND origin=?",
                               (store.stream, row["account"], event))
                if action == "unfreeze" and origin is not None:
                    new, owner = "pending_review", None
                if new.startswith("awaiting"):
                    owner = None
            else:
                return False
        _log(db, event, actor, action, previous, new, reason, at)
        db.execute("""UPDATE casework SET state=?, owner=?, version=version+1, updated=?,
                      notice=CASE WHEN ?='pending_review' THEN notice END WHERE event=?""",
                   (new, owner, at, new, event))
        if fingerprint is not None:
            db.execute("INSERT INTO customer_receipts VALUES (?,?,?)",
                       (store.stream, receipt_id, fingerprint))
        return True

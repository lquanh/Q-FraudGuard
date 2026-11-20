"""Durable event processing, manual review queue and decision log for a local pilot."""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import file_digest
import json
from pathlib import Path
import sqlite3
from threading import Lock
from time import time
from uuid import uuid4
import warnings

import numpy as np
import pandas as pd

from qf_guard.data import CANONICAL_COLUMNS, CANONICAL_DEFAULTS, PAYSIM_ALIASES, normalize_transactions
from qf_guard import casework
from qf_guard.explainability import recommended_treatment
from qf_guard.runtime import EventScorer
from qf_guard.scoring import RiskScorer


def model_digest(path: Path) -> str:
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def review_policy(model: RiskScorer, model_path: Path, digest: str) -> float:
    profile = model_path.with_suffix(".review.json")
    stats = model.validation_stats_
    threshold = stats.get("review_threshold", stats.get("alert_hold_threshold", model.candidate_threshold_))
    if profile.exists():
        saved = json.loads(profile.read_text(encoding="utf-8"))
        if saved["model_sha256"] == digest:
            threshold = float(saved["review_threshold"])
        else:
            warnings.warn(f"Model changed; using its embedded review threshold instead of {profile}", stacklevel=2)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("invalid manual review threshold")
    return max(model.candidate_threshold_, threshold)


def normalize_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve producer IDs, optional feedback labels and source event time on retries."""
    data = frame.rename(columns=PAYSIM_ALIASES).copy()
    if data.columns.has_duplicates:
        raise ValueError("duplicate transaction columns after resolving aliases")
    for key in ("transaction_id", "timestamp", "sender_id", "receiver_id", "amount"):
        if key not in data or data[key].isna().any() or data[key].astype(str).str.strip().eq("").any():
            raise ValueError(f"each event requires {key}")
    for key, default in CANONICAL_DEFAULTS.items():
        if key in data:
            data[key] = data[key].fillna(default)
    data["_event_id"] = data["transaction_id"].astype(str)
    if "is_fraud" in data:
        labels = pd.to_numeric(data.pop("is_fraud"), errors="raise")
        if not labels.dropna().isin([0, 1]).all():
            raise ValueError("is_fraud must contain binary labels or null")
        data["_event_label"] = labels
    times = pd.to_datetime(data["timestamp"], utc=True, format="mixed", errors="raise")
    if not times.is_monotonic_increasing:
        raise ValueError("events in a batch must be in chronological source order")
    data = normalize_transactions(data)
    data["transaction_id"] = data.pop("_event_id")
    if "_event_label" in data:
        data["is_fraud"] = data.pop("_event_label").astype("Int64")
    if not np.isfinite(data[["amount", "old_balance_sender", "new_balance_sender",
                             "old_balance_receiver", "new_balance_receiver"]]).all().all():
        raise ValueError("amounts and balances must be finite")
    optional = ["payment_currency", "amount_currency", "receiving_currency", "payment_format", "is_fraud",
                "source_transaction_id", "replay_cycle"]
    data = data.loc[:, [*CANONICAL_COLUMNS, *(key for key in optional if key in data)]]
    if data.transaction_id.duplicated().any():
        unique = data.drop_duplicates()
        if unique.transaction_id.duplicated().any():
            raise ValueError("conflicting payloads share a transaction_id in this batch")
        return unique.reset_index(drop=True)
    return data


def _payloads(data):
    payloads = []
    for record in data.to_dict("records"):
        record["timestamp"] = record["timestamp"].isoformat()
        payloads.append(json.dumps({k: v for k, v in record.items() if pd.notna(v)},
                                   sort_keys=True, ensure_ascii=False, allow_nan=False))
    return payloads


class CaseStore:
    def __init__(self, path: Path, stream: str):
        if not stream.strip():
            raise ValueError("a stream name is required")
        self.path = Path(path)
        self.stream = stream
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS streams (
                    name TEXT PRIMARY KEY, model TEXT NOT NULL, review_threshold REAL NOT NULL,
                    processed INTEGER NOT NULL DEFAULT 0, candidates INTEGER NOT NULL DEFAULT 0,
                    reviews INTEGER NOT NULL DEFAULT 0, labelled INTEGER NOT NULL DEFAULT 0,
                    frauds INTEGER NOT NULL DEFAULT 0, detected INTEGER NOT NULL DEFAULT 0,
                    review_hits INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY, stream TEXT NOT NULL REFERENCES streams(name),
                    txid TEXT NOT NULL, payload TEXT NOT NULL, result TEXT NOT NULL,
                    risk REAL NOT NULL, candidate INTEGER NOT NULL, review INTEGER NOT NULL,
                    created REAL NOT NULL, priority TEXT NOT NULL DEFAULT 'pending',
                    lease TEXT, lease_until REAL, UNIQUE(stream, txid));
                CREATE INDEX IF NOT EXISTS stream_sequence ON events(stream, seq);
                CREATE INDEX IF NOT EXISTS sender_history ON events(stream, ifnull(json_extract(payload,'$.replay_cycle'),0),
                    json_extract(payload,'$.sender_id'), julianday(json_extract(payload,'$.timestamp')), seq);
                CREATE INDEX IF NOT EXISTS receiver_history ON events(stream, ifnull(json_extract(payload,'$.replay_cycle'),0),
                    json_extract(payload,'$.receiver_id'), julianday(json_extract(payload,'$.timestamp')), seq);
                CREATE INDEX IF NOT EXISTS review_queue ON events(stream, review, risk DESC, seq) WHERE candidate=1;
                CREATE INDEX IF NOT EXISTS candidate_queue ON events(stream, risk DESC, seq) WHERE candidate=1;
                CREATE INDEX IF NOT EXISTS pending_windows ON events(stream, priority, seq) WHERE candidate=1;
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY, event INTEGER NOT NULL UNIQUE REFERENCES events(seq),
                    action TEXT NOT NULL, actor TEXT NOT NULL, decided_at TEXT NOT NULL, reason TEXT);
                CREATE TABLE IF NOT EXISTS admissions (
                    event INTEGER PRIMARY KEY REFERENCES events(seq), actor TEXT NOT NULL,
                    reason TEXT NOT NULL, admitted_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS priority_windows (
                    id TEXT PRIMARY KEY, stream TEXT NOT NULL REFERENCES streams(name), created REAL NOT NULL,
                    source TEXT NOT NULL, backend TEXT NOT NULL, config TEXT NOT NULL,
                    status TEXT NOT NULL, lease TEXT NOT NULL, deadline REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS stream_windows ON priority_windows(stream,created DESC);
                CREATE INDEX IF NOT EXISTS active_windows ON priority_windows(stream,status,deadline);
                CREATE TABLE IF NOT EXISTS window_members (
                    event INTEGER PRIMARY KEY REFERENCES events(seq), window_id TEXT NOT NULL
                    REFERENCES priority_windows(id), selected INTEGER);
                CREATE INDEX IF NOT EXISTS window_events ON window_members(window_id,event);
                CREATE TABLE IF NOT EXISTS window_runs (
                    lease TEXT PRIMARY KEY, window_id TEXT NOT NULL REFERENCES priority_windows(id),
                    number INTEGER NOT NULL, started REAL NOT NULL, finished REAL,
                    summary TEXT NOT NULL DEFAULT '{}', error TEXT, UNIQUE(window_id,number));
                CREATE TABLE IF NOT EXISTS window_measurements (
                    lease TEXT NOT NULL REFERENCES window_runs(lease), number INTEGER NOT NULL,
                    detail TEXT NOT NULL, PRIMARY KEY(lease,number));
                CREATE TABLE IF NOT EXISTS replay_progress (
                    stream TEXT PRIMARY KEY REFERENCES streams(name), identity TEXT NOT NULL, total INTEGER NOT NULL,
                    cycle INTEGER NOT NULL DEFAULT 1, position INTEGER NOT NULL DEFAULT 0,
                    first_seq INTEGER NOT NULL DEFAULT 1, next_tick REAL NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS backend_changes (
                    id INTEGER PRIMARY KEY, stream TEXT NOT NULL, backend TEXT NOT NULL,
                    actor TEXT NOT NULL, at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS backend_by_stream ON backend_changes(stream,id DESC);
            """ + casework.SCHEMA)
            db.execute("BEGIN IMMEDIATE")
            if "reason" not in {r["name"] for r in db.execute("PRAGMA table_info(decisions)")}:
                db.execute("ALTER TABLE decisions ADD COLUMN reason TEXT")
            if "auto_routing" not in {r["name"] for r in db.execute("PRAGMA table_info(streams)")}:
                db.execute("ALTER TABLE streams ADD COLUMN auto_routing INTEGER NOT NULL DEFAULT 0")
            if "notice" not in {r["name"] for r in db.execute("PRAGMA table_info(casework)")}:
                db.execute("ALTER TABLE casework ADD COLUMN notice INTEGER")
                db.execute("""UPDATE casework SET notice=(SELECT max(id) FROM case_history
                           WHERE event=casework.event AND action='escalate') WHERE state='pending_review'""")
            db.execute("CREATE INDEX IF NOT EXISTS work_notifications ON casework(stream,notice DESC) "
                       "WHERE state='pending_review' AND owner IS NULL AND notice IS NOT NULL")
            if db.execute("PRAGMA user_version").fetchone()[0] == 0:
                casework.migrate_response_receipts(db)
                db.execute("PRAGMA user_version=1")

    @contextmanager
    def connection(self, *, write=False):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _record(row):
        record = {**json.loads(row["payload"]), **json.loads(row["result"]),
                  "manual_review": bool(row["review"]), "priority_status": row["priority"]}
        for key in ("admitted_by", "admission_reason", "action", "decision_reason", "decided_by",
                    "window_id", "backend", "fallback_reason", "source", "work_state", "work_version",
                    "work_owner", "work_event", "hold_origin"):
            if key in row.keys():
                record[key] = row[key]
        return record

    _CASE_FIELDS = """e.*, a.actor AS admitted_by, a.reason AS admission_reason,
        d.action, d.reason AS decision_reason, d.actor AS decided_by, w.id AS window_id, w.source,
        json_extract(r.summary,'$.backend') AS backend, json_extract(r.summary,'$.fallback_reason') AS fallback_reason,
        c.state AS work_state,c.version AS work_version,c.owner AS work_owner,c.event AS work_event,
        h.origin AS hold_origin"""
    _CASE_TABLES = """ FROM events e LEFT JOIN admissions a ON a.event=e.seq
        LEFT JOIN decisions d ON d.event=e.seq LEFT JOIN window_members m ON m.event=e.seq
        LEFT JOIN priority_windows w ON w.id=m.window_id LEFT JOIN window_runs r ON r.lease=w.lease
        LEFT JOIN casework c ON c.event=e.seq
        LEFT JOIN account_holds h ON h.stream=e.stream AND h.account=json_extract(e.payload,'$.sender_id') """

    def case(self, txid):
        with self.connection() as db:
            row = db.execute("SELECT " + self._CASE_FIELDS + self._CASE_TABLES +
                             "WHERE e.stream=? AND e.txid=?", (self.stream, txid)).fetchone()
            return self._record(row) if row else None

    def account_activity(self, txid, *, hours=24, limit=30, relation="all"):
        """Summarize the ledger as of this transfer and return its most recent related transactions."""
        if hours <= 0 or limit < 1:
            raise ValueError("hours and limit must be positive")
        with self.connection() as db:
            db.execute("BEGIN")
            anchor = db.execute("""SELECT seq,payload,
                (SELECT json_extract(first.payload,'$.timestamp') FROM events first
                 WHERE first.stream=e.stream ORDER BY first.seq LIMIT 1) AS recorded_since
                FROM events e WHERE stream=? AND txid=?""", (self.stream, txid)).fetchone()
            if anchor is None:
                raise KeyError(txid)
            event = json.loads(anchor["payload"])
            end = pd.Timestamp(event["timestamp"])
            unit = (event.get("amount_currency") or event.get("payment_currency") or "").upper()
            equivalent = {"US DOLLAR": "USD", "USD": "US DOLLAR", "EURO": "EUR", "EUR": "EURO"}
            args = dict(stream=self.stream, cycle=event.get("replay_cycle") or 0, seq=anchor["seq"],
                        sender=event["sender_id"] if event["sender_id"] != "UNKNOWN" else None,
                        receiver=event["receiver_id"] if event["receiver_id"] != "UNKNOWN" else None,
                        start=(end - pd.Timedelta(hours=hours)).isoformat(), end=end.isoformat(),
                        unit=unit or None, alias=equivalent.get(unit), limit=limit)
            # Each branch uses its account/time index. UNION counts a transfer
            # just once even when both endpoints or a self-transfer match.
            branches = [f"""SELECT seq FROM events INDEXED BY {index_name} WHERE stream=:stream
                        AND ifnull(json_extract(payload,'$.replay_cycle'),0)=:cycle
                        AND json_extract(payload,'$.{field}') IN (:sender,:receiver)
                        AND julianday(json_extract(payload,'$.timestamp')) BETWEEN julianday(:start) AND julianday(:end)
                        AND seq<:seq""" for field, index_name in (("sender_id", "sender_history"),
                                                                  ("receiver_id", "receiver_history"))]
            linked = "WITH linked AS (" + " UNION ".join(branches) + ") "
            totals = db.execute(linked + """
                SELECT count(*) AS total, count(CASE WHEN candidate=1 THEN 1 END) AS alerts,
                       count(CASE WHEN s=:sender THEN 1 END) AS sent,
                       count(DISTINCT CASE WHEN s=:sender AND r!='UNKNOWN' THEN r END) AS payees,
                       count(CASE WHEN r=:receiver THEN 1 END) AS received,
                       count(DISTINCT CASE WHEN r=:receiver AND s!='UNKNOWN' THEN s END) AS payers,
                       count(CASE WHEN s=:sender AND r=:receiver THEN 1 END) AS same_pair,
                       count(CASE WHEN s=:receiver AND r=:sender THEN 1 END) AS reverse_pair,
                       count(CASE WHEN s=:sender AND unit IN (:unit,:alias) THEN amount END) AS comparable,
                       avg(CASE WHEN s=:sender AND unit IN (:unit,:alias) THEN amount END) AS mean_amount
                FROM (SELECT e.candidate, json_extract(payload,'$.sender_id') AS s,
                      json_extract(payload,'$.receiver_id') AS r, json_extract(payload,'$.amount') AS amount,
                      upper(coalesce(nullif(json_extract(payload,'$.amount_currency'),''),
                                     json_extract(payload,'$.payment_currency'))) AS unit
                      FROM linked JOIN events e ON e.seq=linked.seq)
            """, args).fetchone()
            sender = "json_extract(e.payload,'$.sender_id')"
            receiver = "json_extract(e.payload,'$.receiver_id')"
            filters = {"all": "", "same_pair": f"WHERE {sender}=:sender AND {receiver}=:receiver",
                       "reverse_pair": f"WHERE {sender}=:receiver AND {receiver}=:sender",
                       "sent": f"WHERE {sender}=:sender", "received": f"WHERE {receiver}=:receiver"}
            rows = db.execute(linked + "SELECT e.* FROM linked JOIN events e ON e.seq=linked.seq "
                              + filters[relation] + " ORDER BY e.seq DESC LIMIT :limit", args)
            return {**dict(totals), "recorded_since": anchor["recorded_since"]}, [self._record(r) for r in rows]

    def enable_auto_routing(self, model=None):
        """Migrate open cases once. Historical operator decisions stay terminal."""
        with self.connection(write=True) as db:
            stream = db.execute("SELECT auto_routing FROM streams WHERE name=?", (self.stream,)).fetchone()
            if stream is None or stream[0]:
                return
            for row in db.execute("""
                SELECT e.* FROM events e WHERE stream=? AND candidate=1
                AND NOT EXISTS (SELECT 1 FROM decisions d WHERE d.event=e.seq)
                AND NOT EXISTS (SELECT 1 FROM casework c WHERE c.event=e.seq)
            """, (self.stream,)):
                record = self._record(row)
                if model and "recommended_action" not in record:
                    record.update(recommended_treatment(record, hold_threshold=model.validation_stats_.get("alert_hold_threshold", .6),
                                                        verify_threshold=model.candidate_threshold_))
                state, reason = casework.initial_state(record, row["review"])
                casework.open_case(db, row["seq"], self.stream, record["sender_id"], state, reason)
            db.execute("UPDATE streams SET auto_routing=1 WHERE name=?", (self.stream,))

    def work_queue(self, states, *, limit=10, offset=0):
        marks = ",".join("?" for _ in states)
        args = (self.stream, *states)
        with self.connection() as db:
            db.execute("BEGIN")
            where = f" WHERE c.stream=? AND c.state IN ({marks})"
            count = db.execute("SELECT count(*) FROM casework c" + where, args).fetchone()[0]
            rows = db.execute("SELECT " + self._CASE_FIELDS + self._CASE_TABLES + where +
                              " ORDER BY e.risk DESC,c.event LIMIT ? OFFSET ?", (*args,limit,offset)).fetchall()
            return count, [self._record(row) for row in rows]

    def work_history(self, txid):
        with self.connection() as db:
            return [dict(r) for r in db.execute("""
                SELECT h.* FROM events e JOIN case_history h ON h.event=e.seq
                WHERE e.stream=? AND e.txid=? ORDER BY h.id
            """, (self.stream, txid))]

    def case_changes(self, after=0, limit=1000, *, txid=None):
        with self.connection() as db:
            return [dict(r) for r in db.execute("""
                SELECT h.id,e.stream,e.txid AS transaction_id,c.account,h.action,h.actor,h.at,
                       h.previous,h.state,h.reason,c.version AS current_version,c.state AS current_state
                FROM case_history h JOIN casework c ON c.event=h.event JOIN events e ON e.seq=h.event
                WHERE c.stream=? AND h.id>? AND (? IS NULL OR e.txid=?)
                ORDER BY h.id LIMIT ?
            """, (self.stream, after, txid, txid, limit))]

    def oldest_pending(self):
        with self.connection() as db:
            oldest = db.execute("SELECT min(updated) FROM casework WHERE stream=? AND state='pending_review'",
                                (self.stream,)).fetchone()[0]
            return None if oldest is None else max(0, int(time() - pd.Timestamp(oldest).timestamp()))

    def work_counts(self):
        with self.connection() as db:
            return dict(db.execute("SELECT state,count(*) FROM casework WHERE stream=? GROUP BY state", (self.stream,)))

    def latest_alert(self):
        with self.connection() as db:
            row = db.execute("""
                SELECT c.notice AS id,e.txid FROM casework c JOIN events e ON e.seq=c.event
                WHERE c.stream=? AND c.state='pending_review' AND c.owner IS NULL
                AND c.notice IS NOT NULL ORDER BY c.notice DESC LIMIT 1
            """, (self.stream,)).fetchone()
            return dict(row) if row else None

    def customer_response(self, txid, action, reason, version, *, account, receipt_id=None):
        if not isinstance(account, str) or not account.strip():
            return False
        source = "Khách hàng" if action == "info_received" else "Kênh xác minh"
        return casework.transition(self, txid, action, f"{source} · {account}", reason, version,
                                   customer_account=account, receipt_id=receipt_id)

    def operator_action(self, txid, action, actor, reason, version):
        return casework.transition(self, txid, action, actor, reason, version)

    def stats(self):
        with self.connection() as db:
            row = db.execute("SELECT * FROM streams WHERE name=?", (self.stream,)).fetchone()
            return dict(row) if row else dict(processed=0, candidates=0, reviews=0, labelled=0,
                                               frauds=0, detected=0, review_hits=0)

    def _backend_choice(self, db, default):
        row = db.execute("SELECT id,backend,actor,at FROM backend_changes WHERE stream=? ORDER BY id DESC LIMIT 1",
                         (self.stream,)).fetchone()
        return dict(row) if row else {"id": 0, "backend": default, "actor": None, "at": None}

    def backend_choice(self, default):
        with self.connection() as db:
            return self._backend_choice(db, default)

    def change_backend(self, backend, actor, *, default, expected_id):
        from qf_guard.review_runs import SWITCHABLE_BACKENDS

        if backend not in SWITCHABLE_BACKENDS or not actor.strip():
            raise ValueError("Cần chọn bộ giải hợp lệ và nhập tên người vận hành.")
        with self.connection(write=True) as db:
            current = self._backend_choice(db, default)
            if current["id"] != expected_id or current["backend"] == backend:
                return False
            db.execute("INSERT INTO backend_changes(stream,backend,actor,at) VALUES(?,?,?,?)",
                       (self.stream, backend, actor.strip(), pd.Timestamp.now(tz="UTC").isoformat()))
            return True

    def review_workload(self):
        with self.connection() as db:
            row = db.execute("""
                SELECT count(a.event) AS admissions, min(CASE WHEN d.event IS NULL AND c.state IS NOT 'response_recorded'
                    AND (e.review=1 OR a.event IS NOT NULL) THEN
                    coalesce(cast(strftime('%s',a.admitted_at) AS REAL),e.created) END) AS oldest
                FROM events e LEFT JOIN admissions a ON a.event=e.seq LEFT JOIN decisions d ON d.event=e.seq
                LEFT JOIN casework c ON c.event=e.seq WHERE e.stream=? AND e.candidate=1
            """, (self.stream,)).fetchone()
            return row["admissions"], None if row["oldest"] is None else max(0, int(time() - row["oldest"]))

    def recent(self, limit=200, *, through=None):
        with self.connection() as db:
            query = "SELECT * FROM events WHERE stream=?"
            args = [self.stream]
            if through is not None:
                query += " AND seq <= (SELECT seq FROM events WHERE stream=? AND txid=?)"
                args.extend((self.stream, through))
            rows = db.execute(query + " ORDER BY seq DESC LIMIT ?", (*args, limit)).fetchall()
        return [self._record(r) for r in reversed(rows)]

    def queue(self, *, all_candidates=False, proposals=False, use_qubo=False, limit=30, offset=0):
        scope = " AND (e.review=1 OR a.event IS NOT NULL)"
        if proposals:
            scope = " AND e.review=0 AND e.priority='selected' AND a.event IS NULL"
        elif all_candidates:
            scope = ""
        where = self._CASE_TABLES + ("WHERE e.stream=? AND e.candidate=1 AND d.event IS NULL "
                                     "AND c.state IS NOT 'response_recorded'") + scope
        order = "(e.priority='selected') DESC," if use_qubo else ""
        with self.connection() as db:
            db.execute("BEGIN")
            count = db.execute("SELECT count(*)" + where, (self.stream,)).fetchone()[0]
            rows = db.execute("SELECT " + self._CASE_FIELDS + where + " ORDER BY " + order + "e.risk DESC, e.seq LIMIT ? OFFSET ?",
                              (self.stream, limit, offset)).fetchall()
        return count, [self._record(row) for row in rows]

    def admit(self, txid, actor, reason):
        if not actor.strip() or not reason.strip():
            raise ValueError("actor and reason are required")
        with self.connection(write=True) as db:
            result = db.execute("""
                INSERT INTO admissions(event,actor,reason,admitted_at)
                SELECT e.seq,?,?,? FROM events e WHERE e.stream=? AND e.txid=? AND e.candidate=1
                AND e.review=0 AND e.priority='selected' AND NOT EXISTS (SELECT 1 FROM decisions d WHERE d.event=e.seq)
                AND NOT EXISTS (SELECT 1 FROM casework c WHERE c.event=e.seq)
                ON CONFLICT(event) DO NOTHING
            """, (actor.strip(), reason.strip(), pd.Timestamp.now(tz="UTC").isoformat(), self.stream, txid))
            return result.rowcount == 1

    def decide(self, txid, action, actor, *, reason=None, require_review=False):
        if not str(actor).strip() or not str(action).strip():
            raise ValueError("actor and action are required")
        with self.connection(write=True) as db:
            result = db.execute("""
                INSERT INTO decisions(event,action,actor,decided_at,reason)
                SELECT e.seq,?,?,?,? FROM events e WHERE stream=? AND txid=? AND candidate=1
                AND NOT EXISTS (SELECT 1 FROM casework c WHERE c.event=e.seq AND
                    (c.state='response_recorded' OR ?=1 OR
                     ? NOT IN ('verify','request_verification','step_up_verification','monitor_and_review','continue_monitoring')))
                AND (?=0 OR review=1 OR EXISTS (SELECT 1 FROM admissions a WHERE a.event=e.seq))
                ON CONFLICT(event) DO NOTHING
            """, (action, actor, pd.Timestamp.now(tz="UTC").isoformat(), reason, self.stream, txid,
                  int(require_review), action, int(require_review)))
            return result.rowcount == 1

    def decisions(self, after=0, limit=1000):
        with self.connection() as db:
            return [dict(r) for r in db.execute("""
                SELECT d.id, e.stream, e.txid AS transaction_id, d.action, d.actor, d.decided_at, d.reason
                FROM decisions d JOIN events e ON e.seq=d.event
                WHERE e.stream=? AND d.id>? ORDER BY d.id LIMIT ?
            """, (self.stream, after, limit))]

    def journal(self, query="", limit=30, offset=0):
        """Search actions and solver runs by exact case or window ID within this stream."""
        query = query.strip()
        case_filter = "WHERE e.stream=:stream AND (:q='' OR e.txid=:q OR m.window_id=:q)"
        with self.connection() as db:
            return [dict(row) for row in db.execute("""
                SELECT * FROM (
                  SELECT 'admit' AS kind, e.txid, m.window_id, a.actor, a.reason AS detail,
                         a.admitted_at AS at,a.event AS entry_id FROM admissions a JOIN events e ON e.seq=a.event
                         LEFT JOIN window_members m ON m.event=e.seq """ + case_filter + """
                  UNION ALL
                  SELECT 'decide',e.txid,m.window_id,d.actor,d.action || coalesce(' · ' || nullif(d.reason,''),''),
                         d.decided_at,d.id FROM decisions d JOIN events e ON e.seq=d.event
                         LEFT JOIN window_members m ON m.event=e.seq """ + case_filter + """
                  UNION ALL
                  SELECT 'casework',e.txid,m.window_id,h.actor,h.action || ' · ' || h.reason,h.at,h.id
                         FROM case_history h JOIN events e ON e.seq=h.event
                         LEFT JOIN window_members m ON m.event=e.seq """ + case_filter + """
                  UNION ALL
                  SELECT 'run',NULL,w.id,'Hệ thống',
                         CASE WHEN r.error IS NOT NULL THEN 'error' WHEN r.finished IS NULL THEN 'running' ELSE 'completed' END,
                         strftime('%Y-%m-%dT%H:%M:%fZ',coalesce(r.finished,r.started),'unixepoch'),r.rowid
                         FROM window_runs r JOIN priority_windows w ON w.id=r.window_id
                         WHERE w.stream=:stream AND (:q='' OR w.id=:q OR EXISTS
                           (SELECT 1 FROM window_members m JOIN events e ON e.seq=m.event
                            WHERE m.window_id=w.id AND e.txid=:q))
                ) ORDER BY julianday(at) DESC, at DESC, kind DESC, entry_id DESC
                  LIMIT :lim OFFSET :off
            """, dict(stream=self.stream, q=query, lim=limit, off=offset))]

    def window_list(self, limit=30, offset=0):
        with self.connection() as db:
            return [dict(r) for r in db.execute("""
                SELECT w.*,r.number AS attempt, json_extract(r.summary,'$.k') AS k, json_extract(r.summary,'$.n') AS n,
                       json_extract(r.summary,'$.backend') AS actual_backend,
                       json_extract(r.summary,'$.fallback_reason') AS fallback_reason
                FROM priority_windows w JOIN window_runs r ON r.lease=w.lease
                WHERE w.stream=? ORDER BY w.created DESC,w.id LIMIT ? OFFSET ?
            """, (self.stream, limit, offset))]

    def window(self, window_id):
        with self.connection() as db:
            db.execute("BEGIN")
            row = db.execute("SELECT * FROM priority_windows WHERE stream=? AND id=?",
                             (self.stream, window_id)).fetchone()
            if row is None:
                return None
            members = db.execute("""
                SELECT e.*,m.selected FROM window_members m JOIN events e ON e.seq=m.event
                WHERE m.window_id=? ORDER BY e.seq
            """, (window_id,)).fetchall()
            runs = []
            for run in db.execute("SELECT * FROM window_runs WHERE window_id=? ORDER BY number", (window_id,)):
                measurements = db.execute("SELECT detail FROM window_measurements WHERE lease=? ORDER BY number",
                                          (run["lease"],))
                runs.append({**dict(run), "summary": json.loads(run["summary"]),
                             "measurements": [json.loads(r[0]) for r in measurements]})
            return {**dict(row), "config": json.loads(row["config"]), "runs": runs,
                    "members": [{**self._record(r), "selected": r["selected"]} for r in members]}

    def _expire(self, db, now):
        leases = db.execute("SELECT lease FROM priority_windows WHERE stream=? AND status='running' AND deadline<=?",
                            (self.stream, now)).fetchall()
        for row in leases:
            self._finish(db, row["lease"], None, {}, "Đã hết thời gian chờ kết quả.", now)

    def expire_windows(self):
        with self.connection(write=True) as db:
            self._expire(db, time())

    def active_run(self, lease):
        with self.connection() as db:
            return db.execute("SELECT 1 FROM priority_windows WHERE stream=? AND lease=? "
                              "AND status='running' AND deadline>?", (self.stream, lease, time())).fetchone() is not None

    def _start(self, db, rows, backend, source, config, timeout, now):
        window_id = uuid4().hex
        db.execute("INSERT INTO priority_windows(id,stream,created,source,backend,config,status,lease,deadline) "
                   "VALUES (?,?,?,?,?,?,'running',?,?)",
                   (window_id, self.stream, now, source, backend, json.dumps(config), window_id, now + timeout))
        db.executemany("INSERT INTO window_members(event,window_id) VALUES (?,?)",
                       [(r["seq"], window_id) for r in rows])
        return self._lease(db, rows, window_id, window_id, 1, now, timeout)

    @staticmethod
    def _lease(db, rows, window_id, lease, number, now, timeout):
        db.execute("INSERT INTO window_runs(lease,window_id,number,started) VALUES (?,?,?,?)",
                   (lease, window_id, number, now))
        db.executemany("UPDATE events SET priority='running', lease=?, lease_until=? WHERE seq=?",
                       [(lease, now + timeout, r["seq"]) for r in rows])
        return lease, [CaseStore._record(r) for r in rows]

    def claim_window(self, minimum, maximum, wait_seconds, *, flush=False, backend="",
                     source="live", config=None, timeout=3600, expected_backend_id=None):
        now = time()
        with self.connection(write=True) as db:
            if expected_backend_id is not None and self._backend_choice(db, backend)["id"] != expected_backend_id:
                return None
            self._expire(db, now)
            rows = db.execute("""
                SELECT * FROM events e WHERE stream=? AND candidate=1 AND
                (priority IN ('pending','error') OR (priority='running' AND lease_until<=?))
                AND NOT EXISTS (SELECT 1 FROM window_members m WHERE m.event=e.seq)
                AND NOT EXISTS (SELECT 1 FROM decisions d WHERE d.event=e.seq)
                ORDER BY seq LIMIT ?
            """, (self.stream, now, maximum)).fetchall()
            if not rows or (not flush and len(rows) < minimum and now - rows[0]["created"] < wait_seconds):
                return None
            return self._start(db, rows, backend, source, config or {}, timeout, now)

    def retry_window(self, window_id, *, max_attempts, timeout):
        now = time()
        with self.connection(write=True) as db:
            self._expire(db, now)
            previous = db.execute("""
                SELECT r.number FROM priority_windows w JOIN window_runs r ON r.lease=w.lease
                WHERE w.stream=? AND w.id=? AND w.status='error' AND w.source!='recorded'
            """, (self.stream, window_id)).fetchone()
            if previous is None or previous["number"] >= max_attempts:
                return None
            rows = db.execute("SELECT e.* FROM events e JOIN window_members m ON m.event=e.seq "
                              "WHERE m.window_id=? ORDER BY e.seq", (window_id,)).fetchall()
            lease = uuid4().hex
            db.execute("UPDATE priority_windows SET status='running',lease=?,deadline=? WHERE id=?",
                       (lease, now + timeout, window_id))
            return self._lease(db, rows, window_id, lease, previous["number"] + 1, now, timeout)

    def record_measurement(self, lease, number, detail):
        with self.connection(write=True) as db:
            self._expire(db, time())
            inserted = db.execute("""
                INSERT INTO window_measurements(lease,number,detail)
                SELECT lease,?,? FROM priority_windows WHERE stream=? AND lease=? AND status='running'
                ON CONFLICT(lease,number) DO NOTHING
            """, (number, json.dumps(detail, ensure_ascii=False), self.stream, lease))
            return inserted.rowcount == 1

    def _finish(self, db, lease, selected, summary, error, now):
        window = db.execute("SELECT id FROM priority_windows WHERE stream=? AND lease=? AND status='running'",
                            (self.stream, lease)).fetchone()
        if window is None:
            return False
        rows = db.execute("SELECT seq,txid FROM events WHERE stream=? AND lease=?", (self.stream, lease)).fetchall()
        if selected is not None and not set(selected).issubset(r["txid"] for r in rows):
            raise ValueError("selection contains an event outside the leased window")
        status = "error" if selected is None else "completed"
        if selected is None and not error:
            error = "Bộ giải chưa trả về danh sách hồ sơ."
        db.execute("UPDATE priority_windows SET status=? WHERE id=?", (status, window["id"]))
        db.execute("UPDATE window_runs SET finished=?,summary=?,error=? WHERE lease=?",
                   (now, json.dumps(summary, ensure_ascii=False, allow_nan=False), error, lease))
        for row in rows:
            picked = None if selected is None else row["txid"] in selected
            priority = "error" if picked is None else ("selected" if picked else "not_selected")
            db.execute("UPDATE events SET priority=?,lease=NULL,lease_until=NULL WHERE seq=?", (priority, row["seq"]))
            db.execute("UPDATE window_members SET selected=? WHERE event=?", (picked, row["seq"]))
        return True

    def finish_window(self, lease, selected, *, summary=None, error=None):
        with self.connection(write=True) as db:
            now = time()
            self._expire(db, now)
            return self._finish(db, lease, selected, summary or {}, error, now)

    def load_recording(self, scored, report, *, recording_id, policy):
        """Install the frozen scored transactions and their hardware result in one transaction."""
        audit = report["audit"]
        candidates = scored.loc[scored.is_candidate]
        selected = set(report["selected_ids"])
        summary = dict(report["dashboard"])
        k = summary["k"]
        review_threshold = float(policy["review_threshold"])
        bits = audit["winning_provider_bitstring"]
        if audit["reverse_provider_bits"]:
            bits = bits[::-1]
        order = audit["winning_circuit_candidate_order"]
        ids = audit["candidate_ids"]
        if (len(bits) != len(ids) or set(bits) - {"0", "1"} or sorted(order) != list(range(len(ids)))
                or {ids[i] for bit, i in zip(bits, order, strict=True) if bit == "1"} != selected):
            raise ValueError("recorded selection disagrees with the measured bit string")
        if (len(scored) != report["processed"] or scored.transaction_id.duplicated().any()
                or candidates.transaction_id.tolist() != audit["candidate_ids"]
                or len(candidates) != summary["n"] or len(selected) != k
                or not selected.issubset(audit["candidate_ids"])
                or candidates.nlargest(k, "risk_score").transaction_id.tolist() != audit["risk_only_topk_ids"]):
            raise ValueError("recorded events disagree with the hardware report")
        summary.update(optimum_energy=audit["exact_energy"], selected_energy=audit["selected_energy"],
                       qpu_access_ms=summary.get("qpu_access_ms") or None, task_id=str(summary["task_id"]))
        data = normalize_events(scored)
        with self.connection(write=True) as db:
            stream = db.execute("SELECT model FROM streams WHERE name=?", (self.stream,)).fetchone()
            if stream is not None:
                if stream["model"] != recording_id:
                    raise ValueError("recorded replay requires a separate stream")
                # Refresh the immutable hardware evidence in an already installed replay.
                encoded = json.dumps(summary, ensure_ascii=False, allow_nan=False)
                db.execute("""UPDATE window_runs SET summary=? WHERE summary<>?
                           AND window_id IN (SELECT id FROM priority_windows WHERE stream=? AND source='recorded')""",
                           (encoded, encoded, self.stream))
                return
            db.execute("INSERT INTO streams(name,model,review_threshold) VALUES (?,?,?)",
                       (self.stream, recording_id, review_threshold))
            _write_scores(db, self.stream, scored, _payloads(data), review_threshold)
            rows = db.execute("SELECT * FROM events WHERE stream=? AND candidate=1 ORDER BY seq",
                              (self.stream,)).fetchall()
            now = time()
            lease, _ = self._start(db, rows, summary["backend"], "recorded", policy, 1, now)
            self._finish(db, lease, selected, summary, None, now)


class EventProcessor:
    """One transactional scoring path for CSV replay and producer-supplied events."""

    def __init__(self, store: CaseStore, model: RiskScorer, digest: str, threshold: float, *, from_start=False):
        self.store = store
        self.model = model
        self.threshold = threshold
        self.lock = Lock()
        self.scorer = None
        self.seq = 0
        self.from_start = from_start
        self.cycle = 0
        with store.connection(write=True) as db:
            db.execute("INSERT INTO streams(name,model,review_threshold) VALUES (?,?,?) "
                       "ON CONFLICT(name) DO NOTHING", (store.stream, digest, threshold))
            row = db.execute("SELECT model,review_threshold FROM streams WHERE name=?", (store.stream,)).fetchone()
            if tuple(row) != (digest, threshold):
                raise ValueError("model or review policy changed; use a new QFG_STREAM for this version")
        store.enable_auto_routing(model)

    @classmethod
    def load(cls, store, path, *, from_start=False):
        path = Path(path)
        model = RiskScorer.load(path)
        model._check_fitted()
        model.candidate_threshold_ = model.validation_stats_.get(
            "operational_candidate_threshold", model.candidate_threshold_
        )
        digest = model_digest(path)
        return cls(store, model, digest, review_policy(model, path, digest), from_start=from_start)

    def _restore(self, db):
        cursor = db.execute("SELECT seq,payload FROM events WHERE stream=? AND seq>? ORDER BY seq",
                            (self.store.stream, self.seq))
        while rows := cursor.fetchmany(4096):
            batch = pd.DataFrame([json.loads(r["payload"]) for r in rows], columns=CANONICAL_COLUMNS)
            self.scorer.warm(normalize_transactions(batch))
            self.seq = rows[-1]["seq"]

    def ingest(self, frame: pd.DataFrame, *, expected_offset=None, replay=None) -> int:
        if frame.empty:
            return 0
        data = normalize_events(frame)
        payloads = _payloads(data)
        with self.lock:
            dirty = False
            try:
                if not replay and self.from_start:
                    raise ValueError("Full replay requires a cursor.")
                if not replay and self.scorer is None:
                    dirty = True
                    self.scorer = EventScorer(self.model)
                    # Replay the durable prefix under a read snapshot. New arrivals are
                    # caught up under the writer lock immediately before the prediction.
                    with self.store.connection() as db:
                        self._restore(db)
                    dirty = False
                with self.store.connection(write=True) as db:
                    cursor = db.execute("SELECT * FROM replay_progress WHERE stream=?", (self.store.stream,)).fetchone()
                    if replay:
                        previous, cycle, position, interval = replay
                        if cursor is None or (cursor["cycle"], cursor["position"]) != (previous["cycle"], previous["position"]):
                            return 0
                        if cycle != self.cycle or self.scorer is None:
                            self.scorer = EventScorer(self.model, warm_history=False)
                            self.cycle = cycle
                            self.seq = previous["first_seq"] - 1
                        if position == 0:
                            self.seq = db.execute("SELECT coalesce(max(seq),0) FROM events WHERE stream=?",
                                                  (self.store.stream,)).fetchone()[0]
                    elif cursor is not None:
                        raise ValueError("Luồng này dành riêng cho nguồn phát lại toàn bộ.")
                    if expected_offset is not None:
                        current = db.execute("SELECT processed FROM streams WHERE name=?",
                                             (self.store.stream,)).fetchone()[0]
                        if expected_offset != current:
                            return 0
                    positions = []
                    for pos, txid in enumerate(data.transaction_id):
                        row = db.execute("SELECT payload FROM events WHERE stream=? AND txid=?",
                                         (self.store.stream, txid)).fetchone()
                        if row is None:
                            positions.append(pos)
                        elif row[0] != payloads[pos]:
                            raise ValueError(f"conflicting payload for transaction_id {txid}")
                    if replay and len(positions) != len(data):
                        raise ValueError("Mã giao dịch của vòng phát lại đã tồn tại trước vị trí đang xử lý.")
                    if not positions:
                        return 0
                    dirty = True
                    self._restore(db)
                    scored = self.scorer.score(data.iloc[positions].reset_index(drop=True))
                    last_seq = self._save(db, scored, [payloads[i] for i in positions])
                    if replay:
                        first_seq = last_seq - len(positions) + 1 if position == 0 else previous["first_seq"]
                        db.execute("UPDATE replay_progress SET cycle=?,position=?,first_seq=?,next_tick=? WHERE stream=?",
                                   (cycle, position + len(positions), first_seq, time() + interval, self.store.stream))
                self.seq = last_seq
            except BaseException:
                # Feature updates and DB writes must start from the same committed prefix on retry.
                if dirty:
                    self.scorer = None
                    self.seq = 0
                raise
        return len(positions)

    def _save(self, db, scored, payloads):
        return _write_scores(db, self.store.stream, scored, payloads, self.threshold, self.model)


def _write_scores(db, stream, scored, payloads, threshold, model=None):
    candidates = scored.is_candidate.to_numpy(dtype=bool)
    reviews = candidates & (scored.risk_score.to_numpy() >= threshold)
    labels = scored.get("is_fraud", pd.Series(pd.NA, index=scored.index, dtype="Int64"))
    frauds = labels.fillna(0).to_numpy(dtype=int)
    now = time()
    columns = ["risk_score", "supervised_score", "anomaly_score", "graph_score",
               "rule_score", "is_candidate", "reason_codes"]
    if "graph_model" in scored:
        columns.append("graph_model")
    summaries = scored[columns].to_dict("records")
    automatic = db.execute("SELECT auto_routing FROM streams WHERE name=?", (stream,)).fetchone()[0]
    held = {r[0] for r in db.execute("SELECT account FROM account_holds WHERE stream=?", (stream,))} if automatic else set()
    for txid, payload, result, candidate, review in zip(
        scored.transaction_id, payloads, summaries, candidates, reviews, strict=True
    ):
        if candidate and model is not None:
            result.update(recommended_treatment(result, hold_threshold=model.validation_stats_.get("alert_hold_threshold", .6),
                                                verify_threshold=model.candidate_threshold_))
        cursor = db.execute("""
            INSERT INTO events(stream,txid,payload,result,risk,candidate,review,created)
            VALUES (?,?,?,?,?,?,?,?)
        """, (stream, txid, payload, json.dumps(result, ensure_ascii=False, allow_nan=False),
              result["risk_score"], int(candidate), int(review), now))
        if automatic:
            account = json.loads(payload)["sender_id"]
            if candidate or account in held:
                state, reason = casework.initial_state(result, review, account in held)
                casework.open_case(db, cursor.lastrowid, stream, account, state, reason)
    db.execute("""
        UPDATE streams SET processed=processed+?, candidates=candidates+?, reviews=reviews+?,
            labelled=labelled+?, frauds=frauds+?, detected=detected+?, review_hits=review_hits+?
        WHERE name=?
    """, (len(scored), int(candidates.sum()), int(reviews.sum()), int(labels.notna().sum()),
          int(frauds.sum()), int(frauds[candidates].sum()), int(frauds[reviews].sum()), stream))
    return cursor.lastrowid

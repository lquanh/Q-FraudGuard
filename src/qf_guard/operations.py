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
    optional = ["payment_currency", "receiving_currency", "payment_format", "is_fraud"]
    data = data.loc[:, [*CANONICAL_COLUMNS, *(key for key in optional if key in data)]]
    if data.transaction_id.duplicated().any():
        unique = data.drop_duplicates()
        if unique.transaction_id.duplicated().any():
            raise ValueError("conflicting payloads share a transaction_id in this batch")
        return unique.reset_index(drop=True)
    return data


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
                CREATE INDEX IF NOT EXISTS review_queue ON events(stream, review, risk DESC, seq) WHERE candidate=1;
                CREATE INDEX IF NOT EXISTS candidate_queue ON events(stream, risk DESC, seq) WHERE candidate=1;
                CREATE INDEX IF NOT EXISTS pending_windows ON events(stream, priority, seq) WHERE candidate=1;
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY, event INTEGER NOT NULL UNIQUE REFERENCES events(seq),
                    action TEXT NOT NULL, actor TEXT NOT NULL, decided_at TEXT NOT NULL);
            """)

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
        return {**json.loads(row["payload"]), **json.loads(row["result"]),
                "manual_review": bool(row["review"]), "priority_status": row["priority"]}

    def stats(self):
        with self.connection() as db:
            row = db.execute("SELECT * FROM streams WHERE name=?", (self.stream,)).fetchone()
            return dict(row) if row else dict(processed=0, candidates=0, reviews=0, labelled=0,
                                               frauds=0, detected=0, review_hits=0)

    def recent(self, limit=200, *, through=None):
        with self.connection() as db:
            query = "SELECT * FROM events WHERE stream=?"
            args = [self.stream]
            if through is not None:
                query += " AND seq <= (SELECT seq FROM events WHERE stream=? AND txid=?)"
                args.extend((self.stream, through))
            rows = db.execute(query + " ORDER BY seq DESC LIMIT ?", (*args, limit)).fetchall()
        return [self._record(r) for r in reversed(rows)]

    def queue(self, *, all_candidates=False, use_qubo=False, limit=30, offset=0):
        scope = "" if all_candidates else " AND review=1"
        order = "(priority='selected') DESC," if use_qubo else ""
        where = (" FROM events e WHERE stream=? AND candidate=1" + scope +
                 " AND NOT EXISTS (SELECT 1 FROM decisions d WHERE d.event=e.seq)")
        with self.connection() as db:
            db.execute("BEGIN")
            count = db.execute("SELECT count(*)" + where, (self.stream,)).fetchone()[0]
            rows = db.execute("SELECT *" + where + " ORDER BY " + order + "risk DESC, seq LIMIT ? OFFSET ?",
                              (self.stream, limit, offset)).fetchall()
        return count, [self._record(row) for row in rows]

    def decide(self, txid, action, actor):
        if not str(actor).strip() or not str(action).strip():
            raise ValueError("actor and action are required")
        with self.connection(write=True) as db:
            result = db.execute("""
                INSERT INTO decisions(event,action,actor,decided_at)
                SELECT seq,?,?,? FROM events WHERE stream=? AND txid=? AND candidate=1
                ON CONFLICT(event) DO NOTHING
            """, (action, actor, pd.Timestamp.now(tz="UTC").isoformat(), self.stream, txid))
            return result.rowcount == 1

    def decisions(self, after=0, limit=1000):
        with self.connection() as db:
            return [dict(r) for r in db.execute("""
                SELECT d.id, e.stream, e.txid AS transaction_id, d.action, d.actor, d.decided_at
                FROM decisions d JOIN events e ON e.seq=d.event
                WHERE e.stream=? AND d.id>? ORDER BY d.id LIMIT ?
            """, (self.stream, after, limit))]

    def claim_window(self, minimum, maximum, wait_seconds, *, flush=False):
        now = time()
        with self.connection(write=True) as db:
            rows = db.execute("""
                SELECT * FROM events e WHERE stream=? AND candidate=1 AND
                (priority='pending' OR (priority='running' AND lease_until<=?))
                AND NOT EXISTS (SELECT 1 FROM decisions d WHERE d.event=e.seq)
                ORDER BY seq LIMIT ?
            """, (self.stream, now, maximum)).fetchall()
            if not rows or (not flush and len(rows) < minimum and now - rows[0]["created"] < wait_seconds):
                return None
            lease = uuid4().hex
            db.executemany("UPDATE events SET priority='running', lease=?, lease_until=? WHERE seq=?",
                           [(lease, now + 3600, r["seq"]) for r in rows])
        return lease, [self._record(r) for r in rows]

    def finish_window(self, lease, selected):
        with self.connection(write=True) as db:
            rows = db.execute("SELECT seq,txid FROM events WHERE stream=? AND lease=?",
                              (self.stream, lease)).fetchall()
            for row in rows:
                status = "error" if selected is None else ("selected" if row["txid"] in selected else "not_selected")
                db.execute("UPDATE events SET priority=?, lease=NULL, lease_until=NULL WHERE seq=?",
                           (status, row["seq"]))


class EventProcessor:
    """One transactional scoring path for CSV replay and producer-supplied events."""

    def __init__(self, store: CaseStore, model: RiskScorer, digest: str, threshold: float):
        self.store = store
        self.model = model
        self.threshold = threshold
        self.lock = Lock()
        self.scorer = None
        self.seq = 0
        with store.connection(write=True) as db:
            db.execute("INSERT INTO streams(name,model,review_threshold) VALUES (?,?,?) "
                       "ON CONFLICT(name) DO NOTHING", (store.stream, digest, threshold))
            row = db.execute("SELECT model,review_threshold FROM streams WHERE name=?", (store.stream,)).fetchone()
            if tuple(row) != (digest, threshold):
                raise ValueError("model or review policy changed; use a new QFG_STREAM for this version")

    @classmethod
    def load(cls, store, path):
        path = Path(path)
        model = RiskScorer.load(path)
        model.candidate_threshold_ = model.validation_stats_.get(
            "operational_candidate_threshold", model.candidate_threshold_
        )
        digest = model_digest(path)
        return cls(store, model, digest, review_policy(model, path, digest))

    def _restore(self, db):
        cursor = db.execute("SELECT seq,payload FROM events WHERE stream=? AND seq>? ORDER BY seq",
                            (self.store.stream, self.seq))
        while rows := cursor.fetchmany(4096):
            batch = pd.DataFrame([json.loads(r["payload"]) for r in rows], columns=CANONICAL_COLUMNS)
            self.scorer.features.warm(normalize_transactions(batch))
            self.seq = rows[-1]["seq"]

    def ingest(self, frame: pd.DataFrame, *, expected_offset=None) -> int:
        if frame.empty:
            return 0
        data = normalize_events(frame)
        payloads = []
        for record in data.to_dict("records"):
            record["timestamp"] = record["timestamp"].isoformat()
            payloads.append(json.dumps({k: v for k, v in record.items() if pd.notna(v)},
                                       sort_keys=True, ensure_ascii=False, allow_nan=False))
        with self.lock:
            dirty = False
            try:
                if self.scorer is None:
                    dirty = True
                    self.scorer = EventScorer(self.model)
                    # Replay the durable prefix under a read snapshot. New arrivals are
                    # caught up under the writer lock immediately before the prediction.
                    with self.store.connection() as db:
                        self._restore(db)
                    dirty = False
                with self.store.connection(write=True) as db:
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
                    if not positions:
                        return 0
                    dirty = True
                    self._restore(db)
                    scored = self.scorer.score(data.iloc[positions].reset_index(drop=True))
                    last_seq = self._save(db, scored, [payloads[i] for i in positions])
                self.seq = last_seq
            except BaseException:
                # Feature updates and DB writes must start from the same committed prefix on retry.
                if dirty:
                    self.scorer = None
                    self.seq = 0
                raise
        return len(positions)

    def _save(self, db, scored, payloads):
        candidates = scored.is_candidate.to_numpy(dtype=bool)
        reviews = candidates & (scored.risk_score.to_numpy() >= self.threshold)
        labels = scored.get("is_fraud", pd.Series(pd.NA, index=scored.index, dtype="Int64"))
        frauds = labels.fillna(0).to_numpy(dtype=int)
        now = time()
        summaries = scored[["risk_score", "supervised_score", "anomaly_score", "graph_score",
                            "rule_score", "is_candidate", "reason_codes"]].to_dict("records")
        hold = self.model.validation_stats_.get("alert_hold_threshold", .6)
        for txid, payload, result, candidate, review in zip(
            scored.transaction_id, payloads, summaries, candidates, reviews, strict=True
        ):
            if candidate:
                result.update(recommended_treatment(result, hold_threshold=hold,
                                                    verify_threshold=self.model.candidate_threshold_))
            cursor = db.execute("""
                INSERT INTO events(stream,txid,payload,result,risk,candidate,review,created)
                VALUES (?,?,?,?,?,?,?,?)
            """, (self.store.stream, txid, payload, json.dumps(result, ensure_ascii=False),
                  result["risk_score"], int(candidate), int(review), now))
        db.execute("""
            UPDATE streams SET processed=processed+?, candidates=candidates+?, reviews=reviews+?,
                labelled=labelled+?, frauds=frauds+?, detected=detected+?, review_hits=review_hits+?
            WHERE name=?
        """, (len(scored), int(candidates.sum()), int(reviews.sum()), int(labels.notna().sum()),
              int(frauds.sum()), int(frauds[candidates].sum()), int(frauds[reviews].sum()), self.store.stream))
        return cursor.lastrowid

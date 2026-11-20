"""Disk-backed, chronological replay of the entire configured source."""
from __future__ import annotations

from contextlib import closing, contextmanager
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Lock
from time import time
import zipfile

import pandas as pd

from qf_guard.data import normalize_transactions
from qf_guard.datasets import amlworld_to_canonical


def _identity(source):
    info = {"path": str(source.resolve()), "bytes": source.stat().st_size, "mtime": source.stat().st_mtime_ns}
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            member = archive.getinfo("HI-Small_Trans.csv")
        info.update(member=member.filename, crc=member.CRC)
    return json.dumps(info, sort_keys=True)


@contextmanager
def _open_source(source):
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive, archive.open("HI-Small_Trans.csv") as reader:
            yield reader
    else:
        with source.open("rb") as reader:
            yield reader


def prepare_replay(source: Path, destination: Path, *, chunk_size=100_000):
    """Build once, in bounded memory. Publish a complete immutable source atomically."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    signature = _identity(source)
    if destination.exists():
        with closing(sqlite3.connect(destination.as_uri() + "?mode=ro", uri=True)) as db:
            if db.execute("SELECT identity FROM manifest").fetchone()[0] != signature:
                raise ValueError("Nguồn đã thay đổi. Dùng tệp chỉ mục và tên luồng mới.")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as work:
        staged = Path(work) / "replay.sqlite"
        with closing(sqlite3.connect(staged)) as db, _open_source(source) as reader:
            db.execute("PRAGMA journal_mode=OFF")  # staged file, rebuilt on interruption
            db.execute("PRAGMA synchronous=OFF")
            total = 0
            for chunk in pd.read_csv(reader, chunksize=chunk_size):
                if "Amount Paid" in chunk:
                    frame = amlworld_to_canonical(chunk, id_offset=total)
                    frame["amount_currency"] = "USD"
                else:
                    if "transaction_id" not in chunk:
                        chunk["transaction_id"] = [f"TX-{i:08d}" for i in range(total, total + len(chunk))]
                    if "timestamp" not in chunk and "step" not in chunk:
                        chunk["timestamp"] = pd.Timestamp("2026-01-01", tz="UTC") + pd.to_timedelta(chunk.index, unit="min")
                    frame = normalize_transactions(chunk)
                frame["timestamp"] = frame.timestamp.map(pd.Timestamp.isoformat)
                frame.to_sql("staging", db, if_exists="append", index=False, chunksize=2500)
                total += len(frame)
            if not total:
                raise ValueError("Nguồn phát lại rỗng.")
            db.execute("CREATE TABLE replay_rows AS SELECT * FROM staging ORDER BY timestamp,rowid")
            db.execute("CREATE UNIQUE INDEX replay_id ON replay_rows(transaction_id)")
            db.execute("DROP TABLE staging")
            db.execute("CREATE TABLE manifest(identity TEXT NOT NULL,total INTEGER NOT NULL)")
            db.execute("INSERT INTO manifest VALUES (?,?)", (signature, total))
            db.commit()
            db.execute("VACUUM")
        staged.replace(destination)


class FullReplay:
    """A durable cursor and an independent feature history for each replay pass."""

    def __init__(self, store, model_path, source_index):
        from qf_guard.operations import EventProcessor

        self.store = store
        self.path = Path(source_index).resolve()
        with self.source() as db:
            self.identity, self.total = db.execute("SELECT identity,total FROM manifest").fetchone()
        self.processor = EventProcessor.load(store, model_path, from_start=True)
        self.lock = Lock()
        with store.connection(write=True) as db:
            row = db.execute("SELECT identity,total FROM replay_progress WHERE stream=?", (store.stream,)).fetchone()
            if row is None:
                if db.execute("SELECT processed FROM streams WHERE name=?", (store.stream,)).fetchone()[0]:
                    raise ValueError("Chọn tên luồng riêng để bắt đầu phát lại toàn bộ.")
                db.execute("INSERT INTO replay_progress(stream,identity,total) VALUES(?,?,?)",
                           (store.stream, self.identity, self.total))
            elif tuple(row) != (self.identity, self.total):
                raise ValueError("Nguồn của luồng đã thay đổi. Chọn QFG_STREAM riêng.")

    def source(self):
        return closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True))

    def progress(self):
        with self.store.connection() as db:
            return dict(db.execute("SELECT * FROM replay_progress WHERE stream=?", (self.store.stream,)).fetchone())

    def tick(self, batch_size, *, interval=0):
        if batch_size < 1:
            raise ValueError("Kích thước lô phải dương.")
        with self.lock:
            progress = self.progress()
            if time() < progress["next_tick"]:
                return 0
            position, cycle = progress["position"], progress["cycle"]
            if position == self.total:
                position, cycle = 0, cycle + 1
            with self.source() as db:
                frame = pd.read_sql_query("SELECT * FROM replay_rows WHERE rowid>? ORDER BY rowid LIMIT ?",
                                          db, params=(position, batch_size))
            frame["source_transaction_id"] = frame.transaction_id.astype(str)
            frame["transaction_id"] = "R" + str(cycle) + "/" + frame.source_transaction_id
            frame["replay_cycle"] = cycle
            return self.processor.ingest(frame, replay=(progress, cycle, position, interval))

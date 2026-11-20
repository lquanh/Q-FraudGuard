"""Feed the stream, prioritize cases and manage the operational backend."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from getpass import getuser
import json
from math import isfinite
import os
from pathlib import Path
import sys
from time import sleep

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from qf_guard.casework import RESPONSE_REQUEST  # noqa: E402
from qf_guard.operations import CaseStore, EventProcessor  # noqa: E402
from qf_guard.replay import FullReplay, prepare_replay  # noqa: E402
from qf_guard.review_runs import SWITCHABLE_BACKENDS, backend_requirement, configured_priority, solve_review_window  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=os.environ.get("QFG_CASE_DB", ROOT / "artifacts" / "operations.sqlite3"))
    parser.add_argument("--stream", default=os.environ.get("QFG_STREAM"))
    commands = parser.add_subparsers(dest="command", required=True)
    feed = commands.add_parser("ingest")
    feed.add_argument("--model", type=Path, default=os.environ.get("QFG_MODEL_PATH"),
                      required=not os.environ.get("QFG_MODEL_PATH"))
    feed.add_argument("--input", type=Path, help="JSONL file; reads stdin if omitted")
    feed.add_argument("--batch-size", type=int, help="default 1 for stdin, 256 for a file")
    actions = commands.add_parser("decisions")
    actions.add_argument("--after", type=int, default=0, help="last consumed decision id")
    actions.add_argument("--limit", type=int, default=1000)
    changes = commands.add_parser("case-history", help="read automatic checks, customer replies and analyst actions")
    changes.add_argument("--after", type=int, default=0, help="last consumed history id")
    changes.add_argument("--limit", type=int, default=1000)
    changes.add_argument("--transaction", help="filter to a single transaction ID")
    feed_loop = commands.add_parser("replay-loop", help="replay the whole source continuously, independent of a browser")
    feed_loop.add_argument("--source", type=Path, default=os.environ.get("QFG_REPLAY_SOURCE", ROOT / "data/downloads/ibm_amlworld.zip"))
    feed_loop.add_argument("--index", type=Path, default=os.environ.get("QFG_REPLAY_INDEX", ROOT / "artifacts/amlworld_full-replay.sqlite"))
    feed_loop.add_argument("--model", type=Path, default=os.environ.get("QFG_MODEL_PATH", ROOT / "artifacts/models/amlworld_full_scorer.joblib"))
    feed_loop.add_argument("--batch-size", type=int, default=int(os.environ.get("QFG_REPLAY_BATCH", 64)))
    feed_loop.add_argument("--interval", type=float, default=float(os.environ.get("QFG_REPLAY_INTERVAL", 1)))
    worker = commands.add_parser("prioritize", help="prioritize incoming windows independently of a browser; CPU by default")
    worker.add_argument("--window-size", type=int, default=int(os.environ.get("QFG_QUANTUM_MAX_BATCH", 18)))
    commands.add_parser("backend", help="show the configured backend for future windows")
    switch = commands.add_parser("set-backend", help="apply a backend to future windows in this stream")
    switch.add_argument("name", choices=SWITCHABLE_BACKENDS)
    switch.add_argument("--actor", default=os.environ.get("QFG_OPERATOR") or getuser())
    reply = commands.add_parser("customer-response", help="record a customer or verification receipt in the case log",
                                description="Log a response from a trusted channel. The bank handles transaction approval.")
    reply.add_argument("--transaction", required=True)
    reply.add_argument("--account", required=True)
    reply.add_argument("--version", type=int, required=True)
    reply.add_argument("--action", choices=tuple(RESPONSE_REQUEST), required=True)
    reply.add_argument("--reason", required=True)
    args = parser.parse_args()
    if args.command == "replay-loop":
        prefix = "loop"
    elif args.command in ("prioritize", "backend", "set-backend"):
        prefix = os.environ.get("QFG_SOURCE", "loop")
    elif args.command == "ingest":
        prefix = "live"
    else:
        prefix = os.environ.get("QFG_SOURCE", "live")
    store = CaseStore(args.db, args.stream or prefix + "-" + os.environ.get("QFG_DATASET", "amlworld_full"))
    default_backend = os.environ.get("QFG_SOLVER", "exact").strip().lower() or "exact"
    if args.command in ("backend", "set-backend"):
        choice = store.backend_choice(default_backend)
        if args.command == "set-backend" and choice["backend"] != args.name:
            if problem := backend_requirement(args.name):
                parser.error(problem)
            if not store.change_backend(args.name, args.actor, default=default_backend, expected_id=choice["id"]):
                parser.error("another operator changed the backend; inspect the current choice and retry")
            choice = store.backend_choice(default_backend)
        print(json.dumps({"stream": store.stream, **choice}))
        return
    if args.command == "prioritize":
        seconds = float(os.environ.get("QFG_QUANTUM_WINDOW_SECONDS", 10))
        timeout = float(os.environ.get("QFG_WINDOW_TIMEOUT_SECONDS", 180))
        jobs = int(os.environ.get("QFG_WINDOW_MAX_JOBS", 8))
        if not 1 <= args.window_size <= 18 or min(jobs, timeout) <= 0 or not isfinite(timeout) or not 0 <= seconds < float("inf"):
            parser.error("use 1–18 candidates, positive jobs and timeout, and a nonnegative window duration")
        config = configured_priority()
        try:
            while True:
                choice = store.backend_choice(default_backend)
                backend = choice["backend"]
                job = store.claim_window(args.window_size, args.window_size, seconds, backend=backend,
                                         source=prefix, config=asdict(config), timeout=timeout, expected_backend_id=choice["id"])
                if job is None:
                    sleep(1)
                    continue
                lease, batch = job
                solve_review_window(store, lease, batch, backend=backend, config=config, max_jobs=jobs, timeout=timeout)
        except KeyboardInterrupt:
            return
    if args.command == "customer-response":
        saved = store.customer_response(args.transaction, args.action, args.reason, args.version, account=args.account)
        print(json.dumps({"recorded": saved, "accepted": saved}))
        if not saved:
            raise SystemExit(1)
        return
    if args.command == "replay-loop":
        if args.batch_size < 1 or not 0 <= args.interval < float("inf"):
            parser.error("batch size must be positive and interval finite and nonnegative")
        prepare_replay(args.source, args.index)
        replay = FullReplay(store, args.model, args.index)
        print(json.dumps({"stream": store.stream, "rows_per_pass": replay.total}), flush=True)
        try:
            while True:
                replay.tick(args.batch_size, interval=args.interval)
                sleep(max(.05, args.interval))
        except KeyboardInterrupt:
            return
    if args.command in ("decisions", "case-history"):
        if args.after < 0 or args.limit < 1:
            parser.error("after must be non-negative and limit positive")
        rows = (store.decisions(args.after, args.limit) if args.command == "decisions" else
                store.case_changes(args.after, args.limit, txid=args.transaction))
        for row in rows:
            print(json.dumps(row, ensure_ascii=True))
        return
    size = args.batch_size if args.batch_size is not None else (256 if args.input else 1)
    if size < 1:
        parser.error("batch size must be positive")
    processor = EventProcessor.load(store, args.model)

    def commit(batch):
        inserted = processor.ingest(pd.DataFrame(batch))
        print(json.dumps({"received": len(batch), "inserted": inserted}), flush=True)

    batch = []
    with (args.input.open(encoding="utf-8-sig") if args.input else nullcontext(sys.stdin)) as source:
        for number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError as exc:
                parser.error(f"invalid JSON on line {number}: {exc}")
            if not isinstance(event, dict):
                parser.error(f"line {number} must contain a transaction object")
            batch.append(event)
            if len(batch) == size:
                commit(batch)
                batch = []
        if batch:
            commit(batch)


if __name__ == "__main__":
    main()

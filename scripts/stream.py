"""Ingest chronological JSONL transactions or read recorded operator decisions."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from qf_guard.operations import CaseStore, EventProcessor  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=os.environ.get("QFG_CASE_DB", ROOT / "artifacts" / "operations.sqlite3"))
    parser.add_argument("--stream", default=os.environ.get("QFG_STREAM") or "live-" + os.environ.get("QFG_DATASET", "amlworld_full"))
    commands = parser.add_subparsers(dest="command", required=True)
    feed = commands.add_parser("ingest")
    feed.add_argument("--model", type=Path, default=os.environ.get("QFG_MODEL_PATH"),
                      required=not os.environ.get("QFG_MODEL_PATH"))
    feed.add_argument("--input", type=Path, help="JSONL file; reads stdin if omitted")
    feed.add_argument("--batch-size", type=int, help="default 1 for stdin, 256 for a file")
    actions = commands.add_parser("decisions")
    actions.add_argument("--after", type=int, default=0, help="last consumed decision id")
    actions.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    store = CaseStore(args.db, args.stream)
    if args.command == "decisions":
        if args.after < 0 or args.limit < 1:
            parser.error("after must be non-negative and limit positive")
        for row in store.decisions(args.after, args.limit):
            print(json.dumps(row, ensure_ascii=False))
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

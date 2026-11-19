"""Write a canonical-schema AMLworld sample for training and the dashboard stream.

The CSV is streamed out of ``data/downloads/ibm_amlworld.zip``; the archive is
never extracted.  AMLworld is synthetic (IBM, CDLA-Sharing-1.0).

Usage:
    python scripts/prepare_amlworld.py                       # 2022/09/03-04 of HI-Small
    python scripts/prepare_amlworld.py --start 2022/09/05 --days 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qf_guard.datasets import read_amlworld_days  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, default=ROOT / "data" / "downloads" / "ibm_amlworld.zip")
    parser.add_argument("--member", default="HI-Small_Trans.csv")
    parser.add_argument("--start", default="2022/09/03")
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "amlworld_hi_small.csv")
    args = parser.parse_args()

    frame = read_amlworld_days(args.archive, args.member, start_day=args.start, days=args.days)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(
        f"{len(frame):,} rows, {int(frame['is_fraud'].sum()):,} laundering "
        f"({frame['is_fraud'].mean():.3%}), {frame['sender_id'].nunique():,} senders -> {args.output}"
    )


if __name__ == "__main__":
    main()

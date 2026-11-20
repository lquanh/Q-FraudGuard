"""Train the realtime risk scorer with a chronological 60/20/20 split.

Usage:
    python scripts/train.py --output artifacts/models/paysim_lightgbm_scorer.joblib
    python scripts/prepare_amlworld.py
    python scripts/train.py --data data/amlworld_hi_small.csv --dataset-name amlworld \\
        --output artifacts/models/amlworld_lightgbm_scorer.joblib
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from qf_guard.training import train_and_evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "paysim.csv")
    parser.add_argument("--output", type=Path, required=True, help="joblib model path")
    parser.add_argument("--metrics", type=Path, help="metrics JSON (default: next to model)")
    parser.add_argument("--target-candidate-rate", type=float, default=.05)
    parser.add_argument("--review-rate", type=float, default=.02, help="manual review fraction on validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-name", default="paysim", help="label stored in the metrics")
    parser.add_argument("--tgnn-epochs", type=int, default=0, help="positive to train the TGNN graph signal")
    parser.add_argument("--tgnn-min-weight", type=float, default=0.0, help="minimum TGNN fusion weight for a controlled pilot")
    args = parser.parse_args()
    if args.tgnn_epochs < 0:
        parser.error("--tgnn-epochs must be nonnegative")
    if not 0 <= args.tgnn_min_weight <= .95 or (args.tgnn_min_weight and not args.tgnn_epochs):
        parser.error("--tgnn-min-weight must be 0–0.95 and requires TGNN training when positive")
    if args.tgnn_epochs:
        import torch

        torch.set_num_threads(1)

    scorer, metrics = train_and_evaluate(
        pd.read_csv(args.data),
        target_candidate_rate=args.target_candidate_rate,
        review_rate=args.review_rate,
        random_state=args.seed,
        dataset_name=args.dataset_name,
        temporal_graph_epochs=args.tgnn_epochs,
        min_tgnn_weight=args.tgnn_min_weight,
    )
    scorer.save(args.output)
    metrics_path = args.metrics or args.output.with_name(
        args.output.stem.replace("_scorer", "") + "_metrics.json"
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({key: metrics[key] for key in metrics if key.startswith("test_")}, indent=2))
    print(f"model  -> {args.output}\nmetrics -> {metrics_path}")


if __name__ == "__main__":
    main()

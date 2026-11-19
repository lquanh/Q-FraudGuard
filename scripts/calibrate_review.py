"""Set a model-bound manual review threshold from its chronological validation split."""

import argparse
import json
from math import isclose
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from qf_guard.operations import model_digest  # noqa: E402
from qf_guard.scoring import RiskScorer, _confirmed_minutes, review_threshold  # noqa: E402
from qf_guard.training import temporal_split  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--data", type=Path, help="original CSV used by the 60/20/20 training split")
    sources.add_argument("--validation-scores", type=Path, help="trusted local validation-score pickle")
    parser.add_argument("--review-rate", type=float, default=.02)
    args = parser.parse_args()
    digest = model_digest(args.model)
    scorer = RiskScorer.load(args.model)
    history = scorer.history_transactions_
    if args.data:
        train, validation, _ = temporal_split(pd.read_csv(args.data))
    else:
        validation = pd.read_pickle(args.validation_scores)
    if history is None:
        parser.error("the model has no saved training history to identify its validation period")
    # Only the validation partition of this training run may set its operating point.
    val = validation.reset_index(drop=True)
    expected = scorer.validation_stats_.get("validation_rows", round(len(history) * .25))
    # Training splits round down independently, so allow the one-row boundary difference.
    tail = history.tail(len(val)).reset_index(drop=True)
    if (abs(len(val) - expected) > 1 or not val.transaction_id.equals(tail.transaction_id)
            or not pd.to_datetime(val.timestamp, utc=True).equals(tail.timestamp)):
        parser.error("rows do not match the model's saved chronological validation partition")
    if args.data:
        scorer.history_transactions_ = scorer._feature_history(train)
        scorer.confirmed_minute_ = _confirmed_minutes(train)
        scores = scorer.score(val).risk_score
    else:
        scores = val.risk_score
    candidate = scorer.validation_stats_.get("operational_candidate_threshold", scorer.candidate_threshold_)
    if args.validation_scores:
        actual = {"candidate_rate": float((scores >= candidate).mean()),
                  "pr_auc": float(average_precision_score(val.is_fraud, scores))}
        for key, value in actual.items():
            saved = scorer.validation_stats_.get(key)
            if saved is not None and not isclose(value, saved, rel_tol=1e-8, abs_tol=1e-10):
                parser.error(f"validation scores disagree with the model's saved {key}")
    threshold = review_threshold(scores, candidate, args.review_rate)
    flags = scores >= threshold
    profile = dict(model_sha256=digest, review_rate_target=args.review_rate, review_threshold=threshold,
                   validation_rows=len(val), validation_reviews=int(flags.sum()),
                   validation_frauds=int(val.is_fraud.sum()), validation_review_hits=int(val.is_fraud[flags].sum()))
    target = args.model.with_suffix(".review.json")
    with NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as f:
        json.dump(profile, f, indent=2)
        f.write("\n")
    os.replace(f.name, target)
    print(f"{target}: {profile}")


if __name__ == "__main__":
    main()

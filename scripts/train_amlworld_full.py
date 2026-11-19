"""Train on all of AMLworld HI-Small and set the alert threshold for a target recall.

* Rows are streamed from ``data/downloads/ibm_amlworld.zip`` (never extracted).
* Days after 2022/09/10 are excluded: in HI-Small they contain ~1,100
  transactions of which ~60% are laundering (patterns finishing after normal
  traffic stops), a generator artefact that would distort any metric.
* Chronological 60/20/20 split.  The alert threshold is chosen on validation so
  that ``--target-recall`` of validation laundering is flagged; the test period
  is then scored once.  The confirmed-account list (SIMO-like) uses labels only
  after ``--label-delay-hours``.

AMLworld is synthetic (IBM, CDLA-Sharing-1.0): results are not bank results.

Usage:
    python scripts/train_amlworld_full.py --target-recall 0.95
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from qf_guard.config import ScoringConfig  # noqa: E402
from qf_guard.datasets import read_amlworld_days  # noqa: E402
from qf_guard.scoring import RiskScorer, _confirmed_minutes  # noqa: E402
from qf_guard.training import temporal_split  # noqa: E402

TUNED_NOFORMAT = {"num_leaves": 63, "min_child_samples": 100, "n_estimators": 400,
                  "learning_rate": 0.05, "colsample_bytree": 0.8, "reg_lambda": 1.0}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def flag_rate_for_recall(labels: np.ndarray, score: np.ndarray, recall: float) -> float:
    positives = np.sort(score[labels == 1])
    if len(positives) == 0:
        return float("nan")
    threshold = positives[int(np.floor((1 - recall) * len(positives)))]
    return float((score >= threshold).mean())


def attempt_keys(patterns: Path) -> dict[tuple[str, str, str], int]:
    """(minute, from account, to account) -> laundering attempt id, from *_Patterns.txt."""

    keys: dict[tuple[str, str, str], int] = {}
    attempt = -1
    for line in patterns.read_text(encoding="utf-8").splitlines():
        if line.startswith("BEGIN LAUNDERING ATTEMPT"):
            attempt += 1
        elif line and not line.startswith("END") and attempt >= 0:
            p = line.split(",")
            minute = pd.Timestamp(p[0].replace("/", "-"), tz="UTC").strftime("%Y-%m-%d %H:%M")
            keys[(minute, p[2], p[4])] = attempt
    return keys


def evaluate(name: str, scored: pd.DataFrame, threshold_recall: float, keys: dict) -> dict:
    y = scored["is_fraud"].to_numpy()
    s = scored["risk_score"].to_numpy()
    flagged = scored["is_candidate"].to_numpy()
    reviews = scored["manual_review"].to_numpy()
    review_hits = int(y[reviews].sum())
    out = {
        "name": name,
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "test_laundering": int(y.sum()),
        "caught_at_threshold": int((flagged & (y == 1)).sum()),
        "recall_at_threshold": float((flagged & (y == 1)).sum() / max(1, y.sum())),
        "flag_rate_at_threshold": float(flagged.mean()),
        "precision_at_threshold": float((flagged & (y == 1)).sum() / max(1, flagged.sum())),
        "manual_reviews": int(reviews.sum()),
        "manual_review_hits": review_hits,
        "manual_review_precision": review_hits / max(1, int(reviews.sum())),
        "manual_review_recall": review_hits / max(1, int(y.sum())),
        "flag_rate_needed_for_90": flag_rate_for_recall(y, s, .90),
        "flag_rate_needed_for_95": flag_rate_for_recall(y, s, .95),
        "flag_rate_needed_for_97": flag_rate_for_recall(y, s, .97),
    }
    k = max(1, int(round(.04 * len(s))))
    out["recall_at_top4"] = float(y[np.argsort(-s, kind="stable")[:k]].sum() / max(1, y.sum()))
    if keys:
        minute = scored["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        sender = scored["sender_id"].str.split(":", n=1).str[-1]
        receiver = scored["receiver_id"].str.split(":", n=1).str[-1]
        attempt = [keys.get((m, a, b), -1) for m, a, b in zip(minute, sender, receiver, strict=True)]
        frame = pd.DataFrame({"attempt": attempt, "flagged": flagged})
        frame = frame[frame.attempt >= 0]
        per_attempt = frame.groupby("attempt").flagged.any()
        out["attempts_in_test"] = int(len(per_attempt))
        out["attempts_with_a_flagged_tx"] = int(per_attempt.sum())
    log(f"{name}: " + ", ".join(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
                                 for key, value in out.items() if key != "name"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, default=ROOT / "data" / "downloads" / "ibm_amlworld.zip")
    parser.add_argument("--patterns", type=Path, default=ROOT / "data" / "ibm_amlworld" / "HI-Small_Patterns.txt")
    parser.add_argument("--start", default="2022/09/01")
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--target-recall", type=float, default=.95)
    parser.add_argument("--label-delay-hours", type=float, default=24)
    parser.add_argument("--review-rate", type=float, default=.02, help="manual review fraction on validation")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "models" / "amlworld_full_scorer.joblib")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "artifacts" / "training")
    parser.add_argument("--save-test-scores", action="store_true",
                        help="also save full_scores_test.pkl in the report directory for QPU benchmarks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stream-output", type=Path,
                        help="also write a dashboard stream: last 20,000 validation rows + the test period")
    args = parser.parse_args()

    started = time.time()
    data = read_amlworld_days(args.archive, start_day=args.start, days=args.days)
    train, validation, test = temporal_split(data)
    del data
    if args.stream_output:
        pd.concat([validation.tail(20_000).assign(split="history"), test.assign(split="stream")],
                  ignore_index=True).to_csv(args.stream_output, index=False)
        log(f"stream file -> {args.stream_output}")
    log(f"rows train/val/test {len(train):,}/{len(validation):,}/{len(test):,}; laundering "
        f"{int(train.is_fraud.sum())}/{int(validation.is_fraud.sum())}/{int(test.is_fraud.sum())}; "
        f"test from {test.timestamp.min()} to {test.timestamp.max()}")
    keys = attempt_keys(args.patterns) if args.patterns.is_file() else {}
    base = replace(ScoringConfig(), target_recall=args.target_recall, review_rate=args.review_rate, random_state=args.seed,
                   label_delay_minutes=int(args.label_delay_hours * 60))
    results = []

    # the model now on the dashboard (4 days), with its confirmed list brought up to date
    four_day = ROOT / "artifacts" / "models" / "amlworld_noformat_tuned_scorer.joblib"
    if four_day.is_file():
        scorer = RiskScorer.load(four_day)
        scorer.history_transactions_ = RiskScorer._feature_history(train, validation)
        scorer.confirmed_minute_ = _confirmed_minutes(train, validation)
        results.append({**evaluate("4-day model (own threshold, list updated)", scorer.score(test),
                                   args.target_recall, keys), "validation_recall": None})

    variants = [
        ("all data, base features", replace(base, feature_groups=())),
        ("all data, confirmed list, no payment format", replace(
            base, feature_groups=("attributes", "confirmed"), lightgbm_params=TUNED_NOFORMAT)),
    ]
    saved = None
    for name, config in variants:
        t0 = time.time()
        scorer = RiskScorer(config).fit(train, validation)
        log(f"{name}: fitted in {time.time() - t0:.0f} s; validation recall "
            f"{scorer.validation_stats_['fraud_retention']:.3f} at flag rate "
            f"{scorer.validation_stats_['candidate_rate']:.3f}")
        scored = scorer.score(test)
        row = evaluate(name, scored, args.target_recall, keys)
        row.update(validation_recall=scorer.validation_stats_["fraud_retention"],
                   validation_flag_rate=scorer.validation_stats_["candidate_rate"],
                   validation_pr_auc=scorer.validation_stats_["pr_auc"],
                   threshold=scorer.candidate_threshold_, fit_seconds=round(time.time() - t0))
        results.append(row)
        if "confirmed" in config.feature_groups:
            scorer.save(args.output)
            saved = asdict(config)
            if args.save_test_scores:
                columns = ["transaction_id", "timestamp", "sender_id", "receiver_id",
                           "is_fraud", "payment_format", "amount", "risk_score"]
                args.report_dir.mkdir(parents=True, exist_ok=True)
                destination = args.report_dir / "full_scores_test.pkl"
                with TemporaryDirectory(dir=args.report_dir) as temporary:
                    staged = Path(temporary) / destination.name
                    scored[columns].to_pickle(staged)
                    staged.replace(destination)
                log(f"benchmark scores -> {destination}")
        del scored

    report = {"rows": [len(train), len(validation), len(test)],
              "laundering": [int(train.is_fraud.sum()), int(validation.is_fraud.sum()), int(test.is_fraud.sum())],
              "target_recall": args.target_recall, "label_delay_hours": args.label_delay_hours,
              "results": results, "saved_model": str(args.output), "saved_config": saved,
              "runtime_seconds": round(time.time() - started)}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "amlworld_full_report.json").write_text(json.dumps(report, indent=2, default=str),
                                                               encoding="utf-8")
    log(f"done in {report['runtime_seconds']} s -> {args.output}")


if __name__ == "__main__":
    main()

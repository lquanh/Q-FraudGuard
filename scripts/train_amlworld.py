"""Train and tune the risk scorer on IBM AMLworld with an honest protocol.

* chronological 60/20/20 split of the prepared sample (default 4 days);
* every choice (feature groups, LightGBM parameters) is made on the
  validation period only; the test period is scored once at the end;
* an ablation compares the variants on the same test rows, with bootstrap
  95% intervals and paired differences against the reference scorer;
* the "confirmed" feature group uses laundering labels only after a delay
  (default 24 h), like a SIMO-style list of confirmed mule accounts.

AMLworld is synthetic (IBM, CDLA-Sharing-1.0): results are not bank results.

Usage:
    python scripts/prepare_amlworld.py --days 4 --output data/amlworld_hi_small_4d.csv
    python scripts/train_amlworld.py                       # all feature groups allowed
    python scripts/train_amlworld.py --exclude-groups formats \\
        --output artifacts/models/amlworld_noformat_tuned_scorer.joblib   # 4-day dashboard option
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from qf_guard.config import ScoringConfig  # noqa: E402
from qf_guard.scoring import RiskScorer, training_matrices  # noqa: E402
from qf_guard.training import temporal_split  # noqa: E402

VARIANTS = [
    ("base features", ()),
    ("+ cross-bank / self-transfer / currency mismatch", ("attributes",)),
    ("+ payment format", ("attributes", "formats")),
    ("+ confirmed-account list (24 h delay)", ("attributes", "formats", "confirmed")),
    ("confirmed list without payment format", ("attributes", "confirmed")),
]
PARAM_GRID = [
    dict(num_leaves=nl, min_child_samples=mcs, n_estimators=ne, learning_rate=lr,
         colsample_bytree=cs, reg_lambda=rl)
    for nl, mcs, ne, lr, cs, rl in [
        (31, 20, 240, .05, 1.0, 0.0),   # current settings
        (15, 50, 400, .05, 1.0, 1.0),
        (31, 100, 400, .05, .8, 1.0),
        (63, 100, 400, .05, .8, 1.0),
        (31, 200, 600, .03, .8, 5.0),
        (63, 200, 600, .03, .8, 5.0),
        (127, 400, 600, .03, .7, 5.0),
        (31, 50, 900, .02, .8, 1.0),
        (63, 50, 900, .02, .7, 1.0),
        (15, 20, 900, .03, .9, 0.0),
        (255, 800, 500, .03, .7, 10.0),
        (31, 400, 300, .08, .8, 5.0),
    ]
]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def recall_at(labels: np.ndarray, score: np.ndarray, rate: float) -> float:
    k = max(1, int(round(rate * len(score))))
    top = np.argsort(-score, kind="stable")[:k]
    return float(labels[top].sum() / max(1, labels.sum()))


def bootstrap(labels: np.ndarray, scores: dict[str, np.ndarray], rate: float, reference: str,
              rounds: int, seed: int) -> dict[str, dict]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    draws: dict[str, list[tuple[float, float]]] = {name: [] for name in scores}
    for _ in range(rounds):
        idx = rng.integers(0, n, n)
        y = labels[idx]
        if y.sum() == 0:
            continue
        for name, s in scores.items():
            draws[name].append((average_precision_score(y, s[idx]), recall_at(y, s[idx], rate)))
    out = {}
    ref = np.asarray(draws[reference])
    for name, values in draws.items():
        v = np.asarray(values)
        diff = v - ref
        out[name] = {
            "pr_auc_ci": [float(np.quantile(v[:, 0], .025)), float(np.quantile(v[:, 0], .975))],
            "recall_ci": [float(np.quantile(v[:, 1], .025)), float(np.quantile(v[:, 1], .975))],
            "pr_auc_diff_ci": [float(np.quantile(diff[:, 0], .025)), float(np.quantile(diff[:, 0], .975))],
            "recall_diff_ci": [float(np.quantile(diff[:, 1], .025)), float(np.quantile(diff[:, 1], .975))],
        }
    return out


def evaluate(name: str, scored: pd.DataFrame, rate: float) -> dict:
    y = scored["is_fraud"].to_numpy()
    s = scored["risk_score"].to_numpy()
    candidates = scored["is_candidate"].to_numpy()
    return {
        "name": name,
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        f"recall_at_{rate:.0%}": recall_at(y, s, rate),
        "caught_at_threshold": int((candidates & (y == 1)).sum()),
        "flagged_rate": float(candidates.mean()),
        "precision_at_threshold": float((candidates & (y == 1)).sum() / max(1, candidates.sum())),
    }


def training_markdown(report: dict) -> str:
    """Render the CPU scoring report from the same measurements saved as JSON."""
    names = {
        "current model (2-day sample, base features)": "Mô hình đối chứng huấn luyện trên mẫu hai ngày",
        "base features": "Đặc trưng cơ bản",
        "+ cross-bank / self-transfer / currency mismatch": "Thêm đặc trưng liên ngân hàng, tự chuyển và chênh lệch tiền tệ",
        "+ payment format": "Thêm hình thức thanh toán",
        "+ confirmed-account list (24 h delay)": "Thêm danh sách tài khoản đã xác nhận",
        "confirmed list without payment format": "Đặc trưng giao dịch và danh sách đã xác nhận, bỏ hình thức thanh toán",
        "tuned LightGBM on best feature set": "LightGBM tinh chỉnh trên nhóm đặc trưng chọn ở tập xác thực",
    }

    def number(value, spec=".3f"):
        return format(value, spec).replace(".", ",")

    def interval(values, spec, scale=1):
        return " đến ".join(number(v * scale, spec) for v in values) or "—"

    rows = report["results"]
    chosen = next(row for row in rows if row["name"] == report["chosen_by_validation"])
    # The first row is the two-day comparator when supplied, otherwise the base features.
    reference = names.get(rows[0]["name"], rows[0]["name"])
    rate = report["recall_rate"]
    budget = number(rate * 100, "g") + "%"
    recall_key = f"recall_at_{rate:.0%}"
    data_name = PureWindowsPath(report["data"]).name
    model_name = PureWindowsPath(report["saved_model"]).name
    test_size = f"{report['rows'][-1]:,}".replace(",", ".")
    lines = [
        "# So sánh mô hình sàng lọc trên AMLworld", "",
        f"Thí nghiệm dùng `{data_name}`, một mẫu giao dịch tổng hợp IBM AMLworld. "
        "Mỗi cấu hình được đánh giá trên cùng tập kiểm tra theo thời gian. "
        "Tiêu chí chọn mô hình là Average Precision trên tập xác thực. "
        f"Ở các cấu hình dùng danh sách tài khoản đã xác nhận, nhãn được sử dụng sau "
        f"{number(report['label_delay_minutes'] / 60, 'g')} giờ.", "",
        f"Trên {test_size} giao dịch kiểm tra của mẫu này, mô hình `{model_name}` "
        f"đạt PR-AUC **{number(chosen['pr_auc'])}**. "
        f"Trong {budget} giao dịch có điểm cao nhất, mô hình giữ được "
        f"**{number(chosen[recall_key], '.1%')}** số giao dịch rửa tiền của tập kiểm tra.", "",
        "## Dữ liệu theo thứ tự thời gian", "",
        "| Tập | Giao dịch | Giao dịch rửa tiền |", "|---|---:|---:|",
    ]
    for split, count, positives in zip(("Huấn luyện", "Xác thực", "Kiểm tra"),
                                       report["rows"], report["laundering"], strict=True):
        lines.append(f"| {split} | {count:,} | {positives:,} |".replace(",", "."))
    lines += [
        "", "## So sánh các cấu hình", "",
        f"Recall ở top {budget} là tỷ lệ giao dịch rửa tiền tìm thấy khi chỉ xem {budget} "
        "giao dịch có điểm cao nhất. PR-AUC trong báo cáo được tính bằng average precision. "
        "Các khoảng trong bảng là khoảng tin cậy 95% ước lượng bằng bootstrap trên giao dịch. "
        f"Cột cuối so sánh recall với dòng “{reference}” trên cùng các mẫu bootstrap.", "",
        f"| Cấu hình | PR-AUC xác thực | PR-AUC kiểm tra | ROC-AUC kiểm tra | Recall ở top {budget} "
        "| Khoảng chênh lệch recall so với đối chứng (điểm phần trăm) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        label = names.get(row["name"], row["name"])
        if row["name"] == chosen["name"]:
            label = f"**{label} (được chọn)**"
        validation = row["validation_pr_auc"]
        val_ap = number(validation) if validation is not None else "—"
        ap_ci = interval(row.get("pr_auc_ci", []), ".3f")
        recall_ci = interval(row.get("recall_ci", []), ".1%")
        difference = interval(row.get("recall_diff_ci", []), "+.1f", scale=100)
        lines.append(f"| {label} | {val_ap} | {number(row['pr_auc'])} ({ap_ci}) "
                     f"| {number(row['roc_auc'])} | {number(row[recall_key], '.1%')} ({recall_ci}) | {difference} |")
    lines += [
        "", f"Thời gian huấn luyện, dò tham số và đánh giá là {number(report['runtime_seconds'], '.1f')} giây. "
        "Các tham số LightGBM, điểm từng lượt dò và số liệu trước khi làm tròn nằm trong "
        "[báo cáo JSON của cùng lần chạy](amlworld_training_report.json).",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "amlworld_hi_small_4d.csv")
    parser.add_argument("--current-model", type=Path,
                        default=ROOT / "artifacts" / "models" / "amlworld_lightgbm_scorer.joblib")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts" / "models" / "amlworld_tuned_scorer.joblib")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "artifacts" / "training")
    parser.add_argument("--rate", type=float, default=.04, help="flag rate used for recall@rate")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--tune-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-groups", nargs="*", default=[],
                        help="feature groups never allowed, decided before looking at test "
                             "(e.g. 'formats', a likely AMLworld generator artefact)")
    args = parser.parse_args()
    variants = [(name, groups) for name, groups in VARIANTS if not set(groups) & set(args.exclude_groups)]

    started = time.time()
    train, validation, test = temporal_split(pd.read_csv(args.data))
    log(f"rows train/val/test {len(train):,}/{len(validation):,}/{len(test):,}; laundering "
        f"{int(train.is_fraud.sum())}/{int(validation.is_fraud.sum())}/{int(test.is_fraud.sum())}")
    base = replace(ScoringConfig(), target_candidate_rate=args.rate, random_state=args.seed)

    results, test_scores, models = [], {}, {}
    if args.current_model.is_file():
        current = RiskScorer.load(args.current_model)
        current.history_transactions_ = RiskScorer._feature_history(train, validation)
        scored = current.score(test)
        results.append({**evaluate("current model (2-day sample, base features)", scored, args.rate),
                        "validation_pr_auc": None, "feature_groups": [], "lightgbm_params": None})
        test_scores["current"] = scored["risk_score"].to_numpy()
        labels = scored["is_fraud"].to_numpy()
        log(f"current model: {results[-1]}")

    for name, groups in variants:
        config = replace(base, feature_groups=groups)
        t0 = time.time()
        scorer = RiskScorer(config).fit(train, validation)
        scored = scorer.score(test)
        labels = scored["is_fraud"].to_numpy()
        row = {**evaluate(name, scored, args.rate), "validation_pr_auc": scorer.validation_stats_["pr_auc"],
               "feature_groups": list(groups), "lightgbm_params": None,
               "weights": list(scorer.learned_weights_), "fit_seconds": round(time.time() - t0, 1)}
        results.append(row)
        test_scores[name] = scored["risk_score"].to_numpy()
        models[name] = scorer
        log(f"{name}: val PR-AUC {row['validation_pr_auc']:.3f} | test {row}")

    # tune LightGBM on the two strongest feature sets (by validation PR-AUC only)
    ranked = sorted((r for r in results if r["validation_pr_auc"] is not None),
                    key=lambda r: -r["validation_pr_auc"])
    tuning = []
    for candidate in ranked[:2]:
        groups = tuple(candidate["feature_groups"])
        config = replace(base, feature_groups=groups)
        X_train, y_train, X_val, y_val = training_matrices(train, validation, config)
        from lightgbm import LGBMClassifier
        from sklearn.utils.class_weight import compute_sample_weight

        weight = compute_sample_weight("balanced", y_train)
        for params in PARAM_GRID:
            t0 = time.time()
            model = LGBMClassifier(deterministic=True, force_col_wise=True, n_jobs=args.tune_threads,
                                   random_state=args.seed, verbosity=-1, **params)
            model.fit(X_train, y_train, sample_weight=weight)
            ap = float(average_precision_score(y_val, model.predict_proba(X_val)[:, 1]))
            tuning.append({"feature_groups": list(groups), "params": params, "validation_pr_auc_lgbm": ap,
                           "seconds": round(time.time() - t0, 1)})
            log(f"tune {groups} {params}: val PR-AUC (LightGBM only) {ap:.4f}")
    best = max(tuning, key=lambda r: r["validation_pr_auc_lgbm"])
    config = replace(base, feature_groups=tuple(best["feature_groups"]), lightgbm_params=best["params"])
    t0 = time.time()
    tuned = RiskScorer(config).fit(train, validation)
    scored = tuned.score(test)
    name = "tuned LightGBM on best feature set"
    row = {**evaluate(name, scored, args.rate), "validation_pr_auc": tuned.validation_stats_["pr_auc"],
           "feature_groups": best["feature_groups"], "lightgbm_params": best["params"],
           "weights": list(tuned.learned_weights_), "fit_seconds": round(time.time() - t0, 1)}
    results.append(row)
    test_scores[name] = scored["risk_score"].to_numpy()
    models[name] = tuned
    log(f"{name}: {row}")

    # final choice by validation PR-AUC only (never by test)
    eligible = [r for r in results if r["validation_pr_auc"] is not None]
    chosen = max(eligible, key=lambda r: r["validation_pr_auc"])
    models[chosen["name"]].save(args.output)
    reference = "current" if "current" in test_scores else VARIANTS[0][0]
    intervals = bootstrap(labels, test_scores, args.rate, reference, args.bootstrap, args.seed)
    for r in results:
        key = "current" if r["name"].startswith("current") else r["name"]
        r.update(intervals.get(key, {}))
    report = {
        "data": str(args.data),
        "rows": [len(train), len(validation), len(test)],
        "laundering": [int(train.is_fraud.sum()), int(validation.is_fraud.sum()), int(test.is_fraud.sum())],
        "recall_rate": args.rate,
        "label_delay_minutes": base.label_delay_minutes,
        "results": results,
        "tuning": tuning,
        "chosen_by_validation": chosen["name"],
        "saved_model": str(args.output),
        "config": asdict(replace(base, feature_groups=tuple(chosen["feature_groups"]),
                                 lightgbm_params=chosen["lightgbm_params"])),
        "runtime_seconds": round(time.time() - started, 1),
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "amlworld_training_report.json").write_text(json.dumps(report, indent=2, default=str),
                                                                   encoding="utf-8")
    metrics_path = args.output.with_name(args.output.stem.replace("_scorer", "") + "_metrics.json")
    metrics_path.write_text(json.dumps({**chosen, "chosen_by_validation": True}, indent=2, default=str),
                            encoding="utf-8")
    (args.report_dir / "amlworld_training_report.md").write_text(training_markdown(report), encoding="utf-8", newline="\n")
    log(f"done: chosen '{chosen['name']}' -> {args.output}")


if __name__ == "__main__":
    main()

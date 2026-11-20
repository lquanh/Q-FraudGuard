"""Tính lại các chỉ tiêu sàng lọc từ điểm từng giao dịch. Chỉ cần Python chuẩn."""
import csv
import gzip
import json
from itertools import groupby
from math import floor
from pathlib import Path

root = Path(__file__).resolve().parents[1]
def read_json(name):
    return json.loads((root / name).read_bytes())

with gzip.open(root / "minh_chung/diem_va_nhan.csv.gz", "rt", encoding="utf-8", newline="") as stream:
    rows = [(float(r["risk_score"]), int(r["is_fraud"])) for r in csv.DictReader(stream)]

report = read_json("artifacts/training/amlworld_full_report.json")
base, selected = report["results"][1:]
operating = read_json("artifacts/training/amlworld_full_operating_points.json")["0.96"]
review = read_json("artifacts/models/amlworld_full_scorer.review.json")
n = len(rows)
positive = sum(label for _, label in rows)
ranked = sorted(rows, key=lambda row: -row[0])  # Giữ thứ tự nguồn khi hai điểm bằng nhau.
k = round(.04 * n)
top = sum(label for _, label in ranked[:k])
seen = hits = 0
average_precision = 0.0
for score, group in groupby(ranked, key=lambda row: row[0]):
    group_hits = 0
    for _, label in group:
        seen += 1
        group_hits += label
    hits += group_hits
    if group_hits:
        average_precision += group_hits / positive * hits / seen

def count_at(threshold):
    selected_rows = [label for score, label in rows if score >= threshold]
    return {"flagged": len(selected_rows), "labelled": sum(selected_rows)}

positive_scores = sorted(score for score,label in rows if label == 1)
cutoff = positive_scores[floor((1 - .95) * positive)]
base_flags = round(base["flag_rate_needed_for_95"] * n)
model_flags = count_at(cutoff)["flagged"]
baseline_top = round(base["recall_at_top4"] * positive)
result = {
    "source": "IBM AMLworld HI-Small | tập kiểm tra mười ngày",
    "rows": n, "labelled": positive, "average_precision": average_precision,
    "top_4_percent": {"budget": k, "baseline_from_run_json": baseline_top, "selected_from_scores": top,
                      "additional_labelled": top - baseline_top},
    "alerts_at_95_percent_recall_retrospective": {
        "baseline_from_run_json": base_flags, "selected_from_scores": model_flags,
        "fewer_alerts": base_flags-model_flags, "reduction_fraction": (base_flags-model_flags)/base_flags},
    "screening_at_threshold_fixed_on_validation": count_at(operating["threshold"]),
    "manual_review_at_threshold_fixed_on_validation": count_at(review["review_threshold"]),
}
assert n == report["rows"][-1] and positive == report["laundering"][-1]
assert abs(average_precision - selected["pr_auc"]) < 1e-10
assert top == round(selected["recall_at_top4"] * positive)
assert model_flags == round(selected["flag_rate_needed_for_95"] * n)
assert result["screening_at_threshold_fixed_on_validation"] == {
    "flagged": round(operating["test_flag"] * n), "labelled": operating["caught"]}
print(json.dumps(result, ensure_ascii=False, indent=2))

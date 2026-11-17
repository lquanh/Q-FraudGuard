"""Human-readable reason codes and investigation descriptions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


REASON_LABELS = {
    "AMOUNT_SPIKE": "Số tiền cao bất thường so với mẫu huấn luyện",
    "BALANCE_DRAIN": "Giao dịch rút gần hết số dư nguồn",
    "BALANCE_MISMATCH": "Biến động số dư không khớp số tiền giao dịch",
    "SHARED_DEVICE": "Thiết bị được dùng bởi nhiều tài khoản",
    "RAPID_ACTIVITY": "Tần suất giao dịch tăng nhanh trong cửa sổ ngắn",
    "CROSS_BORDER": "Giao dịch ngoài thị trường hoạt động chính",
    "NETWORK_PATTERN": "Quan hệ mạng có fan-in, fan-out hoặc dòng tiền đáng ngờ",
    "MODEL_HIGH_RISK": "Mô hình có giám sát ước lượng rủi ro cao",
    "ANOMALY": "Mẫu giao dịch khác biệt so với dữ liệu huấn luyện",
    "CONNECTED_SELECTION": "Được chọn cùng các giao dịch có quan hệ đáng ngờ",
}


def reason_codes(row: Mapping[str, Any], max_reasons: int = 4) -> list[str]:
    """Return stable machine-readable codes ordered by evidential strength."""

    checks = (
        ("BALANCE_DRAIN", float(row.get("rule_balance_drain", 0)) >= 0.5),
        ("SHARED_DEVICE", float(row.get("rule_shared_device", 0)) >= 0.5),
        ("AMOUNT_SPIKE", float(row.get("rule_amount_spike", 0)) >= 0.5),
        ("BALANCE_MISMATCH", float(row.get("rule_balance_mismatch", 0)) >= 0.5),
        ("RAPID_ACTIVITY", float(row.get("rule_rapid_activity", 0)) >= 0.5),
        ("CROSS_BORDER", float(row.get("rule_cross_border", 0)) >= 0.5),
        ("NETWORK_PATTERN", float(row.get("graph_score", 0)) >= 0.55),
        ("MODEL_HIGH_RISK", float(row.get("supervised_score", 0)) >= 0.65),
        ("ANOMALY", float(row.get("anomaly_score", 0)) >= 0.70),
    )
    codes = [code for code, matched in checks if matched]
    return codes[:max_reasons] or ["MODEL_HIGH_RISK"]


def reason_text(codes: list[str] | tuple[str, ...]) -> str:
    return "; ".join(REASON_LABELS.get(code, code) for code in codes)


def risk_band(score: float) -> str:
    if score >= 0.80:
        return "critical"
    if score >= 0.60:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def investigation_narrative(record: Mapping[str, Any]) -> str:
    cluster = record.get("cluster_id", "chưa xác định")
    score = float(record.get("risk_score", 0))
    codes_value = record.get("reason_codes", [])
    if isinstance(codes_value, str):
        codes = [code for code in codes_value.split("|") if code]
    else:
        codes = list(codes_value)
    return (
        f"Ứng viên thuộc cụm {cluster}, điểm rủi ro {score:.1%}. "
        f"Tín hiệu chính: {reason_text(codes)}. Cần chuyên viên xác minh trước khi hành động."
    )


def recommended_treatment(
    record: Mapping[str, Any],
    *,
    hold_threshold: float = 0.80,
    verify_threshold: float = 0.60,
) -> dict[str, str]:
    """Return a review-safe treatment plan; no adverse action is automatic.

    Thresholds are on the model's own risk scale, so callers should pass the
    operating points learned for the deployed scorer.
    """

    score = float(record.get("risk_score", 0))
    codes = set(str(record.get("reason_codes", "")).split("|"))
    if score >= hold_threshold or {"BALANCE_DRAIN", "CONNECTED_SELECTION"} <= codes:
        return {
            "recommended_action": "hold_and_escalate",
            "action_detail": (
                "Tạm giữ giao dịch theo chính sách hiện hành, xác minh khách hàng và "
                "kiểm tra toàn cụm liên quan; chỉ phong tỏa khi chuyên viên phê duyệt."
            ),
            "review_sla": "15_minutes",
        }
    if score >= verify_threshold or "NETWORK_PATTERN" in codes:
        return {
            "recommended_action": "step_up_verification",
            "action_detail": (
                "Yêu cầu xác thực tăng cường, đối chiếu thiết bị/người thụ hưởng và "
                "chuyển chuyên viên gian lận duyệt."
            ),
            "review_sla": "2_hours",
        }
    return {
        "recommended_action": "monitor_and_review",
        "action_detail": (
            "Cho vào hàng đợi giám sát, rà soát lịch sử giao dịch và chỉ can thiệp "
            "khi có thêm bằng chứng."
        ),
        "review_sla": "24_hours",
    }

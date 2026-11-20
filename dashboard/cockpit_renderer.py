"""Render the standalone cockpit preview with a snapshot from CaseStore."""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

from qf_guard import casework
from qf_guard.explainability import review_signals

if TYPE_CHECKING:
    from qf_guard.operations import CaseStore

UI_DIR = Path(__file__).resolve().parent
HTML_TEMPLATE_PATH = UI_DIR / "ui" / "index.html"
ENGRAVING_PATH = UI_DIR / "assets" / "engraving.webp"
PREVIEW_STATES = {
    "pending_review": "urgent",
    "awaiting_info": "waiting",
    "awaiting_verification": "waiting",
    "frozen": "blocked",
    "suspended": "blocked",
}
RUN_STATES = {"running": "Đang xử lý", "error": "Cần xử lý lại", "completed": "Đã hoàn tất"}


def _numeric(value: object) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _preview_case(item: dict) -> dict:
    state = item.get("work_state")
    signals, _ = review_signals(item)
    return {
        "txid": item["transaction_id"],
        "amount": _numeric(item.get("amount")),
        "currency": item.get("amount_currency") or item.get("currency") or "",
        "format": item.get("payment_format") or item.get("type") or "—",
        "sender": item.get("sender_id") or "—",
        "receiver": item.get("receiver_id") or "—",
        "riskScore": _numeric(item.get("risk_score")),
        "graphModel": item.get("graph_model", "heuristic"),
        "signals": [_numeric(item.get(f"{code}_score")) for code in ("supervised", "anomaly", "graph", "rule")],
        "category": PREVIEW_STATES.get(state, "urgent"),
        "time": str(item.get("timestamp") or "—"),
        "tags": signals[:2],
        "sla": casework.STATES.get(state, "Chờ xem xét"),
        "notes": "",
    }


def _preview_log(entry: dict, actions: dict[str, str]) -> dict:
    detail = entry["detail"] or ""
    kind = entry["kind"]
    if kind in ("casework", "decide"):
        code, _, reason = detail.partition(" · ")
        labels = casework.OPERATIONS if kind == "casework" else actions
        action, detail = labels.get(code, "Xử lý hồ sơ"), reason
    elif kind == "run":
        action, detail = "Phiên ưu tiên", RUN_STATES.get(detail, "Đã ghi nhận")
    else:
        action = "Tiếp nhận"
    return {
        "txid": entry["txid"] or entry["window_id"] or "—",
        "action": action,
        "actor": entry["actor"] or "Hệ thống",
        "time": str(entry["at"] or "")[11:19] or "—",
        "desc": detail,
    }


def build_cockpit_html(store: CaseStore, actions: dict[str, str]) -> str:
    raw_html = HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
    if ENGRAVING_PATH.is_file():
        image = base64.b64encode(ENGRAVING_PATH.read_bytes()).decode("ascii")
        raw_html = raw_html.replace(
            "url('../assets/engraving.webp')", f"url('data:image/webp;base64,{image}')"
        )
    stats = store.stats()
    _, cases = store.work_queue(tuple(PREVIEW_STATES), limit=30)
    if not cases:
        _, cases = store.queue(all_candidates=True, use_qubo=True, limit=30)
    payload = {
        "processed": stats["processed"],
        "candidates": stats["candidates"],
        "cases": [_preview_case(item) for item in cases],
        "audit": [_preview_log(entry, actions) for entry in store.journal(limit=12)],
    }
    script_data = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", r"\u003c")
    return raw_html.replace(
        "<!-- QFG_COCKPIT_DATA -->", f"<script>window.qfgCockpitData = {script_data};</script>", 1
    )


from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
from html import escape
from math import ceil, cos, pi, sin
import os
from pathlib import Path
import random
import sys
from time import monotonic, perf_counter
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd
import streamlit as st

from qf_guard.data import normalize_transactions
from qf_guard.explainability import recommended_treatment
from qf_guard.prioritization import PrioritySelection, select_priority
from qf_guard.runtime import RealtimeService
from qf_guard.scoring import RiskScorer
from qf_guard.solvers import FallbackSolver, solver_from_env

RED = "#d92d4f"
MAX_VISIBLE_ALERTS = 30
MAX_PENDING_ALERTS = 500
REALTIME_HISTORY_LIMIT = 5_000
GRAPH_HISTORY_LIMIT = 2_000
MODEL_PATH = Path(
    os.environ.get(
        "QFG_MODEL_PATH",
        PROJECT_ROOT / "artifacts" / "models" / "paysim_lightgbm_scorer.joblib",
    )
)
# Candidates are grouped into windows so each QPU call carries a meaningful
# QUBO and the per-call network latency does not throttle the 1 s UI tick.
QUANTUM_MIN_BATCH = int(os.environ.get("QFG_QUANTUM_MIN_BATCH", "8"))
QUANTUM_MAX_BATCH = int(os.environ.get("QFG_QUANTUM_MAX_BATCH", "40"))
QUANTUM_WINDOW_SECONDS = float(os.environ.get("QFG_QUANTUM_WINDOW_SECONDS", "10"))
BACKEND_LABELS = {
    "dwave-qpu": "D-Wave QPU",
    "dwave-hybrid": "D-Wave Hybrid",
    "cpu-simulated-quantum-annealing": "SQA (CPU)",
    "cpu-exhaustive": "Exact (CPU)",
    "none": "Không cần giải (K = N)",
    "error": "Lỗi solver",
}
QUANTUM_STATUS_LABELS = {
    "pending": "⏳ chờ batch",
    "running": "⚛ đang giải",
    "selected": "✅ ưu tiên",
    "not_selected": "—",
    "error": "⚠ lỗi solver",
}

st.set_page_config(
    page_title="Q-FraudGuard",
    page_icon="QF",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    f"""
    <style>
      .stApp {{background:#f5f7fb;color:#172033}}
      [data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],
      [data-testid="stHeader"],[data-testid="stToolbar"],
      [data-testid="stDecoration"],[data-testid="stStatusWidget"] {{display:none}}
      .block-container {{max-width:1500px;padding-top:.55rem;padding-bottom:3rem}}
      .qfg-header {{display:flex;align-items:center;gap:.55rem;min-height:2.2rem;
        margin-bottom:.7rem;padding:.2rem 0 .55rem;border-bottom:1px solid #e2e8f0}}
      .qfg-logo {{width:1.8rem;height:1.8rem;border-radius:.5rem;display:grid;
        place-items:center;color:white;font-size:.72rem;font-weight:900;
        background:linear-gradient(135deg,#4338ca,#7c3aed)}}
      .qfg-name {{font-size:.95rem;font-weight:750}}
      .qfg-live {{margin-left:auto;color:#008f83;font-size:.64rem;font-weight:700;
        text-transform:uppercase;letter-spacing:.12em}}
      .qfg-dot {{display:inline-block;width:.45rem;height:.45rem;margin-right:.4rem;
        border-radius:50%;background:#008f83}}
      .qfg-card {{min-height:6.6rem;padding:1rem;border-radius:1rem;background:white;
        border:1px solid #e2e8f0;box-shadow:0 10px 30px rgba(15,23,42,.06)}}
      .qfg-label {{font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;
        color:#64748b;font-weight:650}}
      .qfg-value {{font-size:1.7rem;font-weight:760;margin:.35rem 0 .2rem}}
      .qfg-note {{font-size:.75rem;color:#64748b;overflow-wrap:anywhere}}
      .qfg-value.qfg-quantum {{font-size:1.15rem;color:#5b21b6;margin:.55rem 0 .35rem}}
      .qfg-risk {{color:{RED}}}
      .qfg-warn {{color:#b86b00}}
      div[data-testid="stDataFrame"] {{border:1px solid #e2e8f0;border-radius:.8rem}}
      [data-testid="stToastContainer"] {{top:.65rem!important;bottom:auto!important;
        right:1.25rem!important}}
      .st-key-fraud_alert_stack {{position:fixed;right:1.25rem;top:6.25rem;
        width:370px;max-width:calc(100vw - 2.5rem);max-height:calc(100vh - 7.5rem);
        overflow-y:auto;z-index:999999;
        background:rgba(255,255,255,.98);border:1px solid #fecdd3;border-radius:1rem;
        padding:.9rem;box-shadow:0 24px 70px rgba(15,23,42,.18)}}
      .st-key-fraud_alert_reopen {{position:fixed;right:1.25rem;top:6.25rem;
        z-index:999999}}
      .st-key-fraud_alert_reopen .stButton>button {{border-radius:999px;
        border-color:#fecdd3;color:{RED};box-shadow:0 10px 30px rgba(15,23,42,.12)}}
      .st-key-hide_fraud_alerts .stButton>button {{border:0;background:transparent;
        box-shadow:none;font-size:1.35rem}}
      [class*="st-key-execute_"] .stButton>button {{background:#15803d!important;
        border-color:#15803d!important;color:white!important;
        transition:transform .16s ease,box-shadow .2s ease,background-color .2s ease}}
      [class*="st-key-execute_"] .stButton>button:hover {{
        background:#16a34a!important;transform:scaleX(1.025);
        box-shadow:-18px 0 24px -13px rgba(22,163,74,.9),
          18px 0 24px -13px rgba(22,163,74,.9),0 8px 18px rgba(21,128,61,.22)}}
      [class*="st-key-execute_"] .stButton>button:active {{
        transform:scaleX(.99);box-shadow:0 3px 8px rgba(21,128,61,.2)}}
      [class*="st-key-open_graph_"] .stButton>button {{border-color:#c4b5fd;
        color:#5b21b6;background:#f5f3ff;transition:background .16s ease,transform .16s ease}}
      [class*="st-key-open_graph_"] .stButton>button:hover {{
        background:#ede9fe;transform:translateY(-1px)}}
      .fraud-title {{color:#b4233f;font-size:1.02rem;font-weight:800}}
      .fraud-id {{font-size:1.18rem;font-weight:760;margin:.3rem 0 .6rem}}
      .risk-viz {{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));
        gap:.35rem;margin:.45rem 0;padding:.38rem;border-radius:.65rem;background:#fff7f8;
        border:1px solid #ffe4e8}}
      .risk-viz-title {{font-size:.68rem;font-weight:800;letter-spacing:.08em;
        text-transform:uppercase;color:#9f1239;margin-bottom:.55rem}}
      .risk-viz-row {{display:grid;grid-template-columns:auto minmax(1rem,1fr) auto;
        gap:.25rem;align-items:center;min-width:0;font-size:.58rem;color:#475569}}
      .risk-viz-name {{font-weight:750;color:#64748b}}
      .risk-viz-track {{height:.3rem;border-radius:999px;background:#e2e8f0;overflow:hidden}}
      .risk-viz-fill {{height:100%;border-radius:999px;background:linear-gradient(90deg,
        #fb7185,#be123c)}}
      .risk-viz-value {{text-align:right;font-weight:800;color:#881337}}
      .fraud-graph {{margin:.5rem 0 .75rem;padding:.35rem;border-radius:.75rem;
        background:#f8fafc;border:1px solid #e2e8f0}}
      .fraud-graph-label {{font-size:.67rem;color:#64748b;text-align:center;margin-top:.15rem}}
      .graph-flow {{stroke-dasharray:7 5;animation:qfg-flow 1.15s linear infinite}}
      @keyframes qfg-flow {{to {{stroke-dashoffset:-24}}}}
      .graph-focus {{animation:qfg-focus 1.25s ease-in-out infinite}}
      @keyframes qfg-focus {{
        0%,100% {{opacity:.55;stroke-width:2}}
        50% {{opacity:1;stroke-width:4}}
      }}
      .graph-legend {{display:flex;flex-wrap:wrap;justify-content:center;gap:.45rem 1rem;
        margin:.2rem 0 .35rem;color:#64748b;font-size:.62rem}}
      .graph-dot {{display:inline-block;width:.5rem;height:.5rem;border-radius:50%;
        margin-right:.25rem}}
      @media (max-width:700px) {{
        .st-key-fraud_alert_stack {{right:.5rem;width:calc(100vw - 1rem);
          max-width:calc(100vw - 1rem)}}
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def scorer_template() -> RiskScorer:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")
    return RiskScorer.load(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def quantum_solver() -> Any:
    """Process-wide solver chosen by QFG_SOLVER (QPU budget is shared too)."""

    return solver_from_env()


@st.cache_resource(show_spinner=False)
def quantum_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="qfg-quantum")


def service() -> RealtimeService:
    """Per-session service: each browser session keeps its own feature history."""

    return st.session_state["qfg_service"]


@st.cache_data(show_spinner=False)
def realtime_source() -> pd.DataFrame:
    source = PROJECT_ROOT / "data" / "paysim.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Không tìm thấy nguồn luồng: {source}")
    return normalize_transactions(pd.read_csv(source))


def metric_card(label: str, value: str, note: str, css_class: str = "") -> None:
    st.markdown(
        f"""<div class="qfg-card"><div class="qfg-label">{escape(label)}</div>
        <div class="qfg-value {css_class}">{escape(value)}</div>
        <div class="qfg-note">{escape(note)}</div></div>""",
        unsafe_allow_html=True,
    )


def risk_visualization(alert: Mapping[str, object]) -> str:
    components = (
        ("Supervised", "Supervised model score", float(alert.get("Supervised", 0))),
        ("Anomaly", "Anomaly detection score", float(alert.get("Anomaly", 0))),
        ("Graph", "Graph relationship score", float(alert.get("Graph", 0))),
        ("Rule", "Business rule score", float(alert.get("Rule", 0))),
    )
    rows = "".join(
        f"""<div class="risk-viz-row" title="{title}">
        <span class="risk-viz-name">{label}</span>
        <div class="risk-viz-track"><div class="risk-viz-fill"
        style="width:{max(0.0, min(1.0, value)) * 100:.1f}%"></div></div>
        <span class="risk-viz-value">{value:.0%}</span></div>"""
        for label, title, value in components
    )
    return f"""<div class="risk-viz-title">Thành phần Risk Score</div>
    <div class="risk-viz">{rows}</div>"""


def _transaction_accounts(transaction: Mapping[str, object]) -> set[str]:
    return {
        str(transaction.get("Người gửi", "")),
        str(transaction.get("Người nhận", "")),
    } - {"", "UNKNOWN"}


def _candidate_relation(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> tuple[str, float] | None:
    if _transaction_accounts(left) & _transaction_accounts(right):
        return "shared", 1.0
    risk_gap = abs(float(left.get("Risk", 0)) - float(right.get("Risk", 0)))
    graph_gap = abs(float(left.get("Graph", 0)) - float(right.get("Graph", 0)))
    rule_gap = abs(float(left.get("Rule", 0)) - float(right.get("Rule", 0)))
    same_type = str(left.get("Loại", "")) == str(right.get("Loại", ""))
    left_reasons = set(str(left.get("Lý do", "")).split("|")) - {""}
    right_reasons = set(str(right.get("Lý do", "")).split("|")) - {""}
    shared_rule = bool(left_reasons & right_reasons)
    left_amount = max(1.0, float(left.get("Số tiền", 0)))
    right_amount = max(1.0, float(right.get("Số tiền", 0)))
    amount_ratio = min(left_amount, right_amount) / max(left_amount, right_amount)
    if same_type and amount_ratio >= 0.42 and risk_gap <= 0.16:
        return "behavior", 0.55 + 0.25 * amount_ratio + 0.20 * (1 - risk_gap / 0.16)
    if risk_gap <= 0.055 and graph_gap <= 0.10:
        return "risk", 0.60 + 0.25 * (1 - risk_gap / 0.055) + 0.15 * (1 - graph_gap / 0.10)
    if shared_rule and rule_gap <= 0.12:
        return "rule", 0.62 + 0.38 * (1 - rule_gap / 0.12)
    return None


def _has_directed_cycle(transactions: list[Mapping[str, object]]) -> bool:
    adjacency: dict[str, set[str]] = {}
    for transaction in transactions:
        source = str(transaction.get("Người gửi", ""))
        target = str(transaction.get("Người nhận", ""))
        adjacency.setdefault(source, set()).add(target)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(visit(neighbor) for neighbor in adjacency.get(node, ())):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in tuple(adjacency))


def _separate_graph_nodes(
    account_positions: dict[str, tuple[float, float]],
    transaction_positions: dict[str, tuple[float, float]],
    degree: Counter[str],
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    positions = {
        **{f"account:{key}": value for key, value in account_positions.items()},
        **{f"transaction:{key}": value for key, value in transaction_positions.items()},
    }
    anchors = dict(positions)
    radii = {
        **{
            f"account:{key}": min(33.0, 22.0 + 2.2 * degree[key])
            for key in account_positions
        },
        **{f"transaction:{key}": 25.0 for key in transaction_positions},
    }
    node_ids = tuple(positions)
    for _ in range(120):
        displacement = {node_id: [0.0, 0.0] for node_id in node_ids}
        for left_index, left_id in enumerate(node_ids):
            left_x, left_y = positions[left_id]
            for right_index, right_id in enumerate(node_ids[left_index + 1 :], left_index + 1):
                right_x, right_y = positions[right_id]
                delta_x, delta_y = right_x - left_x, right_y - left_y
                distance = (delta_x**2 + delta_y**2) ** 0.5
                if distance < 0.01:
                    angle = (left_index * 17 + right_index * 31) * pi / 180
                    delta_x, delta_y = cos(angle), sin(angle)
                    distance = 1.0
                minimum = radii[left_id] + radii[right_id] + 7.0
                if distance >= minimum:
                    continue
                push = (minimum - distance) * 0.52
                unit_x, unit_y = delta_x / distance, delta_y / distance
                displacement[left_id][0] -= unit_x * push
                displacement[left_id][1] -= unit_y * push
                displacement[right_id][0] += unit_x * push
                displacement[right_id][1] += unit_y * push
        for node_id in node_ids:
            x, y = positions[node_id]
            anchor_x, anchor_y = anchors[node_id]
            move_x = displacement[node_id][0] + (anchor_x - x) * 0.03
            move_y = displacement[node_id][1] + (anchor_y - y) * 0.03
            magnitude = max(1.0, (move_x**2 + move_y**2) ** 0.5 / 8.0)
            radius = radii[node_id]
            positions[node_id] = (
                min(520.0 - radius, max(radius, x + move_x / magnitude)),
                min(340.0 - radius, max(radius, y + move_y / magnitude)),
            )
    return (
        {key: positions[f"account:{key}"] for key in account_positions},
        {key: positions[f"transaction:{key}"] for key in transaction_positions},
    )


def suspicious_graph(
    alert: Mapping[str, object],
    history: list[dict[str, object]],
    candidate_pool: list[dict[str, object]],
) -> str:
    alert_id = str(alert["Mã giao dịch"])
    if len(_transaction_accounts(alert)) < 2:
        return """<div class="fraud-graph"><div class="fraud-graph-label">
        Đang thu thập dữ liệu quan hệ cho ứng viên này</div></div>"""
    unique_candidates = {
        str(item["Mã giao dịch"]): item
        for item in candidate_pool
        if len(_transaction_accounts(item)) == 2
    }
    unique_candidates[alert_id] = dict(alert)
    candidate_ids = tuple(unique_candidates)
    potential_relations: list[tuple[str, str, str, float]] = []
    for left_index, left_id in enumerate(candidate_ids):
        for right_id in candidate_ids[left_index + 1 :]:
            relation = _candidate_relation(unique_candidates[left_id], unique_candidates[right_id])
            if relation:
                potential_relations.append((left_id, right_id, relation[0], relation[1]))

    kept_relations = [relation for relation in potential_relations if relation[2] == "shared"]
    for candidate_id in candidate_ids:
        analytical = sorted(
            (
                relation
                for relation in potential_relations
                if relation[2] != "shared" and candidate_id in relation[:2]
            ),
            key=lambda relation: relation[3],
            reverse=True,
        )
        kept_relations.extend(analytical[:2])
    kept_relations = list(dict.fromkeys(kept_relations))

    adjacency: dict[str, list[tuple[str, str, float]]] = {item: [] for item in candidate_ids}
    for left_id, right_id, kind, strength in kept_relations:
        adjacency[left_id].append((right_id, kind, strength))
        adjacency[right_id].append((left_id, kind, strength))
    selected_ids: list[str] = []
    queue = [alert_id]
    while queue and len(selected_ids) < 6:
        current = queue.pop(0)
        if current in selected_ids:
            continue
        selected_ids.append(current)
        neighbors = sorted(adjacency[current], key=lambda item: item[2], reverse=True)
        queue.extend(neighbor for neighbor, _, _ in neighbors if neighbor not in selected_ids)

    transactions: list[Mapping[str, object]] = [unique_candidates[item] for item in selected_ids]
    known_accounts: set[str] = set().union(
        *(_transaction_accounts(transaction) for transaction in transactions)
    )
    seen_transactions = set(selected_ids)
    support_limit = min(4, max(0, 10 - len(transactions)))
    sender = str(alert.get("Người gửi", ""))
    receiver = str(alert.get("Người nhận", ""))
    anchor_accounts = tuple(dict.fromkeys((sender, receiver)))
    alert_time = pd.to_datetime(alert.get("Thời gian"), utc=True, errors="coerce")

    def add_supporting_transaction(row: dict[str, object]) -> bool:
        row_id = str(row.get("Mã giao dịch", ""))
        row_accounts = _transaction_accounts(row)
        row_time = pd.to_datetime(row.get("Thời gian"), utc=True, errors="coerce")
        if (
            row_id in seen_transactions
            or not row_accounts
            or len(known_accounts | row_accounts) > 16
            or (not pd.isna(alert_time) and not pd.isna(row_time) and row_time >= alert_time)
        ):
            return False
        transactions.append(row)
        seen_transactions.add(row_id)
        known_accounts.update(row_accounts)
        return True

    for account in anchor_accounts:
        for direction in ("incoming", "outgoing"):
            for row in reversed(history):
                if len(transactions) - len(selected_ids) >= support_limit:
                    break
                matches_direction = (
                    str(row.get("Người nhận", "")) == account
                    if direction == "incoming"
                    else str(row.get("Người gửi", "")) == account
                )
                if matches_direction and add_supporting_transaction(row):
                    break

    for row in reversed(history):
        if len(transactions) - len(selected_ids) >= support_limit:
            break
        if _transaction_accounts(row) & known_accounts:
            add_supporting_transaction(row)

    candidate_transactions = transactions[: len(selected_ids)]
    send_counts = Counter(str(item.get("Người gửi", "")) for item in candidate_transactions)
    receive_counts = Counter(str(item.get("Người nhận", "")) for item in candidate_transactions)
    chain_count = sum(
        str(left.get("Người nhận", "")) == str(right.get("Người gửi", ""))
        for left in candidate_transactions
        for right in candidate_transactions
        if left is not right
    )
    motif_threshold = max(2, ceil(len(candidate_transactions) * 0.5))
    analytical_kinds = Counter(
        kind
        for left_id, right_id, kind, _ in kept_relations
        if left_id in selected_ids and right_id in selected_ids and kind != "shared"
    )
    if len(candidate_transactions) == 1:
        motif, motif_label = "isolated", "Ứng viên độc lập"
    elif _has_directed_cycle(candidate_transactions):
        motif, motif_label = "cycle", "Dòng tiền vòng"
    elif receive_counts and max(receive_counts.values()) >= motif_threshold:
        motif, motif_label = "fan_in", "Nhiều nguồn → một đích"
    elif send_counts and max(send_counts.values()) >= motif_threshold:
        motif, motif_label = "fan_out", "Một nguồn → nhiều đích"
    elif chain_count >= 2:
        motif, motif_label = "chain", "Chuỗi chuyển tiếp"
    elif analytical_kinds:
        motif, motif_label = "behavior", "Cụm tương đồng hành vi"
    else:
        motif, motif_label = "mixed", "Mạng quan hệ hỗn hợp"

    account_order = sorted(known_accounts)
    degree = Counter(
        account
        for transaction in transactions
        for account in _transaction_accounts(transaction)
    )
    account_positions: dict[str, tuple[float, float]] = {}
    if motif in {"fan_in", "fan_out"}:
        hub = (
            receive_counts.most_common(1)[0][0]
            if motif == "fan_in"
            else send_counts.most_common(1)[0][0]
        )
        hub_x, leaf_x = ((410.0, 170.0), 95.0) if motif == "fan_in" else ((110.0, 170.0), 425.0)
        account_positions[hub] = hub_x
        leaves = [account for account in account_order if account != hub]
        for index, account in enumerate(leaves):
            y = 45.0 + index * 250.0 / max(1, len(leaves) - 1)
            account_positions[account] = (leaf_x + 18 * (index % 2), y)
    elif motif == "chain":
        flow_score = {
            account: send_counts.get(account, 0) - receive_counts.get(account, 0)
            for account in account_order
        }
        ordered = sorted(account_order, key=lambda account: flow_score[account], reverse=True)
        for index, account in enumerate(ordered):
            x = 55.0 + index * 410.0 / max(1, len(ordered) - 1)
            account_positions[account] = (x, 125.0 if index % 2 == 0 else 215.0)
    elif motif == "mixed":
        hub = max(account_order, key=lambda account: degree[account])
        account_positions[hub] = (260.0, 170.0)
        outer = [account for account in account_order if account != hub]
        for index, account in enumerate(outer):
            angle = -pi / 2 + 2 * pi * index / max(1, len(outer))
            account_positions[account] = (260 + 205 * cos(angle), 170 + 128 * sin(angle))
    else:
        for index, account in enumerate(account_order):
            angle = -pi / 2 + 2 * pi * index / max(1, len(account_order))
            account_positions[account] = (260 + 205 * cos(angle), 170 + 128 * sin(angle))

    transaction_positions: dict[str, tuple[float, float]] = {}
    for index, transaction in enumerate(transactions):
        transaction_id = str(transaction.get("Mã giao dịch", ""))
        source = str(transaction.get("Người gửi", ""))
        target = str(transaction.get("Người nhận", ""))
        source_x, source_y = account_positions[source]
        target_x, target_y = account_positions[target]
        delta_x, delta_y = target_x - source_x, target_y - source_y
        distance = max(1.0, (delta_x**2 + delta_y**2) ** 0.5)
        spread = (index // 2 + 1) * (7 if index % 2 else -7)
        transaction_positions[transaction_id] = (
            (source_x + target_x) / 2 - delta_y / distance * spread,
            (source_y + target_y) / 2 + delta_x / distance * spread,
        )
    account_positions, transaction_positions = _separate_graph_nodes(
        account_positions,
        transaction_positions,
        degree,
    )

    relation_styles = {
        "behavior": ("#d97706", "6 4"),
        "risk": ("#7c3aed", "3 4"),
        "rule": ("#64748b", "2 5"),
    }
    relation_edges: list[str] = []
    used_relation_kinds: set[str] = set()
    for left_id, right_id, kind, strength in kept_relations:
        if (
            kind == "shared"
            or left_id not in transaction_positions
            or right_id not in transaction_positions
        ):
            continue
        left_x, left_y = transaction_positions[left_id]
        right_x, right_y = transaction_positions[right_id]
        color, dash = relation_styles[kind]
        used_relation_kinds.add(kind)
        relation_edges.append(
            f'<g><title>{kind} · {strength:.2f}</title>'
            f'<line x1="{left_x:.1f}" y1="{left_y:.1f}" x2="{right_x:.1f}" '
            f'y2="{right_y:.1f}" stroke="{color}" stroke-width="1.5" '
            f'stroke-dasharray="{dash}"/></g>'
        )

    flow_edges: list[str] = []
    transaction_nodes: list[str] = []
    candidate_accounts: set[str] = set()
    for transaction in transactions:
        transaction_id = str(transaction.get("Mã giao dịch", ""))
        tx_x, tx_y = transaction_positions[transaction_id]
        source = str(transaction.get("Người gửi", ""))
        target = str(transaction.get("Người nhận", ""))
        source_x, source_y = account_positions[source]
        target_x, target_y = account_positions[target]
        candidate = transaction_id in selected_ids
        focused = transaction_id == alert_id
        color = "#e11d48" if candidate else "#0f766e"
        if candidate:
            candidate_accounts.update((source, target))
        width = 3.5 if focused else (2.5 if candidate else 1.4)
        flow_edges.extend(
            (
                f'<line class="graph-flow" x1="{source_x:.1f}" y1="{source_y:.1f}" '
                f'x2="{tx_x:.1f}" y2="{tx_y:.1f}" stroke="{color}" '
                f'stroke-width="{width}"/>',
                f'<line class="graph-flow" x1="{tx_x:.1f}" y1="{tx_y:.1f}" '
                f'x2="{target_x:.1f}" y2="{target_y:.1f}" stroke="{color}" '
                f'stroke-width="{width}" marker-end="url(#qfg-arrow)"/>',
            )
        )
        amount = float(transaction.get("Số tiền", 0))
        focus_halo = (
            f'<rect class="graph-focus" x="{tx_x - 24:.1f}" y="{tx_y - 15:.1f}" '
            'width="48" height="30" rx="9" fill="none" stroke="#fbbf24"/>'
            if focused
            else ""
        )
        node_width = 42 if focused else 36
        node_height = 22 if focused else 18
        transaction_nodes.append(
            f'<g><title>{escape(transaction_id)} · {amount:,.0f}</title>{focus_halo}'
            f'<rect x="{tx_x - node_width / 2:.1f}" y="{tx_y - node_height / 2:.1f}" '
            f'width="{node_width}" height="{node_height}" rx="5" fill="{color}" '
            f'stroke="{"#fbbf24" if focused else "#fff"}" '
            f'stroke-width="{"2.5" if focused else "1.5"}"/>'
            f'<text x="{tx_x:.1f}" y="{tx_y + 3:.1f}" text-anchor="middle" '
            f'font-size="7.5" font-weight="700" fill="#fff">'
            f'{escape(transaction_id[-6:])}</text></g>'
        )

    account_nodes: list[str] = []
    for account in account_order:
        x, y = account_positions[account]
        suspicious = account in candidate_accounts
        radius = min(27.0, 17.0 + 2.2 * degree[account])
        stroke = "#be123c" if suspicious else "#0f766e"
        fill = "#fff1f2" if suspicious else "#ecfdf5"
        role = "GỬI" if account == sender else ("NHẬN" if account == receiver else "TK")
        account_nodes.append(
            f'<g><title>{escape(account)} · {degree[account]} cạnh</title>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="2.5"/>'
            f'<text x="{x:.1f}" y="{y - 3:.1f}" text-anchor="middle" '
            f'font-size="7.5" fill="#64748b">{role}</text>'
            f'<text x="{x:.1f}" y="{y + 9:.1f}" text-anchor="middle" '
            f'font-size="8" font-weight="700" fill="#172033">'
            f'{escape(account[-8:])}</text></g>'
        )

    legend_labels = {
        "behavior": ("#d97706", "Tương đồng hành vi"),
        "risk": ("#7c3aed", "Tương đồng Risk/Graph"),
        "rule": ("#64748b", "Chung tín hiệu rule"),
    }
    analytical_legend = "".join(
        f'<span><i class="graph-dot" style="background:{legend_labels[kind][0]}"></i>'
        f'{legend_labels[kind][1]}</span>'
        for kind in sorted(used_relation_kinds)
    )
    node_count = len(account_nodes) + len(transaction_nodes)
    support_count = len(transactions) - len(selected_ids)
    relation_label = f"{node_count} node · {len(selected_ids)} ứng viên"
    if support_count:
        relation_label += f" · {support_count} giao dịch hỗ trợ"
    return f"""<div class="fraud-graph">
    <div class="risk-viz-title" style="padding:.45rem .5rem 0">
      {escape(motif_label)}</div>
    <svg viewBox="0 0 520 340" width="100%" role="img"
      aria-label="{escape(motif_label)}">
      <defs><marker id="qfg-arrow" markerWidth="8" markerHeight="8" refX="7"
        refY="3" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L8,3 z" fill="#64748b"/></marker></defs>
      {''.join(relation_edges)}
      {''.join(flow_edges)}
      {''.join(transaction_nodes)}
      {''.join(account_nodes)}
    </svg>
    <div class="graph-legend">
      <span><i class="graph-dot" style="background:#fbbf24"></i>Giao dịch hiện tại</span>
      <span><i class="graph-dot" style="background:#e11d48"></i>Ứng viên</span>
      <span><i class="graph-dot" style="background:#0f766e"></i>Giao dịch hỗ trợ</span>
      {analytical_legend}
    </div>
    <div class="fraud-graph-label">{escape(relation_label)}</div></div>"""


def initialize_stream() -> None:
    data = realtime_source()
    scorer = copy.deepcopy(scorer_template())
    threshold = scorer.validation_stats_.get("operational_candidate_threshold")
    if threshold is not None:
        scorer.candidate_threshold_ = float(threshold)
    boundary = max(1, int(len(data) * .8))
    scorer.history_transactions_ = scorer._feature_history(
        data.iloc[max(0, boundary - REALTIME_HISTORY_LIMIT) : boundary]
    )
    st.session_state["qfg_service"] = RealtimeService.from_scorer(scorer)
    graph_source = data.iloc[max(0, boundary - GRAPH_HISTORY_LIMIT) : boundary]
    graph_rows = [
        {
            "Mã giao dịch": str(row.transaction_id),
            "Thời gian": row.timestamp,
            "Người gửi": str(row.sender_id),
            "Người nhận": str(row.receiver_id),
            "Loại": str(row.type),
            "Số tiền": float(row.amount),
            "Ứng viên": False,
        }
        for row in graph_source.itertuples(index=False)
    ]
    st.session_state.update(
        stream_cursor=boundary,
        stream_start=boundary,
        stream_processed=0,
        stream_candidates=0,
        stream_fraud_seen=0,
        stream_fraud_detected=0,
        stream_cycles=0,
        stream_sequence=0,
        stream_rows=[],
        graph_rows=graph_rows,
        stream_latency=[],
        fraud_alerts=[],
        alerts_overflow=0,
        graph_dialog_alert_id=None,
        alerts_panel_hidden=False,
        quantum_pending=[],
        quantum_window_started=None,
        quantum_future=None,
        quantum_batch_ids=(),
        quantum_status={},
        quantum_last=None,
        quantum_runs=0,
        qfg_error=None,
    )


def skip_tick() -> None:
    st.session_state["_skip_tick"] = True


def minimize_alerts() -> None:
    st.session_state["alerts_panel_hidden"] = True
    skip_tick()


def reopen_alerts() -> None:
    st.session_state["alerts_panel_hidden"] = False
    skip_tick()


def execute_alert(alert_id: str, choices: Mapping[str, str]) -> None:
    selected = st.session_state.get(f"selected_action_{alert_id}")
    action = choices.get(str(selected), next(iter(choices.values())))
    st.session_state["fraud_alerts"] = [
        item
        for item in st.session_state.get("fraud_alerts", [])
        if str(item["Mã giao dịch"]) != alert_id
    ]
    st.session_state.setdefault("alert_actions", []).append(
        {"transaction_id": alert_id, "action": action}
    )
    st.session_state["action_success"] = alert_id
    skip_tick()


def close_graph_dialog() -> None:
    st.session_state["graph_dialog_alert_id"] = None


@st.dialog(
    "Đồ thị giao dịch nghi ngờ",
    width="medium",
    on_dismiss=close_graph_dialog,
)
def render_graph_dialog(alert_id: str) -> None:
    alert = next(
        (
            item
            for item in st.session_state.get("fraud_alerts", [])
            if str(item["Mã giao dịch"]) == alert_id
        ),
        None,
    )
    if alert is None:
        st.info("Cảnh báo này đã được xử lý.")
    else:
        st.markdown(
            f"""<div class="fraud-title">RISK ALERT · {alert["Risk"]:.1%}</div>
            <div class="fraud-id">{escape(alert_id)}</div>""",
            unsafe_allow_html=True,
        )
        st.write(f"{alert['Số tiền']:,.0f} · {alert['Loại']}")
        st.markdown(
            suspicious_graph(
                alert,
                st.session_state.get("graph_rows", []),
                st.session_state.get("fraud_alerts", []),
            ),
            unsafe_allow_html=True,
        )
    if st.button("Đóng", key="close_graph_dialog", width="stretch"):
        close_graph_dialog()
        st.rerun(scope="app")


def alert_sort_key(item: Mapping[str, object]) -> tuple[int, bool, float]:
    status = st.session_state.get("quantum_status", {})
    return (
        int(item["action_priority"]),
        status.get(str(item["Mã giao dịch"])) == "selected",
        float(item["Risk"]),
    )


def sort_alerts() -> None:
    alerts = sorted(st.session_state["fraud_alerts"], key=alert_sort_key, reverse=True)
    overflow = max(0, len(alerts) - MAX_PENDING_ALERTS)
    st.session_state["alerts_overflow"] += overflow
    st.session_state["fraud_alerts"] = alerts[:MAX_PENDING_ALERTS]


def summarize_selection(selection: PrioritySelection) -> dict[str, object]:
    meta = dict(selection.metadata)
    result = selection.result
    return {
        "backend": meta.get("backend", "none"),
        "hardware": meta.get("hardware", ""),
        "n": len(selection.transaction_ids),
        "k": selection.k,
        "edges": len(selection.edges),
        "runtime_ms": (result.runtime * 1000) if result else 0.0,
        "qpu_access_ms": float(meta.get("qpu_access_time_us") or 0.0) / 1000,
        "fallback_reason": meta.get("fallback_reason"),
        "problem_id": meta.get("problem_id"),
        "feasible": bool(result.feasible) if result else True,
    }


def advance_quantum() -> None:
    """Collect a finished QUBO window, then submit the next one if it is due."""

    status: dict[str, str] = st.session_state["quantum_status"]
    future = st.session_state["quantum_future"]
    if future is not None:
        if not future.done():
            return
        st.session_state["quantum_future"] = None
        ids = st.session_state["quantum_batch_ids"]
        try:
            selection = future.result()
        except Exception as exc:  # solver errors must not stop the stream
            for transaction_id in ids:
                status[transaction_id] = "error"
            st.session_state["quantum_last"] = {
                "backend": "error",
                "fallback_reason": f"{type(exc).__name__}: {exc}"[:300],
            }
        else:
            for transaction_id in ids:
                status[transaction_id] = (
                    "selected" if transaction_id in selection.selected_ids else "not_selected"
                )
            st.session_state["quantum_last"] = summarize_selection(selection)
            st.session_state["quantum_runs"] += 1
            sort_alerts()
    pending: list[dict[str, object]] = st.session_state["quantum_pending"]
    started = st.session_state["quantum_window_started"]
    if not pending or started is None:
        return
    if len(pending) < QUANTUM_MIN_BATCH and monotonic() - started < QUANTUM_WINDOW_SECONDS:
        return
    batch, rest = pending[:QUANTUM_MAX_BATCH], pending[QUANTUM_MAX_BATCH:]
    st.session_state["quantum_pending"] = rest
    st.session_state["quantum_window_started"] = monotonic() if rest else None
    ids = tuple(str(candidate["transaction_id"]) for candidate in batch)
    for transaction_id in ids:
        status[transaction_id] = "running"
    st.session_state["quantum_batch_ids"] = ids
    st.session_state["quantum_future"] = quantum_executor().submit(
        select_priority,
        batch,
        quantum_solver(),
        name=f"window_{st.session_state['quantum_runs']:05d}",
    )


def prune_quantum_status() -> None:
    keep = {str(row["Mã giao dịch"]) for row in st.session_state["stream_rows"]}
    keep |= {str(item["Mã giao dịch"]) for item in st.session_state["fraud_alerts"]}
    keep |= {str(item["transaction_id"]) for item in st.session_state["quantum_pending"]}
    keep |= set(st.session_state["quantum_batch_ids"])
    status = st.session_state["quantum_status"]
    for transaction_id in [key for key in status if key not in keep]:
        del status[transaction_id]


def process_tick() -> None:
    data = realtime_source()
    cursor = int(st.session_state["stream_cursor"])
    if cursor >= len(data):
        st.session_state["stream_cycles"] += 1
        cursor = int(st.session_state["stream_start"])
        st.session_state["stream_cursor"] = cursor
        service().pipeline.scorer.history_transactions_ = (
            service().pipeline.scorer._feature_history(
                data.iloc[max(0, cursor - REALTIME_HISTORY_LIMIT) : cursor]
            )
        )
    stop = min(len(data), cursor + random.randint(6, 24))
    batch = data.iloc[cursor:stop].reset_index(drop=True).copy()
    sequence = int(st.session_state["stream_sequence"])
    cycle = int(st.session_state["stream_cycles"])
    batch["transaction_id"] = [
        f"TX-{cycle:03d}-{sequence + offset:010d}" for offset in range(len(batch))
    ]
    batch["timestamp"] = pd.Timestamp.now(tz="UTC") + pd.to_timedelta(
        range(len(batch)), unit="ms"
    )
    st.session_state["stream_sequence"] += len(batch)
    started = perf_counter()
    scored = pd.DataFrame(service().score_transactions(batch))
    scorer = service().pipeline.scorer
    if scorer.history_transactions_ is not None:
        scorer.history_transactions_ = (
            scorer.history_transactions_.iloc[-REALTIME_HISTORY_LIMIT:].reset_index(drop=True)
        )
    elapsed = (perf_counter() - started) * 1000 / max(1, len(scored))
    hold_threshold = float(scorer.validation_stats_.get("alert_hold_threshold", .6))
    verify_threshold = float(scorer.candidate_threshold_)
    rows = list(st.session_state["stream_rows"])
    graph_rows = list(st.session_state.get("graph_rows", []))
    alerts = list(st.session_state["fraud_alerts"])
    pending = st.session_state["quantum_pending"]
    status = st.session_state["quantum_status"]
    for _, row in scored.iterrows():
        is_candidate = bool(row["is_candidate"])
        is_fraud = int(row.get("is_fraud", 0))
        risk = float(row["risk_score"])
        st.session_state["stream_processed"] += 1
        st.session_state["stream_candidates"] += int(is_candidate)
        st.session_state["stream_fraud_seen"] += is_fraud
        st.session_state["stream_fraud_detected"] += int(is_fraud and is_candidate)
        record = {
            "Thời gian": row["timestamp"],
            "Mã giao dịch": row["transaction_id"],
            "Người gửi": str(row["sender_id"]),
            "Người nhận": str(row["receiver_id"]),
            "Loại": row["type"],
            "Số tiền": float(row["amount"]),
            "Risk": risk,
            "Supervised": float(row.get("supervised_score", 0)),
            "Anomaly": float(row.get("anomaly_score", 0)),
            "Graph": float(row.get("graph_score", 0)),
            "Rule": float(row.get("rule_score", 0)),
            "Ứng viên": is_candidate,
            "Nhãn fraud": bool(is_fraud),
            "Lý do": row.get("reason_codes", ""),
        }
        rows.append(record)
        graph_rows.append(record)
        if is_candidate:
            treatment = recommended_treatment(
                {"risk_score": risk, "reason_codes": record["Lý do"]},
                hold_threshold=hold_threshold,
                verify_threshold=verify_threshold,
            )
            priority = {"hold_and_escalate": 3, "step_up_verification": 2}.get(
                treatment["recommended_action"], 1
            )
            alerts.append({**record, **treatment, "action_priority": priority})
            transaction_id = str(row["transaction_id"])
            pending.append(
                {
                    "transaction_id": transaction_id,
                    "risk_score": risk,
                    "sender_id": str(row["sender_id"]),
                    "receiver_id": str(row["receiver_id"]),
                }
            )
            status[transaction_id] = "pending"
            if st.session_state["quantum_window_started"] is None:
                st.session_state["quantum_window_started"] = monotonic()
    st.session_state["fraud_alerts"] = alerts
    sort_alerts()
    st.session_state["stream_rows"] = rows[-200:]
    st.session_state["graph_rows"] = graph_rows[-GRAPH_HISTORY_LIMIT:]
    st.session_state["stream_latency"] = (st.session_state["stream_latency"] + [elapsed])[-100:]
    st.session_state["stream_cursor"] = stop
    advance_quantum()
    prune_quantum_status()


def quantum_card() -> tuple[str, str, str, str]:
    solver = quantum_solver()
    last = st.session_state.get("quantum_last")
    configured = os.environ.get("QFG_SOLVER", "sqa")
    if not last:
        return ("Bộ giải QUBO", "Chờ batch đầu", f"QFG_SOLVER={configured}", "qfg-quantum")
    label = BACKEND_LABELS.get(str(last["backend"]), str(last["backend"]))
    if last.get("fallback_reason"):
        label = f"{label} · fallback"
    parts = [f"{st.session_state['quantum_runs']} batch"]
    if last.get("n"):
        parts.append(f"chọn {last['k']}/{last['n']}")
    if last.get("qpu_access_ms"):
        parts.append(f"QPU {last['qpu_access_ms']:.1f} ms")
    else:
        parts.append(f"{last.get('runtime_ms', 0):.0f} ms")
    if isinstance(solver, FallbackSolver):
        stats = solver.stats()
        budget = stats["qpu_budget_seconds"]
        parts.append(
            f"QPU đã dùng {stats['qpu_seconds_used']:.2f}"
            + (f"/{budget:.0f} s" if budget is not None else " s")
        )
    if last.get("fallback_reason"):
        parts.append(f"lý do: {str(last['fallback_reason'])[:80]}")
    elif last.get("hardware"):
        parts.append(str(last["hardware"]))
    return ("Bộ giải QUBO", label, " · ".join(parts), "qfg-quantum")


def render_alerts() -> None:
    completed_alert = st.session_state.pop("action_success", None)
    if completed_alert:
        st.toast(f"Đã thực hiện hành động cho {completed_alert}", icon="✅")
    alerts = st.session_state.get("fraud_alerts", [])
    if not alerts:
        return
    if st.session_state.get("alerts_panel_hidden"):
        with st.container(key="fraud_alert_reopen"):
            st.button(f"🔔 {len(alerts)}", on_click=reopen_alerts)
        return
    action_labels = {
        "hold_and_escalate": "Tạm giữ và chuyển điều tra",
        "step_up_verification": "Yêu cầu xác minh bổ sung",
        "monitor_and_review": "Tiếp tục giám sát",
    }
    alternatives = {
        "Tạm giữ giao dịch": "hold_transaction",
        "Yêu cầu xác minh": "request_verification",
        "Chuyển điều tra": "escalate_to_investigator",
        "Từ chối giao dịch": "reject_transaction",
        "Tiếp tục giám sát": "continue_monitoring",
    }
    with st.container(key="fraud_alert_stack"):
        heading, control = st.columns([6, 1])
        heading.markdown(f"**Cảnh báo cần xử lý · {len(alerts)}**")
        if overflow := int(st.session_state.get("alerts_overflow", 0)):
            st.caption(
                f"{overflow} cảnh báo ưu tiên thấp đã bị loại do hàng đợi vượt "
                f"{MAX_PENDING_ALERTS} mục."
            )
        quantum_status = st.session_state.get("quantum_status", {})
        control.button("−", key="hide_fraud_alerts", on_click=minimize_alerts)
        for alert in alerts[:MAX_VISIBLE_ALERTS]:
            alert_id = str(alert["Mã giao dịch"])
            recommended = str(alert["recommended_action"])
            choices = {
                f"Khuyến nghị · {action_labels.get(recommended, recommended)}": recommended,
                **{f"Hành động khác · {label}": action for label, action in alternatives.items()},
            }
            with st.container(border=True):
                st.markdown(
                    f"""<div class="fraud-title">RISK ALERT · {alert["Risk"]:.1%}</div>
                    <div class="fraud-id">{escape(alert_id)}</div>""",
                    unsafe_allow_html=True,
                )
                st.write(f"{alert['Số tiền']:,.0f} · {alert['Loại']}")
                state = quantum_status.get(alert_id)
                if state == "selected":
                    st.markdown(":violet-badge[⚛ Được QUBO chọn ưu tiên]")
                elif state in ("pending", "running"):
                    st.markdown(f":gray-badge[{QUANTUM_STATUS_LABELS[state]}]")
                st.markdown(risk_visualization(alert), unsafe_allow_html=True)
                if st.button(
                    "Mở đồ thị",
                    key=f"open_graph_{alert_id}",
                    width="content",
                ):
                    st.session_state["graph_dialog_alert_id"] = alert_id
                    skip_tick()
                    st.rerun(scope="app")
                st.markdown(f"**Khuyến nghị:** {action_labels.get(recommended, recommended)}")
                st.info(str(alert["action_detail"]))
                st.selectbox(
                    "Hành động xử lý",
                    tuple(choices),
                    key=f"selected_action_{alert_id}",
                    on_change=skip_tick,
                )
                st.button(
                    "Thực thi hành động",
                    key=f"execute_{alert_id}",
                    type="primary",
                    width="stretch",
                    on_click=execute_alert,
                    args=(alert_id, choices),
                )


st.markdown(
    """<div class="qfg-header"><div class="qfg-logo">QF</div>
    <div class="qfg-name">Q-FraudGuard</div>
    <div class="qfg-live"><span class="qfg-dot"></span>Real-time processing</div></div>""",
    unsafe_allow_html=True,
)

if "stream_cursor" not in st.session_state and not st.session_state.get("qfg_error"):
    try:
        initialize_stream()
    except Exception as exc:
        st.session_state["qfg_error"] = str(exc)

if st.session_state.get("qfg_error"):
    st.error(f"Không thể xử lý luồng giao dịch: {st.session_state['qfg_error']}")


@st.fragment(run_every=1.0)
def realtime_panel() -> None:
    if "stream_cursor" not in st.session_state:
        return
    skip = st.session_state.pop("_skip_tick", False)
    if not skip:
        process_tick()
    processed = int(st.session_state["stream_processed"])
    candidates = int(st.session_state["stream_candidates"])
    fraud_seen = int(st.session_state["stream_fraud_seen"])
    fraud_detected = int(st.session_state["stream_fraud_detected"])
    latency = st.session_state["stream_latency"]
    values = (
        ("Đã xử lý", f"{processed:,}".replace(",", "."), "Giao dịch real-time", ""),
        ("Ứng viên", f"{candidates:,}".replace(",", "."), "Qua ngưỡng rủi ro", "qfg-warn"),
        (
            "Tỷ lệ bắt được fraud",
            f"{fraud_detected / fraud_seen:.1%}" if fraud_seen else "0.0%",
            f"{fraud_detected}/{fraud_seen} fraud được phát hiện",
            "qfg-risk",
        ),
        (
            "Latency",
            f"{sum(latency) / len(latency):.1f} ms" if latency else "0.0 ms",
            "Trung bình mỗi giao dịch",
            "",
        ),
        quantum_card(),
    )
    for column, value in zip(st.columns(5), values):
        with column:
            metric_card(*value)
    rows = st.session_state["stream_rows"]
    if rows:
        frame = pd.DataFrame(rows).iloc[::-1].copy()
        frame["Nhãn thật"] = frame["Nhãn fraud"].map({True: "FRAUD", False: "NORMAL"})
        status = st.session_state["quantum_status"]
        frame["Ưu tiên QUBO"] = [
            QUANTUM_STATUS_LABELS.get(status.get(str(transaction_id), ""), "")
            for transaction_id in frame["Mã giao dịch"]
        ]
        view = frame[
            [
                "Thời gian",
                "Mã giao dịch",
                "Loại",
                "Số tiền",
                "Risk",
                "Ứng viên",
                "Ưu tiên QUBO",
                "Nhãn thật",
            ]
        ]
        st.dataframe(
            view,
            hide_index=True,
            width="stretch",
            height=520,
            column_config={
                "Số tiền": st.column_config.NumberColumn(format="%,.0f"),
                "Risk": st.column_config.ProgressColumn(
                    "Risk score", min_value=0.0, max_value=1.0, format="%.3f", color=RED
                ),
                "Ứng viên": st.column_config.CheckboxColumn(),
                "Ưu tiên QUBO": st.column_config.TextColumn(
                    help="Kết quả bài toán chọn K hồ sơ ưu tiên (QUBO) theo từng cửa sổ"
                ),
            },
        )


@st.fragment(run_every=1.0)
def alert_panel() -> None:
    render_alerts()


realtime_panel()
alert_panel()
if graph_alert_id := st.session_state.get("graph_dialog_alert_id"):
    render_graph_dialog(str(graph_alert_id))

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from html import escape
from math import ceil, cos, pi, sin
import os
from pathlib import Path
import sqlite3
import sys
from time import perf_counter
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd
import streamlit as st

from qf_guard.data import normalize_transactions
from qf_guard.operations import CaseStore, EventProcessor
from qf_guard.prioritization import PriorityConfig, PrioritySelection, select_priority
from qf_guard.quafu import sampling_time_estimate
from qf_guard.solvers import FallbackSolver, solver_from_env

RED = "#d92d4f"
MAX_VISIBLE_ALERTS = 30
GRAPH_HISTORY_LIMIT = 2_000
QUANTUM_HISTORY_LIMIT = 500
# QFG_DATASET picks the stream and its scorer.  AMLworld (synthetic, IBM) keeps
# account identities, so candidates in a window share accounts and the QUBO
# has real pairwise terms; PaySim candidates almost never do.
DATASETS = {
    "paysim": ("PaySim (mô phỏng)", "paysim.csv", "paysim_lightgbm_scorer.joblib",
               "python scripts/train.py --output artifacts/models/paysim_lightgbm_scorer.joblib"),
    "amlworld": ("IBM AMLworld HI-Small (tổng hợp)", "amlworld_hi_small.csv",
                 "amlworld_lightgbm_scorer.joblib",
                 "python scripts/prepare_amlworld.py && python scripts/train.py --data "
                 "data/amlworld_hi_small.csv --dataset-name amlworld --output "
                 "artifacts/models/amlworld_lightgbm_scorer.joblib"),
    # new scorer (4 days, confirmed-account list, no payment format, chosen on
    # validation); its stream (last 20% of the 4-day sample) is its untouched test period
    "amlworld_tuned": ("IBM AMLworld HI-Small 4 ngày · model mới (tổng hợp)",
                       "amlworld_hi_small_4d.csv", "amlworld_noformat_tuned_scorer.joblib",
                       "python scripts/prepare_amlworld.py --days 4 --output "
                       "data/amlworld_hi_small_4d.csv && python scripts/train_amlworld.py "
                       "--exclude-groups formats --output "
                       "artifacts/models/amlworld_noformat_tuned_scorer.joblib"),
    # trained on all normal days of HI-Small; threshold set for >= 95% recall; the
    # stream file holds its untouched test period (rows with split == "stream")
    "amlworld_full": ("IBM AMLworld HI-Small toàn bộ · ngưỡng bắt ≥ 95% (tổng hợp)",
                      "amlworld_hi_small_full_stream.csv", "amlworld_full_scorer.joblib",
                      "python scripts/train_amlworld_full.py --target-recall 0.96 "
                      "--stream-output data/amlworld_hi_small_full_stream.csv"),
}
DATASET = os.environ.get("QFG_DATASET", "amlworld_full").strip().lower()
DATASET_LABEL, DATA_FILE, MODEL_FILE, PREPARE_HINT = DATASETS.get(
    DATASET, (DATASET, "", "", f"QFG_DATASET phải là một trong: {', '.join(DATASETS)}")
)
MODEL_PATH = Path(
    os.environ.get("QFG_MODEL_PATH", PROJECT_ROOT / "artifacts" / "models" / MODEL_FILE)
)
STREAM_PATH = PROJECT_ROOT / "data" / DATA_FILE
SOURCE_MODE = os.environ.get("QFG_SOURCE", "replay").strip().lower()
STREAM_NAME = os.environ.get("QFG_STREAM") or f"{SOURCE_MODE}-{DATASET}"
CASE_DB = Path(os.environ.get("QFG_CASE_DB") or PROJECT_ROOT / "artifacts" / "operations.sqlite3")
CONFIGURED_SOLVER = os.environ.get("QFG_SOLVER", "").strip().lower()
REQUIRE_MEASURED_OPTIMUM = CONFIGURED_SOLVER.startswith("quafu-fast-")
DEFAULT_POLICY = "qubo" if CONFIGURED_SOLVER not in ("", "sqa") else "risk"
PRIORITY_MODE = os.environ.get("QFG_PRIORITY_MODE", DEFAULT_POLICY).strip().lower()
# AMLworld validation settings. Each shared-account pair adds 0.01 * 0.5 after scaling.
PRIORITY_CONFIG = PriorityConfig(
    connectivity_strength=float(os.environ.get("QFG_CONNECTIVITY_STRENGTH", "0.01")),
    risk_normalization=os.environ.get("QFG_RISK_NORMALIZATION", "max").strip().lower() or "max",
)
# Candidates are grouped into windows so each QPU call carries a meaningful
# QUBO and the per-call network latency does not throttle the 1 s UI tick.
QUAFU_BATCH_SIZE = "18" if REQUIRE_MEASURED_OPTIMUM else "8"
QUANTUM_MIN_BATCH = int(os.environ.get("QFG_QUANTUM_MIN_BATCH", QUAFU_BATCH_SIZE))
QUANTUM_MAX_BATCH = int(
    os.environ.get(
        "QFG_QUANTUM_MAX_BATCH",
        QUAFU_BATCH_SIZE if CONFIGURED_SOLVER.startswith("quafu") else "40",
    )
)
QUANTUM_WINDOW_SECONDS = float(os.environ.get("QFG_QUANTUM_WINDOW_SECONDS", "10"))
BACKEND_LABELS = {
    "dwave-qpu": "D-Wave QPU",
    "dwave-mock": "D-Wave (mô phỏng cục bộ)",
    "dwave-hybrid": "D-Wave Hybrid",
    "quafu-qpu": "Quafu QPU",
    "quafu-mock": "Quafu (mô phỏng cục bộ)",
    "quafu-anneal-qpu": "Quafu QPU · Ủ lượng tử số hóa",
    "quafu-anneal-mock": "Quafu mô phỏng · Ủ lượng tử số hóa",
    "quafu-fast-qpu": "Quafu QPU · Khởi tạo theo rủi ro",
    "quafu-fast-mock": "Quafu mô phỏng · Khởi tạo theo rủi ro",
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
def case_store(path: Path, name: str) -> CaseStore:
    return CaseStore(path, name)


def ledger() -> CaseStore:
    return case_store(CASE_DB, STREAM_NAME)


@st.cache_resource(show_spinner=False)
def replay_processor(path: Path, database: Path, name: str) -> EventProcessor:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy model: {path}. Tạo bằng: {PREPARE_HINT}")
    return EventProcessor.load(case_store(database, name), path)


@st.cache_resource(show_spinner=False)
def quantum_solver() -> Any:
    """Process-wide solver chosen by QFG_SOLVER (QPU budget is shared too)."""

    return solver_from_env()


@st.cache_resource(show_spinner=False)
def quantum_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="qfg-quantum")


@st.cache_resource(show_spinner=False)
def realtime_source(path: Path) -> tuple[pd.DataFrame, int]:
    if DATASET not in DATASETS:
        raise ValueError(PREPARE_HINT)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy nguồn luồng: {path}. Tạo bằng: {PREPARE_HINT}")
    data = normalize_transactions(pd.read_csv(path))
    boundary = int(len(data) * .8)
    if "split" in data and data["split"].eq("stream").any():
        boundary = int(data["split"].eq("stream").to_numpy().argmax())
    return data, boundary


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
    if SOURCE_MODE not in ("live", "replay") or PRIORITY_MODE not in ("risk", "qubo"):
        raise ValueError("QFG_SOURCE: live/replay; QFG_PRIORITY_MODE: risk/qubo")
    if PRIORITY_MODE == "qubo" and not (0 < QUANTUM_MIN_BATCH <= QUANTUM_MAX_BATCH and QUANTUM_WINDOW_SECONDS >= 0):
        raise ValueError("Cấu hình kích thước hoặc thời gian cửa sổ QUBO không hợp lệ")
    ledger()
    if SOURCE_MODE == "replay":
        realtime_source(STREAM_PATH)
        replay_processor(MODEL_PATH, CASE_DB, STREAM_NAME)
    st.session_state.update(
        stream_ready=True, stream_latency=[], fraud_alerts=[], alerts_panel_hidden=False,
        graph_dialog_alert_id=None, quantum_future=None, quantum_last=None,
        quantum_history=[], quantum_runs=0,
    )


def display_record(event):
    return {
        "Thời gian": pd.Timestamp(event["timestamp"]), "Mã giao dịch": event["transaction_id"],
        "Người gửi": event["sender_id"], "Người nhận": event["receiver_id"],
        "Loại": event["type"], "Số tiền": event["amount"], "Risk": event.get("risk_score", 0.),
        "Supervised": event.get("supervised_score", 0.), "Anomaly": event.get("anomaly_score", 0.),
        "Graph": event.get("graph_score", 0.), "Rule": event.get("rule_score", 0.),
        "Ứng viên": event.get("is_candidate", False), "Nhãn fraud": event.get("is_fraud"),
        "Lý do": event.get("reason_codes", ""), "priority_status": event.get("priority_status"),
        "manual_review": event.get("manual_review", False),
        "recommended_action": event.get("recommended_action"), "action_detail": event.get("action_detail"),
    }


def graph_history(txid):
    history = ledger().recent(GRAPH_HISTORY_LIMIT, through=txid)
    missing = GRAPH_HISTORY_LIMIT - len(history)
    if missing and SOURCE_MODE == "replay":
        data, boundary = realtime_source(STREAM_PATH)
        history = data.iloc[max(0, boundary - missing):boundary].to_dict("records") + history
    return [display_record(row) for row in history]


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
    # Local operator identity can be supplied by the deployment or Streamlit login.
    actor = st.user.get("email") or os.environ.get("QFG_OPERATOR") or "local-operator"
    recorded = ledger().decide(alert_id, action, actor)
    st.session_state["action_notice"] = (
        f"Đã lưu quyết định cho {alert_id}" if recorded else f"{alert_id} đã được xử lý"
    )
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
                graph_history(alert_id),
                st.session_state.get("fraud_alerts", []),
            ),
            unsafe_allow_html=True,
        )
    if st.button("Đóng", key="close_graph_dialog", width="stretch"):
        close_graph_dialog()
        st.rerun(scope="app")




def summarize_selection(
    selection: PrioritySelection, batch: list[Mapping[str, object]] | None = None
) -> dict[str, object]:
    meta = dict(selection.metadata)
    result = selection.result
    cert = dict(selection.certificate or {})
    summary: dict[str, object] = {
        "backend": meta.get("backend", "none"),
        "hardware": meta.get("hardware", ""),
        "n": len(selection.transaction_ids),
        "k": selection.k,
        "edges": len(selection.edges),
        "runtime_ms": (result.runtime * 1000) if result else 0.0,
        "qpu_access_ms": float(meta.get("qpu_access_time_us") or 0.0) / 1000,
        "server_time_s": meta.get("server_time_s"),
        "task_id": meta.get("task_id"),
        "fallback_reason": meta.get("fallback_reason"),
        "problem_id": meta.get("problem_id"),
        "feasible": bool(result.feasible) if result else True,
        "certified": bool(cert.get("certified")),
        "matches_optimum": cert.get("matches_optimum"),
        "raw_matches_optimum": cert.get("raw_matches_optimum"),
        "final_gap": cert.get("final_gap"),
        "raw_gap": cert.get("raw_gap"),
        "certify_ms": cert.get("certify_ms"),
        "measured_hit": cert.get("measured_hit"),
        "optimal_reads": cert.get("optimal_reads"),
        "measured_reads": cert.get("measured_reads"),
        "time_to_first_hit_s": cert.get("time_to_first_hit_s"),
    }
    if batch and selection.k < len(batch):
        # classical reference: the K highest risk scores in the same window
        by_risk = sorted(range(len(batch)), key=lambda i: -float(batch[i]["risk_score"]))
        top_ids = {str(batch[i]["transaction_id"]) for i in by_risk[: selection.k]}
        summary["differs_from_topk"] = top_ids != set(selection.selected_ids)
        if all(c.get("is_fraud") is not None for c in batch):
            fraud = {str(c["transaction_id"]): int(c["is_fraud"]) for c in batch}
            summary.update(fraud_selected=sum(fraud[i] for i in selection.selected_ids),
                           fraud_topk=sum(fraud[i] for i in top_ids))
    return summary


def solve_window(store, lease, batch, solver):
    try:
        selection = select_priority(batch, solver, PRIORITY_CONFIG, name=f"window_{lease}", certify_optimum=True,
                                    max_measurement_jobs=None if REQUIRE_MEASURED_OPTIMUM else 3)
        if REQUIRE_MEASURED_OPTIMUM and selection.result is not None:
            primary_hit = (selection.metadata.get("backend") == CONFIGURED_SOLVER
                           and not selection.metadata.get("fallback_reason")
                           and (selection.certificate or {}).get("measured_hit"))
            if not primary_hit:
                raise RuntimeError(selection.metadata.get("fallback_reason") or "Chưa nhận được mẫu đo tối ưu từ bộ giải đã chọn")
    except Exception:
        store.finish_window(lease, None)
        raise
    store.finish_window(lease, selection.selected_ids)
    return summarize_selection(selection, batch)


def advance_quantum(*, flush=False):
    state = st.session_state
    future = state["quantum_future"]
    if future is not None:
        if not future.done():
            return
        state["quantum_future"] = None
        try:
            summary = future.result()
        except Exception as exc:
            summary = {"backend": "error", "fallback_reason": f"{type(exc).__name__}: {exc}"[:300]}
        else:
            state["quantum_runs"] += 1
        if summary.get("certified") or (REQUIRE_MEASURED_OPTIMUM and summary["backend"] == "error"):
            state["quantum_history"] = (state["quantum_history"] + [summary])[-QUANTUM_HISTORY_LIMIT:]
        state["quantum_last"] = summary
    solver = quantum_solver()
    store = ledger()
    window = store.claim_window(QUANTUM_MIN_BATCH, QUANTUM_MAX_BATCH, QUANTUM_WINDOW_SECONDS, flush=flush)
    if window:
        lease, batch = window
        state["quantum_future"] = quantum_executor().submit(solve_window, store, lease, batch, solver)


def process_tick():
    finished = False
    if SOURCE_MODE == "replay":
        data, boundary = realtime_source(STREAM_PATH)
        offset = ledger().stats()["processed"]
        start = boundary + offset
        if start < len(data):
            started = perf_counter()
            processor = replay_processor(MODEL_PATH, CASE_DB, STREAM_NAME)
            inserted = processor.ingest(data.iloc[start:start + 32], expected_offset=offset)
            if inserted:
                elapsed = (perf_counter() - started) * 1000 / inserted
                st.session_state["stream_latency"] = (st.session_state["stream_latency"] + [elapsed])[-100:]
        finished = start + 32 >= len(data)
    if PRIORITY_MODE == "qubo":
        advance_quantum(flush=finished)
    return finished


def quantum_card() -> tuple[str, str, str, str]:
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
    if REQUIRE_MEASURED_OPTIMUM:
        if last.get("fallback_reason"):
            return ("Bộ giải QUBO", "Lỗi bộ giải", last["fallback_reason"], "qfg-quantum")
        if last.get("measured_hit") is not None:
            outcome = "Đã đo được nghiệm tối ưu" if last["measured_hit"] else "Chưa đo được nghiệm tối ưu"
            parts.append(f"{last['optimal_reads']}/{last['measured_reads']} lần đo đạt tối ưu")
            if last.get("time_to_first_hit_s") is not None:
                parts.append(f"máy khách nhận lô có nghiệm tối ưu sau {last['time_to_first_hit_s']:.2f} s")
            estimate = sampling_time_estimate(last["measured_reads"])
            if estimate:
                parts.append(f"lấy mẫu ước tính {estimate['estimated_sampling_ms']:.1f} ms "
                             f"(giả định {estimate['assumed_shot_period_us']:g} µs/phép đo)")
            return (str(last.get("hardware") or label), outcome, " · ".join(parts), "qfg-quantum")
        return ("Bộ giải QUBO", label, " · ".join(parts), "qfg-quantum")
    solver = quantum_solver()
    if last.get("server_time_s"):
        parts.append(f"Quafu {float(last['server_time_s']):.1f} s/job")
    elif last.get("qpu_access_ms"):
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
    if last.get("measured_hit") is not None:
        parts.append(f"{last['optimal_reads']}/{last['measured_reads']} lần đo đạt tối ưu")
        if last.get("time_to_first_hit_s") is not None:
            parts.append(f"nhận đáp án sau {last['time_to_first_hit_s']:.2f} s")
    elif last.get("matches_optimum") is True:
        parts.append("khớp tối ưu MILP ✓")
    elif last.get("final_gap") is not None:
        parts.append(f"lệch tối ưu {float(last['final_gap']):.1%}")
    if last.get("fallback_reason"):
        parts.append(f"lý do: {str(last['fallback_reason'])[:80]}")
    elif last.get("hardware"):
        parts.append(str(last["hardware"]))
    return ("Bộ giải QUBO", label, " · ".join(parts), "qfg-quantum")


def _share(values: list[object]) -> str:
    known = [bool(v) for v in values if v is not None]
    return f"{sum(known) / len(known):.0%}" if known else "—"


def verification_panel() -> None:
    """Honest per-window check: solver vs certified optimum vs plain top-K sort."""

    history = st.session_state.get("quantum_history", [])
    if not history:
        st.caption(
            f"Dữ liệu: {DATASET_LABEL} · trọng số liên kết c = "
            f"{PRIORITY_CONFIG.connectivity_strength:g} (chuẩn hóa: "
            f"{PRIORITY_CONFIG.risk_normalization}) · chờ cửa sổ đầu tiên để kiểm chứng tối ưu"
        )
        return
    if REQUIRE_MEASURED_OPTIMUM:
        hits = sum(h.get("measured_hit") is True for h in history)
        st.caption(f"**{hits}/{len(history)} cụm đã kết thúc có mẫu đo đạt nghiệm tối ưu QUBO.** "
                   "CPU kiểm chứng giá trị tối ưu bằng MILP.")
        with st.expander("Chi tiết 10 cụm gần nhất"):
            rows = []
            for h in reversed(history[-10:]):
                estimate = sampling_time_estimate(h.get("measured_reads"))
                rows.append({
                    "Ứng viên": h.get("n"), "Hồ sơ chọn": h.get("k"),
                    "Đo được nghiệm tối ưu": h.get("measured_hit") is True,
                    "Mẫu tối ưu": h.get("optimal_reads"), "Tổng phép đo": h.get("measured_reads"),
                    "Máy khách nhận lô đạt (s)": h.get("time_to_first_hit_s"),
                    "Lấy mẫu ước tính (ms)": estimate.get("estimated_sampling_ms"),
                    "Chu kỳ giả định (µs)": estimate.get("assumed_shot_period_us"),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        return
    linked = sum(1 for h in history if int(h.get("edges", 0)) > 0)
    raw_gaps = [float(h["raw_gap"]) for h in history if h.get("raw_gap") is not None]
    differs = [h.get("differs_from_topk") for h in history if "differs_from_topk" in h]
    labelled = [h for h in history if "fraud_selected" in h]
    fraud_qubo = sum(h["fraud_selected"] for h in labelled)
    fraud_topk = sum(h["fraud_topk"] for h in labelled)
    label_summary = (f"QUBO {fraud_qubo}, theo điểm {fraud_topk} trên {len(labelled)} cửa sổ có nhãn"
                     if labelled else "chưa có cửa sổ đủ nhãn")
    st.caption(
        f"**Kiểm chứng tối ưu (MILP, CPU)** · Dữ liệu: {DATASET_LABEL} · c = "
        f"{PRIORITY_CONFIG.connectivity_strength:g} ({PRIORITY_CONFIG.risk_normalization}) · "
        f"{len(history)} cửa sổ, {linked} có liên kết · "
        f"solver khớp tối ưu: {_share([h.get('matches_optimum') for h in history])} "
        f"(trước hậu xử lý: {_share([h.get('raw_matches_optimum') for h in history])}, "
        f"lệch TB {sum(raw_gaps) / len(raw_gaps) if raw_gaps else 0:.1%}) · "
        f"QUBO khác sắp xếp theo điểm ở {sum(bool(d) for d in differs)}/{len(differs)} cửa sổ · "
        f"fraud được ưu tiên: {label_summary}"
    )
    with st.expander("Chi tiết 10 cửa sổ gần nhất"):
        table = pd.DataFrame(
            [
                {
                    "Backend": BACKEND_LABELS.get(str(h.get("backend")), str(h.get("backend"))),
                    "N": h.get("n"),
                    "K": h.get("k"),
                    "Liên kết": h.get("edges"),
                    "Lệch trước hậu xử lý": h.get("raw_gap"),
                    "Lệch sau hậu xử lý": h.get("final_gap"),
                    "Khớp tối ưu": bool(h.get("matches_optimum")),
                    "Lần đo đạt tối ưu": h.get("optimal_reads"),
                    "Nhận đáp án tối ưu (s)": h.get("time_to_first_hit_s"),
                    "Khác sắp xếp": bool(h.get("differs_from_topk")),
                    "Fraud QUBO / sắp xếp": f"{h.get('fraud_selected', '—')} / {h.get('fraud_topk', '—')}",
                    "MILP (ms)": h.get("certify_ms"),
                }
                for h in reversed(history[-10:])
            ]
        )
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "Lệch trước hậu xử lý": st.column_config.NumberColumn(format="percent"),
                "Lệch sau hậu xử lý": st.column_config.NumberColumn(format="percent"),
                "MILP (ms)": st.column_config.NumberColumn(format="%.1f"),
            },
        )


def render_alerts() -> None:
    if not st.session_state.get("stream_ready"):
        return
    notice = st.session_state.pop("action_notice", None)
    if notice:
        st.toast(notice, icon="✅")
    show_all = st.toggle("Xem cả ứng viên dưới ngưỡng điều tra", key="show_all_candidates", on_change=skip_tick)
    query = dict(all_candidates=show_all, use_qubo=PRIORITY_MODE == "qubo", limit=MAX_VISIBLE_ALERTS)
    total, records = ledger().queue(**query)
    pages = ceil(total / MAX_VISIBLE_ALERTS)
    if pages > 1:
        if st.session_state.get("queue_page", 1) > pages:
            st.session_state["queue_page"] = pages
        page = st.number_input("Trang hồ sơ", 1, pages, key="queue_page", on_change=skip_tick)
        if page > 1:
            _, records = ledger().queue(**query, offset=(page - 1) * MAX_VISIBLE_ALERTS)
    alerts = [display_record(row) for row in records]
    st.session_state["fraud_alerts"] = alerts
    if not alerts:
        return
    if st.session_state.get("alerts_panel_hidden"):
        with st.container(key="fraud_alert_reopen"):
            st.button(f"🔔 {total}", on_click=reopen_alerts)
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
        heading.markdown(f"**Hồ sơ chờ xử lý · {total}**")
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
                state = alert["priority_status"] if PRIORITY_MODE == "qubo" else None
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
                    "Ghi quyết định",
                    key=f"execute_{alert_id}",
                    type="primary",
                    width="stretch",
                    on_click=execute_alert,
                    args=(alert_id, choices),
                )


st.markdown(
    f"""<div class="qfg-header"><div class="qfg-logo">QF</div>
    <div class="qfg-name">Q-FraudGuard</div>
    <div class="qfg-note">· {escape(DATASET_LABEL if SOURCE_MODE == "replay" else STREAM_NAME)}</div>
    <div class="qfg-live"><span class="qfg-dot"></span>{"Phát lại CSV" if SOURCE_MODE == "replay" else "Luồng giao dịch"}</div></div>""",
    unsafe_allow_html=True,
)

if not st.session_state.get("stream_ready"):
    try:
        initialize_stream()
    except (ValueError, OSError, sqlite3.Error) as exc:
        st.error(str(exc))
        st.stop()


@st.fragment(run_every=1.0)
def realtime_panel() -> None:
    if not st.session_state.pop("_skip_tick", False):
        try:
            st.session_state["source_finished"] = process_tick()
        except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
            st.error(f"Luồng dừng tại giao dịch chưa ghi. {exc}")
            return
    stats = ledger().stats()
    frauds, caught, hits = stats["frauds"], stats["detected"], stats["review_hits"]
    recall = f"{caught / frauds:.1%}" if frauds else "—"
    values = [
        ("Đã xử lý", f"{stats['processed']:,}", STREAM_NAME, ""),
        ("Ứng viên", f"{stats['candidates']:,}", "Qua ngưỡng sàng lọc", "qfg-warn"),
        ("Đủ ngưỡng điều tra", f"{stats['reviews']:,}", "Hồ sơ ưu tiên kiểm tra thủ công", ""),
        ("Recall sàng lọc", recall, f"{caught}/{frauds} giao dịch gian lận đã có nhãn", "qfg-risk"),
    ]
    if PRIORITY_MODE == "qubo":
        values.append(quantum_card())
    for column, value in zip(st.columns(len(values)), values):
        with column:
            metric_card(*value)
    if frauds:
        st.caption(f"Tầng điều tra giữ {hits}/{frauds} giao dịch gian lận ({hits / frauds:.1%}).")
    latency = st.session_state["stream_latency"]
    if latency:
        st.caption(f"Thời gian chấm và ghi dữ liệu trung bình {sum(latency) / len(latency):.1f} ms/giao dịch.")
    if st.session_state.get("source_finished"):
        st.caption("Đã phát hết giao dịch trong tập kiểm tra.")
    if PRIORITY_MODE == "qubo":
        verification_panel()
    rows = [display_record(row) for row in ledger().recent()]
    if rows:
        frame = pd.DataFrame(rows).iloc[::-1].copy()
        frame["Nhãn thật"] = frame["Nhãn fraud"].map({True: "FRAUD", False: "NORMAL"}).fillna("Chưa có nhãn")
        if PRIORITY_MODE == "qubo":
            frame["Ưu tiên"] = frame["priority_status"].map(QUANTUM_STATUS_LABELS).where(frame["Ứng viên"], "")
        else:
            frame["Ưu tiên"] = frame["manual_review"].map({True: "Điều tra", False: ""})
        view = frame[
            [
                "Thời gian",
                "Mã giao dịch",
                "Loại",
                "Số tiền",
                "Risk",
                "Ứng viên",
                "Ưu tiên",
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
                "Ưu tiên": st.column_config.TextColumn(),
            },
        )


@st.fragment(run_every=1.0)
def alert_panel() -> None:
    render_alerts()


realtime_panel()
alert_panel()
if graph_alert_id := st.session_state.get("graph_dialog_alert_id"):
    render_graph_dialog(str(graph_alert_id))

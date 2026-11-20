from __future__ import annotations

from base64 import b64encode
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from hmac import compare_digest
from html import escape
from math import ceil, cos, isfinite, pi, sin
import os
import logging
from pathlib import Path
import re
import sqlite3
import sys
from threading import BoundedSemaphore
from time import perf_counter
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
# Streamlit watches out-of-tree imports through PYTHONPATH during dashboard updates.
source_paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
if str(SRC_ROOT) not in source_paths:
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC_ROOT), *source_paths]))

import pandas as pd
import streamlit as st

from qf_guard.data import normalize_transactions
from qf_guard import casework
from qf_guard.explainability import review_signals
from qf_guard.operations import CaseStore, EventProcessor
from qf_guard.prioritization import PriorityConfig, candidate_edges, selection_size
from qf_guard.replay import FullReplay, prepare_replay
from qf_guard.review_runs import RECORDING, configured_priority, is_qpu_backend, load_haituo_recording, solve_review_window

PAGE_SIZE = 6
GRAPH_HISTORY_LIMIT = 2_000
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
    "amlworld_full": ("IBM AMLworld HI-Small · Tập kiểm tra toàn bộ (tổng hợp)",
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
LOOP_SOURCE = Path(os.environ.get("QFG_REPLAY_SOURCE") or
                   (PROJECT_ROOT / "data/downloads/ibm_amlworld.zip" if DATASET == "amlworld_full" else STREAM_PATH))
default_source = "loop" if DATASET == "amlworld_full" and LOOP_SOURCE.is_file() else "replay"
SOURCE_MODE = os.environ.get("QFG_SOURCE", default_source).strip().lower()
STREAM_NAME = os.environ.get("QFG_STREAM") or ("recorded-haituo" if SOURCE_MODE == "recorded" else f"{SOURCE_MODE}-{DATASET}")
CASE_DB = Path(os.environ.get("QFG_CASE_DB") or PROJECT_ROOT / "artifacts" / "operations.sqlite3")
LOOP_INDEX = Path(os.environ.get("QFG_REPLAY_INDEX") or PROJECT_ROOT / "artifacts" / f"{DATASET}-replay.sqlite")
LOOP_BATCH = int(os.environ.get("QFG_REPLAY_BATCH", "64"))
LOOP_INTERVAL = float(os.environ.get("QFG_REPLAY_INTERVAL", "1"))
EXTERNAL_FEED = os.environ.get("QFG_REPLAY_EXTERNAL", "0") == "1"
EXTERNAL_PRIORITY = os.environ.get("QFG_EXTERNAL_PRIORITY", "0") == "1"
CONFIGURED_SOLVER = os.environ.get("QFG_SOLVER", "exact" if SOURCE_MODE == "loop" else "").strip().lower()
REQUIRE_MEASURED_OPTIMUM = CONFIGURED_SOLVER.startswith("quafu-fast-")
DEFAULT_POLICY = "qubo" if CONFIGURED_SOLVER not in ("", "sqa") else "risk"
PRIORITY_MODE = os.environ.get("QFG_PRIORITY_MODE", DEFAULT_POLICY).strip().lower()
# AMLworld validation settings. Each shared-account pair adds 0.01 * 0.5 after scaling.
PRIORITY_CONFIG = configured_priority()
# Candidates are grouped into windows so each QPU call carries a meaningful
# QUBO and the per-call network latency does not throttle the 1 s UI tick.
QUAFU_BATCH_SIZE = "18" if REQUIRE_MEASURED_OPTIMUM or CONFIGURED_SOLVER == "exact" else "8"
QUANTUM_MIN_BATCH = int(os.environ.get("QFG_QUANTUM_MIN_BATCH", QUAFU_BATCH_SIZE))
QUANTUM_MAX_BATCH = int(
    os.environ.get(
        "QFG_QUANTUM_MAX_BATCH",
        QUAFU_BATCH_SIZE if CONFIGURED_SOLVER.startswith("quafu") or CONFIGURED_SOLVER == "exact" else "40",
    )
)
QUANTUM_WINDOW_SECONDS = float(os.environ.get("QFG_QUANTUM_WINDOW_SECONDS", "10"))
WINDOW_TIMEOUT = float(os.environ.get("QFG_WINDOW_TIMEOUT_SECONDS", "180"))
MAX_JOBS = int(os.environ.get("QFG_WINDOW_MAX_JOBS", "8"))
MAX_ATTEMPTS = int(os.environ.get("QFG_WINDOW_MAX_ATTEMPTS", "3"))
QUANTUM_STATUS_LABELS = {
    "pending": "Chờ gom nhóm",
    "running": "Đang xét ưu tiên",
    "selected": "Được ưu tiên",
    "not_selected": "—",
    "error": "Cần xử lý lại",
}

st.set_page_config(
    page_title="Q-FraudGuard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

UI_DIR = Path(__file__).resolve().parent
tokens_file = UI_DIR / "design" / "design-system" / "q-fraudguard" / "tokens.css"
if tokens_file.is_file():
    st.html(tokens_file)
st.html(UI_DIR / "workspace.css")
engraving = b64encode((UI_DIR / "assets/engraving.webp").read_bytes()).decode("ascii")
st.html(f"<style>.qfg-header,.qfg-login-brand{{--qfg-engraving:url(data:image/webp;base64,{engraving})}}</style>")

DASHBOARD_ACCOUNTS = {
    "chuyenvien1": os.environ.get("QFG_DASHBOARD_OPERATOR_PASSWORD") or "1",
    "admin": os.environ.get("QFG_DASHBOARD_ADMIN_PASSWORD") or "admin",
}


def login():
    if st.session_state.get("dashboard_operator"):
        st.switch_page(DASHBOARD_PAGE, query_params=cockpit_query())
    with st.container(key="operator_login_shell"):
        brand, form = st.columns([1, 1], gap=None, vertical_alignment="center")
        with brand:
            st.html(
                '<section class="qfg-login-brand" aria-label="Q-FraudGuard">'
                '<p class="qfg-login-product">Q-FraudGuard</p>'
                '<p class="qfg-login-workspace">Hệ thống nội bộ</p>'
                '</section>'
            )
        with form, st.container(key="operator_login_form"):
            st.title("Đăng nhập")
            st.caption("Sử dụng tài khoản chuyên viên để mở bàn làm việc.")
            with st.form("operator_login", clear_on_submit=True, border=False):
                username = st.text_input("Tên đăng nhập", key="login_username", autocomplete="username")
                password = st.text_input("Mật khẩu", key="login_password", type="password",
                                         autocomplete="current-password")
                submitted = st.form_submit_button("Đăng nhập", type="primary", width="stretch")
            if submitted:
                username = username.strip()
                expected = DASHBOARD_ACCOUNTS.get(username)
                if expected is not None and compare_digest(password.encode(), expected.encode()):
                    st.session_state.dashboard_operator = username
                    st.switch_page(DASHBOARD_PAGE, query_params=cockpit_query())
                st.error("Tên đăng nhập hoặc mật khẩu không đúng.")


def operator_session_bar():
    with st.container(key="operator_session", horizontal=True, vertical_alignment="center",
                      horizontal_alignment="right", gap="small"):
        st.caption("Đang đăng nhập · " + st.session_state.dashboard_operator)
        st.button("Đăng xuất", key="operator_logout", on_click=st.session_state.clear)


@st.cache_resource(show_spinner=False, hash_funcs={type: id})
def case_store(store_class: type[CaseStore], path: Path, name: str) -> CaseStore:
    return store_class(path, name)


def ledger() -> CaseStore:
    return case_store(CaseStore, CASE_DB, STREAM_NAME)


@st.cache_resource(show_spinner=False)
def replay_processor(path: Path, database: Path, name: str) -> EventProcessor:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy model: {path}. Tạo bằng: {PREPARE_HINT}")
    return EventProcessor.load(case_store(CaseStore, database, name), path)


@st.cache_resource(show_spinner=False)
def recorded_source(database: Path, name: str) -> CaseStore:
    store = case_store(CaseStore, database, name)
    load_haituo_recording(store, PROJECT_ROOT)
    return store


@st.cache_resource(show_spinner="Đang mở nguồn phát lại toàn bộ…")
def loop_source(source, index, model, database, name):
    prepare_replay(source.resolve(), index.resolve())
    return FullReplay(case_store(CaseStore, database, name), model, index)


@st.cache_resource(show_spinner=False)
def quantum_executor() -> tuple[ThreadPoolExecutor, BoundedSemaphore]:
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="qfg-quantum"), BoundedSemaphore(2)


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


def list_label(identifier: str, detail: str) -> str:
    # IDs, actor names and currencies are data even in Streamlit's Markdown button labels.
    def literal(value: str) -> str:
        return re.sub(r"([\\`*_\[\]<>~$|])", r"\\\1", " ".join(value.split()))

    return f"**{literal(identifier)}**  \n{literal(detail)}"


def queue_heading(title: str, count: int) -> None:
    st.html(f'<div class="queue-heading"><h2>{escape(title)}</h2>'
            f'<span class="queue-count">{number(count)}</span></div>')


def risk_visualization(alert: Mapping[str, object]) -> str:
    tgnn = alert.get("graph_model") == "tgnn"
    graph_label = "TGNN Score" if tgnn else "Graph Score"
    graph_title = ("Điểm từ lịch sử liên kết theo thời gian của hai tài khoản" if tgnn
                   else "Điểm từ quan hệ và hoạt động của các tài khoản")
    components = (
        ("Supervised Learning", "Điểm từ mô hình học trên dữ liệu có nhãn", float(alert.get("Supervised", 0))),
        ("Anomaly Detection", "Mức độ bất thường so với dữ liệu tham chiếu", float(alert.get("Anomaly", 0))),
        (graph_label, graph_title, float(alert.get("Graph", 0))),
        ("Rule-based Score", "Điểm từ các quy tắc sàng lọc", float(alert.get("Rule", 0))),
    )
    rows = "".join(
        f"""<div class="risk-viz-row" title="{title}">
        <span class="risk-viz-name">{label}</span>
        <div class="risk-viz-track"><div class="risk-viz-fill"
        style="width:{max(0.0, min(1.0, value)) * 100:.1f}%"></div></div>
        <span class="risk-viz-value">{number(value, 3)}</span></div>"""
        for label, title, value in components
    )
    return f"""<div class="risk-viz-title">Điểm thành phần · thang 0–1</div>
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
    left_reasons = set(left.get("Tín hiệu", ()))
    right_reasons = set(right.get("Tín hiệu", ()))
    shared_rule = bool(left_reasons & right_reasons)
    left_amount = max(1.0, float(left.get("Số tiền", 0)))
    right_amount = max(1.0, float(right.get("Số tiền", 0)))
    amount_ratio = min(left_amount, right_amount) / max(left_amount, right_amount)
    same_unit = bool(left.get("Đơn vị tiền")) and left["Đơn vị tiền"] == right.get("Đơn vị tiền")
    if same_type and same_unit and amount_ratio >= 0.42 and risk_gap <= 0.16:
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
        if source != target:
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
    *, expanded=False,
) -> str:
    alert_id = str(alert["Mã giao dịch"])
    def complete_endpoints(item):
        return all(item.get(field) not in (None, "", "UNKNOWN") for field in ("Người gửi", "Người nhận"))

    if not complete_endpoints(alert):
        return """<div class="fraud-graph"><div class="fraud-graph-label">
        Cần mã tài khoản gửi và nhận để hiển thị Transaction Graph.</div></div>"""
    unique_candidates = {
        str(item["Mã giao dịch"]): item
        for item in candidate_pool
        if complete_endpoints(item)
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
            or not complete_endpoints(row)
            or len(known_accounts | row_accounts) > 16
            or (not pd.isna(alert_time) and not pd.isna(row_time) and row_time > alert_time)
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

    transfers = {(item["Người gửi"], item["Người nhận"]) for item in transactions if item["Người gửi"] != item["Người nhận"]}
    send_counts = Counter(source for source, _ in transfers)
    receive_counts = Counter(target for _, target in transfers)
    analytical_kinds = Counter(
        kind
        for left_id, right_id, kind, _ in kept_relations
        if left_id in selected_ids and right_id in selected_ids and kind != "shared"
    )
    if len(known_accounts) == 1:
        motif, motif_label = "self", "Giao dịch trong cùng một tài khoản"
    elif not analytical_kinds and len(known_accounts) == 2:
        motif = "pair"
        motif_label = "Có chuyển tiền hai chiều" if len(transfers) == 2 else "Chuyển tiền giữa hai tài khoản"
    elif _has_directed_cycle(transactions):
        motif, motif_label = "cycle", "Có vòng chuyển tiền (Cycle)"
    elif max(receive_counts.values(), default=0) >= 2:
        motif, motif_label = "fan_in", "Nhiều tài khoản chuyển đến cùng một tài khoản (Fan-in)"
    elif max(send_counts.values(), default=0) >= 2:
        motif, motif_label = "fan_out", "Một tài khoản chuyển đến nhiều tài khoản (Fan-out)"
    elif send_counts.keys() & receive_counts.keys():
        motif, motif_label = "chain", "Tài khoản xuất hiện ở cả đầu nhận và đầu gửi"
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
    if motif == "self":
        account_positions[sender] = (180.0, 165.0)
    elif motif == "pair":
        first, second = (sender, receiver) if sender != receiver else tuple(account_order)
        account_positions = {first: (105.0, 165.0), second: (415.0, 165.0)}
    elif motif in {"fan_in", "fan_out"}:
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
        if source == target:
            transaction_positions[transaction_id] = (source_x + (105 if source_x < 260 else -105), source_y + 24 * index)
            continue
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
        "risk": ("#4D91B8", "3 4"),
        "rule": ("#64748b", "2 5"),
    }
    relation_labels = {
        "behavior": "Tương đồng hành vi",
        "risk": "Risk Score tương đồng",
        "rule": "Chung tín hiệu từ quy tắc",
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
            f'<g class="flow-similarity" data-trace="context"><title>{relation_labels[kind]} · {number(strength, 2)}</title>'
            f'<line x1="{left_x:.1f}" y1="{left_y:.1f}" x2="{right_x:.1f}" '
            f'y2="{right_y:.1f}" stroke="{color}" stroke-width="1.5" '
            f'stroke-dasharray="{dash}"/></g>'
        )

    flow_edges: list[str] = []
    transaction_nodes: list[str] = []
    tooltips: list[str] = []
    tooltip_rules: list[str] = []
    candidate_accounts: set[str] = set()
    marker = "qfg-arrow-detail" if expanded else "qfg-arrow"
    for index, transaction in enumerate(transactions):
        transaction_id = str(transaction.get("Mã giao dịch", ""))
        tag = f"flow-tx-{index}"
        tip_id = f"{marker}-tip-{index}"
        tx_x, tx_y = transaction_positions[transaction_id]
        source = str(transaction.get("Người gửi", ""))
        target = str(transaction.get("Người nhận", ""))
        source_x, source_y = account_positions[source]
        target_x, target_y = account_positions[target]
        candidate = transaction_id in selected_ids
        focused = transaction_id == alert_id
        trace_group = "current" if focused else "context"
        color = "#a33443" if candidate else "#4D91B8"
        if candidate:
            candidate_accounts.update((source, target))
        width = 3.5 if focused else (2.5 if candidate else 1.4)
        if source == target:
            middle_x = (tx_x + source_x) / 2
            segments = [f'path d="M {source_x:.1f} {source_y-12:.1f} Q {middle_x:.1f} {source_y-58:.1f} '
                        f'{tx_x:.1f} {tx_y:.1f} Q {middle_x:.1f} {source_y+58:.1f} {source_x:.1f} {source_y+20:.1f}"']
        else:
            # Terminate the arrow at the receiving node's border so the circle doesn't cover it.
            dx, dy = target_x - tx_x, target_y - tx_y
            scale = min(1.0, (min(27, 17 + 2.2 * degree[target]) + 3) / max(1, (dx*dx + dy*dy)**.5))
            segments = [f'line x1="{source_x:.1f}" y1="{source_y:.1f}" x2="{tx_x:.1f}" y2="{tx_y:.1f}"',
                        f'line x1="{tx_x:.1f}" y1="{tx_y:.1f}" x2="{target_x-dx*scale:.1f}" y2="{target_y-dy*scale:.1f}"']
        edges = []
        for i, segment in enumerate(segments):
            arrow = f' marker-end="url(#{marker})"' if i == len(segments) - 1 else ""
            edges.append(f'<{segment} stroke="transparent" stroke-width="12" pointer-events="stroke"/>'
                         f'<{segment} stroke="{color}" stroke-width="{width}"{arrow}/>')
        flow_edges.append(f'<g class="flow-target flow-route {tag}" data-trace="{trace_group}" '
                          f'fill="none" aria-hidden="true">{"".join(edges)}</g>')
        amount_label = transaction.get("Số tiền hiển thị") or number(float(transaction.get("Số tiền", 0)), 2)
        method = transaction.get("Kênh thanh toán")
        fields = [("Số tiền", amount_label), ("Thời điểm", utc(transaction["Thời gian"])),
                  ("Tài khoản gửi", source), ("Tài khoản nhận", target)]
        if isinstance(method, str) and method.strip():
            fields.append(("Kênh thanh toán", method))
        elif transaction.get("Loại"):
            fields.append(("Loại giao dịch", transaction["Loại"]))
        details = "".join(f'<dt>{label}</dt><dd>{escape(str(value))}</dd>' for label, value in fields)
        side = "above" if tx_y >= 170 else "below"
        tooltips.append(f'<div class="flow-tooltip" role="tooltip" id="{tip_id}" data-side="{side}" '
                        f'style="--node-x:{tx_x/520:.2%};--node-y:{tx_y/340:.2%}">'
                        f'<strong>{escape(transaction_id)}</strong><dl>{details}</dl></div>')
        # The SVG node and both legs of its transfer share a single HTML card in this graph.
        tooltip_rules.append(f'.flow-stage:has(.{tag}:hover) > #{tip_id},'
                             f'.flow-stage:not(:has(.flow-target:hover)):has(.{tag}:focus-visible) > #{tip_id}'
                             '{display:block}')
        focus_halo = (
            f'<rect class="graph-focus" x="{tx_x - 24:.1f}" y="{tx_y - 15:.1f}" '
            'width="48" height="30" rx="8" fill="none" stroke="#4D91B8"/>'
            if focused
            else ""
        )
        node_width = 42 if focused else 36
        node_height = 22 if focused else 18
        transaction_nodes.append(
            f'<g class="flow-target flow-node {tag}" data-trace="{trace_group}" tabindex="0" role="img" '
            f'aria-label="Giao dịch {escape(transaction_id)}" aria-describedby="{tip_id}">{focus_halo}'
            f'<rect class="flow-marker" x="{tx_x - node_width / 2:.1f}" y="{tx_y - node_height / 2:.1f}" '
            f'width="{node_width}" height="{node_height}" rx="5" fill="{color}" '
            f'stroke="{"#fff9ed" if focused else "#fff"}" '
            f'stroke-width="{"2.5" if focused else "1.5"}"/>'
            f'<text x="{tx_x:.1f}" y="{tx_y + 3:.1f}" text-anchor="middle" '
            f'font-size="9" font-weight="700" fill="#fff">'
            f'{escape(transaction_id[-6:])}</text></g>'
        )

    account_nodes: list[str] = []
    for account in account_order:
        x, y = account_positions[account]
        suspicious = account in candidate_accounts
        radius = min(27.0, 17.0 + 2.2 * degree[account])
        stroke = "#a33443" if suspicious else "#4D91B8"
        fill = "#fff5f4" if suspicious else "#DCECF5"
        role = "GỬI / NHẬN" if account == sender == receiver else ("GỬI" if account == sender else "NHẬN" if account == receiver else "TK")
        trace_group = "current" if account in anchor_accounts else "context"
        account_nodes.append(
            f'<g class="flow-account" data-trace="{trace_group}"><title>{escape(account)} · {degree[account]} giao dịch</title>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="2.5"/>'
            f'<text x="{x:.1f}" y="{y - 3:.1f}" text-anchor="middle" '
            f'font-size="7.5" fill="#4D91B8">{role}</text>'
            f'<text x="{x:.1f}" y="{y + 9:.1f}" text-anchor="middle" '
            f'font-size="9" font-weight="700" fill="#183B56">'
            f'{escape(account[-6:])}</text></g>'
        )

    analytical_legend = "".join(
        f'<span><i class="graph-link-key" style="color:{relation_styles[kind][0]}"></i>'
        f'{relation_labels[kind]}</span>'
        for kind in sorted(used_relation_kinds)
    )
    relation_label = f"{len(account_nodes)} tài khoản · {len(transactions)} giao dịch"
    trace_control = ""
    if len(transactions) > 1:
        trace_control = (f'<details class="qfg-disclosure qfg-trace"><summary>Theo dấu giao dịch</summary>'
                         f'<p><bdi>{escape(sender)}</bdi> → <bdi>{escape(receiver)}</bdi></p></details>')
    markup = f"""<div class="fraud-graph">
    <div class="graph-heading"><div class="graph-caption"><strong>{escape(motif_label)}</strong>
    <span>{escape(relation_label)}</span></div>{trace_control}</div>
    <div class="flow-stage"><svg viewBox="0 0 520 340" width="100%" role="group"
      aria-label="Transaction Graph · {escape(motif_label)}">
      <defs><marker id="{marker}" markerWidth="8" markerHeight="8" refX="7"
        refY="3" orient="auto" markerUnits="userSpaceOnUse">
        <path d="M0,0 L0,6 L8,3 z" fill="#4D91B8"/></marker></defs>
      {''.join(relation_edges)}
      {''.join(flow_edges)}
      {''.join(transaction_nodes)}
      {''.join(account_nodes)}
    </svg>{''.join(tooltips)}<style>{''.join(tooltip_rules)}</style></div>
    <details class="qfg-disclosure qfg-legend" open="open"><summary>Chú giải</summary>
    <div class="graph-legend">
      <span><i class="graph-dot graph-current"></i>Giao dịch hiện tại</span>
      <span><i class="graph-dot graph-candidate"></i>Giao dịch đang xét</span>
      <span><i class="graph-dot graph-history"></i>Giao dịch trước đó</span>
      {analytical_legend}</div></details>
    <div class="fraud-graph-label">Trỏ, chạm hoặc dùng phím Tab để xem chi tiết giao dịch.</div></div>"""
    # A blank line for an empty edge layer would terminate Markdown's raw HTML block.
    return markup.replace("\n", " ")


def display_record(event):
    return {
        "Thời gian": pd.Timestamp(event["timestamp"]), "Mã giao dịch": event["transaction_id"],
        "Người gửi": event["sender_id"], "Người nhận": event["receiver_id"],
        "Loại": event["type"], "Kênh thanh toán": event.get("payment_format"),
        "Số tiền": event["amount"], "Số tiền hiển thị": amount(event),
        "Đơn vị tiền": amount_unit(event), "Risk": event.get("risk_score", 0.),
        "Supervised": event.get("supervised_score", 0.), "Anomaly": event.get("anomaly_score", 0.),
        "Graph": event.get("graph_score", 0.), "Rule": event.get("rule_score", 0.),
        "graph_model": event.get("graph_model"),
        "Ứng viên": event.get("is_candidate", False), "Nhãn fraud": event.get("is_fraud"),
        "Tín hiệu": review_signals(event)[0], "priority_status": event.get("priority_status"),
        "manual_review": event.get("manual_review", False),
        "recommended_action": event.get("recommended_action"), "action_detail": event.get("action_detail"),
    }


def graph_history(row, recent, hours):
    history = list(reversed(recent))
    prefix = []
    if SOURCE_MODE == "replay":
        data, boundary = realtime_source(STREAM_PATH)
        prior = data.iloc[max(0, boundary - GRAPH_HISTORY_LIMIT):boundary]
        accounts = [row["sender_id"], row["receiver_id"]]
        end = pd.Timestamp(row["timestamp"])
        matched = prior.loc[prior.timestamp.between(end-pd.Timedelta(hours=hours), end)
                            & (prior.sender_id.isin(accounts) | prior.receiver_id.isin(accounts))]
        prefix = matched.tail(30).to_dict("records")
    return [display_record(item) for item in prefix + history], bool(prefix)


DESK, RUNS, JOURNAL, CUSTOMER = "Bàn điều tra", "Phiên ưu tiên", "Nhật ký", "Phản hồi khách hàng"
STATE_LABELS = {"running": "Đang xử lý", "error": "Cần xử lý lại", "completed": "Đã hoàn tất"}
ACTIONS = {
    "request_verification": "Yêu cầu xác minh", "continue_monitoring": "Tiếp tục giám sát",
    "step_up_verification": "Yêu cầu xác minh tăng cường", "monitor_and_review": "Giám sát và rà soát",
    "escalate_to_investigator": "Chuyển điều tra chuyên sâu", "hold_and_escalate": "Đề nghị tạm giữ và chuyển điều tra",
    "hold_transaction": "Đề nghị tạm giữ giao dịch", "reject_transaction": "Đề nghị từ chối giao dịch",
}




def number(value, places=0):
    return f"{value:,.{places}f}".translate(str.maketrans(".,", ",."))


def amount_unit(event):
    for field in ("amount_currency", "payment_currency"):
        unit = event.get(field)
        if isinstance(unit, str) and unit.strip():
            return {"US Dollar": "USD", "Euro": "EUR"}.get(unit, unit)
    return ""


def amount(event):
    unit = amount_unit(event)
    value = number(event["amount"], 2)
    return f"{value} {unit}" if unit else f"{value} (chưa rõ loại tiền)"


def utc(value):
    timestamp = pd.Timestamp(value, unit="s", tz="UTC") if isinstance(value, (int, float)) else pd.Timestamp(value)
    return timestamp.strftime("%d/%m/%Y %H:%M:%S UTC")


def set_destination(area, *, case=None, window=None, trail=None):
    if case is not None:
        if trail is not None or case != st.session_state.get("focus_case"):
            st.session_state.case_trail = trail or []
        st.session_state.focus_case = case
    if window is not None:
        st.session_state.focus_window = window
    st.session_state.next_area = area
    st.session_state.skip_ingest = True


def navigate(area, *, case=None, window=None):
    set_destination(area, case=case, window=window)
    st.rerun(scope="app")


def initialize_stream():
    if SOURCE_MODE not in ("live", "replay", "loop", "recorded") or PRIORITY_MODE not in ("risk", "qubo"):
        raise ValueError("Cấu hình QFG_SOURCE hoặc QFG_PRIORITY_MODE không hợp lệ.")
    if (not 0 < QUANTUM_MIN_BATCH <= QUANTUM_MAX_BATCH or not isfinite(QUANTUM_WINDOW_SECONDS)
            or QUANTUM_WINDOW_SECONDS < 0 or not isfinite(WINDOW_TIMEOUT) or WINDOW_TIMEOUT <= 0
            or min(MAX_ATTEMPTS, MAX_JOBS, LOOP_BATCH) < 1 or not isfinite(LOOP_INTERVAL) or LOOP_INTERVAL < 0):
        raise ValueError("Kích thước cửa sổ, thời gian chờ hoặc giới hạn lần thử không hợp lệ.")
    if SOURCE_MODE == "recorded":
        recorded_source(CASE_DB, STREAM_NAME)
    else:
        if ledger().stats().get("model") == RECORDING:
            raise ValueError("Tên luồng này đã dành cho bản ghi lịch sử. Chọn tên riêng cho luồng mới.")
        if SOURCE_MODE == "replay":
            realtime_source(STREAM_PATH)
            replay_processor(MODEL_PATH, CASE_DB, STREAM_NAME)
        elif SOURCE_MODE == "loop" and not EXTERNAL_FEED:
            loop_source(LOOP_SOURCE, LOOP_INDEX, MODEL_PATH, CASE_DB, STREAM_NAME)
        elif SOURCE_MODE == "live":
            ledger().enable_auto_routing()
    st.session_state.update(stream_ready=True, quantum_future=None, quantum_lease=None, focus_case=None, focus_window=None,
                            stream_latency=[], source_finished=False)


def dispatch(claim, backend, config):
    """Reserve worker capacity before leasing cases, including across browser sessions."""
    executor, slots = quantum_executor()
    if not slots.acquire(blocking=False):
        return False
    try:
        job = claim()
        if job is None:
            slots.release()
            return False
        lease, batch = job
        try:
            future = executor.submit(solve_review_window, ledger(), lease, batch, backend=backend, config=config,
                                     max_jobs=MAX_JOBS, timeout=WINDOW_TIMEOUT)
        except RuntimeError as exc:
            ledger().finish_window(lease, None, error=str(exc))
            raise
    except BaseException:
        slots.release()
        raise
    future.add_done_callback(lambda _: slots.release())
    st.session_state.quantum_lease = lease
    st.session_state.quantum_future = future
    return True


def worker_busy():
    future = st.session_state.quantum_future
    return future is not None and not future.done() and ledger().active_run(st.session_state.quantum_lease)


def advance_quantum(flush):
    store = ledger()
    store.expire_windows()
    previous = st.session_state.quantum_future
    if previous is not None:
        if worker_busy():
            return
        st.session_state.quantum_future = None
        if previous.done():
            previous.result()
    choice = store.backend_choice(CONFIGURED_SOLVER or "sqa")
    backend = choice["backend"]
    maximum = min(QUANTUM_MAX_BATCH, 18) if backend == "exact" or backend.startswith("quafu-fast-") else QUANTUM_MAX_BATCH
    dispatch(lambda: store.claim_window(min(QUANTUM_MIN_BATCH, maximum), maximum, QUANTUM_WINDOW_SECONDS, flush=flush,
                                        backend=backend, source=SOURCE_MODE, config=asdict(PRIORITY_CONFIG),
                                        timeout=WINDOW_TIMEOUT, expected_backend_id=choice["id"]), backend, PRIORITY_CONFIG)


def process_tick():
    if SOURCE_MODE == "recorded":
        return
    if SOURCE_MODE == "loop" and not EXTERNAL_FEED:
        loop_source(LOOP_SOURCE, LOOP_INDEX, MODEL_PATH, CASE_DB, STREAM_NAME).tick(LOOP_BATCH, interval=LOOP_INTERVAL)
    if SOURCE_MODE == "replay" and not st.session_state.source_finished:
        data, boundary = realtime_source(STREAM_PATH)
        offset = ledger().stats()["processed"]
        start = boundary + offset
        if start < len(data):
            started = perf_counter()
            inserted = replay_processor(MODEL_PATH, CASE_DB, STREAM_NAME).ingest(
                data.iloc[start:start + 32], expected_offset=offset)
            if inserted:
                recent = st.session_state.stream_latency + [(perf_counter() - started) * 1000 / inserted]
                st.session_state.stream_latency = recent[-100:]
        st.session_state.source_finished = start + 32 >= len(data)
    if PRIORITY_MODE == "qubo" and not EXTERNAL_PRIORITY:
        advance_quantum(st.session_state.source_finished)


@st.fragment(run_every=None if SOURCE_MODE == "recorded" else 1.0)
def stream_metrics():
    try:
        if not st.session_state.pop("skip_ingest", False):
            process_tick()
    except (OSError, ValueError, RuntimeError, sqlite3.Error):
        logging.exception("Operational stream tick failed")
        st.error("Luồng tạm dừng ở vị trí đã lưu. Người vận hành có thể kiểm tra nhật ký máy chủ.")
    stats = ledger().stats()
    windows = ledger().window_list(limit=1)
    result = "—"
    scope = "Chưa có cửa sổ"
    if windows:
        latest = windows[0]
        if latest["status"] == "completed" and latest["k"] is not None:
            result = f"{latest['k']}/{latest['n']}"
            scope = "Đã hoàn tất"
        else:
            scope = STATE_LABELS[latest["status"]]
    values = [("Giao dịch đã chấm điểm", number(stats["processed"]), "Trong luồng hiện tại"),
              ("Ứng viên", number(stats["candidates"]), "Qua ngưỡng sàng lọc"),
              ("Đủ ngưỡng điều tra", number(stats["reviews"]), "Theo Risk Score ban đầu"),
              ("Đã chọn / ứng viên", result, scope + " · phiên mới nhất")]
    pending_reviews = 0
    if SOURCE_MODE != "recorded":
        counts = ledger().work_counts()
        waiting = counts.get("awaiting_info",0) + counts.get("awaiting_verification",0)
        pending_reviews = counts.get("pending_review", 0)
        values[2] = ("Chờ khách hàng", number(waiting), "Bổ sung thông tin hoặc xác minh")
        values[3] = ("Cần chuyên viên", number(pending_reviews), "Hồ sơ chờ quyết định")
        if windows:
            description = f"Chọn {result}" if result != "—" else scope
            values[1] = ("Ứng viên", number(stats["candidates"]), f"Phiên gần nhất · {description}")
    cells = []
    for label, value, note in values:
        emphasis = ' class="needs-review"' if label == "Cần chuyên viên" and pending_reviews else ""
        cells.append(f'<div{emphasis}><dt>{escape(label)}</dt><dd>{escape(value)}</dd>'
                     f'<dd class="metric-note">{escape(note)}</dd></div>')
    st.html('<dl class="scoreboard" aria-label="Tổng quan luồng giao dịch">' + "".join(cells) + "</dl>")
    if SOURCE_MODE != "recorded":
        notification = ledger().latest_alert()
        if notification and notification["id"] > st.session_state.get("last_alert_id", 0):
            st.session_state.last_alert_id = notification["id"]
            st.toast(f"Cần chuyên viên quyết định · {notification['txid']}")


def queue_rows(label, group):
    page_key = "page_" + group
    page = st.session_state.get(page_key, 1)
    store = ledger()
    if group in ("urgent", "customer_wait", "restrictions"):
        states = {"urgent": ("pending_review",), "customer_wait": ("awaiting_info", "awaiting_verification"),
                  "restrictions": ("frozen", "suspended")}[group]
        query, options = store.work_queue, dict(states=states, limit=PAGE_SIZE)
    else:
        query, options = store.queue, dict(proposals=group == "proposals", use_qubo=True, limit=PAGE_SIZE)
    count, records = query(**options, offset=(page-1)*PAGE_SIZE)
    pages = max(1, ceil(count/PAGE_SIZE))
    if page > pages:
        page = st.session_state[page_key] = pages
        _, records = query(**options, offset=(page-1)*PAGE_SIZE)
    queue_heading(label, count)
    if not count:
        st.caption("Chưa có hồ sơ trong mục này.")
    for row in records:
        txid = row["transaction_id"]
        if group == "urgent":
            origin = row.get("work_owner") or f"Điểm {number(row['risk_score'], 5)}"
        elif row.get("work_state"):
            origin = row["work_owner"] or casework.STATES[row["work_state"]]
        elif row.get("admitted_by"):
            origin = "Đã tiếp nhận"
        elif row["priority_status"] == "selected":
            origin = "Được ưu tiên"
        else:
            origin = "Đủ ngưỡng điều tra"
        if st.button(list_label(txid, f"{amount(row)} · {origin}"), key=f"desk_row_{group}_{txid}", width="stretch",
                     type="primary" if txid == st.session_state.focus_case else "secondary"):
            navigate(DESK, case=txid)
    if pages > 1:
        st.number_input("Trang " + label.lower(), 1, pages, key=page_key)
    return records


@st.fragment(run_every=None if SOURCE_MODE == "recorded" else 2.0)
def queues():
    if SOURCE_MODE != "recorded":
        urgent = queue_rows("Cần chuyên viên quyết định", "urgent")
        with st.expander("Các hồ sơ đang hạn chế"):
            queue_rows("Đang đóng băng hoặc tạm ngưng", "restrictions")
        if st.session_state.focus_case is None and urgent:
            navigate(DESK, case=urgent[0]["transaction_id"])
        return
    review = queue_rows("Hàng chờ điều tra", "review")
    proposals = queue_rows("Đề xuất bổ sung", "proposals")
    if st.session_state.focus_case is None and (review or proposals):
        navigate(DESK, case=(review or proposals)[0]["transaction_id"])


def decision_form(row, history):
    txid = row["transaction_id"]
    if row.get("work_state"):
        workflow_form(row, history)
        return
    if SOURCE_MODE != "recorded" and not row["action"]:
        st.caption("Giao dịch được tiếp tục theo kết quả sàng lọc.")
        return
    if row["action"]:
        st.success(f"{ACTIONS.get(row['action'], row['action'])} · {row['decided_by']}")
        if row.get("decision_reason"):
            st.write(row["decision_reason"])
        return
    pending = row["manual_review"] or bool(row["admitted_by"])
    if not pending and row["priority_status"] != "selected":
        return
    actor = st.session_state.dashboard_operator
    with st.form(f"case_form_{txid}_{pending}"):
        action = None
        if pending:
            keys = list(ACTIONS)
            index = keys.index(row["recommended_action"]) if row.get("recommended_action") in ACTIONS else 0
            action = st.selectbox("Quyết định xử lý", keys, index=index,
                                  format_func=ACTIONS.get)
        reason = st.text_input("Lý do quyết định" if pending else "Lý do tiếp nhận",
                               placeholder="Ghi tín hiệu, chứng từ hoặc giao dịch đã đối chiếu.")
        submit = st.form_submit_button("Ghi quyết định" if pending else "Tiếp nhận điều tra", type="primary")
    if submit:
        if not reason.strip():
            st.warning("Nhập lý do để ghi vào nhật ký.")
            return
        store = ledger()
        saved = (store.decide(txid, action, actor, reason=reason.strip(), require_review=True)
                 if pending else store.admit(txid, actor, reason))
        st.session_state.notice = ("Đã ghi quyết định" if pending else "Đã tiếp nhận hồ sơ") if saved else "Hồ sơ đã đổi trạng thái"
        navigate(DESK, case=txid)


def submit_operator_action(txid, action, version, form_key):
    actor = st.session_state.get("dashboard_operator")
    if not actor:
        return
    reason = st.session_state[form_key + "_reason"].strip()
    if not reason or (action in casework.RESTRICTIONS and not st.session_state[form_key + "_confirm"]):
        st.session_state[form_key + "_error"] = "Nhập lý do và xác nhận phạm vi nếu áp dụng hạn chế."
        return
    saved = ledger().operator_action(txid, action, actor, reason, version)
    st.session_state.pop(form_key + "_error", None)
    st.session_state.notice = "Đã ghi thao tác vào nhật ký." if saved else "Hồ sơ đã đổi trạng thái hoặc thuộc chuyên viên khác."
    set_destination(DESK, case=txid)


def workflow_form(row, history):
    txid = row["transaction_id"]
    state, version = row["work_state"], row["work_version"]
    st.markdown("**Trạng thái xử lý** · " + casework.STATES[state])
    if row["work_owner"]:
        st.caption("Chuyên viên nhận hồ sơ · " + row["work_owner"])
    key = f"work_{txid}_{version}"
    choices = casework.operator_choices(state, row["hold_origin"] is not None)
    if state == "pending_review":
        choices.insert(0, "unassign" if row["work_owner"] else "claim")
    choices.append("note")
    action = st.selectbox("Thao tác chuyên viên", choices, index=None, placeholder="Chọn thao tác thẩm định",
                          format_func=casework.OPERATIONS.get, key=key + "_action")
    if action:
        effects = {
            "freeze": f"Giao dịch {txid} sẽ bị đóng băng cho đến khi chuyên viên gỡ hạn chế.",
            "suspend": f"Tài khoản {row['sender_id']} sẽ tạm ngưng. Giao dịch mới từ tài khoản này sẽ chuyển cho chuyên viên.",
            "allow": f"Giao dịch {txid} sẽ được phép tiếp tục.",
            "request_info": "Thông tin khách hàng bổ sung sẽ được lưu vào nhật ký hồ sơ.",
            "request_verification": "Kết quả xác minh sẽ được lưu vào nhật ký. Ngân hàng quyết định việc duyệt giao dịch.",
            "restore_account": f"Khôi phục tài khoản gửi {row['sender_id']} trong sổ vận hành.",
            "unfreeze": "Gỡ hạn chế của giao dịch này. Nếu tài khoản đang tạm ngưng, hồ sơ sẽ quay lại hàng chờ.",
        }
        if action in effects:
            st.caption(effects[action])
        form_key = f"{key}_{action}"
        with st.form(form_key):
            st.text_area("Lý do hoặc ghi chú", max_chars=2000, key=form_key + "_reason",
                         placeholder="Ghi giao dịch, phản hồi hoặc chứng từ làm căn cứ cho thao tác này.")
            if action in casework.RESTRICTIONS:
                target = txid if action == "freeze" else row["sender_id"]
                st.checkbox(f"Xác nhận {casework.OPERATIONS[action].lower()} · {target}", key=form_key + "_confirm")
            st.form_submit_button("Ghi thao tác", type="primary", on_click=submit_operator_action,
                                  args=(txid, action, version, form_key))
            if error := st.session_state.get(form_key + "_error"):
                st.warning(error)
    if state in ("awaiting_verification", "awaiting_info") and SOURCE_MODE != "live":
        if st.button("Mở phản hồi khách hàng mô phỏng", key=key + "_customer"):
            st.session_state.customer_case = txid
            navigate(CUSTOMER)
    show_case_history(history)


def show_case_history(history):
    with st.expander("Lịch sử xử lý hồ sơ", expanded=True):
        for item in reversed(history):
            st.markdown(f"**{casework.OPERATIONS[item['action']]}** · {item['actor']}")
            st.caption(f"{utc(item['at'])} · {casework.STATES[item['state']]} · {item['reason']}")


def show_latest_response(history):
    receipt = next((entry for entry in reversed(history) if entry["action"] in casework.RESPONSE_REQUEST), None)
    if receipt:
        st.write(f"**Phản hồi gần nhất** · {casework.OPERATIONS[receipt['action']]}")
        st.caption(f"{utc(receipt['at'])} · {receipt['actor']}")
        st.text(receipt["reason"])


def open_response_log(txid):
    st.session_state.update(journal_query=txid, journal_page=1)
    set_destination(JOURNAL)


def submit_customer_reply(txid, account, version, state):
    if not st.session_state.get("dashboard_operator"):
        return
    key = f"response_{txid}_{version}"
    action = "info_received" if state == "awaiting_info" else st.session_state[key + "_action"]
    if action not in casework.RESPONSE_REQUEST:
        st.session_state.notice = "Chọn kết quả do kênh xác minh trả về."
        return
    reason = st.session_state[key + "_details"] if state == "awaiting_info" else "Kết quả kiểm tra qua kênh mô phỏng."
    if not reason.strip():
        st.session_state.notice = "Nhập thông tin bổ sung trước khi gửi."
        return
    saved = ledger().customer_response(txid, action, reason, version, account=account)
    st.session_state.notice = "Đã lưu phản hồi vào nhật ký." if saved else "Yêu cầu đã được xử lý hoặc đã thay đổi."
    st.session_state.skip_ingest = True


def customer_panel():
    store = ledger()
    left, right = st.columns([1, 2.35], gap="medium")
    with left, st.container(border=True, key="customer_queue"):
        first = customer_queue(store)
    txid = st.session_state.get("customer_case") or first
    with right, st.container(border=True, key="customer_workspace"):
        if txid:
            st.session_state.customer_case = txid
            customer_request(store, txid)
        else:
            st.info("Chưa có yêu cầu bổ sung thông tin hoặc xác minh.")


@st.fragment(run_every=3.0)
def live_customer_panel():
    if st.session_state.get("next_area"):
        st.rerun(scope="app")
    customer_panel()


def customer_queue(store):
    if SOURCE_MODE == "live":
        st.caption("Phản hồi được nhận từ kênh khách hàng đã xác thực qua tiến trình nhập sự kiện.")
    else:
        st.caption("Kênh mô phỏng tiếp nhận thông tin bổ sung và kết quả xác minh trên dữ liệu tổng hợp.")
    page = st.session_state.get("customer_page", 1)
    states = ("awaiting_info", "awaiting_verification")
    count, waiting = store.work_queue(states, limit=PAGE_SIZE, offset=(page-1)*PAGE_SIZE)
    pages = max(1, ceil(count/PAGE_SIZE))
    if page > pages:
        page = st.session_state.customer_page = pages
        _, waiting = store.work_queue(states, limit=PAGE_SIZE, offset=(page-1)*PAGE_SIZE)
    verify_count, verify = store.work_queue(("awaiting_verification",), limit=1)
    queue_heading("Yêu cầu đang chờ", count)
    if verify:
        if st.button(f"Mở yêu cầu xác minh · {number(verify_count)} đang chờ", key="customer_verify_next"):
            st.session_state.customer_case = verify[0]["transaction_id"]
    with st.form("customer_lookup"):
        search = st.text_input("Tìm theo mã giao dịch", key="customer_lookup_tx")
        if st.form_submit_button("Mở yêu cầu"):
            if store.case(search.strip()):
                st.session_state.customer_case = search.strip()
            else:
                st.warning("Không tìm thấy giao dịch.")
    first = waiting[0]["transaction_id"] if waiting else None
    active = st.session_state.get("customer_case") or first
    for item in waiting:
        tid = item["transaction_id"]
        st.button(list_label(tid, casework.STATES[item["work_state"]]), key="customer_pick_" + tid,
                  width="stretch", type="primary" if tid == active else "secondary",
                  on_click=st.session_state.update, args=({"customer_case": tid},))
    if pages > 1:
        st.number_input("Trang yêu cầu khách hàng", 1, pages, key="customer_page")
    return first


def customer_request(store, txid):
    row = store.case(txid)
    if row is None or not row.get("work_state"):
        st.caption("Giao dịch này chưa có yêu cầu đối với khách hàng.")
        return
    state = row["work_state"]
    case_identity(row)
    waiting = state in ("awaiting_info", "awaiting_verification")
    request_status = casework.STATES[state] if waiting or state == "response_recorded" else "Không có yêu cầu đang chờ"
    st.html(fact_cards([("Tài khoản gửi", row["sender_id"], ""),
                        ("Yêu cầu khách hàng", request_status, "")]))
    show_latest_response(store.work_history(txid))
    if state == "response_recorded":
        st.caption("Ngân hàng quyết định duyệt hoặc từ chối giao dịch trong hệ thống của mình.")
        st.button("Xem nhật ký hồ sơ", key="customer_log_" + txid, on_click=open_response_log, args=(txid,))
        return
    if state in ("pending_review", "frozen", "suspended") and SOURCE_MODE != "live":
        if st.button("Mở hồ sơ tại Bàn điều tra", key="customer_analyst_" + txid):
            navigate(DESK, case=txid)
    if not waiting or SOURCE_MODE == "live":
        return
    version = row["work_version"]
    key = f"response_{txid}_{version}"
    with st.form(f"customer_{txid}_{version}"):
        if state == "awaiting_info":
            st.text_area("Thông tin bổ sung về mục đích giao dịch", max_chars=2000, key=key + "_details")
        else:
            st.radio("Kết quả do kênh xác minh trả về", ["verify_ok", "verify_failed"], index=None,
                     format_func=casework.OPERATIONS.get, key=key + "_action")
        st.form_submit_button("Ghi phản hồi vào nhật ký", type="primary", on_click=submit_customer_reply,
                              args=(txid, row["sender_id"], version, state))


@st.dialog("Transaction Graph", width="large")
def graph_dialog(alert, history, pool):
    st.markdown(suspicious_graph(alert, history, pool, expanded=True), unsafe_allow_html=True)


def fact_cards(items):
    cards = "".join(f"<div><span>{escape(label)}</span><strong>{escape(value)}</strong><small>{escape(note)}</small></div>"
                    for label, value, note in items)
    return f'<div class="case-facts">{cards}</div>'


def open_related(txid, target):
    trail = st.session_state.get("case_trail", [])
    set_destination(DESK, case=target, trail=[*trail, txid][-12:])


def case_identity(row):
    with st.container(key="case_identity", horizontal=True, vertical_alignment="center"):
        st.subheader(row["transaction_id"], anchor=False)
        st.html(f'<span class="case-time">{utc(row["timestamp"])}</span>', width="content")


def investigation_header(row, history, threshold):
    txid = row["transaction_id"]
    trail = st.session_state.get("case_trail", [])
    if trail:
        st.button("Quay lại " + trail[-1], key="case_back_" + txid, on_click=set_destination,
                  args=(DESK,), kwargs={"case": trail[-1], "trail": trail[:-1]})
    case_identity(row)
    method = row.get("payment_format") or row["type"]
    state = casework.STATES.get(row.get("work_state"))
    if not state:
        state = "Đã có quyết định" if row["action"] else "Trong hàng điều tra" if row["manual_review"] or row["admitted_by"] else "Đã sàng lọc"
    facts = fact_cards([("Số tiền", amount(row), str(method)),
                ("Risk Score", number(row["risk_score"], 5), f"Ngưỡng điều tra {number(threshold, 5)}"),
                ("Trạng thái hồ sơ", state, row.get("work_owner") or ("Chưa phân công" if row.get("work_state") == "pending_review" else ""))])
    accounts = (f'<div class="case-accounts"><div><span>Tài khoản gửi</span><bdi>{escape(row["sender_id"])}</bdi></div>'
                '<i aria-hidden="true">→</i>'
                f'<div><span>Tài khoản nhận</span><bdi>{escape(row["receiver_id"])}</bdi></div></div>')
    recorded = row.get("work_state") == "response_recorded"
    routes = ("route", "request_verification", "request_info") if recorded else ("route", "escalate", "request_verification", "request_info")
    route = next((h for h in reversed(history) if h["action"] in routes), None)
    route_label = "Lý do yêu cầu khách hàng phản hồi" if recorded else "Lý do chuyển xử lý"
    notice = f'<div class="case-notice"><strong>{route_label}</strong> {escape(route["reason"])}</div>' if route else ""
    st.html(f'<div class="case-brief">{facts}{accounts}{notice}</div>')
    if not route and row["admitted_by"]:
        st.write(f"Tiếp nhận dưới ngưỡng · {row['admitted_by']} · {row['admission_reason']}")
    elif not route and row["manual_review"]:
        st.caption("Risk Score đạt ngưỡng điều tra.")
    show_latest_response(history)
    if row.get("hold_origin") is not None:
        st.warning("Tài khoản gửi đang tạm ngưng trong sổ vận hành.")


def account_summary(row, totals, period):
    st.html(fact_cards([("TK gửi đã chuyển", number(totals["sent"]), f"đến {number(totals['payees'])} tài khoản"),
                        ("TK nhận đã nhận", number(totals["received"]), f"từ {number(totals['payers'])} tài khoản"),
                        ("Cùng cặp, cùng chiều", number(totals["same_pair"]), period + " trước giao dịch")]))
    if row["sender_id"] != row["receiver_id"] and totals["reverse_pair"]:
        st.caption(f"Đã có {number(totals['reverse_pair'])} giao dịch theo chiều ngược lại trong khoảng tra cứu.")
    if totals["comparable"] >= 3 and totals["mean_amount"] and totals["mean_amount"] > 0:
        mean = amount({**row, "amount": totals["mean_amount"]})
        ratio = number(row["amount"]/totals["mean_amount"], 2)
        st.caption(f"Số tiền hiện tại bằng {ratio} lần mức trung bình {mean} của "
                   f"{number(totals['comparable'])} giao dịch gửi trước đó cùng đơn vị tiền.")


def account_journal(row, recent, totals, hours):
    if not totals["total"]:
        st.info("Chưa ghi nhận giao dịch trước đó của hai tài khoản trong khoảng đã chọn.")
        return
    txid = row["transaction_id"]
    filters = {"all": "Cả hai tài khoản", "same_pair": "Cùng cặp, cùng chiều", "sent": "Từ tài khoản gửi",
               "received": "Đến tài khoản nhận"}
    if row["sender_id"] != row["receiver_id"]:
        filters["reverse_pair"] = "Cùng cặp, ngược chiều"
    relation = st.selectbox("Lọc theo dòng tiền", list(filters), format_func=filters.get, key="history_filter_" + txid)
    if relation != "all":
        _, recent = ledger().account_activity(txid, hours=hours, relation=relation)
    count = totals["total" if relation == "all" else relation]
    if not count:
        st.info("Chưa có giao dịch theo hướng đã chọn trong khoảng tra cứu.")
        return
    st.caption(f"Hiển thị {number(len(recent))} trên {number(count)} giao dịch theo bộ lọc, từ mới đến cũ.")
    st.dataframe([{"Giao dịch": r["transaction_id"], "Thời điểm": utc(r["timestamp"]), "Từ": r["sender_id"],
                   "Đến": r["receiver_id"], "Số tiền": amount(r), "Từng gắn cờ": bool(r["is_candidate"])} for r in recent],
                 hide_index=True, width="stretch")
    target = st.selectbox("Xem một giao dịch liên quan", [r["transaction_id"] for r in recent], index=None,
                          placeholder="Chọn mã giao dịch", key="history_tx_" + txid)
    st.button("Mở giao dịch đã chọn", disabled=not target, key="history_open_" + txid, on_click=open_related, args=(txid, target))


def case_detail(txid):
    row = ledger().case(txid) if txid else None
    if row is None:
        if SOURCE_MODE == "recorded" or txid:
            st.info("Chọn một hồ sơ để xem dòng tiền và căn cứ điều tra.")
        else:
            st.info("Hồ sơ rủi ro cao hoặc có hạn chế tài khoản cần chuyên viên xem xét sẽ xuất hiện tại đây.")
            if st.button("Xem yêu cầu đang chờ khách hàng", key="desk_pending_customers"):
                navigate(CUSTOMER)
        return
    store = ledger()
    window = store.window(row["window_id"]) if row["window_id"] else None
    selected = [r for r in window["members"] if r["selected"]] if window else []
    history = store.work_history(txid) if row.get("work_state") else []
    investigation_header(row, history, store.stats()["review_threshold"])
    recorded_response = row.get("work_state") == "response_recorded"
    case_log_label = "Nhật ký hồ sơ" if recorded_response else "Quyết định & nhật ký"
    overview, transactions, decision = st.tabs(["Transaction Graph", "Giao dịch liên quan", case_log_label],
                                               key="case_tabs_" + txid)
    periods = {1: "1 giờ", 24: "24 giờ", 168: "7 ngày"}
    with overview:
        with st.container(key="graph_tools", horizontal=True, vertical_alignment="center", gap="small"):
            hours = st.segmented_control("Khoảng tra cứu", list(periods), default=24, required=True,
                                         format_func=periods.get, key="case_hours_" + txid, label_visibility="collapsed")
            by_window = bool(selected) and st.toggle("Thêm hồ sơ cùng phiên", key="graph_peers_" + txid)
            expand = st.button("Mở rộng Graph", key="inspect_graph_" + txid)
        totals, recent = store.account_activity(txid, hours=hours)
        start = pd.Timestamp(totals["recorded_since"])
        coverage = f"Sổ giao dịch bắt đầu từ {utc(start)}." if start > pd.Timestamp(row["timestamp"]) - pd.Timedelta(hours=hours) else ""
        prior, warm = graph_history(row, recent, hours)
        alert = display_record(row)
        pool = [display_record(r) for r in selected] if by_window else []
        if expand:
            graph_dialog(alert, prior, pool)
        st.markdown(suspicious_graph(alert, prior, pool), unsafe_allow_html=True)
        if by_window:
            st.caption("Mũi tên chỉ hướng chuyển tiền. Nét đứt nối những hồ sơ tương đồng trong cùng phiên ưu tiên.")
        if warm:
            st.caption("Graph có thêm giao dịch tham chiếu từ trước khi luồng bắt đầu.")
        st.caption(f"Sổ giao dịch · {periods[hours]} trước hồ sơ này · {number(totals['total'])} giao dịch liên quan. {coverage}")
        account_summary(row, totals, periods[hours])
    with transactions:
        st.caption(f"Khoảng đang tra cứu · {periods[hours]} trước hồ sơ, theo thời điểm giao dịch. {coverage}")
        account_journal(row, recent, totals, hours)
        related = [r for r in selected if r["transaction_id"] != txid and candidate_edges([row, r])]
        if related:
            st.markdown("**Chung tài khoản trong danh sách ưu tiên của phiên**")
            for other in related:
                other_id = other["transaction_id"]
                st.button(f"{other_id} · {amount(other)} · {other['sender_id']} → {other['receiver_id']}",
                          key=f"related_{txid}_{other_id}", width="stretch", on_click=open_related, args=(txid, other_id))
        if window:
            st.button("Mở phiên " + window["id"][:10], key="case_window_" + txid, on_click=set_destination,
                      args=(RUNS,), kwargs={"window": window["id"]})
    with decision:
        if recorded_response:
            show_case_history(history)
            return
        st.markdown("**Căn cứ trước khi quyết định**")
        signals, gaps = review_signals(row)
        if signals:
            st.write("Tín hiệu sàng lọc · " + " · ".join(signals))
        if gaps:
            st.write("Cần đối chiếu · " + " ".join(gaps))
        st.write(f"Trong {periods[hours]} trước hồ sơ, có {number(totals['same_pair'])} giao dịch cùng cặp và cùng chiều. "
                 f"Có {number(totals['alerts'])} giao dịch từng được gắn cờ trong {number(totals['total'])} giao dịch liên quan. {coverage}")
        with st.expander("Risk Score Breakdown"):
            st.html(risk_visualization(alert))
        decision_form(row, history)


def stream_activity():
    store = ledger()
    stats = store.stats()
    with st.expander("Tra cứu giao dịch và số liệu sàng lọc", on_change="rerun", key="activity_panel") as panel:
        if not panel.open:
            return
        if SOURCE_MODE == "recorded":
            admissions, oldest = store.review_workload()
            st.caption(f"Đã tiếp nhận {admissions} đề xuất." +
                       (f" Hồ sơ chờ lâu nhất đã ở trong hàng {oldest // 60} phút." if oldest is not None else ""))
        else:
            oldest = store.oldest_pending()
            if oldest is not None:
                st.caption(f"Hồ sơ chờ chuyên viên lâu nhất đã qua {oldest // 60} phút kể từ lần cập nhật gần nhất.")
        with st.form("find_case"):
            lookup = st.text_input("Mã giao dịch trong luồng", key="stream_lookup_tx")
            if st.form_submit_button("Mở giao dịch"):
                if store.case(lookup.strip()):
                    navigate(DESK, case=lookup.strip())
                st.warning("Không tìm thấy giao dịch trong luồng này.")
        if stats["labelled"]:
            st.caption(f"{number(stats['labelled'])} giao dịch có nhãn tham chiếu. "
                       f"Recall tại ngưỡng sàng lọc {stats['detected']}/{stats['frauds']} · "
                       f"Recall tại ngưỡng điều tra {stats['review_hits']}/{stats['frauds']}."
                       if stats["frauds"] else
                       f"{number(stats['labelled'])} giao dịch có nhãn tham chiếu. Chưa có nhãn dương để tính Recall.")
        else:
            st.caption("Nguồn giao dịch chưa cung cấp nhãn tham chiếu.")
        if st.session_state.stream_latency:
            values = st.session_state.stream_latency
            st.caption(f"Chấm và ghi dữ liệu trên máy khách · {sum(values)/len(values):.2f} ms/giao dịch ở các lô gần đây.")
        candidates = st.toggle("Xem toàn bộ ứng viên chưa đóng", key="browse_open_candidates")
        if candidates:
            page = st.number_input("Trang ứng viên", min_value=1, step=1, key="all_candidates_page")
            if SOURCE_MODE == "recorded":
                count, records = store.queue(all_candidates=True, limit=25, offset=(page - 1) * 25)
            else:
                count, records = store.work_queue(tuple(s for s in casework.STATES if s not in ("released", "response_recorded")),
                                                  limit=25, offset=(page - 1) * 25)
            st.caption(f"{number(count)} hồ sơ đang mở.")
        else:
            records = list(reversed(store.recent(100)))
        if records:
            st.dataframe([{"Giao dịch": r["transaction_id"], "Thời gian": utc(r["timestamp"]),
                           "Số tiền": amount(r), "Risk Score": r["risk_score"], "Ứng viên": r["is_candidate"]}
                          for r in records], width="stretch", hide_index=True)


def desk():
    # Select before mounting the two panes. Only a later arrival at an empty desk
    # needs the queue fragment to request a full rerun.
    if st.session_state.focus_case is None:
        store = ledger()
        if SOURCE_MODE == "recorded":
            _, first = store.queue(use_qubo=True, limit=1)
            if not first:
                _, first = store.queue(proposals=True, use_qubo=True, limit=1)
        else:
            _, first = store.work_queue(("pending_review",), limit=1)
        if first:
            st.session_state.focus_case = first[0]["transaction_id"]
    left, right = st.columns([1, 2.35], gap="medium")
    with left, st.container(border=True, key="case_queue"):
        queues()
    with right, st.container(border=True, key="case_workspace"):
        case_detail(st.session_state.focus_case)
    stream_activity()


def labelled_result(rows):
    known = [r["is_fraud"] for r in rows if r.get("is_fraud") is not None]
    if not known:
        return "Chưa có nhãn"
    return f"{sum(known)} nhãn dương · {len(known)}/{len(rows)} hồ sơ có nhãn"


def show_run(window):
    last = window["runs"][-1]
    summary = last["summary"]
    members = window["members"]
    k = summary.get("k", selection_size(len(members), window["config"].get("selection_fraction", .25)))
    ranked = sorted(members, key=lambda r: -r["risk_score"])[:k]
    selected = [r for r in members if r["selected"]]
    st.subheader(f"Chọn {k} trong {len(members)} ứng viên", anchor=False)
    st.caption(f"Ngân sách phiên · {k} suất xem xét. Bước ưu tiên dùng điểm đã sàng lọc và các liên kết chung tài khoản "
               "để phân bổ các suất này. Chuyên viên quyết định cách xử lý từng hồ sơ.")
    st.code(window["id"], language=None)
    st.caption(f"{STATE_LABELS[window['status']]} · {utc(window['created'])}")
    if window["status"] == "completed":
        ids = {r["transaction_id"] for r in selected}
        top = {r["transaction_id"] for r in ranked}
        st.markdown(f"**Đối chiếu với Top-K theo Risk Score** · {len(ids & top)} trùng · "
                    f"{len(ids - top)} chỉ có trong danh sách ưu tiên · {len(top - ids)} chỉ có trong Top-K")
        st.dataframe([{"Mã giao dịch": r["transaction_id"], "Được ưu tiên": bool(r["selected"]),
                       "Top-K theo Risk Score": r["transaction_id"] in top, "Risk Score": r["risk_score"]}
                      for r in members], hide_index=True, width="stretch")
        st.caption("Đối chiếu với nhãn nguồn · Danh sách ưu tiên " + labelled_result(selected) + " · Top-K " + labelled_result(ranked))
    txid = st.selectbox("Mở hồ sơ trong cửa sổ", [r["transaction_id"] for r in members],
                        index=None, key="run_member_" + window["id"], placeholder="Chọn mã giao dịch")
    if st.button("Xem hồ sơ điều tra", disabled=not txid, key="run_to_case"):
        navigate(DESK, case=txid)
    for run in reversed(window["runs"]):
        info = run["summary"]
        status = "error" if run["error"] else "running" if run["finished"] is None else "completed"
        with st.expander(f"Lần xử lý {run['number']} · {STATE_LABELS[status]}", expanded=run["lease"] == last["lease"]):
            st.caption(f"Bắt đầu · {utc(run['started'])}")
            if run["finished"] is not None:
                st.caption(f"Kết thúc · {utc(run['finished'])}")
            if info.get("estimated_sampling_ms") is not None and info.get("assumed_shot_period_us") is not None:
                st.caption(f"Ước tính thời gian lấy mẫu · {number(info['estimated_sampling_ms'], 2)} ms "
                           f"với giả định {number(info['assumed_shot_period_us'], 2)} µs/lượt.")
            elif not is_qpu_backend(window["backend"]) and info.get("client_elapsed_s") is not None:
                st.caption(f"Thời gian xử lý · {number(info['client_elapsed_s'], 3)} giây")
    if window["status"] == "error" and window["source"] != "recorded" and PRIORITY_MODE == "qubo":
        st.caption(f"Đã dùng {len(window['runs'])}/{MAX_ATTEMPTS} lần xử lý. Mỗi lần chờ tối đa {WINDOW_TIMEOUT:g} giây.")
        can_retry = ledger().backend_choice(CONFIGURED_SOLVER or "sqa")["backend"] == window["backend"]
        if st.button("Thử lại cửa sổ", disabled=not can_retry or worker_busy() or len(window["runs"]) >= MAX_ATTEMPTS):
            submitted = dispatch(lambda: ledger().retry_window(window["id"], max_attempts=MAX_ATTEMPTS,
                                                               timeout=WINDOW_TIMEOUT),
                                 window["backend"], PriorityConfig(**window["config"]))
            if not submitted:
                st.session_state.notice = "Phiên đang chờ xử lý hoặc đã được tiếp nhận."
            navigate(RUNS, window=window["id"])




@st.fragment(run_every=None if SOURCE_MODE == "recorded" else 3.0)
def run_view():
    store = ledger()
    page = st.session_state.get("run_page", 1)
    windows = store.window_list(PAGE_SIZE, (page - 1) * PAGE_SIZE)
    if not st.session_state.focus_window and windows:
        st.session_state.focus_window = windows[0]["id"]
    left, right = st.columns([1, 2.35], gap="medium")
    with left, st.container(border=True, key="run_queue"):
        st.markdown("## Cửa sổ xử lý")
        if not windows:
            st.caption("Trang này chưa có cửa sổ xử lý.")
        for row in windows:
            title = list_label(row["id"][:10], f"{STATE_LABELS[row['status']]} · lần {row['attempt']}")
            if st.button(title, key="window_" + row["id"], width="stretch",
                         type="primary" if row["id"] == st.session_state.focus_window else "secondary"):
                navigate(RUNS, window=row["id"])
        st.number_input("Trang danh sách phiên", min_value=1, step=1, key="run_page")
    active = store.window(st.session_state.focus_window)
    if active:
        with right, st.container(border=True, key="run_workspace"):
            show_run(active)


def journal():
    store = ledger()
    search, paging = st.columns([3, 1])
    with search:
        query = st.text_input("Tìm bằng mã hồ sơ hoặc mã phiên", key="journal_query",
                              on_change=lambda: st.session_state.update(journal_page=1))
    with paging:
        page = st.number_input("Trang nhật ký", min_value=1, step=1, key="journal_page")
    entries = store.journal(query, PAGE_SIZE, (page-1)*PAGE_SIZE)
    if not entries:
        st.info("Chưa có mục nhật ký phù hợp.")
    for i,entry in enumerate(entries):
        txid, wid = entry["txid"], entry["window_id"]
        kind = {"admit": "Tiếp nhận", "decide": "Quyết định",
                "run": "Lần xử lý cửa sổ", "casework": "Xử lý hồ sơ"}[entry["kind"]]
        detail = entry["detail"]
        if entry["kind"] in ("decide", "casework"):
            action, separator, reason = detail.partition(" · ")
            if entry["kind"] == "casework" and action in casework.RESPONSE_REQUEST:
                kind = "Phản hồi khách hàng"
            labels = casework.OPERATIONS if entry["kind"] == "casework" else ACTIONS
            detail = labels.get(action, action) + separator + reason
        else:
            detail = STATE_LABELS.get(detail, detail)
        st.markdown(
            '<article class="log-entry">'
            f'<div class="log-meta"><strong>{escape(kind)}</strong>'
            f'<span>{utc(entry["at"])}</span></div><div class="log-id">{escape(txid or wid or "")}</div>'
            f'<p><b>{escape(entry["actor"] or "Hệ thống")}</b> · {escape(detail)}</p></article>',
            unsafe_allow_html=True,
        )
        with st.container(key=f"journal_links_{i}", horizontal=True):
            suffix = f"_{entry['entry_id']}" if entry["kind"] == "casework" else ""
            if txid:
                st.button("Mở hồ sơ", key=f"log_case_{entry['kind']}_{txid}{suffix}",
                          on_click=set_destination, args=(DESK,), kwargs={"case": txid})
            if wid:
                st.button("Mở phiên", key=f"log_window_{wid}_{i}",
                          on_click=set_destination, args=(RUNS,), kwargs={"window": wid})


def cockpit_query():
    return {"view": "cockpit"} if st.query_params.get("view") == "cockpit" else None


def workspace():
    if not st.session_state.get("dashboard_operator"):
        st.switch_page(LOGIN_PAGE, query_params=cockpit_query())

    try:
        if not st.session_state.get("stream_ready"):
            initialize_stream()
    except (ValueError, OSError, sqlite3.Error):
        logging.exception("Cannot initialize operational stream")
        st.error("Chưa mở được nguồn giao dịch. Người vận hành có thể kiểm tra cấu hình và nhật ký máy chủ.")
        return

    if cockpit_query():
        from dashboard.cockpit_renderer import build_cockpit_html
        st.html("""
        <style>
        .stMain .block-container { max-width: 100% !important; padding: 0 !important; margin: 0 !important; }
        iframe { width: 100% !important; min-height: 100vh !important; border: none !important; }
        </style>
        """)
        operator_session_bar()
        st.components.v1.html(build_cockpit_html(ledger(), ACTIONS), height=1280, scrolling=True)
        return

    source_name = {"recorded": "IBM AMLworld tổng hợp", "replay": DATASET_LABEL,
                   "loop": "IBM AMLworld HI-Small · toàn bộ dữ liệu tổng hợp" if DATASET == "amlworld_full" else DATASET_LABEL,
                   "live": "Giao dịch đang tiếp nhận"}.get(SOURCE_MODE, SOURCE_MODE)
    mode_name = {"recorded": "Bản ghi", "live": "Luồng trực tiếp", "replay": "Phát lại CSV",
                 "loop": "Phát lặp"}.get(SOURCE_MODE, SOURCE_MODE)
    source_caption = "Dữ liệu tổng hợp" if "tổng hợp" in source_name or "mô phỏng" in source_name else "Nguồn dữ liệu"
    st.html(
        '<header class="qfg-header"><div class="wrap"><div class="qfg-brand"><h1>Q-FraudGuard</h1></div>'
        f'<div class="qfg-status">{escape(mode_name)}</div>'
        f'<details class="qfg-disclosure qfg-source"><summary>{escape(source_caption)}</summary>'
        f'<div>{escape(source_name)}</div></details></div></header>',
    )
    if area := st.session_state.pop("next_area", None):
        st.session_state.area = area
    areas = [DESK, RUNS, JOURNAL] if SOURCE_MODE == "recorded" else [DESK, CUSTOMER, RUNS, JOURNAL]
    with st.container(key="workspace_controls", horizontal=True, vertical_alignment="center",
                      horizontal_alignment="distribute"):
        area = st.segmented_control("Không gian làm việc", areas, default=None if "area" in st.session_state else DESK,
                                    required=True, key="area", label_visibility="collapsed")
        operator_session_bar()
    if notice := st.session_state.pop("notice", None):
        st.toast(notice)
    stream_metrics()
    if area == DESK:
        desk()
    elif area == RUNS:
        run_view()
    elif area == CUSTOMER:
        if SOURCE_MODE == "live":
            live_customer_panel()
        else:
            customer_panel()
    else:
        with st.container(border=True, key="journal_workspace"):
            journal()


def home():
    target = DASHBOARD_PAGE if st.session_state.get("dashboard_operator") else LOGIN_PAGE
    st.switch_page(target, query_params=cockpit_query())


LOGIN_PAGE = st.Page(login, title="Đăng nhập · Q-FraudGuard", url_path="login")
DASHBOARD_PAGE = st.Page(workspace, title="Bàn điều tra · Q-FraudGuard", url_path="dashboard")
st.navigation([st.Page(home, default=True), LOGIN_PAGE, DASHBOARD_PAGE], position="hidden").run()

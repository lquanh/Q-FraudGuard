from __future__ import annotations

from html import escape
from pathlib import Path
import random
import sys
from time import perf_counter
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd
import streamlit as st

from qf_guard.data import normalize_transactions
from qf_guard.explainability import recommended_treatment
from qf_guard.qubo import build_investigation_qubo
from qf_guard.runtime import RealtimeService, get_service
from qf_guard.scoring import RiskScorer
from qf_guard.sqa import OpenJijSQASolver

RED = "#d92d4f"
MAX_VISIBLE_ALERTS = 30
REALTIME_HISTORY_LIMIT = 5_000

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
      .qfg-note {{font-size:.75rem;color:#64748b}}
      .qfg-risk {{color:{RED}}}
      .qfg-warn {{color:#b86b00}}
      div[data-testid="stDataFrame"] {{border:1px solid #e2e8f0;border-radius:.8rem}}
      [data-testid="stToastContainer"] {{top:.65rem!important;bottom:auto!important;
        right:1.25rem!important}}
      .st-key-fraud_alert_stack {{position:fixed;right:1.25rem;top:6.25rem;
        width:370px;max-height:calc(100vh - 7.5rem);overflow-y:auto;z-index:999999;
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
      .fraud-title {{color:#b4233f;font-size:1.02rem;font-weight:800}}
      .fraud-id {{font-size:1.18rem;font-weight:760;margin:.3rem 0 .6rem}}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def service() -> RealtimeService:
    return get_service()


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


def initialize_stream() -> None:
    data = realtime_source()
    model_path = PROJECT_ROOT / "artifacts" / "models" / "paysim_lightgbm_scorer.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
    scorer = RiskScorer.load(model_path)
    threshold = scorer.validation_stats_.get("operational_candidate_threshold")
    if threshold is not None:
        scorer.candidate_threshold_ = float(threshold)
    boundary = max(1, int(len(data) * .8))
    scorer.history_transactions_ = scorer._feature_history(
        data.iloc[max(0, boundary - REALTIME_HISTORY_LIMIT) : boundary]
    )
    pipeline = service().pipeline
    pipeline.scorer = scorer
    pipeline.is_fitted = True
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
        stream_latency=[],
        fraud_alerts=[],
        alerts_panel_hidden=False,
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


def optimize_candidates(candidates: pd.DataFrame, cursor: int) -> set[int]:
    if candidates.empty:
        return set()
    edges: list[tuple[int, int, float]] = []
    for left in range(len(candidates)):
        left_accounts = {candidates.at[left, "sender_id"], candidates.at[left, "receiver_id"]}
        for right in range(left + 1, len(candidates)):
            right_accounts = {
                candidates.at[right, "sender_id"],
                candidates.at[right, "receiver_id"],
            }
            if left_accounts & right_accounts:
                edges.append((left, right, .5))
    risks = candidates["risk_score"].astype(float).tolist()
    k = max(1, round(len(risks) * .25))
    model = build_investigation_qubo(
        risks,
        edges,
        k=k,
        connectivity_strength=.25,
        penalty_strength=max(4.0, sum(risks) + sum(edge[2] for edge in edges) + 1),
        name=f"realtime_{cursor}",
    )
    result = OpenJijSQASolver(num_reads=32, num_sweeps=60, trotter=4).solve(model)
    return {int(candidates.at[index, "index"]) for index in result.selected_indices}


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
    candidates = scored.loc[scored["is_candidate"]].reset_index()
    selected = optimize_candidates(candidates, cursor)
    hold_threshold = float(
        service().pipeline.scorer.validation_stats_.get("alert_hold_threshold", .6)
    )
    rows = list(st.session_state["stream_rows"])
    alerts = list(st.session_state["fraud_alerts"])
    for index, row in scored.iterrows():
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
            "Loại": row["type"],
            "Số tiền": float(row["amount"]),
            "Risk": risk,
            "Supervised": float(row.get("supervised_score", 0)),
            "Anomaly": float(row.get("anomaly_score", 0)),
            "Graph": float(row.get("graph_score", 0)),
            "Rule": float(row.get("rule_score", 0)),
            "Ứng viên": is_candidate,
            "SQA chọn": index in selected,
            "Nhãn fraud": bool(is_fraud),
            "Lý do": row.get("reason_codes", ""),
        }
        rows.append(record)
        if is_candidate:
            treatment = recommended_treatment(
                {
                    "risk_score": .85 if risk >= hold_threshold else .65,
                    "reason_codes": record["Lý do"],
                }
            )
            priority = {"hold_and_escalate": 3, "step_up_verification": 2}.get(
                treatment["recommended_action"], 1
            )
            alerts.append({**record, **treatment, "action_priority": priority})
    alerts.sort(key=lambda item: (item["action_priority"], item["Risk"]), reverse=True)
    st.session_state["fraud_alerts"] = alerts
    st.session_state["stream_rows"] = rows[-200:]
    st.session_state["stream_latency"] = (st.session_state["stream_latency"] + [elapsed])[-100:]
    st.session_state["stream_cursor"] = stop


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
    )
    for column, value in zip(st.columns(4), values):
        with column:
            metric_card(*value)
    rows = st.session_state["stream_rows"]
    if rows:
        frame = pd.DataFrame(rows).iloc[::-1].copy()
        frame["Nhãn thật"] = frame["Nhãn fraud"].map({True: "FRAUD", False: "NORMAL"})
        view = frame[
            [
                "Thời gian",
                "Mã giao dịch",
                "Loại",
                "Số tiền",
                "Risk",
                "Ứng viên",
                "SQA chọn",
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
                "SQA chọn": st.column_config.CheckboxColumn(),
            },
        )


@st.fragment(run_every=1.0)
def alert_panel() -> None:
    render_alerts()


realtime_panel()
alert_panel()

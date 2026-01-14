"""Shared Streamlit UI helpers for the loan assistant pages."""

from datetime import datetime
from pathlib import Path
import html

import streamlit as st

from agents import warm_policy_cache


def apply_base_styles() -> None:
    """Inject shared styles for cards and highlights."""
    st.markdown(
        """
        <style>
        :root {
            --card-bg: rgba(255, 255, 255, 0.85);
            --card-border: rgba(15, 23, 42, 0.1);
            --card-shadow: 0 15px 30px rgba(15, 23, 42, 0.08);
            --accent-bg: linear-gradient(135deg, #2563eb, #7c3aed);
        }
        .card {
            padding: 1.2rem 1.5rem;
            margin-bottom: 1rem;
            border-radius: 18px;
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            box-shadow: var(--card-shadow);
        }
        .card.accent {
            background: var(--accent-bg);
            color: #fff;
        }
        .card-title {
            font-weight: 600;
            font-size: 1rem;
            margin-bottom: 0.6rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .card ul {
            list-style: none;
            padding-left: 0;
            margin: 0;
        }
        .card ul li {
            display: flex;
            justify-content: space-between;
            padding: 0.35rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.15);
        }
        .card ul li:last-child {
            border-bottom: none;
        }
        .card ul li span {
            font-weight: 600;
            opacity: 0.85;
        }
        .memo-box {
            border-radius: 16px;
            padding: 1.5rem;
            background: rgba(15, 23, 42, 0.04);
            border: 1px dashed rgba(15, 23, 42, 0.2);
        }
        .approval-progress {
            margin-top: 0.6rem;
        }
        .approval-progress__track {
            width: 100%;
            height: 10px;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.08);
            overflow: hidden;
            box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
        }
        .approval-progress__fill {
            height: 100%;
            background: linear-gradient(90deg, #16a34a, #22c55e);
            transition: width 0.4s ease;
        }
        .approval-progress__label {
            margin-top: 0.35rem;
            font-weight: 600;
            color: #0f172a;
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_session_state() -> None:
    """Initialize session state defaults used across pages."""
    if "pending_application" not in st.session_state:
        st.session_state.pending_application = None
    if "officer_decision" not in st.session_state:
        st.session_state.officer_decision = "Approve"
    if "officer_reason" not in st.session_state:
        st.session_state.officer_reason = ""
    if "decision_stats" not in st.session_state:
        st.session_state.decision_stats = {"approved": 0, "rejected": 0}
    if "interaction_history" not in st.session_state:
        st.session_state.interaction_history = []


def get_policy_cache_ready() -> bool:
    """Warm the policy cache once per session and return readiness."""
    if "policy_cache_ready" not in st.session_state:
        st.session_state.policy_cache_ready = warm_policy_cache()
    return bool(st.session_state.policy_cache_ready)


def render_info_card(title: str, info_pairs, accent: bool = False) -> None:
    """Render a stylized card with label/value rows."""
    if not info_pairs:
        return
    items = "".join(
        f"<li><span>{html.escape(str(label))}</span><span>{html.escape(str(value))}</span></li>"
        for label, value in info_pairs
    )
    st.markdown(
        f"""
        <div class="card {'accent' if accent else ''}">
            <div class="card-title">{html.escape(title)}</div>
            <ul>{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_text_card(title: str, text: str, accent: bool = False) -> None:
    """Render a text block inside a stylized card."""
    safe_text = html.escape(text or "")
    st.markdown(
        f"""
        <div class="card {'accent' if accent else ''}">
            <div class="card-title">{html.escape(title)}</div>
            <p style="margin:0; line-height:1.5;">{safe_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_decision_stats_sidebar(placeholder=None) -> None:
    """Render sidebar metrics based on the latest decision stats."""
    if placeholder is None:
        target = st.sidebar
    else:
        placeholder.empty()
        target = placeholder.container()
    with target:
        st.header("📊 Decision Stats")
        approved = st.session_state.decision_stats["approved"]
        rejected = st.session_state.decision_stats["rejected"]
        total = approved + rejected
        st.metric("Approved", approved)
        st.metric("Rejected", rejected)
        st.metric("Total Decisions", total)
        if total:
            approval_rate = approved / total * 100.0
            fill_percent = min(max(approval_rate, 0.0), 100.0)
            st.markdown(
                f"""
                <div class="approval-progress">
                    <div class="approval-progress__track">
                        <div class="approval-progress__fill" style="width: {fill_percent:.1f}%;"></div>
                    </div>
                    <div class="approval-progress__label">Approval rate: {approval_rate:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("No decisions recorded yet.")


def append_history_entry(prompt: str, payload: dict) -> None:
    """Record completed interactions for later review."""
    st.session_state.interaction_history.insert(
        0,
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "prompt": prompt,
            "payload": payload,
        },
    )


def get_data_checks() -> list[tuple[str, Path]]:
    """Return list of data assets to validate."""
    base_dir = Path(__file__).resolve().parent
    return [
        ("Credit scores data", base_dir / "data" / "credit_scores.csv"),
        ("Account status data", base_dir / "data" / "account_status.csv"),
        ("PR status data", base_dir / "data" / "pr_status.csv"),
        ("Policy PDF folder", base_dir / "policies"),
    ]

"""Streamlit UI for the loan assistant with modern cards and officer workflow."""

import html
from datetime import datetime
from pathlib import Path

import streamlit as st

from agents import handle_user_input, warm_policy_cache


# Prebuild FAISS so the first Streamlit interaction doesn’t block on embeddings.
POLICY_CACHE_READY = warm_policy_cache()

# ===========================================================
# Streamlit Page Setup
# ===========================================================

st.set_page_config(page_title="Loan Assistant", layout="wide")

if not POLICY_CACHE_READY:
    st.error(
        "Policy database failed to load. Ensure policy PDFs are present and reload the app."
    )

# Inject a lightweight design system to modernize Streamlit's default look.
# Helper renderers and layout utilities

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

# Initialize session state for storing loan evaluations awaiting officer decision
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


# ===========================================================
# Sidebar navigation & statistics
# ===========================================================

sidebar_placeholder = st.sidebar.empty()
render_decision_stats_sidebar(sidebar_placeholder)

page = st.sidebar.radio(
    "Navigate",
    options=["Assistant", "Past Results", "Test/Check"],
)


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


if page == "Assistant":
    st.title("🏦 Loan Assistant Console")
    st.write("Ask any loan-related question or request a loan evaluation.")

    # ===========================================================
    # User Input Section
    # ===========================================================

    # Wide textarea with the CTA button stacked below for clearer flow.
    user_text = st.text_area(
        "Enter your question or loan request:",
        placeholder="Ask for policy guidance or request a customer evaluation...",
        height=140,
    )
    submit = st.button("Submit", use_container_width=True)

    # ===========================================================
    # Handle user input
    # ===========================================================

    if submit:
        if not user_text.strip():
            st.warning("Please enter a valid question.")
        else:
            with st.spinner("Processing your request..."):
                result = handle_user_input(user_text)
            append_history_entry(user_text.strip(), result)
            # ---------------------------
            # Determine response type and render matching view
            # ERROR HANDLING
            # ---------------------------
            if result.get("type") == "error":
                st.error(result.get("message", "Unknown error"))
                st.session_state.pending_application = None

            # ---------------------------
            # GENERAL Q&A RESPONSE
            # ---------------------------
            elif result.get("type") == "qa":
                render_text_card(
                    "Answer", result.get("answer", "No answer provided."), accent=True
                )
                st.session_state.pending_application = None

            # ---------------------------
            # LOAN APPLICATION RESPONSE
            # ---------------------------
            elif result.get("type") == "loan_application":
                customer = result.get("customer", {})
                assessment = result.get("ai_assessment", {})
                memo = result.get("letter", "")

                st.header("📄 Loan Application Evaluation")

                # Customer & assessment snapshot
                info_col, assessment_col = st.columns(2, gap="large")

                customer_pairs = [
                    ("Customer ID", customer.get("id", "—")),
                    ("Name", customer.get("name", "—")),
                    ("Nationality", customer.get("nationality", "—")),
                    ("PR Status", customer.get("pr_status", "—")),
                    ("Account Status", customer.get("account_status", "—")),
                    ("Credit Score", customer.get("credit_score", "—")),
                ]

                assessment_pairs = [
                    ("AI Recommendation", assessment.get("ai_recommendation", "Pending")),
                    ("Risk Tier", assessment.get("risk", "Unknown")),
                    ("Interest Rate", assessment.get("interest_rate", "Not set")),
                    ("PR Status Used", assessment.get("pr_status_used", "—")),
                ]

                with info_col:
                    render_info_card("Customer Snapshot", customer_pairs)
                with assessment_col:
                    render_info_card("AI Assessment", assessment_pairs, accent=True)

                policy_notes = assessment.get("policy_notes") or ""
                if policy_notes:
                    render_text_card("Policy Evidence", policy_notes)

                with st.expander("AI Draft Letter / Memo", expanded=True):
                    safe_memo = html.escape(memo or "No memo provided.")
                    st.markdown(
                        f'<div class="memo-box">{safe_memo}</div>',
                        unsafe_allow_html=True,
                    )

                # Store pending application for officer approval
                st.session_state.pending_application = {
                    "customer": customer,
                    "assessment": assessment,
                    "memo": memo,
                }
                ai_choice = assessment.get("ai_recommendation", "Approve").capitalize()
                st.session_state.officer_decision = (
                    ai_choice if ai_choice in ("Approve", "Reject") else "Approve"
                )
                st.session_state.officer_reason = ""

    # ===========================================================
    # Officer Approval Section
    # ===========================================================

    if st.session_state.pending_application:
        st.divider()
        st.header("📝 Loan Officer Decision")
        st.info(
            "The AI provides recommendations only. Please record the human loan officer's final decision."
        )

        decision = st.radio(
            "Select a final decision:",
            options=["Approve", "Reject"],
            horizontal=True,
            key="officer_decision",
        )

        # Officers must provide a justification to satisfy audit/compliance needs.
        reason = st.text_area(
            "Loan officer justification (required):",
            key="officer_reason",
            placeholder="Explain the rationale for approving or rejecting this application...",
            help="Provide compliance-ready reasoning for the recorded decision.",
        )

        if st.button("Record Final Decision", type="primary", use_container_width=True):
            if not reason.strip():
                st.warning("Please provide a justification before recording the decision.")
            else:
                if decision == "Approve":
                    st.success("Loan Approved ✔ (recorded)")
                    st.session_state.decision_stats["approved"] += 1
                else:
                    st.error("Loan Rejected ✖ (recorded)")
                    st.session_state.decision_stats["rejected"] += 1
                st.write("**Officer justification**")
                st.write(reason.strip())
                st.json(st.session_state.pending_application)
                st.session_state.pending_application = None
                render_decision_stats_sidebar(sidebar_placeholder)

elif page == "Past Results":
    st.title("🗂️ Past Results")
    st.write("Review previous questions, evaluations, and outcomes.")
    history = st.session_state.interaction_history
    if not history:
        st.info("No interactions recorded yet. Submit a question to populate this list.")
    else:
        for entry in history:
            payload = entry.get("payload", {})
            entry_type = payload.get("type", "unknown").replace("_", " ").title()
            with st.expander(
                f"{entry_type} · {entry.get('timestamp', 'Unknown time')}",
                expanded=False,
            ):
                render_text_card("Prompt", entry.get("prompt", ""))
                if payload.get("type") == "qa":
                    render_text_card(
                        "Answer", payload.get("answer", "No answer provided.")
                    )
                elif payload.get("type") == "loan_application":
                    customer = payload.get("customer", {})
                    assessment = payload.get("ai_assessment", {})
                    render_info_card(
                        "Customer Snapshot",
                        [
                            ("Customer ID", customer.get("id", "—")),
                            ("Name", customer.get("name", "—")),
                            ("Credit Score", customer.get("credit_score", "—")),
                            ("Account Status", customer.get("account_status", "—")),
                        ],
                    )
                    render_info_card(
                        "AI Assessment",
                        [
                            ("Recommendation", assessment.get("ai_recommendation", "—")),
                            ("Risk Tier", assessment.get("risk", "—")),
                            ("Interest Rate", assessment.get("interest_rate", "—")),
                        ],
                        accent=True,
                    )
                    st.caption("Memo preview")
                    st.write(payload.get("letter", "No memo provided."))
                elif payload.get("type") == "error":
                    st.error(payload.get("message", "Unknown error"))
                st.json(payload)

elif page == "Test/Check":
    st.title("✅ Test & Check")
    st.write("Quick checks to confirm data and policy resources are ready.")

    st.subheader("System readiness")
    st.metric("Policy cache ready", "Yes" if POLICY_CACHE_READY else "No")

    base_dir = Path(__file__).resolve().parent
    data_checks = [
        ("Credit scores data", base_dir / "data" / "credit_scores.csv"),
        ("Account status data", base_dir / "data" / "account_status.csv"),
        ("PR status data", base_dir / "data" / "pr_status.csv"),
        ("Policy PDF folder", base_dir / "policies"),
    ]

    st.subheader("Data availability")
    for label, path in data_checks:
        if path.exists():
            st.success(f"{label}: found")
        else:
            st.error(f"{label}: missing")

    st.subheader("Checklist")
    st.markdown(
        """
        - ✅ Policy cache built (first page load).
        - ✅ Decision stats updated after each officer approval.
        - ✅ Past results page captures completed interactions.
        """
    )

"""Streamlit UI for the loan assistant with modern cards and officer workflow."""

import html

import streamlit as st

from agents import handle_user_input

# ===========================================================
# Streamlit Page Setup
# ===========================================================

st.set_page_config(page_title="Loan Assistant", layout="wide")

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

st.title("🏦 Loan Assistant Console")
st.write("Ask any loan-related question or request a loan evaluation.")

# Initialize session state for storing loan evaluations awaiting officer decision
if "pending_application" not in st.session_state:
    st.session_state.pending_application = None
if "officer_decision" not in st.session_state:
    st.session_state.officer_decision = "Approve"
if "officer_reason" not in st.session_state:
    st.session_state.officer_reason = ""
if "decision_stats" not in st.session_state:
    st.session_state.decision_stats = {"approved": 0, "rejected": 0}


# ===========================================================
# Sidebar statistics dashboard
# ===========================================================

with st.sidebar:
    st.header("📊 Decision Stats")
    total = (
        st.session_state.decision_stats["approved"]
        + st.session_state.decision_stats["rejected"]
    )
    st.metric("Approved", st.session_state.decision_stats["approved"])
    st.metric("Rejected", st.session_state.decision_stats["rejected"])
    st.metric("Total Decisions", total)
    if total:
        approval_rate = (
            st.session_state.decision_stats["approved"] / total * 100.0
        )
        st.progress(
            min(max(approval_rate / 100.0, 0.0), 1.0),
            text=f"Approval rate: {approval_rate:.1f}%",
        )
    else:
        st.info("No decisions recorded yet.")


# ===========================================================
# User Input Section
# ===========================================================

# Wide textarea keeps long prompts comfortable while a skinny column
# houses the CTA button for a tidy look across breakpoints.
input_col, button_col = st.columns([4, 1])
with input_col:
    user_text = st.text_area(
        "Enter your question or loan request:",
        placeholder="Ask for policy guidance or request a customer evaluation...",
        height=140,
    )
with button_col:
    st.markdown("\n\n")
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
            render_text_card("Answer", result.get("answer", "No answer provided."), accent=True)
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

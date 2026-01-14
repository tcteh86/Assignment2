"""Streamlit UI for the loan assistant with modern cards and officer workflow."""

import html

import streamlit as st

from agents import handle_user_input
from ui_components import (
    append_history_entry,
    apply_base_styles,
    ensure_session_state,
    get_policy_cache_ready,
    render_decision_stats_sidebar,
    render_info_card,
    render_text_card,
)

# ===========================================================
# Streamlit Page Setup
# ===========================================================

st.set_page_config(page_title="Loan Assistant", layout="wide")
apply_base_styles()
ensure_session_state()

policy_cache_ready = get_policy_cache_ready()
if not policy_cache_ready:
    st.error(
        "Policy database failed to load. Ensure policy PDFs are present and reload the app."
    )

sidebar_placeholder = st.sidebar.empty()
render_decision_stats_sidebar(sidebar_placeholder)

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

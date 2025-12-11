import streamlit as st
from agents import handle_user_input

# ===========================================================
# Streamlit Page Setup
# ===========================================================

st.set_page_config(page_title="Loan Assistant", layout="wide")

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

user_text = st.text_input(
    "Enter your question or loan request:",
    placeholder="e.g., What is the policy for high-risk loans? OR Evaluate loan for Andy",
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

        # ---------------------------
        # ERROR HANDLING
        # ---------------------------
        if result.get("type") == "error":
            st.error(result.get("message", "Unknown error"))
            st.session_state.pending_application = None

        # ---------------------------
        # GENERAL Q&A RESPONSE
        # ---------------------------
        elif result.get("type") == "qa":
            st.success("Answer:")
            st.write(result.get("answer", "No answer provided."))
            st.session_state.pending_application = None

        # ---------------------------
        # LOAN APPLICATION RESPONSE
        # ---------------------------
        elif result.get("type") == "loan_application":
            customer = result.get("customer", {})
            assessment = result.get("ai_assessment", {})
            memo = result.get("letter", "")

            st.header("📄 Loan Application Evaluation")

            # Customer summary card
            st.subheader("Customer Information")
            st.json(customer)

            # AI recommendation
            st.subheader("AI Assessment Summary")
            st.json(assessment)

            # Long Memo / Letter
            st.subheader("AI Draft Letter / Memo")
            st.write(memo)

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

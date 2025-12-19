"""Streamlit prototype for a GenAI-assisted loan risk assistant."""

from __future__ import annotations

import os
from typing import Dict, List

import streamlit as st

from services.llm import LLMClient
from services.rag import retrieve_policy_rules
from services.risk_engine import evaluate_application, normalize_application_form


st.set_page_config(page_title="GenAI Loan Risk Assistant", layout="wide")

st.title("🏦 GenAI-Assisted Loan Risk Assessment")
st.write(
    "This prototype demonstrates how a loan officer can triage applications with GenAI-backed "
    "reasoning, policy citations, and human-in-the-loop controls."
)


@st.cache_resource
def _build_llm_client() -> LLMClient:
    return LLMClient(api_key=os.getenv("OPENAI_API_KEY"))


llm_client = _build_llm_client()

with st.sidebar:
    st.header("Configuration")
    api_key_present = bool(os.getenv("OPENAI_API_KEY"))
    st.metric("OPENAI_API_KEY set", "Yes" if api_key_present else "No")
    st.caption("Set the OPENAI_API_KEY environment variable before running Streamlit.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Loan Application Form")
    with st.form("loan_form", clear_on_submit=False):
        name = st.text_input("Full Name")
        national_id = st.text_input("National ID / Passport")
        age = st.number_input("Age", min_value=18, max_value=80, value=30)
        employment_status = st.selectbox(
            "Employment Status",
            ["Salaried", "Self-Employed", "Contract", "Unemployed"],
        )
        monthly_income = st.number_input("Monthly Income (USD)", min_value=0.0, value=6000.0, step=500.0)
        monthly_liabilities = st.number_input(
            "Monthly Liabilities (USD)", min_value=0.0, value=1200.0, step=200.0
        )
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
        loan_amount = st.number_input("Requested Loan Amount (USD)", min_value=1000.0, value=50000.0, step=1000.0)
        loan_purpose = st.text_area(
            "Loan Purpose",
            placeholder="E.g., debt consolidation, education, home renovation, small business",
        )
        documents_provided = st.multiselect(
            "Documents Provided",
            options=[
                "Government ID",
                "Bank Statements (3 months)",
                "Employment Letter",
                "Tax Return",
                "Collateral Documentation",
            ],
        )
        existing_customer = st.checkbox("Existing customer")
        human_review_toggle = st.checkbox("Mark for human review by default", value=False)
        submitted = st.form_submit_button("Evaluate Application", type="primary")

form_data: Dict[str, List[str] | str | float | int | bool] = {
    "name": name,
    "national_id": national_id,
    "age": age,
    "employment_status": employment_status,
    "monthly_income": monthly_income,
    "monthly_liabilities": monthly_liabilities,
    "credit_score": credit_score,
    "loan_amount": loan_amount,
    "loan_purpose": loan_purpose,
    "documents_provided": documents_provided,
    "existing_customer": existing_customer,
    "human_review_toggle": human_review_toggle,
}

with col2:
    st.subheader("Policy Retrieval")
    normalized_snapshot = normalize_application_form(form_data)
    policy_query = f"{normalized_snapshot['loan_purpose']} {normalized_snapshot['employment_status']}"
    policies = retrieve_policy_rules(policy_query)
    for policy in policies:
        st.markdown(f"**{policy['id']} – {policy['title']}**")
        st.caption(policy["text"])
        st.progress(min(policy.get("score", 0.0), 1.0))

if submitted:
    with st.spinner("Running GenAI-assisted assessment..."):
        assessment = evaluate_application(form_data, llm_client)

    st.success("Assessment generated.")
    st.subheader("AI Recommendation")
    st.write(f"**Risk Level:** {assessment.risk_level}")
    st.write(f"**Recommendation:** {assessment.recommendation}")
    st.write(f"**Confidence:** {assessment.confidence:.2f}")

    st.markdown("#### Rationale")
    st.write(assessment.rationale)

    st.markdown("#### Policy Citations")
    if assessment.policy_citations:
        for citation in assessment.policy_citations:
            st.write(f"- {citation}")
    else:
        st.write("No specific citations returned.")

    st.markdown("#### Highlighted Risks")
    if assessment.highlighted_risks:
        st.write("\n".join([f"- {item}" for item in assessment.highlighted_risks]))
    else:
        st.write("No critical risks flagged.")

    st.markdown("#### Follow-up Questions")
    if assessment.follow_up_questions:
        st.write("\n".join([f"- {item}" for item in assessment.follow_up_questions]))
    else:
        st.write("No follow-ups suggested.")

    st.markdown("#### Missing Documents")
    if assessment.missing_documents:
        st.write("\n".join([f"- {item}" for item in assessment.missing_documents]))
    else:
        st.write("All core documents present.")

    st.markdown("#### Human Review")
    requires_review = assessment.human_review_required or human_review_toggle
    st.write("🔎 Human review required" if requires_review else "✅ Auto-approve possible (with officer sign-off)")

    st.divider()
    st.caption("Note: This is a prototype; AI output should always be validated by a human underwriter.")

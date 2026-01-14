"""Streamlit page for reviewing previous interactions."""

import streamlit as st

from ui_components import (
    apply_base_styles,
    ensure_session_state,
    render_info_card,
    render_text_card,
)

st.set_page_config(page_title="Past Results", layout="wide")
apply_base_styles()
ensure_session_state()

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
                render_text_card("Answer", payload.get("answer", "No answer provided."))
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

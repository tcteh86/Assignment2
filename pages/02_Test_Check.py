"""Streamlit page for quick system readiness checks."""

import streamlit as st

from ui_components import (
    apply_base_styles,
    ensure_session_state,
    get_data_checks,
    get_policy_cache_ready,
)

st.set_page_config(page_title="Test & Check", layout="wide")
apply_base_styles()
ensure_session_state()

st.title("✅ Test & Check")
st.write("Quick checks to confirm data and policy resources are ready.")

policy_cache_ready = get_policy_cache_ready()

st.subheader("System readiness")
st.metric("Policy cache ready", "Yes" if policy_cache_ready else "No")

st.subheader("Data availability")
for label, path in get_data_checks():
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

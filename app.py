"""
app.py
----------------
Step 4: Streamlit UI for the Data Analyst AI Agent.

Run with:
    export HF_API_TOKEN="hf_your_token"
    streamlit run app.py
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # reads HF_API_TOKEN (and anything else) from a .env file in this folder

from agent import DataAnalystAgent

st.set_page_config(page_title="Data Analyst AI Agent", layout="wide")
st.title("📊 Data Analyst AI Agent")
st.caption("Upload a dataset, ask questions in plain English, get analysis + charts + insights.")

# ---------------------------------------------------------------
# Session state: keep the agent + chat history alive across reruns
# ---------------------------------------------------------------
if "agent" not in st.session_state:
    st.session_state.agent = None
if "history" not in st.session_state:
    st.session_state.history = []  # list of AgentResponse-like dicts
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None

# ---------------------------------------------------------------
# Sidebar: dataset upload + HF token check
# ---------------------------------------------------------------
with st.sidebar:
    st.header("1. Upload dataset")
    uploaded_file = st.file_uploader("CSV, Excel, or JSON", type=["csv", "xlsx", "xls", "json"])

    if not os.environ.get("HF_API_TOKEN"):
        st.warning("HF_API_TOKEN environment variable is not set. Set it before asking questions.")

    if uploaded_file is not None:
        # Only reload the agent if it's a new file
        if st.session_state.dataset_name != uploaded_file.name:
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                st.session_state.agent = DataAnalystAgent(tmp_path)
                st.session_state.dataset_name = uploaded_file.name
                st.session_state.history = []
                st.success(f"Loaded '{uploaded_file.name}'")
            except Exception as e:
                st.error(f"Failed to load dataset: {e}")

    if st.session_state.agent is not None:
        st.header("2. Dataset overview")
        profile = st.session_state.agent.profile
        st.write(f"**Rows:** {profile['n_rows']}  |  **Columns:** {profile['n_columns']}")
        with st.expander("Full data card (LLM context)"):
            st.text(st.session_state.agent.data_card)

# ---------------------------------------------------------------
# Main panel: question box + chat-style history
# ---------------------------------------------------------------
if st.session_state.agent is None:
    st.info("👈 Upload a dataset in the sidebar to get started.")
else:
    question = st.chat_input("Ask a question about your data...")

    if question:
        with st.spinner("Analyzing..."):
            response = st.session_state.agent.ask(question)
        st.session_state.history.append(response)

    # Render history, most recent first
    for response in reversed(st.session_state.history):
        with st.chat_message("user"):
            st.write(response.question)

        with st.chat_message("assistant"):
            if not response.success:
                st.error(f"Something went wrong: {response.error}")
                if response.generated_code:
                    with st.expander("Generated code (failed)"):
                        st.code(response.generated_code, language="python")
                continue

            if response.summary:
                st.write(response.summary)

            if response.chart_path and os.path.exists(response.chart_path):
                st.image(response.chart_path)

            with st.expander("Raw result"):
                st.write(response.raw_result)

            with st.expander("Generated code"):
                st.code(response.generated_code, language="python")

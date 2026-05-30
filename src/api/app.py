# app.py
import streamlit as st
import requests
import os

st.set_page_config(page_title="PDF RAG Chatbot", layout="centered")
st.title("📄 PDF RAG Chatbot")

# Point to backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Fetch status from backend
try:
    status_res = requests.get(f"{BACKEND_URL}/status").json()
    has_vectors = status_res.get("has_vectors", False)
except Exception:
    st.error("Could not connect to the backend server.")
    st.stop()

# Sidebar Uploader
st.sidebar.header("Upload your Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.spinner(f"Uploading {uploaded_file.name}..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            res = requests.post(f"{BACKEND_URL}/upload", files=files)
            if res.status_code == 200:
                st.sidebar.success(f"Indexed: {uploaded_file.name}")
            else:
                st.sidebar.error(f"Failed to process {uploaded_file.name}")

# Clear DB Option
if st.sidebar.button("Clear Database"):
    with st.spinner("Clearing backend index..."):
        requests.post(f"{BACKEND_URL}/clear")
    st.sidebar.warning("Database cleared! Reloading...")
    st.rerun()

# Chat Interface
if has_vectors:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask a question about your documents..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                res = requests.post(f"{BACKEND_URL}/query", json={"query": user_query})
                if res.status_code == 200:
                    answer = res.json().get("answer")
                else:
                    answer = "Error querying backend."
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("No documents found in backend. Please upload one or more PDFs in the sidebar.")
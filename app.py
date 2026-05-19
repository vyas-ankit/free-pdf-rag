# app.py

import streamlit as st
import os
import tempfile
import rag_logic

st.set_page_config(page_title="PDF RAG Chatbot", layout="centered")
st.title("📄 PDF RAG Chatbot (Modular)")

# 1. Retrieve Secrets
if "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]
else:
    groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if "PINECONE_API_KEY" in st.secrets:
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
else:
    pinecone_api_key = st.sidebar.text_input("Enter Pinecone API Key", type="password")

if not groq_api_key or not pinecone_api_key:
    st.info("Please configure your API keys to continue.")
    st.stop()

# 2. Caching resource generation inside the Streamlit context
@st.cache_resource
def load_cached_embeddings():
    return rag_logic.get_embeddings()

embeddings = load_cached_embeddings()

# 3. Initialize Database Client
pc = rag_logic.init_pinecone(pinecone_api_key)

# Sidebar Uploader
st.sidebar.header("Upload your Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing PDFs and indexing..."):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Delegate backend ingestion
            rag_logic.process_and_upload_pdf(tmp_path, embeddings, pinecone_api_key)
            os.unlink(tmp_path)
        st.sidebar.success("PDFs uploaded and indexed on Pinecone!")

# Load vector store instance
vector_store = rag_logic.get_vector_store(embeddings, pinecone_api_key)

# Administration options
if st.sidebar.button("Clear Pinecone Database"):
    with st.spinner("Clearing index..."):
        rag_logic.clear_database(pc)
    st.sidebar.warning("Database cleared! Please reload the page.")
    st.rerun()

has_vectors = rag_logic.check_index_has_vectors(pc)

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
                # Delegate computation to logic file
                answer = rag_logic.query_rag(user_query, vector_store, groq_api_key)
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("No documents found in Pinecone. Please upload one or more PDFs in the sidebar.")
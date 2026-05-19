# app.py

import streamlit as st
import os
import tempfile
import rag_logic

st.set_page_config(page_title="PDF RAG Chatbot (S3 + Pinecone)", layout="centered")
st.title("📄 S3 + Pinecone PDF RAG Chatbot")

# 1. Retrieve Secrets
# Core Keys
if "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]
else:
    groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if "PINECONE_API_KEY" in st.secrets:
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
else:
    pinecone_api_key = st.sidebar.text_input("Enter Pinecone API Key", type="password")

# AWS Keys
if all(key in st.secrets for key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION", "AWS_S3_BUCKET_NAME"]):
    aws_access_key = st.secrets["AWS_ACCESS_KEY_ID"]
    aws_secret_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
    aws_region = st.secrets["AWS_DEFAULT_REGION"]
    s3_bucket = st.secrets["AWS_S3_BUCKET_NAME"]
else:
    st.sidebar.subheader("AWS Credentials")
    aws_access_key = st.sidebar.text_input("AWS Access Key ID", type="password")
    aws_secret_key = st.sidebar.text_input("AWS Secret Access Key", type="password")
    aws_region = st.sidebar.text_input("AWS Region", value="us-east-1")
    s3_bucket = st.sidebar.text_input("S3 Bucket Name")

if not all([groq_api_key, pinecone_api_key, aws_access_key, aws_secret_key, aws_region, s3_bucket]):
    st.info("Please configure all API Keys and AWS Credentials to continue.")
    st.stop()

# 2. Caching resource generation inside the Streamlit context
@st.cache_resource
def load_cached_embeddings():
    return rag_logic.get_embeddings()

embeddings = load_cached_embeddings()

# 3. Initialize Clients
pc = rag_logic.init_pinecone(pinecone_api_key)
s3_client = rag_logic.get_s3_client(aws_access_key, aws_secret_key, aws_region)

# Sidebar Uploader
st.sidebar.header("Upload your Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save files inside a /documents folder in S3
        s3_key = f"documents/{uploaded_file.name}"
        
        with st.spinner(f"Uploading {uploaded_file.name} to S3 and indexing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # A. Write PDF to AWS S3 bucket
                uploaded = rag_logic.upload_to_s3(tmp_path, s3_bucket, s3_key, s3_client)
                
                if uploaded:
                    # B. Chunk, vectorize and save references on Pinecone
                    rag_logic.process_and_upload_pdf(tmp_path, embeddings, pinecone_api_key, s3_bucket, s3_key)
                    st.sidebar.success(f"Indexed: {uploaded_file.name} (Saved on S3)")
                else:
                    st.sidebar.error(f"S3 upload failed for: {uploaded_file.name}")
            finally:
                os.unlink(tmp_path)

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
                answer = rag_logic.query_rag(user_query, vector_store, groq_api_key)
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("No documents found in Pinecone. Please upload one or more PDFs in the sidebar.")
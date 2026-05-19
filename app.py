import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Page Config
st.set_page_config(page_title="PDF RAG Chatbot", layout="centered")
st.title("📄 PDF RAG Chatbot (Free Tier)")

# 1. Retrieve Groq API Key
if "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]
else:
    groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if not groq_api_key:
    st.info("Please add your Groq API Key in Streamlit Secrets or the sidebar to continue.")
    st.stop()

# 2. Initialize Embeddings (Cached)
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embeddings()

# Sidebar for uploading PDFs
st.sidebar.header("Upload your Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

vector_store = None

if uploaded_files:
    with st.spinner("Processing PDFs..."):
        documents = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            loader = PyPDFLoader(tmp_path)
            documents.extend(loader.load())
            os.unlink(tmp_path)
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        vector_store = FAISS.from_documents(splits, embeddings)
        st.sidebar.success("PDFs uploaded and indexed!")

# Helper function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chat Interface
if vector_store:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if user_query := st.chat_input("Ask a question about your uploaded documents..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Set up modern LCEL retrieval chain
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)
        
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you do not know the answer, say that you do not know.\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # Build LCEL Chain
        rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Invoke LCEL chain (returns a clean string answer directly)
                answer = rag_chain.invoke(user_query)
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Please upload one or more PDFs in the sidebar to start querying.")
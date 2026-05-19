# rag_logic.py

import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from prompts import SYSTEM_RAG_PROMPT

INDEX_NAME = "free-pdf-index"

def get_embeddings():
    """Initializes and returns the Hugging Face embedding model."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def init_pinecone(api_key: str) -> Pinecone:
    """Initializes the Pinecone client and ensures the serverless index exists."""
    pc = Pinecone(api_key=api_key)
    if not pc.has_index(INDEX_NAME):
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
    return pc

def process_and_upload_pdf(file_path: str, embeddings, api_key: str):
    """Loads a PDF, chunks the text, and writes vectors into Pinecone."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    PineconeVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        index_name=INDEX_NAME,
        pinecone_api_key=api_key
    )

def get_vector_store(embeddings, api_key: str) -> PineconeVectorStore:
    """Returns a vector store handler connected to the Pinecone index."""
    return PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=api_key
    )

def check_index_has_vectors(pc: Pinecone) -> bool:
    """Checks if the Pinecone index contains vectors."""
    try:
        index_stats = pc.Index(INDEX_NAME).describe_index_stats()
        return index_stats.total_vector_count > 0
    except Exception:
        return False

def clear_database(pc: Pinecone):
    """Deletes the entire Pinecone index."""
    if pc.has_index(INDEX_NAME):
        pc.delete_index(INDEX_NAME)

def format_docs(docs) -> str:
    """Helper to join retrieved documents together."""
    return "\n\n".join(doc.page_content for doc in docs)

def query_rag(user_query: str, vector_store: PineconeVectorStore, groq_api_key: str) -> str:
    """Executes the RAG pipeline using LCEL and returns the model response."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_RAG_PROMPT),
        ("human", "{input}"),
    ])
    
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(user_query)
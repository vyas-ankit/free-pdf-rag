# rag_logic.py

import os
import time
import boto3
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
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def init_pinecone(api_key: str) -> Pinecone:
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

def get_s3_client(aws_access_key: str = None, aws_secret_key: str = None, region: str = None):
    """Initializes and returns the AWS S3 client."""
    if aws_access_key and aws_secret_key:
        # Fallback for local development using manual keys
        return boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
    else:
        # Production standard: automatically uses the ECS IAM Task Role
        return boto3.client('s3', region_name=region)

def upload_to_s3(local_file_path: str, bucket_name: str, s3_key: str, s3_client) -> bool:
    """Uploads a local temporary file directly to S3."""
    try:
        s3_client.upload_file(local_file_path, bucket_name, s3_key)
        return True
    except Exception as e:
        print(f"S3 Upload Error: {e}")
        return False

def process_and_upload_pdf(local_file_path: str, embeddings, pinecone_api_key: str, s3_bucket: str, s3_key: str):
    """Loads a PDF, chunks it, sets S3 path metadata, and saves vectors to Pinecone."""
    loader = PyPDFLoader(local_file_path)
    documents = loader.load()
    
    # Force document metadata source to use S3 URI
    s3_uri = f"s3://{s3_bucket}/{s3_key}"
    for doc in documents:
        doc.metadata["source"] = s3_uri
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    PineconeVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        index_name=INDEX_NAME,
        pinecone_api_key=pinecone_api_key
    )

def get_vector_store(embeddings, api_key: str) -> PineconeVectorStore:
    return PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=api_key
    )

def check_index_has_vectors(pc: Pinecone) -> bool:
    try:
        index_stats = pc.Index(INDEX_NAME).describe_index_stats()
        return index_stats.total_vector_count > 0
    except Exception:
        return False

def clear_database(pc: Pinecone):
    if pc.has_index(INDEX_NAME):
        pc.delete_index(INDEX_NAME)

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def query_rag(user_query: str, vector_store: PineconeVectorStore, groq_api_key: str) -> str:
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)
    
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
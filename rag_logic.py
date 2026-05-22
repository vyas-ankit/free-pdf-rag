# rag_logic.py

import os
import time
import boto3
from langchain_core.globals import set_llm_cache                    # <── Added for cache
from langchain_community.cache import RedisSemanticCache             # <── Added for cache
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

def init_semantic_cache(embeddings):
    """Initializes the global semantic cache if REDIS_URL is present."""
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        print(f"[~] Initializing global Redis Semantic Cache...")
        set_llm_cache(
            RedisSemanticCache(
                redis_url=redis_url,
                embedding=embeddings,
                score_threshold=0.05  # Lower threshold = stricter semantic match requirement
            )
        )
    else:
        print("[!] REDIS_URL not found. Running without semantic cache.")

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
    """Initializes and returns the AWS S3 client.
    Supports both manual keys (local) and IAM Task Roles (production)."""
    if aws_access_key and aws_secret_key:
        return boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
    else:
        return boto3.client('s3', region_name=region)

def upload_to_s3(local_file_path: str, bucket_name: str, s3_key: str, s3_client) -> bool:
    """Uploads a local temporary file directly to S3."""
    try:
        s3_client.upload_file(local_file_path, bucket_name, s3_key)
        return True
    except Exception as e:
        print(f"S3 Upload Error: {e}")
        return False

def process_and_upload_pdf(
    local_file_path: str, 
    embeddings, 
    pinecone_api_key: str, 
    s3_bucket: str, 
    s3_key: str, 
    required_role: str = "Public"
):
    """Loads a PDF, chunks it, sets S3 path and security metadata, and saves to Pinecone."""
    loader = PyPDFLoader(local_file_path)
    documents = loader.load()
    
    s3_uri = f"s3://{s3_bucket}/{s3_key}"
    for doc in documents:
        doc.metadata["source"] = s3_uri
        doc.metadata["required_role"] = required_role
        
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

def query_rag(
    user_query: str, 
    vector_store: PineconeVectorStore, 
    groq_api_key: str,
    user_id: str = "guest_user",
    session_id: str = "default_session",
    user_role: str = "Public",
    retrieval_k: int = 3,
    prompt_template: str = SYSTEM_RAG_PROMPT,
    model_name: str = "llama-3.1-8b-instant"
) -> str:
    # Enforce metadata filtering on retrieval
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": retrieval_k,
            "filter": {"required_role": user_role}
        }
    )
    
    llm = ChatGroq(model=model_name, groq_api_key=groq_api_key)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        ("human", "{input}"),
    ])
    
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    config = {
        "metadata": {
            "user_id": user_id,
            "session_id": session_id,
            "user_role": user_role,
            "retrieval_k": retrieval_k,
            "model_name": model_name,
            "prompt_version": hash(prompt_template)
        }
    }
    return rag_chain.invoke(user_query, config=config)
# lambda_function.py

# Patch for AWS Lambda: Lambda's sandbox blocks POSIX semaphores.
# Replace multiprocessing.pool.ThreadPool with a threading-based executor.
import concurrent.futures

class _ThreadPoolPatch:
    def __init__(self, processes=None, *args, **kwargs):
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=processes or 1)

    def apply_async(self, func, args=(), kwds={}, callback=None, error_callback=None):
        future = self._executor.submit(func, *args, **kwds)
        class AsyncResult:
            def get(self, timeout=None):
                return future.result(timeout=timeout)
        return AsyncResult()

    def close(self): pass
    def join(self): pass
    def terminate(self): pass

import multiprocessing.pool
multiprocessing.pool.ThreadPool = _ThreadPoolPatch

# --- Normal imports below ---
import os
import urllib.parse
import boto3
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

s3_client = boto3.client('s3')
ssm_client = boto3.client('ssm')

INDEX_NAME = "free-pdf-index"

def lambda_handler(event, context):
    try:
        bucket_name = event['Records'][0]['s3']['bucket']['name']
        s3_key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
        
        print(f"[~] New PDF detected in S3: s3://{bucket_name}/{s3_key}")
        
        if not s3_key.startswith("documents/") or not s3_key.endswith(".pdf"):
            print("[+] File is not a PDF in the 'documents/' folder. Skipping.")
            return {"statusCode": 200, "body": "Skipped non-target file"}

        print("[~] Fetching Pinecone API Key from SSM Parameter Store...")
        parameter = ssm_client.get_parameter(Name="/rag-app/pinecone-api-key", WithDecryption=True)
        pinecone_api_key = parameter['Parameter']['Value']

        local_temp_path = f"/tmp/{os.path.basename(s3_key)}"
        print(f"[~] Downloading s3://{bucket_name}/{s3_key} to {local_temp_path}...")
        s3_client.download_file(bucket_name, s3_key, local_temp_path)

        print("[~] Initializing embedding model and Pinecone...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        pc = Pinecone(api_key=pinecone_api_key)

        print("[~] Loading and chunking PDF...")
        loader = PyPDFLoader(local_temp_path)
        documents = loader.load()
        
        s3_uri = f"s3://{bucket_name}/{s3_key}"
        for doc in documents:
            doc.metadata["source"] = s3_uri
            doc.metadata["required_role"] = "Public"

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        print(f"[~] Uploading {len(splits)} chunks to Pinecone index '{INDEX_NAME}'...")
        PineconeVectorStore.from_documents(
            documents=splits,
            embedding=embeddings,
            index_name=INDEX_NAME,
            pinecone_api_key=pinecone_api_key
        )

        os.remove(local_temp_path)
        print("[+] Ingestion complete!")
        return {"statusCode": 200, "body": f"Successfully processed {s3_key}"}

    except Exception as e:
        print(f"[-] Error processing file: {e}")
        raise e
# lambda_function.py
import os
import urllib.parse
import boto3
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# 1. Initialize AWS Clients inside Lambda
s3_client = boto3.client('s3')
ssm_client = boto3.client('ssm')

INDEX_NAME = "free-pdf-index"

def lambda_handler(event, context):
    try:
        # 2. Extract S3 Bucket and Key from the trigger event
        bucket_name = event['Records'][0]['s3']['bucket']['name']
        s3_key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
        
        print(f"[~] New PDF detected in S3: s3://{bucket_name}/{s3_key}")
        
        if not s3_key.startswith("documents/") or not s3_key.endswith(".pdf"):
            print("[+] File is not a PDF in the 'documents/' folder. Skipping.")
            return {"statusCode": 200, "body": "Skipped non-target file"}

        # 3. Pull Pinecone API Key from SSM Parameter Store
        print("[~] Fetching Pinecone API Key from SSM Parameter Store...")
        parameter = ssm_client.get_parameter(Name="/rag-app/pinecone-api-key", WithDecryption=True)
        pinecone_api_key = parameter['Parameter']['Value']

        # 4. Download PDF from S3 into /tmp
        local_temp_path = f"/tmp/{os.path.basename(s3_key)}"
        print(f"[~] Downloading s3://{bucket_name}/{s3_key} to {local_temp_path}...")
        s3_client.download_file(bucket_name, s3_key, local_temp_path)

        # 5. Initialize Embeddings and Pinecone
        print("[~] Initializing embedding model and Pinecone...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # pool_threads=1 avoids POSIX semaphore creation which Lambda's sandbox blocks
        pc = Pinecone(api_key=pinecone_api_key, pool_threads=1)
        index = pc.Index(INDEX_NAME)

        # 6. Load, Chunk, and Index the PDF
        print("[~] Loading and chunking PDF...")
        loader = PyPDFLoader(local_temp_path)
        documents = loader.load()
        
        # Inject metadata
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
            pinecone_api_key=pinecone_api_key,
            index=index
        )

        # 7. Clean up
        os.remove(local_temp_path)
        print("[+] Ingestion complete!")
        return {"statusCode": 200, "body": f"Successfully processed {s3_key}"}

    except Exception as e:
        print(f"[-] Error processing file: {e}")
        raise e
# backend.py

import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.core import vector_store, aws
from src.core.rag_simple import query_rag_simple

app = FastAPI(title="RAG Backend API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Read variables directly on the backend (LLM key is resolved per-provider inside get_llm)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
S3_BUCKET = os.getenv("AWS_S3_BUCKET_NAME")

# Initialize RAG resources
embeddings = vector_store.get_embeddings()
vector_store.init_semantic_cache(embeddings)

pc = vector_store.init_pinecone(PINECONE_API_KEY)
s3_client = aws.get_s3_client(region=AWS_REGION)  # Uses IAM Task Role
vs = vector_store.get_vector_store(embeddings, PINECONE_API_KEY)

class QueryRequest(BaseModel):
    query: str
    user_id: str = "anonymous_user"
    session_id: str = "default_session"
    user_role: str = "Public"

@app.get("/")
def health_check():
    """Default health check endpoint for AWS Load Balancer."""
    return {"status": "healthy"}

@app.get("/status")
def get_status():
    """Check if database has vectors."""
    has_vectors = vector_store.check_index_has_vectors(pc)
    return {"has_vectors": has_vectors}

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    required_role: str = Form("Public")
):
    """Upload PDF → S3, run extraction pipeline, then ingest chunks into Pinecone."""
    from src.utils.run_pipeline import run_full_pipeline

    temp_path = f"temp_{file.filename}"
    s3_key = f"documents/{file.filename}"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Archive raw PDF to S3
        uploaded = aws.upload_to_s3(temp_path, S3_BUCKET, s3_key, s3_client)
        if not uploaded:
            raise HTTPException(status_code=500, detail="Failed to upload file to S3.")

        # 2. Run extraction pipeline → produces chunks_with_metadata.json
        result = run_full_pipeline(pdf_path=temp_path, output_base_folder="data/processed")
        if not result:
            raise HTTPException(status_code=500, detail="Extraction pipeline failed.")

        # 3. Ingest chunks into Pinecone with role-based metadata
        vector_count = vector_store.ingest_chunks_from_json(
            chunks_json_path=result["output_json"],
            embeddings=embeddings,
            pinecone_api_key=PINECONE_API_KEY,
            required_role=required_role,
            replace_existing=True,
        )
        return {
            "status": "success",
            "message": f"Processed and indexed {file.filename} with role: {required_role}",
            "chunks_indexed": vector_count,
            "output_folder": result["folder"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/query")
def query_endpoint(body: QueryRequest):
    """Query the RAG pipeline enforcing role-based filtering."""
    try:
        answer = query_rag_simple(
            body.query,
            vs,
            user_id=body.user_id,
            session_id=body.session_id,
            user_role=body.user_role,
        )
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


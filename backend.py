# backend.py

import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import rag_logic

app = FastAPI(title="RAG Backend API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Read variables directly on the backend
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
S3_BUCKET = os.getenv("AWS_S3_BUCKET_NAME")

# Initialize RAG resources
embeddings = rag_logic.get_embeddings()

# ─── ADDED: Initialize the global semantic cache on startup ───
rag_logic.init_semantic_cache(embeddings)

pc = rag_logic.init_pinecone(PINECONE_API_KEY)
s3_client = rag_logic.get_s3_client(region=AWS_REGION) # Uses IAM Task Role
vector_store = rag_logic.get_vector_store(embeddings, PINECONE_API_KEY)

class QueryRequest(BaseModel):
    query: str
    user_id: str = "anonymous_user"
    session_id: str = "default_session"
    user_role: str = "Public"  # Add user_role (defaults to 'Public')

@app.get("/")
def health_check():
    """Default health check endpoint for AWS Load Balancer."""
    return {"status": "healthy"}

@app.get("/status")
def get_status():
    """Check if database has vectors."""
    has_vectors = rag_logic.check_index_has_vectors(pc)
    return {"has_vectors": has_vectors}

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...), 
    required_role: str = Form("Public")  # Accept security role via multipart form field
):
    """Upload PDF to S3 and process into Pinecone with role-based metadata."""
    temp_path = f"temp_{file.filename}"
    s3_key = f"documents/{file.filename}"
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        uploaded = rag_logic.upload_to_s3(temp_path, S3_BUCKET, s3_key, s3_client)
        if not uploaded:
            raise HTTPException(status_code=500, detail="Failed to upload file to S3.")
            
        # Pass the required role during indexing
        rag_logic.process_and_upload_pdf(
            temp_path, 
            embeddings, 
            PINECONE_API_KEY, 
            S3_BUCKET, 
            s3_key, 
            required_role
        )
        return {"status": "success", "message": f"Processed and indexed {file.filename} with role: {required_role}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/query")
def query_endpoint(body: QueryRequest):
    """Query the RAG pipeline enforcing role-based filtering."""
    try:
        # Pass the user's role to the query function
        answer = rag_logic.query_rag(
            body.query, 
            vector_store, 
            GROQ_API_KEY, 
            user_id=body.user_id, 
            session_id=body.session_id,
            user_role=body.user_role
        )
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear")
def clear_db():
    """Clear database index."""
    try:
        rag_logic.clear_database(pc)
        return {"status": "success", "message": "Database cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
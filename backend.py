# backend.py
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
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
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
S3_BUCKET = os.getenv("AWS_S3_BUCKET_NAME")

# Initialize RAG resources
embeddings = rag_logic.get_embeddings()
pc = rag_logic.init_pinecone(PINECONE_API_KEY)
s3_client = rag_logic.get_s3_client(AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION)
vector_store = rag_logic.get_vector_store(embeddings, PINECONE_API_KEY)

class QueryRequest(BaseModel):
    query: str

@app.get("/status")
def get_status():
    """Check if database has vectors."""
    has_vectors = rag_logic.check_index_has_vectors(pc)
    return {"has_vectors": has_vectors}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload PDF to S3 and process into Pinecone."""
    temp_path = f"temp_{file.filename}"
    s3_key = f"documents/{file.filename}"
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        uploaded = rag_logic.upload_to_s3(temp_path, S3_BUCKET, s3_key, s3_client)
        if not uploaded:
            raise HTTPException(status_code=500, detail="Failed to upload file to S3.")
            
        rag_logic.process_and_upload_pdf(temp_path, embeddings, PINECONE_API_KEY, S3_BUCKET, s3_key)
        return {"status": "success", "message": f"Processed and indexed {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/query")
def query_endpoint(body: QueryRequest):
    """Query the RAG pipeline."""
    try:
        answer = rag_logic.query_rag(body.query, vector_store, GROQ_API_KEY)
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
# FREE-PDF-RAG: Technical Architecture Report

## Executive Summary
FREE-PDF-RAG is an enterprise-grade Retrieval-Augmented Generation (RAG) system that enables semantic search and question-answering over PDF documents with role-based access control. The system processes user queries by retrieving relevant document chunks from a vector database and generating contextual answers using LLMs, with security filtering enforced at query time.

## Architecture Overview

### System Design Pattern
The architecture follows a **multi-interface, cloud-native design** with three user entry points:
- **Streamlit Web UI** (`app.py`) – interactive chat interface for end users
- **FastAPI REST API** (`backend.py`) – backend service for document operations and query processing
- **CLI Interface** (`main.py`) – local testing and bulk operations

All interfaces converge on a unified **RAG logic core** (`rag_logic.py`) that orchestrates the retrieval and generation pipeline.

### Core Services & Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Embedding Model** | HuggingFace (all-MiniLM-L6-v2) | 384-dim semantic embeddings for documents and queries |
| **Vector Database** | Pinecone (Cloud) | Scalable semantic search with metadata filtering |
| **LLM** | Groq API (Llama 3.1 8B) | Fast, cost-effective answer generation |
| **Document Processing** | LangChain + PyPDF | PDF parsing, chunking (1000 char chunks, 200 char overlap) |
| **Semantic Caching** | Redis (optional) | Cache repeated queries with 0.05 similarity threshold |
| **Document Storage** | AWS S3 | Persistent PDF storage with IAM role-based access |
| **Serverless Processing** | AWS Lambda | Event-driven auto-ingestion triggered by S3 uploads |
| **API Framework** | FastAPI + Uvicorn | RESTful endpoints with CORS support and health checks |
| **Frontend Framework** | Streamlit | Rapid UI development with built-in session management |
| **Monitoring** | LangSmith | Experiment tracking and LLM-as-judge evaluation |

## Data Flow Architecture

```
UPLOAD FLOW:
User → Streamlit/FastAPI → S3 → Lambda (event-driven) → PyPDF Loader 
→ Text Splitting → HuggingFace Embeddings → Pinecone (with role metadata)

QUERY FLOW:
User Query → Streamlit → FastAPI → Pinecone (semantic search + role filtering) 
→ Retrieved Chunks → Groq LLM (with system prompt) → Answer → User
```

## Key Architectural Features

### 1. **Role-Based Access Control**
Documents are tagged with `required_role` metadata (e.g., "Finance", "Public"). At query time, the Pinecone metadata filter ensures users only retrieve chunks they're authorized to access. This enforces security at the retrieval layer, not application layer.

### 2. **Hybrid Caching Strategy**
- **Semantic Cache (Redis)**: Optional Redis backend caches query embeddings and responses, reusing results for similar queries
- **Pinecone Metadata Index**: Enables fast filtering without fetching all documents
- Reduces redundant API calls to Groq and Pinecone

### 3. **Lambda Serverless Ingestion**
S3 event notifications trigger Lambda to automatically process new PDFs:
- Replaces multiprocessing with ThreadPoolExecutor for Lambda compatibility
- Retrieves Pinecone credentials from AWS SSM Parameter Store (secrets management)
- Scales elastically with upload volume; no persistent infrastructure needed

### 4. **Containerized Deployment**
Three separate Docker images (frontend, backend, Lambda) allow independent scaling and deployment. FastAPI runs on Uvicorn with configurable thread pools for AWS Lambda compatibility.

### 5. **Evaluation & Experimentation**
LangSmith integration supports A/B testing different RAG configurations. The evaluator framework uses Groq's 70B model as a judge to score correctness and coherence across experiments.

## Technology Stack Summary

**Python Runtime**: 3.9+ | **Backend**: FastAPI (ASGI) | **Frontend**: Streamlit  
**AI/ML**: LangChain, LangChain-Community, Sentence-Transformers  
**Vector Search**: Pinecone | **Caching**: Redis  
**Cloud**: AWS (S3, Lambda, IAM, SSM Parameter Store)  
**LLM APIs**: Groq | **Monitoring**: LangSmith  
**Containerization**: Docker | **Orchestration**: Serverless (Lambda), Cloud-managed

## Scalability & Reliability Considerations

- **Horizontal Scaling**: Pinecone provides managed vector search; Lambda auto-scales ingestion
- **Cost Optimization**: Uses Llama 3.1 8B (cheaper than GPT-4) via Groq's optimized inference
- **Fault Tolerance**: Lambda dead-letter queues recommended for failed ingestions; Redis optional for graceful degradation
- **Security**: IAM role-based S3 access, Parameter Store for API key rotation, metadata-based authorization

## Summary
FREE-PDF-RAG combines modern generative AI (LLMs, embeddings) with enterprise requirements (role-based access, audit trails via LangSmith) using a cloud-native architecture. The modular design supports multiple interfaces while maintaining a single source of truth for RAG logic, enabling rapid iteration and safe scaling.

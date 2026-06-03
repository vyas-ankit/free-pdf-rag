# src/core/vector_store.py
"""
Pinecone-side concerns: embedding model, index initialization, index management,
and ingestion of pre-extracted chunks into the index.
"""

import json
import os
import re
import time

from langchain_core.globals import set_llm_cache
from langchain_core.documents import Document
from langchain_community.cache import RedisSemanticCache
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.core.config import (
    get_pinecone_config,
    get_embeddings_config,
    get_semantic_cache_config,
)


INDEX_NAME = get_pinecone_config().get("index_name", "free-pdf-index")


# ----------------------- Embeddings + semantic cache -----------------------

def get_embeddings():
    """Build the embedding model from config (embeddings.model)."""
    cfg = get_embeddings_config()
    model_name = cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model_name)


def init_semantic_cache(embeddings):
    """Initializes the global semantic cache if REDIS_URL is present."""
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        print("[~] Initializing global Redis Semantic Cache...")
        cache_cfg = get_semantic_cache_config()
        score_threshold = float(cache_cfg.get("score_threshold", 0.05))
        set_llm_cache(
            RedisSemanticCache(
                redis_url=redis_url,
                embedding=embeddings,
                score_threshold=score_threshold,
            )
        )
    else:
        print("[!] REDIS_URL not found. Running without semantic cache.")


# ----------------------- Index lifecycle -----------------------

def _index_exists(pc: Pinecone) -> bool:
    """Compatible check for both old and new Pinecone client APIs."""
    try:
        # pinecone-client >= 3.x
        return pc.has_index(INDEX_NAME)
    except AttributeError:
        # pinecone-client 5.x removed has_index — use list_indexes instead
        return INDEX_NAME in [idx.name for idx in pc.list_indexes().indexes]


def init_pinecone(api_key: str) -> Pinecone:
    """Initialize Pinecone client; create index from config if it doesn't exist."""
    pc = Pinecone(api_key=api_key)
    if not _index_exists(pc):
        pc_cfg = get_pinecone_config()
        dimension = int(pc_cfg.get("dimension", 384))
        metric = pc_cfg.get("metric", "cosine")
        cloud = pc_cfg.get("cloud", "aws")
        region = pc_cfg.get("region", "us-east-1")
        pc.create_index(
            name=INDEX_NAME,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
    return pc


def get_vector_store(embeddings, api_key: str) -> PineconeVectorStore:
    return PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=api_key,
    )


def check_index_has_vectors(pc: Pinecone) -> bool:
    try:
        index_stats = pc.Index(INDEX_NAME).describe_index_stats()
        return index_stats.total_vector_count > 0
    except Exception:
        return False


def clear_database(pc: Pinecone):
    """Delete the entire Pinecone index (destructive — index must be recreated)."""
    if _index_exists(pc):
        pc.delete_index(INDEX_NAME)


def clear_vectors(pc: Pinecone, namespace: str = "") -> int:
    """
    Delete all vectors from the index but keep the index structure intact.
    Returns the number of vectors that existed before deletion.
    Prefer this over clear_database() for routine re-ingestion.
    """
    if not _index_exists(pc):
        print(f"Index '{INDEX_NAME}' does not exist — nothing to clear")
        return 0

    index = pc.Index(INDEX_NAME)
    before = index.describe_index_stats().total_vector_count
    if before == 0:
        print(f"Index '{INDEX_NAME}' already empty")
        return 0

    index.delete(delete_all=True, namespace=namespace)
    print(f"✓ Deleted {before} vectors from '{INDEX_NAME}' (namespace={namespace or 'default'})")
    return before


# ----------------------- Ingestion -----------------------

def _slug_from_pdf(source_pdf: str) -> str:
    """Turn 'Silver Report.pdf' into 'silver_report' for clean Pinecone IDs."""
    base = os.path.splitext(os.path.basename(source_pdf))[0]
    return re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")


def ingest_chunks_from_json(
    chunks_json_path: str,
    embeddings,
    pinecone_api_key: str,
    required_role: str = "Public",
    batch_size: int = None,
    replace_existing: bool = True,
    vector_store: PineconeVectorStore = None,
) -> int:
    """
    Upload chunks from a chunks_with_metadata.json file to Pinecone.

    Each chunk becomes one vector with deterministic ID
    `{source_pdf_slug}_chunk_{NNN}` so re-runs upsert in place (no duplicates).

    Args:
        chunks_json_path:    Path to chunks_with_metadata.json (from pipeline output)
        embeddings:          Embeddings model (from get_embeddings())
        pinecone_api_key:    Pinecone API key
        required_role:       Access role to attach to every chunk (default: "Public")
        batch_size:          Upsert batch size (defaults to config pinecone.upsert_batch_size)
        replace_existing:    If True, delete prior vectors for this source_pdf first

    Returns:
        Number of vectors upserted.
    """
    if not os.path.exists(chunks_json_path):
        print(f"ERROR: {chunks_json_path} not found")
        return 0

    if batch_size is None:
        batch_size = int(get_pinecone_config().get("upsert_batch_size", 100))

    with open(chunks_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        print(f"⚠ No chunks in {chunks_json_path}")
        return 0

    # source_pdf must be present in the JSON (injected by attach_metadata_to_chunks)
    source_pdf = chunks[0].get("source_pdf")
    if not source_pdf:
        print(f"ERROR: chunks in {chunks_json_path} missing 'source_pdf' field")
        return 0

    slug = _slug_from_pdf(source_pdf)
    print(f"Ingesting {len(chunks)} chunks from {source_pdf} (slug='{slug}', role='{required_role}')")

    # --- Optional: clear previous vectors for this source ---
    if replace_existing:
        pc = Pinecone(api_key=pinecone_api_key)
        if _index_exists(pc):
            try:
                pc.Index(INDEX_NAME).delete(filter={"source_pdf": {"$eq": source_pdf}})
                print(f"  ✓ Cleared existing vectors for source_pdf={source_pdf}")
            except Exception as e:
                print(f"  ⚠ Could not clear existing vectors (continuing anyway): {e}")

    # --- Build LangChain Documents ---
    documents = []
    ids = []
    for chunk in chunks:
        doc_meta = chunk.get("metadata", {})
        chunk_id = chunk["chunk_id"]

        # Flatten metadata for Pinecone (no nested dicts allowed)
        meta = {
            "chunk_id":      chunk_id,
            "title":         doc_meta.get("title", ""),
            "date":          doc_meta.get("date", ""),
            "theme":         doc_meta.get("theme", ""),
            "word_count":    chunk.get("word_count", 0),
            "chunk_type":    chunk.get("chunk_type", "text"),
            "source_pdf":    source_pdf,
            "required_role": required_role,
            "content":       chunk["content"],  # duplicated so retrieval returns text
        }
        if chunk.get("image_num") is not None:
            meta["image_num"] = chunk["image_num"]

        documents.append(Document(page_content=chunk["content"], metadata=meta))
        ids.append(f"{slug}_chunk_{chunk_id:03d}")

    # --- Batch upsert via LangChain PineconeVectorStore ---
    # Reuse a caller-provided vector_store when batching across many PDFs to
    # avoid leaking gRPC connection pools / OS threads.
    if vector_store is None:
        vector_store = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings,
            pinecone_api_key=pinecone_api_key,
        )

    total = 0
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        vector_store.add_documents(documents=batch_docs, ids=batch_ids)
        total += len(batch_docs)

    print(f"  ✓ Upserted {total} vectors to '{INDEX_NAME}'")
    return total

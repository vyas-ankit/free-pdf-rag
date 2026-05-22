# ingest_docs.py

import os
import rag_logic

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    print("[-] Error: Missing PINECONE_API_KEY in environment variables.")
    exit(1)

# 1. Initialize local Pinecone client
print("[~] Connecting to Pinecone...")
embeddings = rag_logic.get_embeddings()
pc = rag_logic.init_pinecone(PINECONE_API_KEY)

# 2. Clear out the old, unrelated documents
print("[~] Deleting old index to ensure a clean evaluation...")
rag_logic.clear_database(pc)

# 3. Re-initialize empty index
print("[~] Re-initializing fresh index...")
pc = rag_logic.init_pinecone(PINECONE_API_KEY)

# 4. Map the new PDFs in 'local_test' to their target access roles
new_docs = [
    {"path": "local_test/silver_report.pdf", "role": "Public"},
    {"path": "local_test/antimony_report.pdf", "role": "Public"},
    {"path": "local_test/gold_report.pdf", "role": "Finance"},
    {"path": "local_test/copper_report.pdf", "role": "Finance"}
]

# 5. Ingest the documents locally
print("[~] Indexing new documents with secure roles...")
for doc in new_docs:
    if os.path.exists(doc["path"]):
        print(f"  -> Ingesting {doc['path']} with role: {doc['role']}")
        rag_logic.process_and_upload_pdf(
            local_file_path=doc["path"],
            embeddings=embeddings,
            pinecone_api_key=PINECONE_API_KEY,
            s3_bucket="local-test-bucket",
            s3_key=doc["path"],
            required_role=doc["role"]  # Tags the chunks with the correct access role
        )
    else:
        print(f"[-] Error: File '{doc['path']}' not found in 'local_test' folder. Skip.")

print("[+] Ingestion complete! Your Pinecone cloud index is now fully populated.")
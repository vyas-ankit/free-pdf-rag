# main.py

import os
import sys
from dotenv import load_dotenv
load_dotenv()
from src.core import vector_store, aws
from src.core.rag_simple import query_rag_simple


def main():
    # 1. Validate required env vars (LLM key is resolved per-provider inside get_llm)
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    s3_bucket = os.getenv("AWS_S3_BUCKET_NAME")

    # Confirm the active provider has its API key set
    from src.core.llm import get_llm_provider, get_llm_api_key
    active_provider = get_llm_provider()
    if not get_llm_api_key(provider=active_provider):
        provider_env = f"{active_provider.upper()}_API_KEY"
        print(f"\n[-] Error: Missing {provider_env} for active provider '{active_provider}'.")
        sys.exit(1)

    # Verify all required variables are set
    missing_vars = []
    if not pinecone_api_key: missing_vars.append("PINECONE_API_KEY")
    if not aws_access_key: missing_vars.append("AWS_ACCESS_KEY_ID")
    if not aws_secret_key: missing_vars.append("AWS_SECRET_ACCESS_KEY")
    if not s3_bucket: missing_vars.append("AWS_S3_BUCKET_NAME")

    if missing_vars:
        print("\n[-] Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set them in your terminal before running the application.")
        print("Example (macOS/Linux): export AWS_S3_BUCKET_NAME='your-bucket'")
        print("Example (Windows PowerShell): $env:AWS_S3_BUCKET_NAME='your-bucket'")
        sys.exit(1)

    print("\n[+] Initializing S3 & Pinecone RAG System...")
    embeddings = vector_store.get_embeddings()
    pc = vector_store.init_pinecone(pinecone_api_key)

    # Initialize S3 client (supports explicit keys on your local machine)
    s3_client = aws.get_s3_client(aws_access_key, aws_secret_key, aws_region)
    vs = vector_store.get_vector_store(embeddings, pinecone_api_key)
    print("[+] System Ready.")

    while True:
        print("\n" + "="*40)
        print("1. Upload and index a local PDF file")
        print("2. Ask a question (Chat Mode)")
        print("3. Clear the database")
        print("4. Exit")
        print("="*40)
        
        choice = input("Select an option (1-4): ").strip()
        
        if choice == "1":
            pdf_path = input("\nEnter the full path to your PDF file: ").strip()
            # Clean enclosing quotes from dragging-and-dropping into some terminals
            pdf_path = pdf_path.strip("'\"")
            
            if not os.path.exists(pdf_path):
                print(f"[-] Error: File '{pdf_path}' not found.")
                continue
            
            required_role = input("Enter required access role for this PDF (Public/Finance, default is Public): ").strip() or "Public"
            
            file_name = os.path.basename(pdf_path)
            s3_key = f"documents/{file_name}"
            
            print(f"[~] 1. Uploading {file_name} to AWS S3 bucket '{s3_bucket}'...")
            try:
                # Upload the PDF to S3
                uploaded = aws.upload_to_s3(pdf_path, s3_bucket, s3_key, s3_client)

                if not uploaded:
                    print("[-] Ingestion failed during S3 upload step.")
                    continue

                # Run extraction pipeline (PDF → chunks_with_metadata.json)
                print("[~] 2. Running extraction pipeline...")
                from src.utils.run_pipeline import run_full_pipeline
                result = run_full_pipeline(pdf_path=pdf_path, output_base_folder="data/processed")
                if not result:
                    print("[-] Extraction pipeline failed.")
                    continue

                # Ingest chunks into Pinecone
                print("[~] 3. Ingesting chunks into Pinecone...")
                count = vector_store.ingest_chunks_from_json(
                    chunks_json_path=result["output_json"],
                    embeddings=embeddings,
                    pinecone_api_key=pinecone_api_key,
                    required_role=required_role,
                    replace_existing=True,
                )
                print(f"[+] Ingestion complete! {count} chunks indexed in Pinecone as '{required_role}'.")
            except Exception as e:
                print(f"[-] Processing failed: {e}")
                
        elif choice == "2":
            if not vector_store.check_index_has_vectors(pc):
                print("[-] Error: Your database is empty. Please index a PDF first.")
                continue
            
            user_role = input("\nEnter your authorized user role (Public/Finance, default is Public): ").strip() or "Public"
            print(f"\nEntering Chat Mode as '{user_role}' (type 'exit' or 'quit' to go back)...")
            
            while True:
                user_query = input("\nYou: ").strip()
                if user_query.lower() in ["exit", "quit"]:
                    break
                if not user_query:
                    continue
                
                print("Assistant is thinking...")
                try:
                    answer = query_rag_simple(
                        user_query,
                        vs,
                        session_id="cli_session",
                        user_role=user_role,
                    )
                    print(f"\nAssistant: {answer}")
                except Exception as e:
                    print(f"[-] Query failed: {e}")
                    
        elif choice == "3":
            confirm = input("\nAre you sure you want to clear the Pinecone Index? (y/n): ").strip().lower()
            if confirm == 'y':
                print("[~] Deleting Index...")
                vector_store.clear_database(pc)
                print("[+] Database cleared. Re-initializing empty index...")
                pc = vector_store.init_pinecone(pinecone_api_key)
                vs = vector_store.get_vector_store(embeddings, pinecone_api_key)
                
        elif choice == "4":
            print("\nGoodbye!")
            break
        else:
            print("[-] Invalid input. Please enter a number from 1 to 4.")

if __name__ == "__main__":
    main()

# main.py

import os
import sys
from dotenv import load_dotenv
load_dotenv()
import rag_logic


def main():
    # 1. Retrieve all API keys and AWS credentials from system environment variables
    llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GROQ_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    s3_bucket = os.getenv("AWS_S3_BUCKET_NAME")

    # Verify all required variables are set
    missing_vars = []
    if not llm_api_key: missing_vars.append("LLM_API_KEY or GOOGLE_API_KEY")
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
    embeddings = rag_logic.get_embeddings()
    pc = rag_logic.init_pinecone(pinecone_api_key)
    
    # Initialize S3 client (Supports explicit keys on your local machine)
    s3_client = rag_logic.get_s3_client(aws_access_key, aws_secret_key, aws_region)
    vector_store = rag_logic.get_vector_store(embeddings, pinecone_api_key)
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
                uploaded = rag_logic.upload_to_s3(pdf_path, s3_bucket, s3_key, s3_client)
                
                if uploaded:
                    print("[~] 2. Processing and indexing in Pinecone...")
                    # Process the local file and save reference to S3 inside metadata
                    rag_logic.process_and_upload_pdf(
                        pdf_path, 
                        embeddings, 
                        pinecone_api_key, 
                        s3_bucket, 
                        s3_key, 
                        required_role=required_role
                    )
                    print(f"[+] Ingestion complete! File securely archived on S3 & indexed on Pinecone as '{required_role}'.")
                else:
                    print("[-] Ingestion failed during S3 upload step.")
            except Exception as e:
                print(f"[-] Processing failed: {e}")
                
        elif choice == "2":
            if not rag_logic.check_index_has_vectors(pc):
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
                    answer = rag_logic.query_rag(
                        user_query, 
                        vector_store, 
                        llm_api_key, 
                        user_role=user_role
                    )
                    print(f"\nAssistant: {answer}")
                except Exception as e:
                    print(f"[-] Query failed: {e}")
                    
        elif choice == "3":
            confirm = input("\nAre you sure you want to clear the Pinecone Index? (y/n): ").strip().lower()
            if confirm == 'y':
                print("[~] Deleting Index...")
                rag_logic.clear_database(pc)
                print("[+] Database cleared. Re-initializing empty index...")
                pc = rag_logic.init_pinecone(pinecone_api_key)
                vector_store = rag_logic.get_vector_store(embeddings, pinecone_api_key)
                
        elif choice == "4":
            print("\nGoodbye!")
            break
        else:
            print("[-] Invalid input. Please enter a number from 1 to 4.")

if __name__ == "__main__":
    main()

# cli.py

import os
import sys
import rag_logic

def main():
    # 1. Retrieve API keys from system environment variables
    groq_api_key = os.getenv("GROQ_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    if not groq_api_key or not pinecone_api_key:
        print("\nError: Missing API keys.")
        print("Please export your keys in your terminal first:")
        print("  export GROQ_API_KEY='your-groq-key'")
        print("  export PINECONE_API_KEY='your-pinecone-key'")
        sys.exit(1)

    print("\n[+] Initializing RAG System...")
    embeddings = rag_logic.get_embeddings()
    pc = rag_logic.init_pinecone(pinecone_api_key)
    vector_store = rag_logic.get_vector_store(embeddings, pinecone_api_key)
    print("[+] System Ready.")

    while True:
        print("\n" + "="*30)
        print("1. Index a local PDF file")
        print("2. Ask a question (Chat Mode)")
        print("3. Clear the database")
        print("4. Exit")
        print("="*30)
        
        choice = input("Select an option (1-4): ").strip()
        
        if choice == "1":
            pdf_path = input("\nEnter the full path to your PDF file: ").strip()
            # Remove enclosing quotes if dragged-and-dropped into terminal
            pdf_path = pdf_path.strip("'\"")
            
            if not os.path.exists(pdf_path):
                print(f"[-] Error: File '{pdf_path}' not found.")
                continue
                
            print(f"[~] Indexing and uploading {os.path.basename(pdf_path)}...")
            try:
                rag_logic.process_and_upload_pdf(pdf_path, embeddings, pinecone_api_key)
                print("[+] Ingestion complete!")
            except Exception as e:
                print(f"[-] Ingestion failed: {e}")
                
        elif choice == "2":
            if not rag_logic.check_index_has_vectors(pc):
                print("[-] Error: Your database is empty. Please index a PDF first.")
                continue
                
            print("\nEntering Chat Mode (type 'exit' or 'quit' to go back)...")
            while True:
                user_query = input("\nYou: ").strip()
                if user_query.lower() in ["exit", "quit"]:
                    break
                if not user_query:
                    continue
                
                print("Assistant is thinking...")
                try:
                    answer = rag_logic.query_rag(user_query, vector_store, groq_api_key)
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
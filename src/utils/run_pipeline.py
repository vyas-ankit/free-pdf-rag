"""
End-to-end PDF to RAG pipeline runner.
Processes a PDF from extraction to final JSON chunks with metadata.
"""

import os
import sys
from pathlib import Path

# Import all pipeline modules
from src.ingestion.pdf_extractor import organize_extracted_pdf, generate_image_descriptions, insert_image_descriptions, add_image_tags_to_document
from src.ingestion.extract_metadata import extract_metadata_with_llm
from src.ingestion.chunk_document import chunk_document_nltk, combine_chunks, save_combined_chunks
from src.ingestion.attach_metadata_to_chunks import attach_metadata_to_chunks, print_chunks_summary


def run_full_pipeline(pdf_path: str, output_base_folder: str = None) -> dict:
    """
    Run complete PDF to RAG pipeline end-to-end.

    Args:
        pdf_path: Path to input PDF file
        output_base_folder: Base folder for output (default: ./data/processed)

    Returns:
        Dictionary with final chunks and metadata
    """

    # Validate input
    if not os.path.exists(pdf_path):
        print(f"❌ ERROR: PDF file not found: {pdf_path}")
        return None

    if output_base_folder is None:
        output_base_folder = os.path.join(os.getcwd(), "data", "processed")

    os.makedirs(output_base_folder, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"PDF to RAG Pipeline")
    print(f"{'='*60}")
    print(f"Input PDF: {pdf_path}")
    print(f"Output folder: {output_base_folder}\n")

    # ================================================================
    # STEP 1: Extract text, images, and organize
    # ================================================================
    print(f"{'─'*60}")
    print("STEP 1: Extract text, images, and metadata")
    print(f"{'─'*60}")

    folder = organize_extracted_pdf(pdf_path, output_base_folder)

    if not folder:
        print("❌ Failed to extract PDF")
        return None

    print(f"✓ Extraction complete: {folder}\n")

    # ================================================================
    # STEP 2: Extract metadata using LLM
    # ================================================================
    print(f"{'─'*60}")
    print("STEP 2: Extract metadata (title, date, theme)")
    print(f"{'─'*60}")

    metadata_txt = os.path.join(folder, "metadata.txt")
    metadata_json = os.path.join(folder, "metadata.json")

    metadata = extract_metadata_with_llm(
        metadata_txt_path=metadata_txt,
        output_json_path=metadata_json,
    )

    if not metadata:
        print("❌ Failed to extract metadata")
        return None

    print()

    # ================================================================
    # STEP 3: Generate image descriptions
    # ================================================================
    print(f"{'─'*60}")
    print("STEP 3: Generate image descriptions using Gemini Vision")
    print(f"{'─'*60}")

    image_desc_failures = generate_image_descriptions(folder)
    print()

    # ================================================================
    # STEP 4: Add image location tags to document (must run before insert)
    # ================================================================
    print(f"{'─'*60}")
    print("STEP 4: Add image location tags to document")
    print(f"{'─'*60}")

    document_path = os.path.join(folder, "document.txt")
    add_image_tags_to_document(document_path)
    print(f"✓ Image tags added\n")

    # ================================================================
    # STEP 5: Insert image descriptions into document
    # ================================================================
    print(f"{'─'*60}")
    print("STEP 5: Insert image descriptions into document")
    print(f"{'─'*60}")

    insert_image_descriptions(folder)
    print()

    # ================================================================
    # STEP 6: Chunk document using NLTK
    # ================================================================
    print(f"{'─'*60}")
    print("STEP 6: Chunk document using NLTK sentence tokenizer")
    print(f"{'─'*60}")

    chunks_folder = os.path.join(folder, "chunks")
    chunks = chunk_document_nltk(document_path, chunks_folder)

    if not chunks:
        print("❌ Failed to chunk document")
        return None

    print()

    # ================================================================
    # STEP 7: Combine chunks to 300 words max
    # ================================================================
    print(f"{'─'*60}")
    print("STEP 7: Combine chunks to 300-word max")
    print(f"{'─'*60}")

    combined_chunks = combine_chunks(chunks)  # max_words resolves from config
    print(f"Combined {len(chunks)} chunks → {len(combined_chunks)} chunks")
    print()

    # ================================================================
    # STEP 8: Save combined chunks
    # ================================================================
    print(f"{'─'*60}")
    print("STEP 8: Save combined chunks")
    print(f"{'─'*60}")

    chunks_final_folder = os.path.join(folder, "chunks_final")
    save_combined_chunks(combined_chunks, chunks_final_folder)
    print()

    # ================================================================
    # STEP 9: Attach metadata to chunks
    # ================================================================
    print(f"{'─'*60}")
    print("STEP 9: Attach metadata to all chunks")
    print(f"{'─'*60}")

    # Pass combined_chunks directly (preserves chunk_type) instead of re-reading from disk
    source_pdf = os.path.basename(pdf_path)
    final_chunks = attach_metadata_to_chunks(
        chunks=combined_chunks,
        metadata_json_path=metadata_json,
        output_folder=folder,
        source_pdf=source_pdf,
    )

    if not final_chunks:
        print("❌ Failed to attach metadata")
        return None

    print()

    # ================================================================
    # Summary
    # ================================================================
    print(f"{'='*60}")
    print("✅ PIPELINE COMPLETE")
    print(f"{'='*60}")

    print_chunks_summary(final_chunks)

    output_json = os.path.join(folder, "chunks_with_metadata.json")
    print(f"\n📁 Final output:")
    print(f"   JSON: {output_json}")
    print(f"   Individual chunks: {os.path.join(folder, 'chunks_with_metadata')}/")

    return {
        "folder": folder,
        "metadata": metadata,
        "chunks": final_chunks,
        "output_json": output_json,
        "image_desc_failures": image_desc_failures,
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # Get PDF path from command line or use default
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        output_folder = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        # Default example
        project_root = Path(__file__).resolve().parent.parent.parent
        pdf_path = str(project_root / "data" / "raw" / "silver_report.pdf")
        output_folder = str(project_root / "data" / "processed")

    result = run_full_pipeline(pdf_path, output_folder)

    if result:
        print(f"\n✅ Success! Output saved to: {result['output_json']}")
        sys.exit(0)
    else:
        print(f"\n❌ Pipeline failed")
        sys.exit(1)

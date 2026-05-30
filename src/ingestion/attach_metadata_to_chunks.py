"""Attach metadata to all chunks, injecting source_pdf + chunk_type."""

import json
import os
import re


def attach_metadata_to_chunks(
    chunks_folder: str = None,
    metadata_json_path: str = None,
    output_folder: str = None,
    chunks: list = None,
    source_pdf: str = None,
) -> list:
    """
    Attach metadata to every chunk and inject source_pdf + chunk_type fields.

    Two input modes:
      - In-memory (preferred): pass `chunks` (list of dicts with keys
        type, content, word_count, image_num) — preserves text/image type.
      - On-disk fallback: pass `chunks_folder` to load chunk_NNN.txt files
        (chunk_type cannot be recovered, so all are treated as "text").

    Args:
        chunks_folder: Folder with chunk_NNN.txt files (fallback mode)
        metadata_json_path: Path to metadata.json (title/date/theme)
        output_folder: Where to write chunks_with_metadata.json + per-chunk files
        chunks: List of chunk dicts (preferred mode)
        source_pdf: PDF filename to inject into each chunk's metadata
                    (e.g., "silver_report.pdf")

    Returns:
        List of chunks with full metadata attached.
    """

    # --- Load doc-level metadata ---
    if not metadata_json_path or not os.path.exists(metadata_json_path):
        print(f"ERROR: {metadata_json_path} not found")
        return None

    with open(metadata_json_path, "r", encoding="utf-8") as f:
        doc_metadata = json.load(f)

    print(f"Loaded metadata: {doc_metadata['title'][:50]}...")

    # --- Build the chunk list (in-memory preferred over folder) ---
    if chunks is not None:
        # In-memory mode: chunks already have type info
        source_chunks = chunks
        print(f"Using {len(source_chunks)} in-memory chunks (preserves chunk_type)")
    else:
        # On-disk fallback: read .txt files, type info lost
        if not chunks_folder or not os.path.exists(chunks_folder):
            print(f"ERROR: chunks_folder {chunks_folder} not found")
            return None

        chunk_files = sorted(
            [f for f in os.listdir(chunks_folder) if f.startswith("chunk_") and f.endswith(".txt")],
            key=lambda x: int(re.search(r"\d+", x).group()),
        )

        if not chunk_files:
            print(f"No chunk files found in {chunks_folder}")
            return None

        source_chunks = []
        for chunk_file in chunk_files:
            with open(os.path.join(chunks_folder, chunk_file), "r", encoding="utf-8") as f:
                content = f.read()
            source_chunks.append({
                "type": "text",  # cannot recover real type from .txt
                "content": content,
                "word_count": len(content.split()),
                "image_num": None,
            })
        print(f"Loaded {len(source_chunks)} chunks from disk (chunk_type defaults to 'text')")

    # --- Inject metadata into each chunk ---
    chunks_with_metadata = []
    for i, chunk in enumerate(source_chunks, start=1):
        chunks_with_metadata.append({
            "chunk_id": i,
            "metadata": doc_metadata,
            "content": chunk["content"],
            "word_count": chunk["word_count"],
            "chunk_type": chunk.get("type", "text"),
            "image_num": chunk.get("image_num"),
            "source_pdf": source_pdf,
        })

    print(f"✓ Attached metadata to {len(chunks_with_metadata)} chunks")

    # --- Save final JSON + per-chunk files ---
    if output_folder is None:
        output_folder = chunks_folder or os.getcwd()

    os.makedirs(output_folder, exist_ok=True)
    output_json = os.path.join(output_folder, "chunks_with_metadata.json")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(chunks_with_metadata, f, indent=2)
    print(f"✓ Saved chunks with metadata to {output_json}")

    # Per-chunk human-readable files
    output_chunks_folder = os.path.join(output_folder, "chunks_with_metadata")
    os.makedirs(output_chunks_folder, exist_ok=True)

    for chunk_data in chunks_with_metadata:
        chunk_num = chunk_data["chunk_id"]
        chunk_file = os.path.join(output_chunks_folder, f"chunk_{chunk_num:03d}.txt")

        with open(chunk_file, "w", encoding="utf-8") as f:
            f.write(
                f"=== METADATA ===\n"
                f"Title: {doc_metadata['title']}\n"
                f"Date: {doc_metadata['date']}\n"
                f"Theme: {doc_metadata['theme']}\n"
                f"Source PDF: {source_pdf}\n"
                f"Chunk Type: {chunk_data['chunk_type']}\n"
                f"\n=== CONTENT ===\n"
                f"{chunk_data['content']}"
            )

    print(f"✓ Saved individual chunks with metadata to {output_chunks_folder}/")
    return chunks_with_metadata


def print_chunks_summary(chunks_with_metadata: list) -> None:
    """Print summary of chunks with metadata."""
    if not chunks_with_metadata:
        return

    text_count = sum(1 for c in chunks_with_metadata if c.get("chunk_type") == "text")
    image_count = sum(1 for c in chunks_with_metadata if c.get("chunk_type") == "image")

    print("\n=== CHUNKS WITH METADATA SUMMARY ===")
    print(f"Total chunks: {len(chunks_with_metadata)}  (text: {text_count}, image: {image_count})")
    print(f"\nMetadata applied to all chunks:")
    print(f"  Title: {chunks_with_metadata[0]['metadata']['title'][:60]}...")
    print(f"  Date: {chunks_with_metadata[0]['metadata']['date']}")
    print(f"  Theme: {chunks_with_metadata[0]['metadata']['theme']}")
    print(f"  Source PDF: {chunks_with_metadata[0].get('source_pdf')}")

    print(f"\nChunk breakdown:")
    for chunk in chunks_with_metadata[:5]:
        marker = "[IMAGE]" if chunk.get("chunk_type") == "image" else "[TEXT] "
        print(f"  {marker} Chunk {chunk['chunk_id']}: {chunk['word_count']} words")

    if len(chunks_with_metadata) > 5:
        print(f"  ... and {len(chunks_with_metadata) - 5} more chunks")

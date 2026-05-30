"""Chunk a document.txt file based on paragraphs and images."""

import os
import re


def clean_document(document_path: str) -> None:
    """Remove extra blank lines, keep only single blank lines between paragraphs."""
    with open(document_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace multiple blank lines with single blank line
    cleaned = re.sub(r"\n\n+", "\n\n", content)

    with open(document_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"✓ Cleaned {document_path}")


def chunk_document(document_path: str, output_folder: str = None) -> list:
    """
    Chunk document.txt using the strategy:
    i) Chunks are full paragraphs (no max size limit)
    ii) If image encountered, end chunk and create image chunk
    iii) Each image description is its own chunk

    Returns: List of chunks with metadata
    """

    if not os.path.exists(document_path):
        print(f"ERROR: {document_path} not found")
        return []

    # Read document
    with open(document_path, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = []
    lines = content.split("\n")

    current_chunk = []
    chunk_type = "text"  # "text" or "image"
    image_num = None

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if this is an image start tag
        image_match = re.match(r"\[IMAGE_(\d+)_DESC_START\]", line)

        if image_match:
            # Save current text chunk if it has content
            if current_chunk:
                chunk_text = "\n".join(current_chunk).strip()
                if chunk_text:
                    chunks.append({
                        "type": "text",
                        "content": chunk_text,
                        "word_count": len(chunk_text.split()),
                        "image_num": None
                    })
                current_chunk = []

            # Extract image chunk
            image_num = image_match.group(1)
            image_chunk = [line]

            # Collect lines until IMAGE_END
            i += 1
            while i < len(lines):
                line = lines[i]
                image_chunk.append(line)

                if f"[IMAGE_{image_num}_DESC_END]" in line:
                    break

                i += 1

            # Save image chunk
            chunk_text = "\n".join(image_chunk).strip()
            chunks.append({
                "type": "image",
                "content": chunk_text,
                "word_count": len(chunk_text.split()),
                "image_num": int(image_num)
            })

        else:
            # Check if this is a blank line (paragraph boundary)
            if not line.strip():
                # End current chunk if it has content
                if current_chunk:
                    chunk_text = "\n".join(current_chunk).strip()
                    if chunk_text:
                        chunks.append({
                            "type": "text",
                            "content": chunk_text,
                            "word_count": len(chunk_text.split()),
                            "image_num": None
                        })
                    current_chunk = []
            else:
                # Add line to current chunk
                current_chunk.append(line)

        i += 1

    # Save any remaining chunk
    if current_chunk:
        chunk_text = "\n".join(current_chunk).strip()
        if chunk_text:
            chunks.append({
                "type": "text",
                "content": chunk_text,
                "word_count": len(chunk_text.split()),
                "image_num": None
            })

    # Save chunks to files if output folder specified
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

        # Save individual chunk files
        for i, chunk in enumerate(chunks):
            chunk_num = i + 1
            chunk_file = os.path.join(output_folder, f"chunk_{chunk_num:03d}.txt")

            with open(chunk_file, "w", encoding="utf-8") as f:
                f.write(chunk['content'])

        # Save summary file
        chunks_file = os.path.join(output_folder, "chunks_summary.txt")

        with open(chunks_file, "w", encoding="utf-8") as f:
            f.write("=== CHUNKS SUMMARY ===\n\n")
            for i, chunk in enumerate(chunks):
                f.write(f"CHUNK {i+1} ({chunk['type'].upper()})\n")
                f.write(f"Words: {chunk['word_count']}\n")
                if chunk['image_num']:
                    f.write(f"Image: {chunk['image_num']}\n")
                f.write(f"File: chunk_{i+1:03d}.txt\n")
                f.write("-" * 80 + "\n")
                f.write(chunk['content'][:200] + "...\n\n")

        print(f"✓ Saved {len(chunks)} individual chunk files")
        print(f"✓ Saved summary to chunks_summary.txt")

    return chunks


def print_chunks_summary(chunks: list) -> None:
    """Print summary of chunks - handles both old format and final metadata format."""
    if not chunks:
        print("No chunks to summarize")
        return

    # Check if this is final format (with metadata) or old format (with type)
    if "metadata" in chunks[0]:
        # Final format: chunks with metadata
        print(f"\n=== FINAL CHUNKS WITH METADATA ===")
        print(f"Total chunks: {len(chunks)}")
        total_words = sum(c['word_count'] for c in chunks)
        print(f"Total words: {total_words}")
        print(f"Avg chunk size: {total_words / len(chunks):.0f} words")

        if chunks:
            print(f"\n=== METADATA (applied to all chunks) ===")
            print(f"Title: {chunks[0]['metadata']['title'][:60]}...")
            print(f"Date: {chunks[0]['metadata']['date']}")
            print(f"Theme: {chunks[0]['metadata']['theme']}")

        print(f"\n=== CHUNK BREAKDOWN ===")
        for i, chunk in enumerate(chunks[:5]):
            print(f"Chunk {chunk['chunk_id']}: {chunk['word_count']} words")

        if len(chunks) > 5:
            print(f"... and {len(chunks) - 5} more chunks")
    else:
        # Old format: chunks with type
        text_chunks = [c for c in chunks if c["type"] == "text"]
        image_chunks = [c for c in chunks if c["type"] == "image"]

        print(f"\n=== CHUNKING SUMMARY ===")
        print(f"Total chunks: {len(chunks)}")
        print(f"Text chunks: {len(text_chunks)}")
        print(f"Image chunks: {len(image_chunks)}")
        print(f"Total words: {sum(c['word_count'] for c in chunks)}")
        print(f"Avg chunk size: {sum(c['word_count'] for c in chunks) / len(chunks):.0f} words")

        print(f"\n=== CHUNK BREAKDOWN ===")
        for i, chunk in enumerate(chunks):
            chunk_type = chunk["type"].upper()
            words = chunk["word_count"]
            image_info = f" (Image {chunk.get('image_num')})" if chunk.get('image_num') else ""
            print(f"Chunk {i+1}: {chunk_type} - {words} words{image_info}")


def chunk_document_heuristic(document_path: str, output_folder: str = None) -> list:
    """
    Chunk using heuristics - ignore blank lines, use punctuation + capitalization:
    - Join all text, remove blank lines
    - Split on sentence boundaries (. ! ?)
    - If next sentence starts with capital, it's a new paragraph
    - Keep image descriptions separate
    """
    if not os.path.exists(document_path):
        print(f"ERROR: {document_path} not found")
        return []

    with open(document_path, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = []

    # Extract images first, replace with markers
    image_contents = {}
    image_pattern = r"\[IMAGE_(\d+)_DESC_START\](.*?)\[IMAGE_(\d+)_DESC_END\]"

    for match in re.finditer(image_pattern, content, re.DOTALL):
        image_num = match.group(1)
        image_text = match.group(2).strip()
        image_contents[int(image_num)] = image_text
        content = content[:match.start()] + f"__IMAGE_{image_num}__" + content[match.end():]

    # Split by image markers to process text sections
    sections = re.split(r"(__IMAGE_\d+__)", content)

    for section in sections:
        # Check if this is an image marker
        image_match = re.match(r"__IMAGE_(\d+)__", section)

        if image_match:
            # This is an image
            image_num = int(image_match.group(1))
            image_text = image_contents[image_num]

            chunks.append({
                "type": "image",
                "content": image_text,
                "word_count": len(image_text.split()),
                "image_num": image_num
            })
        else:
            # This is text - process it
            # Remove all blank lines but preserve sentence structure
            lines = [line.strip() for line in section.split("\n") if line.strip()]
            text = " ".join(lines)

            if not text:
                continue

            # Split into sentences using regex (. ! ?)
            sentences = re.split(r'(?<=[.!?])\s+', text)

            current_paragraph = []

            for sentence in sentences:
                if not sentence.strip():
                    continue

                # Check if this sentence starts a new paragraph (capital letter)
                if current_paragraph:
                    # If sentence starts with capital AND previous sentence ended with .!?
                    sentence_stripped = sentence.strip()
                    if sentence_stripped and sentence_stripped[0].isupper():
                        # Save current paragraph
                        para_text = " ".join(current_paragraph).strip()
                        if para_text:
                            chunks.append({
                                "type": "text",
                                "content": para_text,
                                "word_count": len(para_text.split()),
                                "image_num": None
                            })
                        current_paragraph = [sentence]
                    else:
                        current_paragraph.append(sentence)
                else:
                    current_paragraph.append(sentence)

            # Save remaining paragraph
            if current_paragraph:
                para_text = " ".join(current_paragraph).strip()
                if para_text:
                    chunks.append({
                        "type": "text",
                        "content": para_text,
                        "word_count": len(para_text.split()),
                        "image_num": None
                    })

    # Save to files
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

        for i, chunk in enumerate(chunks):
            chunk_num = i + 1
            chunk_file = os.path.join(output_folder, f"chunk_{chunk_num:03d}.txt")
            with open(chunk_file, "w", encoding="utf-8") as f:
                f.write(chunk['content'])

        chunks_file = os.path.join(output_folder, "chunks_summary.txt")
        with open(chunks_file, "w", encoding="utf-8") as f:
            f.write("=== CHUNKS SUMMARY (HEURISTIC) ===\n\n")
            for i, chunk in enumerate(chunks):
                f.write(f"CHUNK {i+1} ({chunk['type'].upper()})\n")
                f.write(f"Words: {chunk['word_count']}\n")
                if chunk['image_num']:
                    f.write(f"Image: {chunk['image_num']}\n")
                f.write("-" * 80 + "\n")
                f.write(chunk['content'][:200] + "...\n\n")

        print(f"✓ Saved {len(chunks)} chunks (heuristic method)")

    return chunks


# Image block regex - enforces same image number for START and END
# Uses backreference \1 so START and END must match (e.g. IMAGE_2_START...IMAGE_2_END)
IMAGE_BLOCK_RE = re.compile(
    r"\[IMAGE_(\d+)_DESC_START\](.*?)\[IMAGE_\1_DESC_END\]",
    re.DOTALL
)


def _segment_document_by_image_blocks(content: str) -> list:
    """
    Phase 1: Walk through document and split into ordered list of segments.
    Each segment is either:
      {"type": "text",  "content": "..."}
      {"type": "image", "image_num": N, "content": "..."}
    Image blocks are atomic — never split further.
    """
    segments = []
    cursor = 0

    for match in IMAGE_BLOCK_RE.finditer(content):
        # Text BEFORE this image block
        if match.start() > cursor:
            preceding_text = content[cursor:match.start()]
            if preceding_text.strip():
                segments.append({"type": "text", "content": preceding_text})

        # The image block itself (atomic)
        segments.append({
            "type": "image",
            "image_num": int(match.group(1)),
            "content": match.group(2).strip()
        })

        cursor = match.end()

    # Trailing text after last image (or full document if no images)
    if cursor < len(content):
        trailing_text = content[cursor:]
        if trailing_text.strip():
            segments.append({"type": "text", "content": trailing_text})

    return segments


def _chunk_text_segment(text: str, sent_tokenize) -> list:
    """
    Phase 2 (text branch): NLTK-tokenize a text segment into paragraph chunks.
    Paragraph boundary = sentence starting with a capital letter following another sentence.
    """
    # Strip blank lines, join into single line
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    flat_text = " ".join(lines)
    if not flat_text:
        return []

    sentences = sent_tokenize(flat_text)
    chunks = []
    current_paragraph = []

    for sentence in sentences:
        if not sentence.strip():
            continue

        sentence_stripped = sentence.strip()
        if current_paragraph and sentence_stripped and sentence_stripped[0].isupper():
            # New paragraph boundary — flush current
            para_text = " ".join(current_paragraph).strip()
            if para_text:
                chunks.append({
                    "type": "text",
                    "content": para_text,
                    "word_count": len(para_text.split()),
                    "image_num": None
                })
            current_paragraph = [sentence]
        else:
            current_paragraph.append(sentence)

    # Flush trailing paragraph
    if current_paragraph:
        para_text = " ".join(current_paragraph).strip()
        if para_text:
            chunks.append({
                "type": "text",
                "content": para_text,
                "word_count": len(para_text.split()),
                "image_num": None
            })

    return chunks


def chunk_document_nltk(document_path: str, output_folder: str = None) -> list:
    """
    Image-aware NLTK chunker.

    Phase 1: Segment document into text/image blocks (image descriptions are atomic).
    Phase 2: Chunk each segment:
             - text segment → NLTK sentence tokenization → paragraph chunks
             - image segment → single atomic chunk (preserves boundary)

    Output chunks have a `type` field ("text" or "image") so the downstream
    combiner can keep images standalone.
    """
    try:
        from nltk.tokenize import sent_tokenize
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    except ImportError:
        print("ERROR: NLTK not installed. Install with: pip install nltk")
        return []

    if not os.path.exists(document_path):
        print(f"ERROR: {document_path} not found")
        return []

    with open(document_path, "r", encoding="utf-8") as f:
        content = f.read()

    # PHASE 1: Split document into ordered text/image segments
    segments = _segment_document_by_image_blocks(content)

    # PHASE 2: Chunk each segment, preserving order
    chunks = []
    for segment in segments:
        if segment["type"] == "image":
            # Atomic image chunk — never split
            image_text = segment["content"]
            chunks.append({
                "type": "image",
                "content": image_text,
                "word_count": len(image_text.split()),
                "image_num": segment["image_num"]
            })
        else:
            # Text segment → NLTK paragraph chunks
            chunks.extend(_chunk_text_segment(segment["content"], sent_tokenize))

    # Save to files
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

        for i, chunk in enumerate(chunks):
            chunk_num = i + 1
            chunk_file = os.path.join(output_folder, f"chunk_{chunk_num:03d}.txt")
            with open(chunk_file, "w", encoding="utf-8") as f:
                f.write(chunk['content'])

        chunks_file = os.path.join(output_folder, "chunks_summary.txt")
        with open(chunks_file, "w", encoding="utf-8") as f:
            f.write("=== CHUNKS SUMMARY (NLTK, IMAGE-AWARE) ===\n\n")
            for i, chunk in enumerate(chunks):
                f.write(f"CHUNK {i+1} ({chunk['type'].upper()})\n")
                f.write(f"Words: {chunk['word_count']}\n")
                if chunk['image_num']:
                    f.write(f"Image: {chunk['image_num']}\n")
                f.write("-" * 80 + "\n")
                f.write(chunk['content'][:200] + "...\n\n")

        print(f"✓ Saved {len(chunks)} chunks (NLTK image-aware)")

    return chunks


def combine_chunks(chunks: list, max_words: int = None) -> list:
    """
    Combine text chunks up to max_words size.
    Keep image chunks separate (don't combine with text).

    `max_words` defaults to config `ingestion.chunking.max_words` (300 if unset).

    Returns: New list of combined chunks
    """
    if max_words is None:
        from src.core.config import get_ingestion_config
        max_words = int(get_ingestion_config().get("chunking", {}).get("max_words", 300))

    combined = []
    current_chunk = None
    current_words = 0

    for chunk in chunks:
        if chunk["type"] == "image":
            # Save current text chunk if exists
            if current_chunk:
                combined.append(current_chunk)
                current_chunk = None
                current_words = 0

            # Add image as separate chunk
            combined.append(chunk)

        else:  # Text chunk
            chunk_words = chunk["word_count"]

            if current_chunk is None:
                # Start new combined chunk
                current_chunk = {
                    "type": "text",
                    "content": chunk["content"],
                    "word_count": chunk_words,
                    "image_num": None
                }
                current_words = chunk_words

            elif current_words + chunk_words <= max_words:
                # Add to current chunk
                current_chunk["content"] += "\n\n" + chunk["content"]
                current_chunk["word_count"] += chunk_words
                current_words += chunk_words

            else:
                # Save current chunk and start new one
                combined.append(current_chunk)
                current_chunk = {
                    "type": "text",
                    "content": chunk["content"],
                    "word_count": chunk_words,
                    "image_num": None
                }
                current_words = chunk_words

    # Save remaining chunk
    if current_chunk:
        combined.append(current_chunk)

    return combined


def save_combined_chunks(chunks: list, output_folder: str) -> None:
    """Save combined chunks to files."""
    os.makedirs(output_folder, exist_ok=True)

    # Save individual chunk files
    for i, chunk in enumerate(chunks):
        chunk_num = i + 1
        chunk_file = os.path.join(output_folder, f"chunk_{chunk_num:03d}.txt")

        with open(chunk_file, "w", encoding="utf-8") as f:
            f.write(chunk['content'])

    # Save summary
    chunks_file = os.path.join(output_folder, "chunks_summary.txt")

    with open(chunks_file, "w", encoding="utf-8") as f:
        f.write("=== COMBINED CHUNKS SUMMARY ===\n\n")
        for i, chunk in enumerate(chunks):
            f.write(f"CHUNK {i+1} ({chunk['type'].upper()})\n")
            f.write(f"Words: {chunk['word_count']}\n")
            if chunk['image_num']:
                f.write(f"Image: {chunk['image_num']}\n")
            f.write("-" * 80 + "\n")
            f.write(chunk['content'][:200] + "...\n\n")

    print(f"✓ Saved {len(chunks)} combined chunks to {output_folder}")

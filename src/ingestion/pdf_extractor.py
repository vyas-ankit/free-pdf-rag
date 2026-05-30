# pdf_extractor.py

import base64
import fitz
import time
import requests
from src.core.llm import get_llm_api_key
from src.core.config import get_ingestion_config


def _pdf_extraction_cfg() -> dict:
    return get_ingestion_config().get("pdf_extraction", {})


def extract_metadata(pdf_path: str) -> str:
    """Extract raw metadata text from page 1 (before main content starts)."""
    cfg = _pdf_extraction_cfg()
    skip_top_pct = float(cfg.get("skip_top_pct", 0.03))
    content_markers = cfg.get("content_markers", ["Reading Time:", "What I Learned"])

    doc = fitz.open(pdf_path)
    page = doc[0]

    blocks = page.get_text("blocks")
    page_height = page.rect.height

    # Skip header (top skip_top_pct) and get metadata section (up to content start)
    skip_top = page_height * skip_top_pct
    metadata_blocks = []

    for block in blocks:
        if block[6] == 0:  # Text block
            block_y = block[1]
            block_text = block[4].strip()

            # Stop at content markers
            if any(marker in block_text for marker in content_markers):
                break

            if block_y > skip_top and block_text:
                metadata_blocks.append(block_text)

    metadata_text = "\n".join(metadata_blocks)
    doc.close()
    return metadata_text


def find_content_start_index(blocks: list) -> int:
    """Find where metadata ends and content begins on page 1."""
    content_markers = _pdf_extraction_cfg().get(
        "content_markers",
        ["Reading Time:", "What I Learned", "Introduction", "Overview"],
    )

    for i, block in enumerate(blocks):
        block_text = block[4].strip()
        for marker in content_markers:
            if marker.lower() in block_text.lower():
                return i

    # If no marker found, assume metadata is first 3-5 blocks
    return min(5, len(blocks) // 3)


def extract_text_blocks(pdf_path: str, metadata: dict) -> list:
    """Extract text blocks, skipping metadata on page 1."""
    cfg = _pdf_extraction_cfg()
    skip_top_pct = float(cfg.get("skip_top_pct", 0.03))
    skip_bottom_pct = float(cfg.get("skip_bottom_pct", 0.95))

    doc = fitz.open(pdf_path)
    all_items = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_height = page.rect.height

        skip_top = page_height * skip_top_pct
        skip_bottom = page_height * skip_bottom_pct

        blocks = page.get_text("blocks")
        
        # Skip metadata on page 1
        if page_num == 0:
            content_start = find_content_start_index(blocks)
            blocks = blocks[content_start:]
        
        for block in blocks:
            if block[6] == 0:
                block_y = block[1]
                block_text = block[4].strip()
                
                if not (skip_top < block_y < skip_bottom):
                    continue
                
                if block_text:
                    all_items.append({
                        "type": "text",
                        "content": block_text,
                        "y": block_y,
                        "page": page_num,
                        "metadata": metadata  # Attach to every chunk
                    })
    
    doc.close()
    return all_items


def remove_footer_markers(items: list) -> list:
    """Remove content after footer marker."""
    full_text = "\n\n".join([item["content"] for item in items if item["type"] == "text"])
    footer_markers = _pdf_extraction_cfg().get(
        "footer_markers",
        ["Share\n\nPrev", "Share", "Prev", "Next"],
    )
    cut_index = len(full_text)
    
    for marker in footer_markers:
        if marker in full_text:
            cut_index = full_text.find(marker)
            break
    
    return [item for item in items if item["type"] != "text" or 
            (full_text.find(item["content"]) + len(item["content"]) <= cut_index)]


def extract_images(pdf_path: str, max_retries: int = 3) -> list:
    """Extract images from PDF and describe with Google Vision."""
    from src.core.config import get_provider_config

    api_key = get_llm_api_key("google").split('#')[0].strip()
    if not api_key:
        print("ERROR: No valid Google API key found")
        return []

    google_cfg = get_provider_config("google")
    model = google_cfg.get("vision_model") or google_cfg.get("default_model")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
    doc = fitz.open(pdf_path)
    all_items = []
    image_count = 0
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        for img_index in page.get_images(full=True):
            image_count += 1
            xref = img_index[0]
            pix = fitz.Pixmap(doc, xref)
            
            try:
                img_rect = page.get_image_bbox(img_index)
                img_y = img_rect.y0
            except:
                img_y = float('inf')
            
            if pix.n - pix.alpha < 4:
                img_data = pix.tobytes("png")
            else:
                rgb_pix = fitz.Pixmap(fitz.csRGB, pix)
                img_data = rgb_pix.tobytes("png")
            
            img_base64 = base64.standard_b64encode(img_data).decode()
            print(f"Page {page_num+1}, Image {image_count}: {len(img_base64)} bytes", end="")
            
            description = _call_vision_api(url, api_key, img_base64, max_retries)
            if not description:
                description = "[Image description failed]"
            
            all_items.append({
                "type": "image",
                "content": description,
                "y": img_y,
                "page": page_num
            })
            time.sleep(2)
    
    doc.close()
    print(f"✓ Processed {image_count} images")
    return all_items


def _call_vision_api(url: str, api_key: str, img_base64: str, max_retries: int) -> str:
    """Call Google Gemini Vision API with retry logic."""
    for attempt in range(max_retries):
        try:
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": "Describe this image in 1-2 sentences:"},
                            {
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": img_base64
                                }
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(
                url,
                headers={"X-goog-api-key": api_key},
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                description = result["candidates"][0]["content"]["parts"][0]["text"]
                print(f" ✓")
                return description
            elif response.status_code == 429:
                wait_time = (2 ** attempt) * 5
                if attempt < max_retries - 1:
                    print(f" (waiting {wait_time}s)", end="")
                    time.sleep(wait_time)
                else:
                    print(f" ✗")
            else:
                if attempt == max_retries - 1:
                    print(f" ✗")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        except Exception as e:
            if attempt == max_retries - 1:
                print(f" ✗")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    
    return None


def combine_and_sort(text_items: list, image_items: list) -> list:
    """Combine text and image items, sort by reading order."""
    all_items = text_items + image_items
    all_items.sort(key=lambda x: (x["page"], x["y"]))
    return all_items


def clean_text(text: str) -> str:
    """Clean whitespace and blank lines."""
    text = text.strip()
    text = "\n\n".join([line.strip() for line in text.split("\n\n") if line.strip()])
    return text


def save_to_file(content: str, output_file: str) -> None:
    """Save extracted content to file."""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✓ Saved to {output_file} ({len(content)} characters)")


def extract_and_clean_text_with_images(
    pdf_path: str,
    output_file: str = None
) -> dict:
    """
    Main function: Extract metadata + content + images.
    
    Returns:
        dict with metadata and chunks (ready for vector DB)
    """
    print(f"Processing {pdf_path}...")
    
    # Extract metadata from page 1
    print("Extracting metadata...")
    metadata = extract_metadata(pdf_path)
    print(f"✓ Title: {metadata['title']}")
    print(f"✓ Date: {metadata['date']}")
    print(f"✓ Theme: {metadata['theme']}")
    
    # Extract text (skipping metadata on page 1)
    print("Extracting content...")
    text_items = extract_text_blocks(pdf_path, metadata)
    print(f"✓ Extracted {len(text_items)} content blocks")
    
    print("Removing footer markers...")
    text_items = remove_footer_markers(text_items)
    
    # Extract images
    print("Extracting images...")
    image_items = extract_images(pdf_path)
    
    # Combine and sort
    print("Combining content...")
    all_items = combine_and_sort(text_items, image_items)
    
    result = "\n\n".join([item["content"] for item in all_items])
    result = clean_text(result)
    
    if output_file:
        save_to_file(result, output_file)
    
    return {
        "metadata": metadata,
        "chunks": all_items,
        "content": result
    }


def get_source_for_image(document_text: str, image_num: int) -> str:
    """Extract source line associated with an image marker."""
    # Find the image marker
    image_marker = f"[IMAGE_{image_num}]"
    marker_pos = document_text.find(image_marker)

    if marker_pos == -1:
        return ""

    # Look for next line after the marker
    after_marker = document_text[marker_pos + len(image_marker):]
    lines = after_marker.split("\n")

    for line in lines:
        line = line.strip()
        if line.startswith("Source:"):
            return line
        if line and not line.startswith("["):  # Stop if we hit another marker or content
            break

    return ""


def organize_extracted_pdf(pdf_path: str, output_folder: str = "/Users/ankitvyas/free-pdf-rag/data/processed") -> str:
    """
    Organize extracted PDF into file_<num> folder structure:
    - file_<num>/document.txt (full text with [IMAGE_X] markers)
    - file_<num>/images/image_<num>/ (image + before.txt, after.txt)

    Returns: Path to created folder
    """
    import os
    import re

    print(f"\nOrganizing PDF: {pdf_path}")

    # Find next available file_<num>
    existing_folders = []
    if os.path.exists(output_folder):
        for item in os.listdir(output_folder):
            match = re.match(r"file_(\d+)", item)
            if match:
                existing_folders.append(int(match.group(1)))

    next_num = max(existing_folders) + 1 if existing_folders else 1
    file_folder = os.path.join(output_folder, f"file_{next_num}")
    images_folder = os.path.join(file_folder, "images")

    # Create folders
    os.makedirs(images_folder, exist_ok=True)
    print(f"✓ Created folder: {file_folder}")

    # Extract and save metadata as plain text
    metadata_text = extract_metadata(pdf_path)
    metadata_path = os.path.join(file_folder, "metadata.txt")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(metadata_text)
    print(f"✓ Saved metadata.txt")

    # Extract text blocks and images with their positions
    doc = fitz.open(pdf_path)
    text_items = extract_text_blocks(pdf_path, {})
    text_items = remove_footer_markers(text_items)

    # Extract images with positions
    image_items = []
    image_count = 0

    for page_num in range(len(doc)):
        page = doc[page_num]

        for img_index in page.get_images(full=True):
            image_count += 1
            xref = img_index[0]
            pix = fitz.Pixmap(doc, xref)

            try:
                img_rect = page.get_image_bbox(img_index)
                img_y = img_rect.y0
            except:
                img_y = float('inf')

            # Convert to PNG
            if pix.n - pix.alpha < 4:
                img_data = pix.tobytes("png")
            else:
                rgb_pix = fitz.Pixmap(fitz.csRGB, pix)
                img_data = rgb_pix.tobytes("png")

            # Create image folder
            image_subfolder = os.path.join(images_folder, f"image_{image_count}")
            os.makedirs(image_subfolder, exist_ok=True)

            # Save image
            image_path = os.path.join(image_subfolder, f"image_{image_count}.png")
            with open(image_path, "wb") as f:
                f.write(img_data)

            image_items.append({
                "type": "image",
                "index": image_count,
                "y": img_y,
                "page": page_num
            })

            print(f"✓ Extracted image_{image_count}")

    doc.close()

    # Combine text and image items, sort by position (page, y-coordinate)
    all_items = text_items + image_items
    all_items.sort(key=lambda x: (x["page"], x["y"]))

    # Build document.txt with image markers in correct positions
    document_lines = []
    for item in all_items:
        if item["type"] == "text":
            document_lines.append(item["content"])
        elif item["type"] == "image":
            document_lines.append(f"[IMAGE_{item['index']}]")

    document_text = "\n\n".join(document_lines)

    # Save document.txt
    document_path = os.path.join(file_folder, "document.txt")
    with open(document_path, "w", encoding="utf-8") as f:
        f.write(document_text)
    print(f"✓ Saved document.txt with {image_count} image markers in correct positions")

    # Extract and save before/after context for each image
    words = document_text.split()
    context_words = int(_pdf_extraction_cfg().get("image_context_words", 30))

    for item in all_items:
        if item["type"] == "image":
            img_num = item["index"]
            image_subfolder = os.path.join(images_folder, f"image_{img_num}")

            # Find position of this image in the text
            img_marker = f"[IMAGE_{img_num}]"
            marker_pos = document_text.find(img_marker)

            # Count words before marker
            words_before_marker = document_text[:marker_pos].split()
            word_position = len(words_before_marker)

            # Extract `context_words` words before and after
            before_start = max(0, word_position - context_words)
            after_end = min(len(words), word_position + context_words)

            before_words = words[before_start:word_position]
            after_words = words[word_position:after_end]

            before_text = " ".join(before_words).strip()
            after_text = " ".join(after_words).strip()

            before_path = os.path.join(image_subfolder, "before.txt")
            after_path = os.path.join(image_subfolder, "after.txt")

            with open(before_path, "w", encoding="utf-8") as f:
                f.write(before_text)
            with open(after_path, "w", encoding="utf-8") as f:
                f.write(after_text)

            print(f"✓ Saved context for image_{img_num}")

    print(f"\n✓ Organized PDF into {file_folder}")
    return file_folder


def generate_image_descriptions(file_folder: str, max_retries: int = 3, provider: str = None, model_name: str = None) -> list:
    """
    For each image in file_folder/images/:
    1. Read before.txt, image, after.txt
    2. Send to vision model (Groq or Google Gemini)
    3. Save description to image folder

    Retries each image up to `max_retries` times with escalating backoff
    (15s, 30s, 60s) on rate-limit or transient failures. Failed images are
    written to `file_folder/image_desc_failures.json` and also returned.

    Returns: list of image directory names (e.g. ["image_3", "image_7"]) that
             never succeeded.
    """
    import os
    import json
    import re
    import base64
    import io
    from src.core.llm import get_llm_api_key
    from src.core.config import get_image_desc_config, get_config, get_provider_config
    import time

    img_cfg = get_image_desc_config()

    # Resolve provider — explicit arg > image_descriptions.provider > default_provider > "groq".
    provider = (provider or img_cfg.get("provider") or get_config().get("default_provider", "groq")).lower()

    # Resolve model — explicit arg > image_descriptions.model > providers.{provider}.vision_model > default_model.
    if not model_name:
        model_name = img_cfg.get("model")
    if not model_name:
        provider_cfg = get_provider_config(provider)
        model_name = provider_cfg.get("vision_model") or provider_cfg.get("default_model")

    # Optional downscale before sending to vision API — reduces tokens-per-image so
    # large vision payloads fit in tight TPM budgets (Groq free tier especially).
    downscale_cfg = img_cfg.get("downscale", {}) or {}
    downscale_enabled = bool(downscale_cfg.get("enabled", False))
    downscale_max_dim = int(downscale_cfg.get("max_dim", 768))
    downscale_format = str(downscale_cfg.get("format", "PNG")).upper()
    if downscale_enabled:
        from PIL import Image
        print(f"[~] Image downscale ON — max_dim={downscale_max_dim}, format={downscale_format}")

    # Proactive self-throttle — guarantees `min_interval_sec` between consecutive
    # vision API calls (including retries). Avoids tripping Groq's hidden burst
    # penalty, which can balloon retry-after to many minutes even when the TPM
    # bucket reads as full.
    min_interval_sec = float(img_cfg.get("min_interval_sec", 0))
    last_call_ts = 0.0
    if min_interval_sec > 0:
        print(f"[~] Vision API self-throttle ON — min {min_interval_sec}s between calls")

    # Escalating backoff schedule: 15s after 1st fail, 30s after 2nd, 60s before 3rd.
    backoff_schedule = [15, 30, 60]
    failures = []  # list[str] — image_dir names that never produced a description

    # Cross-image rate-limit memory: if image N hits 429 with retry-after=12,
    # image N+1 preemptively waits 12s before its first request. Same TPM bucket.
    pending_wait = 0.0

    images_folder = os.path.join(file_folder, "images")

    if not os.path.exists(images_folder):
        print(f"No images folder found in {file_folder}")
        return

    if provider == "groq":
        api_key = get_llm_api_key("groq").split('#')[0].strip()
        if not api_key:
            print("ERROR: No valid Groq API key found")
            return
        url = f"https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        is_groq = True
    else:
        api_key = get_llm_api_key("google").split('#')[0].strip()
        if not api_key:
            print("ERROR: No valid Google API key found")
            return
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        headers = {"X-goog-api-key": api_key}
        is_groq = False

    # Process each image folder
    for image_dir in sorted(os.listdir(images_folder)):
        image_path = os.path.join(images_folder, image_dir)

        if not os.path.isdir(image_path):
            continue

        # If a previous image left a retry-after on the table, honour it before
        # firing a fresh request — the rate-limit bucket is shared across the file.
        if pending_wait > 0:
            print(f"[~] Pre-image cool-down {pending_wait:.0f}s (carryover from last 429)...", flush=True)
            time.sleep(pending_wait)
            pending_wait = 0.0

        before_file = os.path.join(image_path, "before.txt")
        after_file = os.path.join(image_path, "after.txt")
        img_file = None

        # Find image file (png, jpg, etc.)
        for file in os.listdir(image_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                img_file = os.path.join(image_path, file)
                break

        if not img_file:
            print(f"⚠ No image file found in {image_path}")
            continue

        # Read context
        before_text = ""
        after_text = ""

        if os.path.exists(before_file):
            with open(before_file, "r", encoding="utf-8") as f:
                before_text = f.read().strip()

        if os.path.exists(after_file):
            with open(after_file, "r", encoding="utf-8") as f:
                after_text = f.read().strip()

        # Read and encode image — optionally downscale via Pillow first to
        # keep vision token cost low. Original file on disk is untouched.
        if downscale_enabled:
            with Image.open(img_file) as im:
                im = im.convert("RGB")
                if max(im.size) > downscale_max_dim:
                    im.thumbnail((downscale_max_dim, downscale_max_dim), Image.LANCZOS)
                buf = io.BytesIO()
                save_kwargs = {"optimize": True}
                if downscale_format == "JPEG":
                    save_kwargs["quality"] = 85
                im.save(buf, format=downscale_format, **save_kwargs)
                img_data = buf.getvalue()
        else:
            with open(img_file, "rb") as f:
                img_data = f.read()
        img_base64 = base64.standard_b64encode(img_data).decode()

        # Build prompt with context
        from src.core.prompts import IMAGE_DESCRIPTION_PROMPT
        prompt = IMAGE_DESCRIPTION_PROMPT.format(
            before_text=before_text,
            after_text=after_text,
        )

        print(f"Processing {image_dir}...", end="")

        description = None
        for attempt in range(max_retries):
            try:
                if is_groq:
                    payload = {
                        "model": model_name,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                                    }
                                ]
                            }
                        ],
                        "max_tokens": 300
                    }
                else:
                    payload = {
                        "contents": [
                            {
                                "parts": [
                                    {"text": prompt},
                                    {
                                        "inline_data": {
                                            "mime_type": "image/png",
                                            "data": img_base64
                                        }
                                    }
                                ]
                            }
                        ]
                    }

                # Self-throttle: ensure at least min_interval_sec since last call.
                if min_interval_sec > 0:
                    elapsed = time.time() - last_call_ts
                    if elapsed < min_interval_sec:
                        time.sleep(min_interval_sec - elapsed)
                last_call_ts = time.time()

                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    if is_groq:
                        description = result["choices"][0]["message"]["content"]
                    else:
                        description = result["candidates"][0]["content"]["parts"][0]["text"]
                    print(" ✓")
                    break
                else:
                    # Any non-200 (especially 429) → escalating backoff.
                    # Groq returns rate-limit info via headers; Google/Gemini returns
                    # everything in the JSON body. Surface both so we can see exactly
                    # what's binding without guessing.
                    rl = {
                        "retry_after": response.headers.get("retry-after"),
                        "rpm_remaining": response.headers.get("x-ratelimit-remaining-requests"),
                        "tpm_remaining": response.headers.get("x-ratelimit-remaining-tokens"),
                        "rpm_reset": response.headers.get("x-ratelimit-reset-requests"),
                        "tpm_reset": response.headers.get("x-ratelimit-reset-tokens"),
                    }
                    rl_summary = " ".join(f"{k}={v}" for k, v in rl.items() if v is not None)
                    body_snippet = (response.text or "").strip().replace("\n", " ")[:500]

                    # Prefer the server's retry-after if provided; otherwise fall back to schedule.
                    server_wait = None
                    try:
                        server_wait = float(rl["retry_after"]) if rl["retry_after"] else None
                    except ValueError:
                        server_wait = None
                    # Google's RetryInfo lives in the body, not headers — try to extract.
                    if server_wait is None and body_snippet:
                        match = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', response.text or "")
                        if match:
                            try:
                                server_wait = float(match.group(1))
                            except ValueError:
                                server_wait = None
                    wait = server_wait if server_wait else backoff_schedule[min(attempt, len(backoff_schedule) - 1)]

                    # Remember it for the NEXT image too — the TPM bucket is per-file shared.
                    if server_wait:
                        pending_wait = max(pending_wait, server_wait)

                    err_line = f"[{response.status_code} {rl_summary}] body: {body_snippet}"
                    if attempt < max_retries - 1:
                        print(f"\n  {err_line}\n  retry in {wait:.0f}s...", end="", flush=True)
                        time.sleep(wait)
                    else:
                        print(f"\n  ✗ {err_line}")

            except Exception as e:
                wait = backoff_schedule[min(attempt, len(backoff_schedule) - 1)]
                if attempt < max_retries - 1:
                    print(f" [err] retry in {wait}s...", end="", flush=True)
                    time.sleep(wait)
                else:
                    print(f" ✗ Error: {str(e)[:100]}")

        if description:
            # Extract image number from folder name
            image_num = image_dir.replace("image_", "")

            # Save description
            desc_file = os.path.join(image_path, f"image_{image_num}_desc.txt")
            with open(desc_file, "w", encoding="utf-8") as f:
                f.write(description)

            print(f" ✓")
        else:
            print(f" ✗ (gave up after {max_retries} attempts)")
            failures.append(image_dir)

    # Persist failure list so downstream tooling can read it without re-running
    failures_path = os.path.join(file_folder, "image_desc_failures.json")
    with open(failures_path, "w") as f:
        json.dump(failures, f, indent=2)
    if failures:
        print(f"  ⚠ {len(failures)} image(s) failed: {failures}")

    return failures


def insert_image_descriptions(file_folder: str) -> None:
    """
    Insert generated image descriptions into document.txt
    1. Read image_X_desc.txt from each image folder
    2. Insert between [IMAGE_X_DESC_START] and [IMAGE_X_DESC_END] tags
    3. Keep any source lines inside the tags
    """
    import os
    import re

    document_path = os.path.join(file_folder, "document.txt")
    images_folder = os.path.join(file_folder, "images")

    if not os.path.exists(document_path):
        print(f"ERROR: {document_path} not found")
        return

    if not os.path.exists(images_folder):
        print(f"ERROR: {images_folder} not found")
        return

    # Read document
    with open(document_path, "r", encoding="utf-8") as f:
        document_text = f.read()

    # Process each image
    for image_dir in sorted(os.listdir(images_folder)):
        image_path = os.path.join(images_folder, image_dir)

        if not os.path.isdir(image_path):
            continue

        # Extract image number
        image_num = image_dir.replace("image_", "")
        desc_file = os.path.join(image_path, f"image_{image_num}_desc.txt")

        if not os.path.exists(desc_file):
            print(f"⚠ {desc_file} not found, skipping")
            continue

        # Read description
        with open(desc_file, "r", encoding="utf-8") as f:
            description = f.read().strip()

        # Find the image tags in document
        pattern = fr"\[IMAGE_{image_num}_DESC_START\](.*?)\[IMAGE_{image_num}_DESC_END\]"
        match = re.search(pattern, document_text, re.DOTALL)

        if match:
            old_content = match.group(1)

            # Extract source if it exists
            source_match = re.search(r"Source:.*", old_content)
            source_line = source_match.group(0) if source_match else None

            # Build new content
            if source_line:
                new_content = f"\n{description}\n\n{source_line}\n"
            else:
                new_content = f"\n{description}\n"

            # Replace in document
            new_text = document_text[:match.start(1)] + new_content + document_text[match.end(1):]
            document_text = new_text

            print(f"✓ Inserted description for image_{image_num}")
        else:
            print(f"⚠ Tags not found for image_{image_num}")

    # Save updated document
    with open(document_path, "w", encoding="utf-8") as f:
        f.write(document_text)

    print(f"\n✓ Updated {document_path}")


def add_image_tags_to_document(document_path: str) -> None:
    """
    Process document.txt:
    1. Replace [IMAGE_X] with [IMAGE_X_DESC_START/END] tags
    2. Move "Source:" lines inside the tags
    3. Save updated document.txt
    """
    import re

    # Read document
    with open(document_path, "r", encoding="utf-8") as f:
        document_text = f.read()

    lines = document_text.split("\n")
    output_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if this line is an image marker
        match = re.match(r"\[IMAGE_(\d+)\]", line)
        if match:
            image_num = match.group(1)

            # Check if next non-empty line is a source
            source_line = None
            next_idx = i + 1

            # Skip empty lines
            while next_idx < len(lines) and not lines[next_idx].strip():
                next_idx += 1

            # Check if it's a source line
            if next_idx < len(lines) and lines[next_idx].strip().startswith("Source:"):
                source_line = lines[next_idx].strip()
                i = next_idx  # Skip the source line we just found

            # Add tags
            output_lines.append(f"[IMAGE_{image_num}_DESC_START]")
            if source_line:
                output_lines.append(source_line)
            output_lines.append(f"[IMAGE_{image_num}_DESC_END]")

        else:
            output_lines.append(line)

        i += 1

    # Save updated document
    updated_text = "\n".join(output_lines)
    with open(document_path, "w", encoding="utf-8") as f:
        f.write(updated_text)

    print(f"✓ Added image tags to {document_path}")

# scripts/ingest_docs.py
"""
End-to-end bulk ingest with per-file metrics:
    data/raw/*.pdf → extract → chunk → image desc → Pinecone
                  → per-file stats → data/processed/_ingest_stats.json

Drop PDFs into data/raw/ and run:
    python -m scripts.ingest_docs

Env flags:
  REPROCESS_ALL=1            re-run extraction even if chunks already exist
  CLEAR_ALL_BEFORE_INGEST=1  wipe Pinecone index before upsert
"""

import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# Cool-down between PDFs so the vision API rate-limit window resets.
PER_FILE_COOLDOWN_SEC = int(os.getenv("PER_FILE_COOLDOWN_SEC", "15"))

from src.core import vector_store
from src.utils.run_pipeline import run_full_pipeline
from langchain_pinecone import PineconeVectorStore

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    print("[-] Error: Missing PINECONE_API_KEY in environment variables.")
    exit(1)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
STATS_PATH = PROCESSED_DIR / "_ingest_stats.json"
REPROCESS_ALL = os.getenv("REPROCESS_ALL", "0").lower() in {"1", "true", "yes"}
CLEAR_ALL = os.getenv("CLEAR_ALL_BEFORE_INGEST", "0").lower() in {"1", "true", "yes"}
ROLE_MAPPING = {}  # all PDFs → "Public" by default


# ----------------------- helpers -----------------------

def find_processed_folder(pdf_name: str) -> Optional[Path]:
    """Return the data/processed/file_N folder built from this PDF, or None."""
    if not PROCESSED_DIR.exists():
        return None
    for folder in sorted(PROCESSED_DIR.glob("file_*")):
        chunks_json = folder / "chunks_with_metadata.json"
        if not chunks_json.exists():
            continue
        try:
            with open(chunks_json) as f:
                data = json.load(f)
            if data and data[0].get("source_pdf") == pdf_name:
                return folder
        except Exception:
            continue
    return None


def collect_file_stats(pdf_name: str, folder: Path) -> dict:
    """Per-PDF metrics: chunks, images extracted, image descriptions generated."""
    chunks_json = folder / "chunks_with_metadata.json"
    images_dir = folder / "images"

    chunks_total = 0
    chunks_text = 0
    chunks_image = 0
    if chunks_json.exists():
        with open(chunks_json) as f:
            chunks = json.load(f)
        chunks_total = len(chunks)
        for c in chunks:
            ctype = c.get("chunk_type", "text")
            if ctype == "image":
                chunks_image += 1
            else:
                chunks_text += 1

    images_extracted = 0
    images_described = 0
    if images_dir.exists():
        for image_dir in images_dir.iterdir():
            if not image_dir.is_dir():
                continue
            images_extracted += 1
            image_num = image_dir.name.replace("image_", "")
            desc_file = image_dir / f"image_{image_num}_desc.txt"
            if desc_file.exists() and desc_file.stat().st_size > 0:
                images_described += 1

    failures_path = folder / "image_desc_failures.json"
    image_desc_failures: List[str] = []
    if failures_path.exists():
        try:
            with open(failures_path) as f:
                image_desc_failures = json.load(f) or []
        except Exception:
            pass

    return {
        "pdf": pdf_name,
        "processed_folder": folder.name,
        "chunks_total": chunks_total,
        "chunks_text": chunks_text,
        "chunks_image": chunks_image,
        "images_extracted": images_extracted,
        "images_described": images_described,
        "image_desc_failures": image_desc_failures,
        "image_desc_failed_count": len(image_desc_failures),
        "image_desc_success_rate": (
            round(images_described / images_extracted, 3) if images_extracted else None
        ),
    }


def count_uploaded_by_type(folder: Path) -> Tuple[int, int, int]:
    """What got upserted to Pinecone = what's in chunks_with_metadata.json
    (ingest_chunks_from_json uploads every chunk in the file)."""
    chunks_json = folder / "chunks_with_metadata.json"
    if not chunks_json.exists():
        return 0, 0, 0
    with open(chunks_json) as f:
        chunks = json.load(f)
    text_n = sum(1 for c in chunks if c.get("chunk_type", "text") != "image")
    image_n = sum(1 for c in chunks if c.get("chunk_type") == "image")
    return len(chunks), text_n, image_n


def print_summary_table(stats: List[dict]):
    headers = ["PDF", "chunks", "text", "image", "imgs_extr", "imgs_desc", "img_fail", "upserted(t/i)"]
    rows = []
    totals = {"chunks": 0, "text": 0, "image": 0, "imgs_extr": 0, "imgs_desc": 0,
              "img_fail": 0,
              "uploaded_total": 0, "uploaded_text": 0, "uploaded_image": 0}
    for s in stats:
        rows.append([
            s["pdf"][:40],
            str(s["chunks_total"]),
            str(s["chunks_text"]),
            str(s["chunks_image"]),
            str(s["images_extracted"]),
            str(s["images_described"]),
            str(s["image_desc_failed_count"]),
            f"{s['uploaded_text']}/{s['uploaded_image']}",
        ])
        totals["chunks"] += s["chunks_total"]
        totals["text"] += s["chunks_text"]
        totals["image"] += s["chunks_image"]
        totals["imgs_extr"] += s["images_extracted"]
        totals["imgs_desc"] += s["images_described"]
        totals["img_fail"] += s["image_desc_failed_count"]
        totals["uploaded_total"] += s["uploaded_total"]
        totals["uploaded_text"] += s["uploaded_text"]
        totals["uploaded_image"] += s["uploaded_image"]

    col_widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    fmt = "  ".join("{:<" + str(w) + "}" for w in col_widths)
    print("\n" + "=" * 80)
    print("PER-FILE INGEST STATS")
    print("=" * 80)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in col_widths]))
    for r in rows:
        print(fmt.format(*r))
    print(fmt.format(*["-" * w for w in col_widths]))
    print(
        f"\nTOTAL: {len(stats)} PDFs | chunks={totals['chunks']} "
        f"(text={totals['text']}, image={totals['image']}) | "
        f"images extracted={totals['imgs_extr']}, described={totals['imgs_desc']} "
        f"({round(100 * totals['imgs_desc'] / totals['imgs_extr'], 1) if totals['imgs_extr'] else 0}%) "
        f"| failed={totals['img_fail']} | "
        f"upserted={totals['uploaded_total']} "
        f"(text={totals['uploaded_text']}, image={totals['uploaded_image']})"
    )


# ----------------------- STEP 1: extract -----------------------

if not RAW_DIR.exists():
    print(f"[-] {RAW_DIR} not found. Create it and drop PDFs in there.")
    exit(1)

pdfs = sorted(RAW_DIR.glob("*.pdf"))
if not pdfs:
    print(f"[!] No PDFs found in {RAW_DIR}/")
    exit(0)

print(f"[~] Found {len(pdfs)} PDF(s) in {RAW_DIR}/")
processed_count = 0
for idx, pdf in enumerate(pdfs):
    if not REPROCESS_ALL and find_processed_folder(pdf.name):
        print(f"  ✓ Skipping {pdf.name} (already processed; set REPROCESS_ALL=1 to redo)")
        continue
    print(f"\n[~] Processing {pdf.name}...")
    result = run_full_pipeline(str(pdf), str(PROCESSED_DIR))
    if not result:
        print(f"  ✗ Failed to process {pdf.name}")
    processed_count += 1

    # Only cool down if THIS file hit rate limits — otherwise the vision API
    # is happy and we shouldn't waste minutes idling.
    failures = (result or {}).get("image_desc_failures") or []
    if idx < len(pdfs) - 1 and failures and PER_FILE_COOLDOWN_SEC > 0:
        print(f"  [~] {len(failures)} image(s) failed → cooling down {PER_FILE_COOLDOWN_SEC}s before next file...")
        time.sleep(PER_FILE_COOLDOWN_SEC)


# ----------------------- STEP 2: Pinecone push -----------------------

print(f"\n[~] Connecting to Pinecone...")
embeddings = vector_store.get_embeddings()
pc = vector_store.init_pinecone(PINECONE_API_KEY)

if CLEAR_ALL:
    print("[~] CLEAR_ALL_BEFORE_INGEST=1 → clearing all vectors in index...")
    vector_store.clear_vectors(pc)

# Single shared vector store (avoids the ThreadPool/FD leak on long runs)
shared_vector_store = PineconeVectorStore(
    index_name=vector_store.INDEX_NAME,
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY,
)

# ----------------------- STEP 3: per-PDF stats + ingest -----------------------

all_stats: List[dict] = []

for pdf in pdfs:
    folder = find_processed_folder(pdf.name)
    if folder is None:
        print(f"⚠ No processed folder for {pdf.name}; skipping ingest.")
        continue

    stats = collect_file_stats(pdf.name, folder)
    chunks_json = folder / "chunks_with_metadata.json"

    print(f"\n--- {folder.name} ({pdf.name}) ---")
    role = ROLE_MAPPING.get(pdf.name, "Public")
    try:
        vectors_upserted = vector_store.ingest_chunks_from_json(
            chunks_json_path=str(chunks_json),
            embeddings=embeddings,
            pinecone_api_key=PINECONE_API_KEY,
            required_role=role,
            replace_existing=True,
            vector_store=shared_vector_store,
        )
    except Exception as exc:
        print(f"  ✗ Ingest failed for {pdf.name}: {exc}")
        vectors_upserted = 0

    total, text_n, image_n = count_uploaded_by_type(folder)
    stats["uploaded_total"] = vectors_upserted
    # If upsert returned fewer than the JSON has (e.g. partial failure), scale t/i proportionally;
    # otherwise the breakdown directly mirrors the JSON.
    if vectors_upserted == total or total == 0:
        stats["uploaded_text"] = text_n
        stats["uploaded_image"] = image_n
    else:
        stats["uploaded_text"] = round(text_n * vectors_upserted / total)
        stats["uploaded_image"] = vectors_upserted - stats["uploaded_text"]

    all_stats.append(stats)


# ----------------------- STEP 4: summary -----------------------

print_summary_table(all_stats)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
with open(STATS_PATH, "w") as f:
    json.dump(all_stats, f, indent=2)
print(f"\n[+] Per-file stats written to: {STATS_PATH}")

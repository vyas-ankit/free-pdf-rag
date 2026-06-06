"""Extract metadata using LLM with exact extraction instructions."""

import json
import os
from pydantic import BaseModel, Field
from src.core.llm import get_llm
from src.core.prompts import METADATA_EXTRACTION_PROMPT


class DocumentMetadata(BaseModel):
    """Structured metadata extracted from document."""
    title: str = Field(description="Title - extract exactly as-is from the text, do not add/edit/delete anything, include all sentences/paragraphs")
    date: str = Field(description="Date - extract exactly as-is, do not change format, keep it exactly as it appears")
    theme: str = Field(description="Theme - extract exactly as-is from metadata text, the theme is the text that comes after the date, after navigation elements like Save/Download/Share")


def extract_metadata_with_llm(
    metadata_txt_path: str,
    output_json_path: str = None,
    provider: str = "groq",
    model_name: str = "openai/gpt-oss-120b"
) -> dict:
    """
    Extract metadata using LLM with strict "extract as-is" instructions.
    No inference, no editing, no changes - extract exactly as it appears.

    Args:
        metadata_txt_path: Path to metadata.txt
        output_json_path: Path to save metadata.json
        provider: LLM provider ("google" or "groq")
        model_name: Specific model to use (optional, uses default if None)

    Returns:
        Dictionary with title, date, theme extracted exactly
    """

    if not os.path.exists(metadata_txt_path):
        print(f"ERROR: {metadata_txt_path} not found")
        return None

    # Read metadata.txt
    with open(metadata_txt_path, "r", encoding="utf-8") as f:
        metadata_text = f.read()

    print(f"Read metadata from {metadata_txt_path}")

    # Get LLM with structured output
    if model_name:
        llm = get_llm(model_name=model_name, provider=provider)
        print(f"Using {provider} model: {model_name}")
    else:
        llm = get_llm(provider=provider)
        print(f"Using {provider} default model")

    structured_llm = llm.with_structured_output(DocumentMetadata)

    prompt = METADATA_EXTRACTION_PROMPT.format(metadata_text=metadata_text)

    # Call LLM
    print("Extracting metadata with LLM (exact extraction)...", end="")
    try:
        response = structured_llm.invoke(prompt)
        metadata_dict = response.model_dump()
        print(" ✓")
    except Exception as e:
        print(f" ✗ Error: {str(e)[:100]}")
        return None

    # Display extracted metadata
    print("\n=== EXTRACTED METADATA ===")
    print(f"Title: {metadata_dict['title']}")
    print(f"Date: {metadata_dict['date']}")
    print(f"Theme: {metadata_dict['theme']}")

    # Save to JSON file
    if output_json_path:
        os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2)

        print(f"\n✓ Saved metadata to {output_json_path}")

    return metadata_dict

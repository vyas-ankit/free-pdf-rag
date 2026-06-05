# src/core/config.py

import json
import os
from pathlib import Path

_CONFIG_CACHE = None


def _get_config_path() -> Path:
    """Return path to llm_config.json. Allows override via LLM_CONFIG_PATH env var."""
    env_path = os.getenv("LLM_CONFIG_PATH")
    if env_path:
        return Path(env_path)
    # Default: repo_root/config/llm_config.json
    return Path(__file__).resolve().parent.parent.parent / "config" / "llm_config.json"


def get_config() -> dict:
    """Load and cache the LLM configuration from JSON."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        config_path = _get_config_path()
        if not config_path.exists():
            raise FileNotFoundError(
                f"LLM config not found at {config_path}. "
                f"Copy config/llm_config.example.json to config/llm_config.json."
            )
        with open(config_path, "r", encoding="utf-8") as f:
            _CONFIG_CACHE = json.load(f)
    return _CONFIG_CACHE


def reload_config() -> dict:
    """Force-reload the config (useful for tests or notebooks)."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None
    return get_config()


def get_provider_config(provider: str) -> dict:
    """Return the provider-specific config dict."""
    config = get_config()
    providers = config.get("providers", {})
    if provider not in providers:
        raise ValueError(f"Provider '{provider}' not defined in config. Available: {list(providers.keys())}")
    return providers[provider]


def get_embeddings_config() -> dict:
    return get_config().get("embeddings", {})


def get_pinecone_config() -> dict:
    return get_config().get("pinecone", {})


def get_retrieval_config() -> dict:
    """Return retrieval tuning params (hybrid weight, reranker, candidate_k, final_k)."""
    return get_config().get("retrieval", {})


def get_tool_calling_config() -> dict:
    """Return tool-calling tuning params."""
    return get_config().get("tool_calling", {})


def get_image_desc_config() -> dict:
    """Return image-description tuning params (downscale, etc.)."""
    return get_config().get("image_descriptions", {})


def get_semantic_cache_config() -> dict:
    """Return Redis semantic cache params (score_threshold, etc.)."""
    return get_config().get("semantic_cache", {})


def get_ingestion_config() -> dict:
    """Return ingestion tuning params (pdf_extraction markers, chunking max_words, etc.)."""
    return get_config().get("ingestion", {})


def get_guards_config() -> dict:
    """Return guardrail params (input/output guards and their tuning)."""
    return get_config().get("guards", {})


def get_query_cache_config() -> dict:
    """Return semantic query cache params (enabled, max_entries, threshold, path)."""
    return get_config().get("query_cache", {})

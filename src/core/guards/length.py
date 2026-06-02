# src/core/guards/length.py
"""
Length guard: rejects queries beyond a configurable token budget.

Uses tiktoken's cl100k_base encoder (GPT-4 family). This isn't the *exact*
tokenizer for Groq/Llama or Gemini, but it's within ~10–15% — close enough
to enforce an upper bound. We're rejecting "way too long," not measuring
precisely.
"""

import tiktoken
from langsmith import traceable

from src.core.config import get_guards_config
from src.core.guards._base import GuardResult


_ENCODERS = {}


def _get_encoder(encoder_name: str):
    """Cache encoders so we don't reload them on every call."""
    if encoder_name not in _ENCODERS:
        _ENCODERS[encoder_name] = tiktoken.get_encoding(encoder_name)
    return _ENCODERS[encoder_name]


@traceable(
    name="length_guard",
    run_type="chain",
    metadata={"component": "guardrails"},
)
def length_guard(query: str) -> GuardResult:
    cfg = get_guards_config().get("input", {}).get("length", {})

    if not cfg.get("enabled", True):
        return GuardResult(passed=True)

    max_tokens = int(cfg.get("max_tokens", 5000))
    encoder_name = cfg.get("encoder", "cl100k_base")

    encoder = _get_encoder(encoder_name)
    token_count = len(encoder.encode(query))

    if token_count > max_tokens:
        return GuardResult(
            passed=False,
            reason="length_exceeded",
            message=(
                f"Your message is too long ({token_count} tokens). "
                f"Please keep it under {max_tokens} tokens."
            ),
            details={"token_count": token_count, "max_tokens": max_tokens},
        )

    return GuardResult(passed=True)

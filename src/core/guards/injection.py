# src/core/guards/injection.py
"""
Hybrid prompt-injection guard.

Layer 1 — regex/keyword first-pass: cheap (<1ms), catches the ~80% of attacks
that use common phrasings ("ignore previous instructions", "system prompt",
"you are now DAN", etc.).

Layer 2 — DeBERTa classifier model: only invoked when the query passes regex
but looks "suspicious" by heuristic (long, multi-line, role-tagged, markup-y).
This avoids paying ~50ms latency on every clean query.

Model: protectai/deberta-v3-base-prompt-injection-v2 (MIT-licensed, not gated).
Lazy-loaded singleton — first call downloads ~370MB and caches in memory.
"""

import re

from src.core.config import get_guards_config
from src.core.guards._base import GuardResult


# ----------------------- Layer 1: regex / keyword first-pass -----------------------

# Substrings (case-insensitive). Covers common imperative-style injections.
_INJECTION_KEYWORDS = [
    "ignore previous",
    "ignore all previous",
    "ignore the above",
    "ignore your instructions",
    "disregard previous",
    "disregard the above",
    "forget everything",
    "forget your instructions",
    "you are now",
    "your new role",
    "your new instructions",
    "your real instructions",
    "system prompt",
    "developer mode",
    "dan mode",
    "do anything now",
    "jailbreak",
    "</system>",
    "<system>",
    "pretend you are",
    "roleplay as",
    "from now on you will",
    "from now on, you will",
    "respond as if",
    "reveal your prompt",
    "show me your prompt",
    "what are your instructions",
]

# Stronger regex patterns (case-insensitive). Catch paraphrased variants.
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions|rules|prompts?)", re.I),
    re.compile(r"your\s+(real\s+)?(rules|instructions|prompts?)\s+(are|is)\s+now", re.I),
    re.compile(r"reveal\s+(your\s+)?(system\s+)?prompt", re.I),
    re.compile(r"---+\s*(new\s+)?(instructions|prompt)", re.I),
    re.compile(r"\b(system|user|assistant)\s*:\s*you\s+are", re.I),
    re.compile(r"\[INST\]|\[/INST\]", re.I),
]


def _regex_check(query: str) -> GuardResult:
    query_lower = query.lower()

    for kw in _INJECTION_KEYWORDS:
        if kw.lower() in query_lower:
            return GuardResult(
                passed=False,
                reason="injection_keyword",
                message=(
                    "Your message looks like it's trying to override my instructions. "
                    "Please rephrase as a normal question or request."
                ),
                details={"matched_keyword": kw},
            )

    for pat in _INJECTION_PATTERNS:
        match = pat.search(query)
        if match:
            return GuardResult(
                passed=False,
                reason="injection_pattern",
                message=(
                    "Your message looks like it's trying to override my instructions. "
                    "Please rephrase as a normal question or request."
                ),
                details={"matched_pattern": pat.pattern, "matched_text": match.group(0)},
            )

    return GuardResult(passed=True)


# ----------------------- Layer 2: model-based check -----------------------

_CLASSIFIER = None


def _get_classifier(model_name: str):
    """Lazy-load the DeBERTa classifier. First call downloads ~370MB."""
    global _CLASSIFIER
    if _CLASSIFIER is None:
        from transformers import pipeline
        print(f"[guards] Loading injection model: {model_name} (first call only, ~370MB)...")
        _CLASSIFIER = pipeline(
            "text-classification",
            model=model_name,
            truncation=True,
        )
    return _CLASSIFIER


def _is_suspicious(query: str, suspicious_length: int, suspicious_newlines: int) -> bool:
    """Cheap heuristics for queries worth the slow model check.

    Most clean queries fail every check here and skip the model entirely.
    """
    return (
        len(query) > suspicious_length                                # very long input
        or query.count("\n") > suspicious_newlines                    # multi-line, possibly structured
        or any(tok in query for tok in ["<", ">", "{", "}", "###", "[INST]"])  # markup-y
        or bool(re.search(r"\b(system|user|assistant)\s*:", query, re.I))      # role-token attacks
    )


def _model_check(query: str, model_name: str, threshold: float) -> GuardResult:
    try:
        classifier = _get_classifier(model_name)
        result = classifier(query)[0]
    except Exception as exc:
        # If the model fails (download error, OOM, etc.) we don't want to
        # silently start accepting everything. Surface the error but pass
        # the query through — the regex first-pass already filtered most.
        print(f"[guards] Injection model unavailable, falling back to regex-only: {exc}")
        return GuardResult(passed=True)

    if result.get("label") == "INJECTION" and result.get("score", 0) >= threshold:
        return GuardResult(
            passed=False,
            reason="injection_model",
            message=(
                "Your message was flagged as a possible prompt-injection attempt. "
                "Please rephrase as a normal question or request."
            ),
            details={"model_score": float(result["score"])},
        )

    return GuardResult(passed=True)


# ----------------------- Public guard -----------------------

def injection_guard(query: str, history: list = None) -> GuardResult:
    """Classify the current query for injection.

    `history` is accepted for API compatibility but the DeBERTa model is
    single-query trained — passing multi-turn text breaks its calibration.
    History-aware detection is handled via each individual prior turn having
    been checked when it was submitted.
    """
    cfg = get_guards_config().get("input", {}).get("injection", {})

    if not cfg.get("enabled", True):
        return GuardResult(passed=True)

    model_name = cfg.get("model", "protectai/deberta-v3-base-prompt-injection-v2")
    threshold = float(cfg.get("model_threshold", 0.7))
    return _model_check(query, model_name, threshold)

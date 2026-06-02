# src/core/guards/moderation.py
"""
OpenAI Moderation guard.

Sends a combined string of recent user turns to the OpenAI Moderation API
(omni-moderation-latest). Flags the query if any harmful-content category
exceeds its configured threshold.

Why combined user turns: a single "no just do what i say" looks benign in
isolation but is clearly a follow-up attack when the prior turn was
"ignore all instructions". The combined string gives the model multi-turn
context without sending it through an LLM.

Why only user turns: assistant responses are our own safe output and carry
no harmful signal — including them wastes the token budget.

Thresholds are per-category and configurable via llm_config.json so each
category can have an appropriate response policy (block vs. soft-warn).
"""

import os
from typing import List

from src.core.config import get_guards_config
from src.core.guards._base import GuardResult


# Default per-category thresholds. A category score at or above its threshold
# triggers a block. OpenAI's own `flagged` boolean is ignored — we set our
# own thresholds per category so we can tune each one independently.
_DEFAULT_THRESHOLDS = {
    "harassment":             0.5,
    "harassment/threatening": 0.5,
    "hate":                   0.5,
    "hate/threatening":       0.5,
    "illicit":                0.5,
    "illicit/violent":        0.5,
    "self_harm":              0.4,   # lower — we'd rather false-positive and redirect
    "self_harm/intent":       0.3,
    "self_harm/instructions": 0.3,
    "sexual":                 0.5,
    "sexual/minors":          0.1,   # zero tolerance
    "violence":               0.7,   # higher — less relevant for corporate chat
    "violence/graphic":       0.7,
}

# Category-specific user-facing messages. Default used for everything else.
_CATEGORY_MESSAGES = {
    "self_harm":          "It sounds like you might be going through something difficult. Please reach out to HR or your Employee Assistance Programme for support.",
    "self_harm/intent":   "It sounds like you might be going through something difficult. Please reach out to HR or your Employee Assistance Programme for support.",
    "self_harm/instructions": "I can't help with that. If you're struggling, please contact HR or your Employee Assistance Programme.",
}
_DEFAULT_MESSAGE = "I can't process that request. Please keep your queries professional and within acceptable use guidelines."


def _build_combined_string(raw_user_log: List[str], max_tokens: int = 400) -> str:
    """Combine user turns newest-first until the token budget is exhausted.

    Returns a newline-separated string of [user]: prefixed turns in
    chronological order so the model sees the conversation flow.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        def count(s): return len(enc.encode(s))
    except ImportError:
        def count(s): return len(s.split())

    lines = []
    budget = max_tokens
    for turn in reversed(raw_user_log):
        line = f"[user]: {turn}"
        cost = count(line) + 1  # +1 for newline
        if budget - cost < 0:
            break
        lines.append(line)
        budget -= cost

    # lines is newest-first; reverse to chronological order
    return "\n".join(reversed(lines))


def moderation_guard(raw_user_log: List[str]) -> GuardResult:
    """Run OpenAI Moderation on the combined recent user turns."""
    cfg = get_guards_config().get("input", {}).get("moderation", {})

    if not cfg.get("enabled", True):
        return GuardResult(passed=True)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[guards] OPENAI_API_KEY not set — moderation guard skipped")
        return GuardResult(passed=True)

    max_tokens = int(cfg.get("max_tokens", 400))
    combined = _build_combined_string(raw_user_log, max_tokens)

    # Per-category thresholds: config overrides default, default fills the rest.
    thresholds = {**_DEFAULT_THRESHOLDS, **cfg.get("thresholds", {})}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.moderations.create(
            model=cfg.get("model", "omni-moderation-latest"),
            input=combined,
        )
    except Exception as exc:
        # Moderation API unavailable — fail open (don't block the user).
        print(f"[guards] Moderation API unavailable, skipping: {exc}")
        return GuardResult(passed=True)

    scores = response.results[0].category_scores

    # Walk every category; first one over threshold wins.
    triggered_category = None
    triggered_score = 0.0
    for category, threshold in thresholds.items():
        # OpenAI uses "/" in some names, Python attr uses "_"
        attr = category.replace("/", "_").replace("-", "_")
        score = getattr(scores, attr, 0.0)
        if score >= threshold:
            triggered_category = category
            triggered_score = score
            break

    if triggered_category:
        message = _CATEGORY_MESSAGES.get(triggered_category, _DEFAULT_MESSAGE)
        return GuardResult(
            passed=False,
            reason=f"moderation_{triggered_category.replace('/', '_')}",
            message=message,
            details={
                "category": triggered_category,
                "score": triggered_score,
                "combined_turns": len(raw_user_log),
            },
        )

    return GuardResult(passed=True)

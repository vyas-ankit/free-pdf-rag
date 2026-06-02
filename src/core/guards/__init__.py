# src/core/guards/__init__.py
"""
Input guardrails — run before any LLM/retrieval work.

Public API:
    run_input_guards(query, raw_user_log) -> GuardResult

Two separate stores flow through here:
  raw_user_log  — every user turn (clean + blocked). Only the moderation
                  guard reads this; the LLM pipeline never sees it.
  SESSION_MEMORY — clean turns + AI responses. Never passed here;
                   stays in rag_logic.py for the graph to consume.
"""

from typing import List

from src.core.guards._base import GuardResult
from src.core.guards.length import length_guard
from src.core.guards.moderation import moderation_guard


def run_input_guards(query: str, raw_user_log: List[str] = None) -> GuardResult:
    """Run every enabled input guard. Returns the first failure or a pass.

    Args:
        query:         The current raw user query.
        raw_user_log:  All user turns so far (including current), clean + blocked.
                       Used by the moderation guard for multi-turn context.
    """
    # Length guard — raw query only.
    result = length_guard(query)
    if not result.passed:
        return result

    # Moderation guard — full user history for multi-turn detection.
    result = moderation_guard(raw_user_log or [query])
    if not result.passed:
        return result

    return GuardResult(passed=True)


__all__ = ["GuardResult", "run_input_guards", "length_guard", "moderation_guard"]

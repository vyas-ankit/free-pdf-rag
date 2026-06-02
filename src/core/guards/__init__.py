# src/core/guards/__init__.py
"""
Input guardrails — run before any LLM/retrieval work.

Public API:
    run_input_guards(query, raw_user_log, user_id, session_id) -> GuardResult
"""

from typing import List

from langsmith import traceable

from src.core.guards._base import GuardResult
from src.core.guards.length import length_guard
from src.core.guards.moderation import moderation_guard


@traceable(
    name="input_guards",
    run_type="chain",
    metadata={"component": "guardrails"},
)
def run_input_guards(
    query: str,
    raw_user_log: List[str] = None,
    user_id: str = "guest_user",
    session_id: str = "default_session",
) -> GuardResult:
    """Run every enabled input guard. Returns the first failure or a pass.

    Args:
        query:         The current raw user query.
        raw_user_log:  All user turns so far (clean + blocked) for multi-turn
                       moderation context.
        user_id:       Authenticated user — passed as trace metadata.
        session_id:    Session identifier — passed as trace metadata.
    """
    # Length guard — raw query only
    result = length_guard(query)
    if not result.passed:
        return result

    # Moderation guard — full user history
    result = moderation_guard(raw_user_log or [query])
    if not result.passed:
        return result

    return GuardResult(passed=True)


__all__ = ["GuardResult", "run_input_guards", "length_guard", "moderation_guard"]

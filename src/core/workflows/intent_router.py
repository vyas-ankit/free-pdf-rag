# src/core/workflows/intent_router.py
"""
Intent router — runs on every turn before anything else.

Returns one of:
  - "RAG"              → pass to existing query_rag_simple flow
  - "BOOK_DESK"        → run book_desk workflow
  - "SUBMIT_VACATION"  → run submit_vacation workflow
  - "CONTEXT_SWITCH"   → user switched context mid-action; needs confirmation

Also manages _session_action_state: if an action is in progress and the router
detects a different intent, it returns CONTEXT_SWITCH so the caller can ask the
user whether to abandon the current action.
"""

import json
from typing import Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from src.core.llm import get_llm
from src.core.prompts import INTENT_ROUTER_PROMPT

# Maps workflow name → intent label (keep in sync with YAML filenames)
WORKFLOW_INTENTS = {
    "BOOK_DESK",
    "SUBMIT_VACATION",
}

# Pending context-switch state: stores what the user originally wanted before
# we interrupted to confirm abandonment.
# Shape: {session_id: {"pending_intent": str, "pending_query": str}}
_pending_context_switch: dict[str, dict] = {}


def get_pending_context_switch(session_id: str) -> Optional[dict]:
    return _pending_context_switch.get(session_id)


def clear_pending_context_switch(session_id: str) -> None:
    _pending_context_switch.pop(session_id, None)


def set_pending_context_switch(session_id: str, intent: str, query: str) -> None:
    _pending_context_switch[session_id] = {"pending_intent": intent, "pending_query": query}


@traceable(name="intent_router.classify", run_type="llm",
           metadata={"component": "intent_router"})
def classify_intent(
    user_query: str,
    history: list[BaseMessage],
    active_workflow: Optional[str],
    model_name: str,
) -> str:
    """Call LLM to classify user intent. Returns one of the intent labels."""
    history_text = "\n".join(
        f"{'user' if isinstance(msg, HumanMessage) else 'assistant'}: {msg.content}"
        for msg in history[-6:]
    ) or "(no prior conversation)"

    llm = get_llm(model_name=model_name)
    prompt = ChatPromptTemplate.from_messages([
        ("system", INTENT_ROUTER_PROMPT),
    ])
    chain = prompt | llm | StrOutputParser()
    raw = (chain.invoke({
        "history": history_text,
        "user_query": user_query,
        "active_workflow": active_workflow or "none",
        "available_workflows": ", ".join(WORKFLOW_INTENTS),
    }) or "").strip().upper()

    # Normalise — accept known labels, fall back to RAG
    valid = WORKFLOW_INTENTS | {"RAG", "CONTINUE"}
    intent = raw if raw in valid else "RAG"
    print(f"[intent_router] query={user_query!r} active_workflow={active_workflow!r} → intent={intent}")
    return intent


def route_intent(
    user_query: str,
    history: list[BaseMessage],
    session_id: str,
    active_workflow: Optional[str],
    model_name: str,
) -> str:
    """
    Top-level routing logic.

    If an action is already in progress and the LLM detects a *different* intent,
    we return CONTEXT_SWITCH and stash the new intent so it can be resumed after
    the user confirms abandonment.
    """
    detected = classify_intent(user_query, history, active_workflow, model_name)

    if detected == "CONTINUE":
        return "CONTINUE"

    if active_workflow and detected != active_workflow:
        print(f"[intent_router] context switch detected: {active_workflow} → {detected}")
        set_pending_context_switch(session_id, detected, user_query)
        return "CONTEXT_SWITCH"

    return detected

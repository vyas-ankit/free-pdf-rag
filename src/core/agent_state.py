# src/core/agent_state.py
"""
State schema for the LangGraph RAG + actions pipeline.

Replaces the scattered local variables, in-memory dicts (_pending_context_switch,
_raw_user_logs) and the SQLite workflow_state table from the pre-LangGraph
implementation. The checkpointer persists this dict per session_id (thread_id),
so workflow progress and context-switch state survive across turns and restarts.
"""

from typing import List, Optional, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    # ── Identity (set fresh on every invocation) ─────────────────────────
    session_id: str
    user_id: str
    user_role: str
    user_query: str
    model_name: str
    prompt_template: Optional[str]
    return_contexts: bool
    skip_guards: bool

    # ── Conversation ──────────────────────────────────────────────────────
    history: List[BaseMessage]
    is_first_turn: bool
    raw_user_log: List[str]

    # ── Routing ───────────────────────────────────────────────────────────
    intent: str  # RAG | BOOK_DESK | SUBMIT_VACATION | CONTINUE | CONTEXT_SWITCH | BLOCKED
    active_workflow: Optional[str]
    pending_context_switch: Optional[dict]  # {"pending_intent": str, "pending_query": str}

    # ── Workflow execution state (persisted by checkpointer) ─────────────
    workflow_name: Optional[str]
    current_step: int
    collected_params: dict
    awaiting_confirmation: bool
    last_tool_results: dict
    workflow_last_active: Optional[str]  # ISO timestamp string

    # ── RAG pipeline ──────────────────────────────────────────────────────
    rewritten_query: str
    retrieval_queries: List[str]
    retrieved_contexts: List[str]
    cache_hit: bool

    # ── Output ────────────────────────────────────────────────────────────
    answer: str
    blocked_reason: Optional[str]

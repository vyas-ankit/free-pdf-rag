# src/core/graph.py
"""
Assembles and compiles the LangGraph state machine for the RAG + actions
pipeline. Each turn is a single graph.invoke() call; the checkpointer
persists AgentState per session_id (= thread_id), so workflow progress,
context-switch state, and pause/resume all survive across turns and process
restarts — replacing the old SQLite workflow_state table and in-memory
_pending_context_switch dict.

Flow (mirrors the old query_rag_simple() control flow exactly):

    START
      → guard_node
          ├─ BLOCKED → END
          └─ OK
      → (pending context-switch?)
          ├─ yes → pending_switch_node → route by resolved intent
          └─ no
      → intent_router_node
          ├─ CONTEXT_SWITCH → context_switch_node → END
          ├─ BOOK_DESK / SUBMIT_VACATION / CONTINUE → workflow_node → END
          └─ RAG
      → cache_lookup_node
          ├─ HIT  → END
          └─ MISS
      → query_rewriter_node → query_expander_node → retrieval_answer_node
          → cache_write_node → END
"""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from src.core.agent_state import AgentState
from src.core.config import get_config
from src.core import nodes


def _db_path() -> str:
    path = get_config().get("session_store", {}).get("db_path", "data/sessions.db")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def _route_after_pending_switch(state: AgentState) -> str:
    intent = state.get("intent", "RAG")
    if intent == "RAG":
        return "RAG"
    return "WORKFLOW"


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("guard", nodes.guard_node)
    workflow.add_node("pending_switch", nodes.pending_switch_node)
    workflow.add_node("intent_router", nodes.intent_router_node)
    workflow.add_node("context_switch", nodes.context_switch_node)
    workflow.add_node("workflow_step", nodes.workflow_node)
    workflow.add_node("cache_lookup", nodes.cache_lookup_node)
    workflow.add_node("query_rewriter", nodes.query_rewriter_node)
    workflow.add_node("query_expander", nodes.query_expander_node)
    workflow.add_node("retrieval_answer", nodes.retrieval_answer_node)
    workflow.add_node("cache_write", nodes.cache_write_node)

    workflow.set_entry_point("guard")

    # Single routing function from guard: blocked → END, else dispatch to
    # pending-switch resolution (if a confirmation is outstanding) or
    # straight to intent classification.
    workflow.add_conditional_edges("guard", _route_after_guard, {
        "BLOCKED": END,
        "PENDING": "pending_switch",
        "INTENT": "intent_router",
    })

    workflow.add_conditional_edges("pending_switch", _route_after_pending_switch, {
        "RAG": "cache_lookup",
        "WORKFLOW": "workflow_step",
    })

    workflow.add_conditional_edges("intent_router", nodes.route_by_intent, {
        "CONTEXT_SWITCH": "context_switch",
        "WORKFLOW": "workflow_step",
        "RAG": "cache_lookup",
    })

    workflow.add_conditional_edges("cache_lookup", nodes.route_after_cache, {
        "HIT": END,
        "MISS": "query_rewriter",
    })

    workflow.add_edge("query_rewriter", "query_expander")
    workflow.add_edge("query_expander", "retrieval_answer")
    workflow.add_edge("retrieval_answer", "cache_write")
    workflow.add_edge("cache_write", END)

    workflow.add_edge("context_switch", END)
    workflow.add_edge("workflow_step", END)

    conn = sqlite3.connect(_db_path(), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return workflow.compile(checkpointer=checkpointer)


def _route_after_guard(state: AgentState) -> str:
    if state.get("intent") == "BLOCKED":
        return "BLOCKED"
    if state.get("pending_context_switch"):
        return "PENDING"
    return "INTENT"


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph

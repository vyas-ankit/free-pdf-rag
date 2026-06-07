# src/core/rag_simple.py
"""
Public entry point for the RAG + actions pipeline — now backed by a
LangGraph StateGraph (see graph.py / nodes.py / agent_state.py).

query_rag_simple() keeps its exact original signature and behaviour:
multi-turn RAG Q&A with query rewriting/expansion/tool-calling, YAML-driven
action workflows (book desk, submit vacation), intent routing, context-switch
confirmation, semantic caching, and input guardrails.

The difference from the pre-LangGraph version is purely internal: workflow
progress and context-switch state are persisted by the graph's checkpointer
(keyed on session_id as thread_id) instead of a hand-rolled SQLite table and
an in-memory dict — eliminating the manual "where were we?" resume logic.
"""

from typing import Optional, Union

from langchain_core.messages import HumanMessage
from langsmith import traceable

from src.core import session_store
from src.core.graph import get_compiled_graph
from src.core.llm import get_default_model

ACTIVE_LLM_MODEL = get_default_model()


def clear_session(session_id: str) -> None:
    """Reset conversation history and any in-progress workflow/context-switch state."""
    session_store.clear_history(session_id)
    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": session_id}}
    graph.update_state(config, {
        "active_workflow": None,
        "workflow_name": None,
        "current_step": 0,
        "collected_params": {},
        "awaiting_confirmation": False,
        "last_tool_results": {},
        "workflow_last_active": None,
        "pending_context_switch": None,
        "raw_user_log": [],
    })


@traceable(name="rag_simple.query", run_type="chain", metadata={"component": "entry_point"})
def query_rag_simple(
    user_query: str,
    vector_store,
    session_id: str = "default_session",
    user_role: str = "Public",
    user_id: str = "guest_user",
    candidate_k: Optional[int] = None,
    final_k: Optional[int] = None,
    model_name: str = ACTIVE_LLM_MODEL,
    prompt_template: Optional[str] = None,
    return_contexts: bool = False,
    skip_guards: bool = False,
) -> Union[str, dict]:
    """Multi-turn RAG Q&A + action workflows, orchestrated by a LangGraph graph.

    Args mirror the pre-LangGraph implementation exactly — see graph.py for
    the internal flow this dispatches to.

    Returns:
        str when return_contexts=False (default).
        dict {"answer": str, "contexts": List[str], "rewritten_query": str}
             when return_contexts=True.
    """
    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": session_id, "vector_store": vector_store}}

    history = session_store.get_history(session_id)
    is_first_turn = len(history) == 0

    # Pull forward any persisted cross-turn state (workflow progress,
    # pending context-switch, raw user log) from the checkpointer.
    snapshot = graph.get_state(config)
    prior = snapshot.values if snapshot and snapshot.values else {}

    input_state = {
        "session_id": session_id,
        "user_id": user_id,
        "user_role": user_role,
        "user_query": user_query,
        "model_name": model_name,
        "prompt_template": prompt_template,
        "return_contexts": return_contexts,
        "skip_guards": skip_guards,

        "history": history,
        "is_first_turn": is_first_turn,
        "raw_user_log": prior.get("raw_user_log", []),

        "intent": "",
        "active_workflow": prior.get("active_workflow"),
        "pending_context_switch": prior.get("pending_context_switch"),

        "workflow_name": prior.get("workflow_name"),
        "current_step": prior.get("current_step", 0),
        "collected_params": prior.get("collected_params", {}),
        "awaiting_confirmation": prior.get("awaiting_confirmation", False),
        "last_tool_results": prior.get("last_tool_results", {}),
        "workflow_last_active": prior.get("workflow_last_active"),

        "rewritten_query": "",
        "retrieval_queries": [],
        "retrieved_contexts": [],
        "cache_hit": False,

        "answer": "",
        "blocked_reason": None,
    }

    result = graph.invoke(input_state, config=config)

    answer = result.get("answer", "")

    # Persist conversation turn — mirrors the original _append_history calls.
    # Skipped for cache hits on a fresh session-less re-entry isn't a concern
    # here since get_history/append_history are the same store as before.
    session_store.append_history(session_id, user_query, answer)

    if return_contexts:
        return {
            "answer": answer,
            "contexts": result.get("retrieved_contexts", []),
            "rewritten_query": result.get("rewritten_query") or user_query,
        }
    return answer

# src/core/rag_logic.py
"""
Public entry point for the planning agent.

`query_rag` invokes the compiled LangGraph with a per-session thread_id so
each user has isolated, persistent conversation state via MemorySaver.

Where the pieces live:
  - AgentState schema                      → src/core/agent_state.py
  - Workflow assembly + compiled_graph     → src/core/graph.py
  - LangGraph node functions               → src/core/nodes.py
  - LangChain tools the agent can call     → src/core/tools.py
  - Pinecone management + ingestion        → src/core/vector_store.py
  - Search/rerank pipeline                 → src/core/retrieval.py
  - S3 helpers                             → src/core/aws.py
"""

from typing import List

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langsmith import traceable

from src.core.graph import build_graph
from src.core.guards import run_input_guards
from src.core.nodes import ACTIVE_LLM_MODEL
from src.core.prompts import SYSTEM_RAG_PROMPT
from src.core.retrieval import candidate_k_default, final_k_default


# ── Checkpointer + compiled graph ────────────────────────────────────────
_checkpointer = MemorySaver()
_compiled_graph = build_graph(checkpointer=_checkpointer)

# ── Vector store registry ─────────────────────────────────────────────────
# PineconeVectorStore is not msgpack-serializable so it cannot live in
# AgentState (the checkpointer would fail to snapshot it). Instead, nodes
# that need it import this module and read _vector_store directly.
_vector_store = None

def set_vector_store(vs) -> None:
    """Called once at startup by the backend/CLI before any query."""
    global _vector_store
    _vector_store = vs

# ── RAW_USER_LOG ──────────────────────────────────────────────────────────
_raw_user_logs: dict[str, List[str]] = {}


@traceable(
    name="query_rag",
    run_type="chain",
    metadata={"component": "entry_point"},
)
def query_rag(
    user_query: str,
    vector_store=None,
    llm_api_key: str = None,
    user_id: str = "guest_user",
    session_id: str = "default_session",
    user_role: str = "Public",
    candidate_k: int = None,
    final_k: int = None,
    prompt_template: str = None,
    model_name: str = ACTIVE_LLM_MODEL,
) -> str:
    # Register vector store for this call (nodes read from _vector_store)
    if vector_store is not None:
        set_vector_store(vector_store)

    # ── 1. Append to per-session raw log (before guard) ───────────────────
    if session_id not in _raw_user_logs:
        _raw_user_logs[session_id] = []
    _raw_user_logs[session_id].append(user_query)

    # ── 2. Input guardrails ───────────────────────────────────────────────
    guard_result = run_input_guards(
        user_query,
        raw_user_log=_raw_user_logs[session_id],
        user_id=user_id,
        session_id=session_id,
    )
    if not guard_result.passed:
        print(f"[guards] Blocked: reason={guard_result.reason} details={guard_result.details}")
        return guard_result.message or "I can't process that request."

    # ── 3. Build initial state ────────────────────────────────────────────
    resolved_candidate_k = candidate_k or candidate_k_default()
    resolved_final_k = final_k or final_k_default()

    initial_state = {
        "user_query": user_query,
        "messages": [HumanMessage(content=user_query)],
        "user_id": user_id,
        "user_role": user_role,
        "llm_api_key": llm_api_key,
        "candidate_k": resolved_candidate_k,
        "final_k": resolved_final_k,
        "prompt_template": prompt_template or SYSTEM_RAG_PROMPT,
        "model_name": model_name,
    }

    # ── 4. Invoke graph ───────────────────────────────────────────────────
    config = {"configurable": {"thread_id": session_id}}
    final_state = _compiled_graph.invoke(initial_state, config=config)

    # ── 5. Return last AI message ─────────────────────────────────────────
    messages = final_state.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and not isinstance(msg, HumanMessage):
            return msg.content

    return "I couldn't generate a response. Please try again."

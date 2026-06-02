# src/core/rag_logic.py
"""
The agent's public surface: `query_rag` invokes the compiled LangGraph and
manages in-memory session memory across turns.

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

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_pinecone import PineconeVectorStore

from src.core.graph import compiled_graph
from src.core.guards import run_input_guards
from src.core.nodes import ACTIVE_LLM_MODEL  # re-exposed for callers (tests/evaluate.py)
from src.core.prompts import SYSTEM_RAG_PROMPT
from src.core.retrieval import candidate_k_default, final_k_default


# SESSION_MEMORY: clean turns only (passed guard + AI responses).
# Fed to the rewriter, intent classifier, and executors.
SESSION_MEMORY: List[BaseMessage] = []

# RAW_USER_LOG: every user turn regardless of whether it passed or was blocked.
# Fed to the moderation guard only — the LLM pipeline never sees this.
# This is what enables multi-turn harmful-content detection: even blocked
# turns are visible to the next call's guard.
RAW_USER_LOG: List[str] = []


def query_rag(
    user_query: str,
    vector_store: PineconeVectorStore,
    llm_api_key: str = None,
    user_id: str = "guest_user",
    session_id: str = "default_session",
    user_role: str = "Public",
    candidate_k: int = None,
    final_k: int = None,
    prompt_template: str = None,
    model_name: str = ACTIVE_LLM_MODEL,
) -> str:
    global SESSION_MEMORY, RAW_USER_LOG

    # Every user turn goes into RAW_USER_LOG immediately — before any guard.
    # The moderation guard reads this log to see the full history including
    # prior blocked turns (which never enter SESSION_MEMORY).
    RAW_USER_LOG.append(user_query)

    # --- Input guardrails ---
    # Length guard: raw query only (we're capping what the user typed).
    # Moderation guard: full RAW_USER_LOG (sees blocked turns too).
    guard_result = run_input_guards(user_query, raw_user_log=RAW_USER_LOG)
    if not guard_result.passed:
        print(f"[guards] Blocked: reason={guard_result.reason} details={guard_result.details}")
        # Blocked query and the rejection message are NOT added to SESSION_MEMORY.
        # The LLM pipeline never sees this turn happened.
        return guard_result.message or "I can't process that request."

    # Guard passed — add to SESSION_MEMORY so rewriter/executors have context.
    messages = list(SESSION_MEMORY)
    messages.append(HumanMessage(content=user_query))

    # Resolve k values from config — explicit args still win if passed.
    resolved_candidate_k = candidate_k or candidate_k_default()
    resolved_final_k = final_k or final_k_default()

    initial_state = {
        "user_query": user_query,
        "messages": messages,
        "user_id": user_id,
        "user_role": user_role,
        "llm_api_key": llm_api_key,
        "vector_store": vector_store,
        "candidate_k": resolved_candidate_k,
        "final_k": resolved_final_k,
        "prompt_template": prompt_template or SYSTEM_RAG_PROMPT,
        "model_name": model_name,
        "final_report": "",
    }

    final_state = compiled_graph.invoke(initial_state)

    # Only clean turns + AI responses enter SESSION_MEMORY.
    SESSION_MEMORY = final_state["messages"]

    return final_state["messages"][-1].content

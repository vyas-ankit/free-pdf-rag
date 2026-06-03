# src/core/rag_simple.py
"""
Simple multi-turn RAG pipeline — pure Q&A, no actions, no planning graph.

Architecture per turn:
    user_query
        │
        ▼
    input_guards          (length + moderation)
        │
        ▼
    query_rewriter        (LangChain LCEL — resolves pronouns using history)
        │
        ▼
    retriever             (hybrid dense + BM25 + cross-encoder rerank)
        │
        ▼
    answer_chain          (LangChain LCEL — context + history + query → answer)
        │
        ▼
    answer (str)

Session history is stored in a per-session dict (in-memory). No LangGraph,
no checkpointer — just a plain list of LangChain BaseMessages per session_id.

LangSmith tracing is applied at every stage via @traceable decorators.
"""

from typing import Dict, List, Optional, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_pinecone import PineconeVectorStore
from langsmith import traceable

from src.core.cache import get_cache
from src.core.guards import run_input_guards
from src.core.llm import get_default_model, get_llm
from src.core.prompts import QUERY_REWRITER_PROMPT, SYSTEM_RAG_PROMPT
from src.core.retrieval import (
    candidate_k_default,
    final_k_default,
    format_docs,
    retrieve_hybrid_and_rerank,
)


ACTIVE_LLM_MODEL = get_default_model()

# Per-session conversation history — keyed by session_id.
# Each value is a list of HumanMessage / AIMessage objects.
_session_histories: Dict[str, List[BaseMessage]] = {}

# Per-session raw user log for multi-turn moderation guard.
_raw_user_logs: Dict[str, List[str]] = {}


# ── Helpers ───────────────────────────────────────────────────────────────

def _get_history(session_id: str) -> List[BaseMessage]:
    return _session_histories.get(session_id, [])


def _append_history(session_id: str, human: str, ai: str) -> None:
    if session_id not in _session_histories:
        _session_histories[session_id] = []
    _session_histories[session_id].append(HumanMessage(content=human))
    _session_histories[session_id].append(AIMessage(content=ai))


def clear_session(session_id: str) -> None:
    """Reset conversation history and raw log for a session."""
    _session_histories.pop(session_id, None)
    _raw_user_logs.pop(session_id, None)


# ── Query rewriter ────────────────────────────────────────────────────────

@traceable(name="rag_simple.query_rewriter", run_type="llm",
           metadata={"component": "rewriter"})
def _rewrite_query(user_query: str, history: List[BaseMessage],
                   model_name: str) -> str:
    """Resolve pronouns / references using conversation history.

    Skips the LLM call entirely when there is no prior history — the raw
    query is already standalone.
    """
    if not history:
        return user_query

    llm = get_llm(model_name=model_name)
    prompt = ChatPromptTemplate.from_messages([
        ("system", QUERY_REWRITER_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{user_query} + \nRemember that you must output a rewritten query only."),

    ])
    chain = prompt | llm | StrOutputParser()
    rewritten = (chain.invoke({
        "history": history,
        "user_query": user_query,
    }) or "").strip()

    # Sanity fallback: if rewriter returns empty or echoes an AI message, use original
    if not rewritten or len(rewritten) < 3:
        return user_query

    print(f"[rag_simple] Rewritten query: {rewritten!r}")
    return rewritten


# ── Answer chain ──────────────────────────────────────────────────────────

@traceable(name="rag_simple.answer_chain", run_type="llm",
           metadata={"component": "answer"})
def _generate_answer(rewritten_query: str, context: str,
                     history: List[BaseMessage], model_name: str,
                     prompt_template: Optional[str]) -> str:
    """Generate an answer from retrieved context + conversation history."""
    system = prompt_template or SYSTEM_RAG_PROMPT

    llm = get_llm(model_name=model_name)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    chain = (
        prompt
        | llm.with_config({"run_name": "rag_simple_llm"})
        | StrOutputParser()
    )
    return chain.invoke({
        "context": context,
        "history": history,
        "question": rewritten_query,
    })


# ── Public entry point ────────────────────────────────────────────────────

@traceable(name="rag_simple.query", run_type="chain",
           metadata={"component": "entry_point"})
def query_rag_simple(
    user_query: str,
    vector_store: PineconeVectorStore,
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
    """Multi-turn RAG Q&A — no actions, no planning graph.

    Args:
        user_query:      The user's raw input.
        vector_store:    Pinecone vector store to retrieve from.
        session_id:      Identifies the conversation. State is kept per session_id.
        user_role:       Role-based retrieval filter ("Public" or "Finance").
        user_id:         Authenticated user identifier (used for guard metadata).
        candidate_k:     How many docs to pull from Pinecone (defaults to config).
        final_k:         How many docs survive reranking (defaults to config).
        model_name:      LLM model to use (defaults to config default_provider model).
        prompt_template: Override the system prompt. Use {context} placeholder.
        return_contexts: When True, returns a dict with keys: answer, contexts,
                         rewritten_query. Used by the RAGAS eval pipeline.
        skip_guards:     When True, bypasses input guards. Only set True for
                         offline evaluation — never for production queries.

    Returns:
        str when return_contexts=False (default).
        dict {"answer": str, "contexts": List[str], "rewritten_query": str}
             when return_contexts=True.
    """
    # ── 1. Raw user log (before guard) ───────────────────────────────────
    if session_id not in _raw_user_logs:
        _raw_user_logs[session_id] = []
    _raw_user_logs[session_id].append(user_query)

    # ── 2. Input guards ───────────────────────────────────────────────────
    if not skip_guards:
        guard_result = run_input_guards(
            user_query,
            raw_user_log=_raw_user_logs[session_id],
            user_id=user_id,
            session_id=session_id,
        )
        if not guard_result.passed:
            print(f"[rag_simple] Blocked: reason={guard_result.reason}")
            blocked_msg = guard_result.message or "I can't process that request."
            return blocked_msg if not return_contexts else {
                "answer": blocked_msg, "contexts": [], "rewritten_query": user_query
            }

    history = _get_history(session_id)
    resolved_candidate_k = candidate_k or candidate_k_default()
    resolved_final_k = final_k or final_k_default()
    is_first_turn = len(history) == 0

    # ── 3. Cache lookup (first turn only) ────────────────────────────────
    # Subsequent turns depend on conversation history so must always retrieve.
    if is_first_turn and not return_contexts and not skip_guards:
        cached = get_cache().get(user_query)
        if cached is not None:
            _append_history(session_id, user_query, cached)
            return cached

    # ── 4. Rewrite query (pronoun resolution) ─────────────────────────────
    rewritten = _rewrite_query(user_query, history, model_name)

    # ── 5. Retrieve ───────────────────────────────────────────────────────
    docs = retrieve_hybrid_and_rerank(
        query=rewritten,
        vector_store=vector_store,
        user_role=user_role,
        candidate_k=resolved_candidate_k,
        final_k=resolved_final_k,
    )
    context = format_docs(docs)

    # ── 6. Generate answer ────────────────────────────────────────────────
    answer = _generate_answer(rewritten, context, history, model_name, prompt_template)

    # ── 7. Persist turn to session history ───────────────────────────────
    _append_history(session_id, user_query, answer)

    # ── 8. Cache write (first turn only, not eval mode) ──────────────────
    if is_first_turn and not return_contexts and not skip_guards:
        get_cache().set(user_query, answer)

    if return_contexts:
        return {
            "answer": answer,
            "contexts": [doc.page_content for doc in docs],
            "rewritten_query": rewritten,
        }
    return answer

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
    query_expander        (creates 1-N focused retrieval queries)
        │
        ▼
    hard retrieval        (retrieve_knowledge_base is called for every expanded query)
        │
        ▼
    tool_calling_llm      (can call retrieve_knowledge_base again if useful)
        │
        ▼
    final answer          (LLM answers from retrieved tool results)
        │
        ▼
    answer (str)

Session history is stored in a per-session dict (in-memory). No LangGraph,
no checkpointer — just a plain list of LangChain BaseMessages per session_id.

LangSmith tracing is applied at every stage via @traceable decorators.
"""

import json
import re
from typing import Dict, List, Optional, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_pinecone import PineconeVectorStore
from langsmith import traceable

from src.core.cache import get_cache
from src.core.config import get_tool_calling_config
from src.core.guards import run_input_guards
from src.core.llm import get_default_model, get_llm
from src.core.prompts import (
    QUERY_EXPANSION_PROMPT,
    QUERY_REWRITER_PROMPT,
    SYSTEM_RAG_PROMPT,
    TOOL_RAG_SYSTEM_PROMPT,
)
from src.core.retrieval import (
    candidate_k_default,
    final_k_default,
)
from src.core.tools import make_retrieval_tool
from src.core.workflows.engine import (
    _run_llm_extract,
    run_workflow_step,
)
from src.core.workflows.intent_router import (
    clear_pending_context_switch,
    get_pending_context_switch,
    route_intent,
)
from src.core import session_store


ACTIVE_LLM_MODEL = get_default_model()

# Per-session raw user log for multi-turn moderation guard (still in-memory — not persisted).
_raw_user_logs: Dict[str, List[str]] = {}


# ── Helpers ───────────────────────────────────────────────────────────────

def _get_history(session_id: str) -> List[BaseMessage]:
    return session_store.get_history(session_id)


def _append_history(session_id: str, human: str, ai: str) -> None:
    session_store.append_history(session_id, human, ai)


def clear_session(session_id: str) -> None:
    """Reset conversation history, workflow state, and raw log for a session."""
    session_store.clear_history(session_id)
    session_store.clear_workflow_state(session_id)
    _raw_user_logs.pop(session_id, None)


def max_tool_call_rounds_default() -> int:
    return int(get_tool_calling_config().get("max_tool_call_rounds", 4))


def query_expansion_enabled() -> bool:
    cfg = get_tool_calling_config().get("query_expansion", {})
    return bool(cfg.get("enabled", True))


def max_expanded_queries_default() -> int:
    cfg = get_tool_calling_config().get("query_expansion", {})
    return int(cfg.get("max_queries", 4))


def _dedupe_queries(queries: List[str], max_queries: int) -> List[str]:
    seen = set()
    deduped = []
    for query in queries:
        cleaned = " ".join(str(query).strip().split())
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
        if len(deduped) >= max_queries:
            break
    return deduped


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


# ── Query expansion ──────────────────────────────────────────────────────

@traceable(name="rag_simple.query_expander", run_type="llm",
           metadata={"component": "query_expander"})
def _expand_queries(rewritten_query: str, history: List[BaseMessage],
                    model_name: str) -> List[str]:
    """Generate focused retrieval queries before the hard retrieval pass."""
    max_queries = max(1, max_expanded_queries_default())
    if not query_expansion_enabled() or max_queries == 1:
        return [rewritten_query]

    llm = get_llm(model_name=model_name)
    history_text = "\n".join(
        f"{'user' if isinstance(msg, HumanMessage) else 'assistant'}: {msg.content}"
        for msg in history[-6:]
    ) or "(no prior conversation)"
    prompt = ChatPromptTemplate.from_messages([
        ("system", QUERY_EXPANSION_PROMPT),
    ])
    chain = prompt | llm | StrOutputParser()
    raw = (chain.invoke({
        "history": history_text,
        "question": rewritten_query,
        "max_queries": max_queries,
    }) or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)

    try:
        payload = json.loads(cleaned)
        queries = payload.get("queries", [])
        if not isinstance(queries, list):
            queries = []
    except json.JSONDecodeError:
        queries = []

    expanded = _dedupe_queries(queries, max_queries)
    if not expanded:
        expanded = [rewritten_query]

    print(f"[rag_simple] Expanded retrieval queries: {expanded!r}")
    return expanded


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


# ── Tool-calling answer loop ─────────────────────────────────────────────

@traceable(name="rag_simple.tool_calling_answer", run_type="chain",
           metadata={"component": "tool_calling_answer"})
def _generate_answer_with_retrieval_tool(
    user_query: str,
    rewritten_query: str,
    retrieval_queries: List[str],
    history: List[BaseMessage],
    vector_store: PineconeVectorStore,
    user_role: str,
    candidate_k: int,
    final_k: int,
    model_name: str,
    prompt_template: Optional[str],
) -> tuple[str, List[str]]:
    """Force one retrieval, then let the LLM decide whether to retrieve more."""
    retrieved_contexts: List[str] = []
    retrieve_knowledge_base = make_retrieval_tool(
        vector_store=vector_store,
        user_role=user_role,
        candidate_k=candidate_k,
        final_k=final_k,
        retrieved_contexts=retrieved_contexts,
    )

    system = prompt_template or TOOL_RAG_SYSTEM_PROMPT
    if "{context}" in system:
        system = system.replace(
            "{context}",
            "Use the retrieve_knowledge_base tool to obtain context when needed.",
        )

    llm = get_llm(model_name=model_name)
    llm_with_tools = llm.bind_tools([retrieve_knowledge_base])

    question = rewritten_query if rewritten_query != user_query else user_query
    messages: List[BaseMessage] = [
        SystemMessage(content=system),
        *history,
        HumanMessage(content=question),
    ]

    hard_tool_calls = []
    hard_tool_messages = []
    for index, query in enumerate(retrieval_queries, start=1):
        tool_call_id = f"hard_retrieval_{index}"
        hard_tool_calls.append({
            "name": "retrieve_knowledge_base",
            "args": {"query": query},
            "id": tool_call_id,
        })
        hard_tool_messages.append(
            ToolMessage(
                content=retrieve_knowledge_base.invoke({"query": query}),
                tool_call_id=tool_call_id,
                name="retrieve_knowledge_base",
            )
        )

    messages.append(AIMessage(content="", tool_calls=hard_tool_calls))
    messages.extend(hard_tool_messages)

    for _ in range(max_tool_call_rounds_default()):
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            return ai_msg.content or "", retrieved_contexts

        for call in tool_calls:
            if call.get("name") != "retrieve_knowledge_base":
                content = f"Unknown tool requested: {call.get('name')}"
            else:
                content = retrieve_knowledge_base.invoke(call.get("args", {}))

            messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id=call["id"],
                    name=call.get("name"),
                )
            )

    final_msg = llm.invoke(messages)
    return final_msg.content or "", retrieved_contexts


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

    # ── 3a. Workflow routing ──────────────────────────────────────────────
    # Check if we are mid-action and waiting for context-switch confirmation
    pending_switch = get_pending_context_switch(session_id)
    if pending_switch:
        extracted, _ = _run_llm_extract(
            {"extract": [{"name": "confirmed", "required": True, "type": "boolean"}]},
            user_query, history, {}, model_name,
        )
        confirmed = extracted.get("confirmed") is True
        print(f"[rag_simple] context-switch confirmation: confirmed={confirmed!r}")
        if confirmed:
            # User confirmed abandonment — clear action state and run pending intent
            session_store.clear_workflow_state(session_id)
            clear_pending_context_switch(session_id)
            pending_intent = pending_switch["pending_intent"]
            pending_query = pending_switch["pending_query"]
            if pending_intent == "RAG":
                user_query = pending_query
            else:
                _append_history(session_id, user_query, "")
                result = run_workflow_step(pending_intent.lower(), pending_query, history, session_id, model_name)
                _append_history(session_id, pending_query, result)
                return result if not return_contexts else {"answer": result, "contexts": [], "rewritten_query": pending_query}
        else:
            clear_pending_context_switch(session_id)
            active_state = session_store.get_workflow_state(session_id)
            if active_state:
                result = run_workflow_step(active_state["workflow"], user_query, history, session_id, model_name)
                _append_history(session_id, user_query, result)
                return result if not return_contexts else {"answer": result, "contexts": [], "rewritten_query": user_query}

    # Check if a workflow is already in progress
    active_state = session_store.get_workflow_state(session_id)
    active_workflow = active_state["workflow"] if active_state else None

    if not skip_guards:
        intent = route_intent(user_query, history, session_id, active_workflow, model_name)
    else:
        intent = "RAG"

    if intent == "CONTINUE":
        if not active_workflow:
            # No workflow actually active — treat as RAG
            intent = "RAG"
        else:
            print(f"[rag_simple] continuing workflow: {active_workflow!r}")
            result = run_workflow_step(active_workflow, user_query, history, session_id, model_name)
            _append_history(session_id, user_query, result)
            return result if not return_contexts else {"answer": result, "contexts": [], "rewritten_query": user_query}

    if intent == "CONTEXT_SWITCH":
        pending = get_pending_context_switch(session_id)
        action_label = active_workflow.replace("_", " ") if active_workflow else "current action"
        if pending and pending["pending_intent"] == "RAG":
            confirm_msg = (
                f"You're in the middle of '{action_label}'. "
                f"If you abandon it, you'll need to restart from scratch if you want to do it later. "
                f"Do you want to abandon it and answer your question instead? (yes/no)"
            )
        else:
            confirm_msg = (
                f"You're in the middle of '{action_label}'. "
                f"If you abandon it, you'll need to restart from scratch if you want to do it later. "
                f"Do you want to abandon it and start a different action? (yes/no)"
            )
        _append_history(session_id, user_query, confirm_msg)
        return confirm_msg if not return_contexts else {"answer": confirm_msg, "contexts": [], "rewritten_query": user_query}

    if intent in ("BOOK_DESK", "SUBMIT_VACATION"):
        workflow_name = intent.lower()
        result = run_workflow_step(workflow_name, user_query, history, session_id, model_name)
        _append_history(session_id, user_query, result)
        return result if not return_contexts else {"answer": result, "contexts": [], "rewritten_query": user_query}

    # intent == "RAG" — fall through to existing flow

    # ── 3b. Cache lookup (first turn only) ───────────────────────────────
    # Subsequent turns depend on conversation history so must always retrieve.
    if is_first_turn and not return_contexts and not skip_guards:
        cached = get_cache().get(user_query)
        if cached is not None:
            _append_history(session_id, user_query, cached)
            return cached

    # ── 4. Rewrite query (pronoun resolution) ─────────────────────────────
    rewritten = _rewrite_query(user_query, history, model_name)

    # ── 5. Expand into focused retrieval queries ──────────────────────────
    retrieval_queries = _expand_queries(rewritten, history, model_name)

    # ── 6. Generate answer with retrieval as the only available tool ──────
    answer, contexts = _generate_answer_with_retrieval_tool(
        user_query=user_query,
        rewritten_query=rewritten,
        retrieval_queries=retrieval_queries,
        history=history,
        vector_store=vector_store,
        user_role=user_role,
        candidate_k=resolved_candidate_k,
        final_k=resolved_final_k,
        model_name=model_name,
        prompt_template=prompt_template,
    )

    # ── 7. Persist turn to session history ───────────────────────────────
    _append_history(session_id, user_query, answer)

    # ── 8. Cache write (first turn only, not eval mode) ──────────────────
    if is_first_turn and not return_contexts and not skip_guards:
        get_cache().set(user_query, answer)

    if return_contexts:
        return {
            "answer": answer,
            "contexts": contexts,
            "rewritten_query": rewritten,
        }
    return answer

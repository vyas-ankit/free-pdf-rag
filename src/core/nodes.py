# src/core/nodes.py
"""
LangGraph node implementations for the RAG + actions pipeline.

Each node receives AgentState and returns a partial state update (a dict).
This module is the new home for logic that used to live inline in
rag_simple.query_rag_simple(), workflows.engine.run_workflow_step(), and
workflows.intent_router.route_intent() — moved here wholesale (not wrapped)
so LangGraph owns orchestration and persistence.
"""

import json
import re
from datetime import date as _date, datetime
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langsmith import traceable

from src.core.agent_state import AgentState
from src.core.cache import get_cache
from src.core.config import get_config, get_tool_calling_config
from src.core.guards import run_input_guards
from src.core.llm import get_default_model, get_llm
from src.core.prompts import (
    QUERY_EXPANSION_PROMPT,
    QUERY_REWRITER_PROMPT,
    SYSTEM_RAG_PROMPT,
    TOOL_RAG_SYSTEM_PROMPT,
)
from src.core.retrieval import candidate_k_default, final_k_default
from src.core.tools import make_retrieval_tool
from src.core.workflows.engine import (
    _invoke_tool,
    _load_workflow,
    _run_llm_extract,
    _substitute,
    _substitute_args,
)
from src.core.workflows.intent_router import classify_intent

ACTIVE_LLM_MODEL = get_default_model()


# ── Config helpers ────────────────────────────────────────────────────────

def max_tool_call_rounds_default() -> int:
    return int(get_tool_calling_config().get("max_tool_call_rounds", 4))


def query_expansion_enabled() -> bool:
    cfg = get_tool_calling_config().get("query_expansion", {})
    return bool(cfg.get("enabled", True))


def max_expanded_queries_default() -> int:
    cfg = get_tool_calling_config().get("query_expansion", {})
    return int(cfg.get("max_queries", 4))


def _workflow_timeout_minutes() -> int:
    return int(get_config().get("workflows", {}).get("timeout_minutes", 30))


def _dedupe_queries(queries: list[str], max_queries: int) -> list[str]:
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


def _history_text(history: list[BaseMessage], n: int = 6) -> str:
    return "\n".join(
        f"{'user' if isinstance(msg, HumanMessage) else 'assistant'}: {msg.content}"
        for msg in history[-n:]
    ) or "(no prior conversation)"


# ── 1. Input guard node ───────────────────────────────────────────────────

@traceable(name="lg.guard_node", run_type="chain", metadata={"component": "guard"})
def guard_node(state: AgentState) -> dict:
    """Run length + moderation guards. Sets intent=BLOCKED on failure."""
    if state.get("skip_guards"):
        return {}

    raw_user_log = state.get("raw_user_log", [])
    raw_user_log = [*raw_user_log, state["user_query"]]

    guard_result = run_input_guards(
        state["user_query"],
        raw_user_log=raw_user_log,
        user_id=state["user_id"],
        session_id=state["session_id"],
    )
    if not guard_result.passed:
        print(f"[lg.guard_node] Blocked: reason={guard_result.reason}")
        blocked_msg = guard_result.message or "I can't process that request."
        return {
            "raw_user_log": raw_user_log,
            "intent": "BLOCKED",
            "answer": blocked_msg,
            "blocked_reason": guard_result.reason,
        }
    return {"raw_user_log": raw_user_log}


# ── 2. Pending context-switch confirmation node ──────────────────────────

@traceable(name="lg.pending_switch_node", run_type="chain",
           metadata={"component": "pending_switch"})
def pending_switch_node(state: AgentState) -> dict:
    """
    Resolve a yes/no answer to a previously-asked context-switch confirmation.

    Mirrors the `pending_switch` block that used to sit at the top of
    query_rag_simple(). Sets `intent` so the router can dispatch correctly:
      - "RAG"              → re-run the originally-pending RAG query
      - "<workflow_name>"  → start the originally-pending workflow
      - "CONTINUE"         → resume the still-active workflow (user said no)
    """
    pending = state["pending_context_switch"]
    history = state["history"]

    extracted, _ = _run_llm_extract(
        {"extract": [{"name": "confirmed", "required": True, "type": "boolean"}]},
        state["user_query"], history, {}, state["model_name"],
    )
    confirmed = extracted.get("confirmed") is True
    print(f"[lg.pending_switch_node] confirmed={confirmed!r}")

    if confirmed:
        pending_intent = pending["pending_intent"]
        pending_query = pending["pending_query"]
        update = {
            "pending_context_switch": None,
            # Abandon the active workflow — clear all its state.
            "workflow_name": None,
            "current_step": 0,
            "collected_params": {},
            "awaiting_confirmation": False,
            "last_tool_results": {},
            "workflow_last_active": None,
            "active_workflow": None,
        }
        if pending_intent == "RAG":
            update["intent"] = "RAG"
            update["user_query"] = pending_query
        else:
            update["intent"] = pending_intent
            update["user_query"] = pending_query
            update["workflow_name"] = pending_intent.lower()
        return update

    # User said no — stay in the active workflow.
    return {
        "pending_context_switch": None,
        "intent": "CONTINUE",
    }


# ── 3. Intent router node ─────────────────────────────────────────────────

@traceable(name="lg.intent_router_node", run_type="chain",
           metadata={"component": "intent_router"})
def intent_router_node(state: AgentState) -> dict:
    """
    Classify intent and detect context switches.

    Mirrors workflows.intent_router.route_intent(): if a workflow is active
    and the detected intent differs, stash the new intent as a pending
    context-switch and ask the user to confirm before abandoning.
    """
    active_workflow = state.get("active_workflow")
    detected = classify_intent(
        state["user_query"], state["history"], active_workflow, state["model_name"]
    )

    if detected == "CONTINUE":
        return {"intent": "CONTINUE" if active_workflow else "RAG"}

    if active_workflow and detected != active_workflow:
        print(f"[lg.intent_router_node] context switch: {active_workflow} → {detected}")
        return {
            "intent": "CONTEXT_SWITCH",
            "pending_context_switch": {
                "pending_intent": detected,
                "pending_query": state["user_query"],
            },
        }

    update = {"intent": detected}
    if detected in ("BOOK_DESK", "SUBMIT_VACATION"):
        update["workflow_name"] = detected.lower()
    return update


def route_by_intent(state: AgentState) -> str:
    intent = state.get("intent", "RAG")
    if intent in ("BOOK_DESK", "SUBMIT_VACATION", "CONTINUE"):
        return "WORKFLOW"
    if intent == "CONTEXT_SWITCH":
        return "CONTEXT_SWITCH"
    return "RAG"


# ── 4. Context-switch confirmation prompt node ───────────────────────────

@traceable(name="lg.context_switch_node", run_type="chain",
           metadata={"component": "context_switch"})
def context_switch_node(state: AgentState) -> dict:
    """Builds the 'do you want to abandon X?' confirmation message."""
    active_workflow = state.get("active_workflow")
    pending = state["pending_context_switch"]
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
    return {"answer": confirm_msg}


# ── 5. Cache lookup / write nodes ─────────────────────────────────────────

@traceable(name="lg.cache_lookup_node", run_type="chain", metadata={"component": "cache"})
def cache_lookup_node(state: AgentState) -> dict:
    if not (state["is_first_turn"] and not state["return_contexts"] and not state["skip_guards"]):
        return {}
    cached = get_cache().get(state["user_query"])
    if cached is not None:
        return {"answer": cached, "cache_hit": True}
    return {}


def route_after_cache(state: AgentState) -> str:
    return "HIT" if state.get("cache_hit") else "MISS"


@traceable(name="lg.cache_write_node", run_type="chain", metadata={"component": "cache"})
def cache_write_node(state: AgentState) -> dict:
    if state["is_first_turn"] and not state["return_contexts"] and not state["skip_guards"]:
        get_cache().set(state["user_query"], state["answer"])
    return {}


# ── 6. Query rewriter node ────────────────────────────────────────────────

@traceable(name="lg.query_rewriter_node", run_type="llm", metadata={"component": "rewriter"})
def query_rewriter_node(state: AgentState) -> dict:
    """Resolve pronouns / references using conversation history."""
    user_query = state["user_query"]
    history = state["history"]
    if not history:
        return {"rewritten_query": user_query}

    llm = get_llm(model_name=state["model_name"])
    prompt = ChatPromptTemplate.from_messages([
        ("system", QUERY_REWRITER_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{user_query} + \nRemember that you must output a rewritten query only."),
    ])
    chain = prompt | llm | StrOutputParser()
    rewritten = (chain.invoke({"history": history, "user_query": user_query}) or "").strip()

    if not rewritten or len(rewritten) < 3:
        return {"rewritten_query": user_query}

    print(f"[lg.query_rewriter_node] Rewritten query: {rewritten!r}")
    return {"rewritten_query": rewritten}


# ── 7. Query expander node ────────────────────────────────────────────────

@traceable(name="lg.query_expander_node", run_type="llm", metadata={"component": "query_expander"})
def query_expander_node(state: AgentState) -> dict:
    """Generate focused retrieval queries before the hard retrieval pass."""
    rewritten_query = state["rewritten_query"]
    history = state["history"]
    model_name = state["model_name"]

    max_queries = max(1, max_expanded_queries_default())
    if not query_expansion_enabled() or max_queries == 1:
        return {"retrieval_queries": [rewritten_query]}

    llm = get_llm(model_name=model_name)
    history_text = _history_text(history)
    prompt = ChatPromptTemplate.from_messages([("system", QUERY_EXPANSION_PROMPT)])
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

    print(f"[lg.query_expander_node] Expanded retrieval queries: {expanded!r}")
    return {"retrieval_queries": expanded}


# ── 8. Retrieval + tool-calling answer node ──────────────────────────────

@traceable(name="lg.retrieval_answer_node", run_type="chain",
           metadata={"component": "tool_calling_answer"})
def retrieval_answer_node(state: AgentState, config) -> dict:
    """Force one retrieval pass per expanded query, then let the LLM decide
    whether to retrieve more before producing the final answer."""
    user_query = state["user_query"]
    rewritten_query = state["rewritten_query"]
    retrieval_queries = state["retrieval_queries"]
    history = state["history"]
    vector_store = config["configurable"]["vector_store"]
    user_role = state["user_role"]
    model_name = state["model_name"]
    prompt_template = state.get("prompt_template")
    candidate_k = candidate_k_default()
    final_k = final_k_default()

    retrieved_contexts: list[str] = []
    chunk_registry: list[dict] = []
    retrieve_knowledge_base = make_retrieval_tool(
        vector_store=vector_store,
        user_role=user_role,
        candidate_k=candidate_k,
        final_k=final_k,
        retrieved_contexts=retrieved_contexts,
        chunk_registry=chunk_registry,
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
    messages: list[BaseMessage] = [
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

    answer = ""
    for _ in range(max_tool_call_rounds_default()):
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            answer = ai_msg.content or ""
            break

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
    else:
        final_msg = llm.invoke(messages)
        answer = final_msg.content or ""

    chunk_lookup = {entry["id"]: entry["metadata"] for entry in chunk_registry}
    final_answer, citations = _build_citations(answer, chunk_lookup)

    return {"answer": final_answer, "retrieved_contexts": retrieved_contexts, "citations": citations}


def _build_citations(answer: str, chunk_lookup: dict[str, dict]) -> tuple[str, list[dict]]:
    """Renumber [doc_N] markers to [k] in order of first appearance, deduped
    by source PDF (so multiple chunks from the same document share one [k]),
    and build a Sources list mapping [k] to real document info.

    chunk_lookup: {"doc_1": {"source_pdf": "...", "date": "...", ...}, ...}
    """
    seen_order: list[str] = []
    key_to_number: dict[str, int] = {}
    key_to_metadata: dict[str, dict] = {}

    def replacer(match: re.Match) -> str:
        full_id = f"doc_{match.group(1)}"
        metadata = chunk_lookup.get(full_id, {})
        source = metadata.get("source_pdf")
        dedup_key = source or full_id
        if dedup_key not in key_to_number:
            seen_order.append(dedup_key)
            key_to_number[dedup_key] = len(seen_order)
            key_to_metadata[dedup_key] = metadata
        return f"[{key_to_number[dedup_key]}]"

    rewritten = re.sub(r"\[doc_(\d+)\]", replacer, answer)

    sources = []
    for dedup_key in seen_order:
        metadata = key_to_metadata[dedup_key]
        sources.append({
            "number": key_to_number[dedup_key],
            "source": metadata.get("source_pdf", "Unknown source"),
            "date": metadata.get("date", ""),
            "url": None,
        })

    return rewritten, sources


# ── 9. Workflow node ──────────────────────────────────────────────────────

@traceable(name="lg.workflow_node", run_type="chain", metadata={"component": "workflow_engine"})
def workflow_node(state: AgentState) -> dict:
    """
    Advance a YAML-defined workflow by one logical turn.

    Mirrors workflows.engine.run_workflow_step(), but uses interrupt() in
    place of "return question and let the caller persist + re-enter" — the
    checkpointer snapshots AgentState at the interrupt point and resumes
    here on the next turn with the user's reply already merged into
    user_query / history by the entry point.
    """
    user_query = state["user_query"]
    history = state["history"]
    session_id = state["session_id"]
    model_name = state["model_name"]

    workflow_name = state.get("workflow_name") or state.get("active_workflow")
    if not workflow_name:
        # Defensive: CONTINUE routed here with nothing active — treat as RAG.
        return {"intent": "RAG", "workflow_name": None}

    workflow_def = _load_workflow(workflow_name)
    steps = workflow_def["steps"]

    print(f"[lg.workflow_node] workflow={workflow_name!r} session={session_id!r} query={user_query!r}")

    # Initialise state if this is a fresh start for this workflow name.
    if state.get("active_workflow") != workflow_name:
        print(f"[lg.workflow_node] starting new workflow: {workflow_name!r}")
        current_step = 0
        collected = {}
        awaiting_confirmation = False
        last_tool_results = {}
    else:
        current_step = state["current_step"]
        collected = dict(state["collected_params"])
        awaiting_confirmation = state["awaiting_confirmation"]
        last_tool_results = dict(state["last_tool_results"])

    # Timeout check
    last_active_raw = state.get("workflow_last_active")
    if last_active_raw:
        last_active = datetime.fromisoformat(last_active_raw)
        elapsed_minutes = (datetime.now() - last_active).total_seconds() / 60
        if elapsed_minutes > _workflow_timeout_minutes():
            action_label = workflow_name.replace("_", " ")
            print(f"[lg.workflow_node] session {session_id!r} expired after {elapsed_minutes:.1f} min")
            timeout_msg = (
                f"Your '{action_label}' session expired after {_workflow_timeout_minutes()} "
                f"minutes of inactivity. Please start again if you'd like to continue."
            )
            return {
                "answer": timeout_msg,
                "active_workflow": None,
                "workflow_name": None,
                "current_step": 0,
                "collected_params": {},
                "awaiting_confirmation": False,
                "last_tool_results": {},
                "workflow_last_active": None,
            }

    now_iso = datetime.now().isoformat()

    # Handle a pending user_confirmation response.
    if awaiting_confirmation:
        confirmation_step = {
            "extract": [{"name": "confirmed", "required": True, "type": "boolean"}]
        }
        extracted, _ = _run_llm_extract(confirmation_step, user_query, history, {}, model_name)
        confirmed_raw = extracted.get("confirmed")
        print(f"[lg.workflow_node] confirmation extracted: confirmed={confirmed_raw!r}")
        if confirmed_raw is True:
            awaiting_confirmation = False
            current_step += 1
        else:
            return {
                "answer": "Got it, I've cancelled the action.",
                "active_workflow": None,
                "workflow_name": None,
                "current_step": 0,
                "collected_params": {},
                "awaiting_confirmation": False,
                "last_tool_results": {},
                "workflow_last_active": None,
            }

    # Walk steps, running non-interactive ones immediately.
    while current_step < len(steps):
        step = steps[current_step]
        step_type = step["type"]
        step_id = step.get("id", str(current_step))

        print(f"[lg.workflow_node] step {current_step} id={step_id!r} type={step_type!r}")

        if step_type == "llm_extract":
            updated_collected, question = _run_llm_extract(
                step, user_query, history, collected, model_name
            )
            collected = updated_collected
            print(f"[lg.workflow_node] collected so far: {collected}")
            if question:
                print(f"[lg.workflow_node] missing param — asking user: {question!r}")
                return {
                    "answer": question,
                    "active_workflow": workflow_name,
                    "workflow_name": workflow_name,
                    "current_step": current_step,
                    "collected_params": collected,
                    "awaiting_confirmation": False,
                    "last_tool_results": last_tool_results,
                    "workflow_last_active": now_iso,
                }
            current_step += 1

        elif step_type == "tool_call":
            tool_name = step["tool"]
            raw_args = step.get("args", {})
            resolved_args = _substitute_args(raw_args, collected, last_tool_results)
            print(f"[lg.workflow_node] calling tool={tool_name!r} args={resolved_args}")
            result = _invoke_tool(tool_name, resolved_args)
            print(f"[lg.workflow_node] tool result: {result!r}")
            last_tool_results[step_id] = result
            current_step += 1

        elif step_type == "user_confirmation":
            message = _substitute(step["message"], collected, last_tool_results)
            return {
                "answer": message,
                "active_workflow": workflow_name,
                "workflow_name": workflow_name,
                "current_step": current_step,
                "collected_params": collected,
                "awaiting_confirmation": True,
                "last_tool_results": last_tool_results,
                "workflow_last_active": now_iso,
            }

        elif step_type == "respond":
            message = _substitute(step["message"], collected, last_tool_results)
            return {
                "answer": message,
                "active_workflow": None,
                "workflow_name": None,
                "current_step": 0,
                "collected_params": {},
                "awaiting_confirmation": False,
                "last_tool_results": {},
                "workflow_last_active": None,
            }

        else:
            current_step += 1

    # Fell off the end without a respond step.
    return {
        "answer": "Action completed.",
        "active_workflow": None,
        "workflow_name": None,
        "current_step": 0,
        "collected_params": {},
        "awaiting_confirmation": False,
        "last_tool_results": {},
        "workflow_last_active": None,
    }

# src/core/nodes.py
"""
LangGraph node functions for the planning-agent architecture.

Flow:
    planner → plan_validator → executor ⟲ step_result_accumulator → synthesizer

Each node receives AgentState and returns a partial state update.
"""

import json
import re
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable, get_current_run_tree

from src.core.agent_state import AgentState
from src.core.llm import get_default_model, get_llm
from src.core.prompts import PLANNER_PROMPT, SYNTHESIZER_PROMPT


ACTIVE_LLM_MODEL = get_default_model()

# Step types the plan_validator will allow
_ALLOWED_STEP_TYPES = {"retrieve", "fetch_location", "book_desk_tool", "ask_user"}
_MAX_STEPS = 8


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _format_history(messages: list) -> str:
    """Format message list into readable conversation history for prompts."""
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"[user]: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"[assistant]: {m.content}")
    return "\n".join(lines) if lines else "(no prior conversation)"


_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}

def _parse_floor(value) -> int:
    """Extract a floor integer from natural language: '3rd floor', 'floor 5', 'third', '3'."""
    v = str(value).lower().strip()
    for word, num in _ORDINALS.items():
        if word in v:
            return num
    match = re.search(r"\d+", v)
    if match:
        return int(match.group())
    raise ValueError(f"Cannot parse floor number from: {value!r}")


def _resolve_arg(value: str, step_results: dict) -> str:
    """Replace $step_N references in arg values with actual step results."""
    if not isinstance(value, str):
        return value
    match = re.fullmatch(r"\$step_(\d+)", value.strip())
    if match:
        step_id = int(match.group(1))
        return step_results.get(step_id, f"[result of step {step_id} not found]")
    return value


def _pending_steps_text(plan: Optional[dict], current_step_index: int) -> str:
    """Summarise remaining steps for the planner when resuming a paused plan."""
    if not plan or not plan.get("steps"):
        return "(none)"
    remaining = plan["steps"][current_step_index:]
    if not remaining:
        return "(none)"
    return "\n".join(
        f"  Step {s['id']}: {s['type']} — {s.get('description', '')}"
        for s in remaining
    )


# ────────────────────────────────────────────────────────────────────────────
# 1. Planner
# ────────────────────────────────────────────────────────────────────────────

@traceable(name="planner", run_type="chain", metadata={"component": "planning"})
def planner_node(state: AgentState) -> dict:
    """Produce a structured JSON plan for the current user query.

    If a prior plan was paused waiting for user input, the user's reply is
    incorporated and the plan resumes from where it left off — no re-planning
    from scratch.
    """
    llm = get_llm(model_name=state.get("model_name", ACTIVE_LLM_MODEL),
                  llm_api_key=state.get("llm_api_key"))

    # If we're resuming a paused plan, reuse it rather than re-plan
    existing_plan = state.get("plan")
    current_step_index = state.get("current_step_index", 0)
    awaiting = state.get("awaiting_user_input")

    if existing_plan and awaiting:
        # User just answered the ask_user question. Re-plan from scratch
        # with the user's answer in context — this lets the planner add
        # steps that depend on what the user said (e.g. book_desk_tool with
        # the now-known date and floor). Pure "advance index" without
        # re-planning meant those dependent steps were never executed.
        # Fall through to the fresh-plan path below (awaiting is now cleared).
        pass

    # Fresh query or prior plan is complete — generate a new plan
    all_messages = state.get("messages", [])
    history_text = _format_history(all_messages[:-1])
    pending_text = _pending_steps_text(existing_plan, current_step_index)
    user_id = state.get("user_id", "guest_user")
    user_role = state.get("user_role", "Public")

    print(f"[planner] total messages in state: {len(all_messages)}")
    print(f"[planner] history turns passed to prompt: {len(all_messages) - 1}")
    if history_text != "(no prior conversation)":
        print(f"[planner] history preview: {history_text[:200]}")

    prompt_text = PLANNER_PROMPT.format(
        user_id=user_id,
        user_role=user_role,
        conversation_history=history_text,
        user_query=state["user_query"],
        pending_steps=pending_text,
    )

    # Tag the full prompt as a child span so LangSmith shows it
    chain = (
        ChatPromptTemplate.from_messages([("human", "{prompt}")])
        | llm.with_config({"run_name": "planner_llm"})
        | StrOutputParser()
    )
    raw = chain.invoke({"prompt": prompt_text})

    # Strip markdown fences if the model wrapped the JSON
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    try:
        plan = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: treat the whole query as a single retrieve step
        plan = {
            "reasoning": "Could not parse plan — defaulting to document retrieval.",
            "steps": [{"id": 1, "type": "retrieve", "description": "Search knowledge base",
                        "args": {"query": state["user_query"]}}],
        }

    step_types = [s["type"] for s in plan.get("steps", [])]
    print(f"[planner] reasoning: {plan.get('reasoning', '')}")
    for s in plan.get("steps", []):
        print(f"[planner]   step {s['id']}: {s['type']} — {s.get('description', '')}")

    # Surface plan summary as langsmith metadata
    rt = get_current_run_tree()
    if rt:
        rt.metadata.update({
            "step_count": len(plan.get("steps", [])),
            "step_types": step_types,
            "reasoning": plan.get("reasoning", ""),
        })

    return {
        "plan": plan,
        "current_step_index": 0,
        "step_results": {},
        "plan_complete": False,
        "awaiting_user_input": None,
        "final_answer": None,
    }


# ────────────────────────────────────────────────────────────────────────────
# 2. Plan Validator
# ────────────────────────────────────────────────────────────────────────────

@traceable(name="plan_validator", run_type="chain", metadata={"component": "planning"})
def plan_validator_node(state: AgentState) -> dict:
    """Validate the plan before any execution.

    Checks:
    - Step types are from the allowed list
    - No more than _MAX_STEPS steps
    - book_desk_tool always uses the authenticated user_id
    - At least one step exists
    """
    plan = state.get("plan", {})
    steps = plan.get("steps", [])
    user_id = state.get("user_id", "guest_user")
    errors = []

    if not steps:
        errors.append("Plan has no steps.")

    if len(steps) > _MAX_STEPS:
        errors.append(f"Plan has {len(steps)} steps — maximum allowed is {_MAX_STEPS}.")

    for step in steps:
        step_type = step.get("type")
        if step_type not in _ALLOWED_STEP_TYPES:
            errors.append(f"Step {step['id']}: unknown type '{step_type}'.")

        # Enforce identity binding on booking tool
        if step_type == "book_desk_tool":
            args = step.get("args", {})
            if args.get("user_id") not in (user_id, f"{{user_id}}", "{user_id}"):
                errors.append(
                    f"Step {step['id']}: book_desk_tool user_id "
                    f"'{args.get('user_id')}' does not match session user '{user_id}'."
                )
                # Correct it rather than blocking — belt and suspenders
                args["user_id"] = user_id

    if errors:
        error_text = " | ".join(errors)
        print(f"[plan_validator] Validation errors: {error_text}")
        # Return a single-step fallback plan that explains the issue
        return {
            "plan": {
                "reasoning": f"Plan validation failed: {error_text}",
                "steps": [{"id": 1, "type": "ask_user",
                            "description": "Clarify request",
                            "args": {"question": "I couldn't understand that request. Could you rephrase?"}}],
            },
            "current_step_index": 0,
            "step_results": {},
        }

    print(f"[plan_validator] Plan valid — {len(steps)} step(s)")
    return {}


# ────────────────────────────────────────────────────────────────────────────
# 3. Executor
# ────────────────────────────────────────────────────────────────────────────

@traceable(name="executor_step", run_type="tool", metadata={"component": "execution"})
def executor_node(state: AgentState) -> dict:
    """Execute the current step in the plan.

    If we were awaiting user input, the user's reply is in user_query.
    Consume it as the answer to the pending ask_user step, then advance.
    """
    plan = state["plan"]
    steps = plan["steps"]
    idx = state.get("current_step_index", 0)
    step_results = dict(state.get("step_results") or {})

    # If the prior step was an ask_user and we paused for input,
    # the user's new message is the answer — record it and advance.
    if state.get("awaiting_user_input"):
        user_reply = state["user_query"]
        ask_step = steps[idx]
        step_results[ask_step["id"]] = user_reply
        print(f"[executor] Consumed user reply for step {ask_step['id']} (ask_user): {user_reply!r}")
        return {
            "step_results": step_results,
            "current_step_index": idx + 1,
            "awaiting_user_input": None,
            "plan_complete": idx + 1 >= len(steps),
        }
    user_id = state.get("user_id", "guest_user")

    if idx >= len(steps):
        return {"plan_complete": True}

    step = steps[idx]
    step_type = step["type"]
    args = {k: _resolve_arg(v, step_results) for k, v in step.get("args", {}).items()}

    print(f"[executor] Running step {step['id']}: {step_type} args={args}")

    # ── retrieve ──────────────────────────────────────────────────────────
    if step_type == "retrieve":
        from src.core.retrieval import retrieve_hybrid_and_rerank, format_docs
        import src.core.rag_logic as _rl
        vector_store = _rl._vector_store
        user_role = state.get("user_role", "Public")
        candidate_k = state.get("candidate_k", 20)
        final_k = state.get("final_k", 7)
        try:
            docs = retrieve_hybrid_and_rerank(
                query=args.get("query", state["user_query"]),
                vector_store=vector_store,
                user_role=user_role,
                candidate_k=candidate_k,
                final_k=final_k,
            )
            result = format_docs(docs)
        except Exception as exc:
            result = f"[retrieve error: {exc}]"

    # ── fetch_location ────────────────────────────────────────────────────
    elif step_type == "fetch_location":
        from src.core.tools import fetch_location
        try:
            result = fetch_location.invoke({"user_id": args.get("user_id", user_id)})
        except Exception as exc:
            result = f"[fetch_location error: {exc}]"

    # ── book_desk_tool ────────────────────────────────────────────────────
    elif step_type == "book_desk_tool":
        from src.core.tools import book_desk_tool
        try:
            floor_val = _parse_floor(args.get("floor", 1))
            result = book_desk_tool.invoke({
                "user_id": user_id,
                "date": args.get("date", ""),
                "floor": floor_val,
            })
        except Exception as exc:
            # Tool failed — pause and ask the user to correct the input
            # rather than silently marking the step complete.
            error_question = (
                f"I couldn't complete the booking: {exc}. "
                f"Could you please clarify the date and floor? "
                f"(e.g. 'June 17, floor 3')"
            )
            print(f"[executor] book_desk_tool failed: {exc} — asking user to retry")
            return {
                "step_results": step_results,
                "awaiting_user_input": error_question,
                "messages": [AIMessage(content=error_question)],
                "plan_complete": False,
            }

    # ── ask_user ──────────────────────────────────────────────────────────
    elif step_type == "ask_user":
        question = args.get("question", "Could you provide more information?")
        print(f"[executor] Pausing for user input: {question}")
        return {
            "awaiting_user_input": question,
            "step_results": step_results,
            "messages": [AIMessage(content=question)],
            "plan_complete": False,
        }

    else:
        result = f"[unknown step type: {step_type}]"

    step_results[step["id"]] = result
    print(f"[executor] Step {step['id']} result: {str(result)[:120]}")

    rt = get_current_run_tree()
    if rt:
        rt.metadata.update({
            "step_id": step["id"],
            "step_type": step_type,
            "step_description": step.get("description", ""),
            "result_preview": str(result)[:200],
        })

    return {
        "step_results": step_results,
        "current_step_index": idx + 1,
        "plan_complete": idx + 1 >= len(steps),
    }


# ────────────────────────────────────────────────────────────────────────────
# 4. Step Result Accumulator
# ────────────────────────────────────────────────────────────────────────────

@traceable(name="step_accumulator", run_type="chain", metadata={"component": "execution"})
def step_result_accumulator_node(state: AgentState) -> dict:
    """Lightweight pass-through that logs step progress.

    Real accumulation happens inside executor_node — this node exists as a
    clean separation point for the graph's conditional edge logic.
    """
    completed = state.get("current_step_index", 0)
    total = len((state.get("plan") or {}).get("steps", []))
    print(f"[accumulator] {completed}/{total} steps complete, plan_complete={state.get('plan_complete')}")
    return {}


# ────────────────────────────────────────────────────────────────────────────
# 5. Synthesizer
# ────────────────────────────────────────────────────────────────────────────

@traceable(name="synthesizer", run_type="chain", metadata={"component": "synthesis"})
def synthesizer_node(state: AgentState) -> dict:
    """Produce the final user-facing answer from all step results."""
    llm = get_llm(model_name=state.get("model_name", ACTIVE_LLM_MODEL),
                  llm_api_key=state.get("llm_api_key"))

    step_results = state.get("step_results") or {}
    plan = state.get("plan") or {}

    # Build a readable summary of step results
    lines = []
    for step in plan.get("steps", []):
        sid = step["id"]
        result = step_results.get(sid, "(no result)")
        lines.append(f"Step {sid} [{step['type']}]: {result}")
    step_results_text = "\n\n".join(lines) if lines else "(no steps were executed)"

    history_text = _format_history(state.get("messages", [])[:-1])

    prompt_text = SYNTHESIZER_PROMPT.format(
        user_query=state["user_query"],
        conversation_history=history_text,
        step_results_text=step_results_text,
    )

    chain = (
        ChatPromptTemplate.from_messages([("human", "{prompt}")])
        | llm.with_config({"run_name": "synthesizer_llm"})
        | StrOutputParser()
    )
    answer = chain.invoke({"prompt": prompt_text})

    print(f"[synthesizer] Answer: {answer[:120]}")

    rt = get_current_run_tree()
    if rt:
        rt.metadata.update({
            "steps_used": len(plan.get("steps", [])),
            "answer_preview": answer[:200],
        })

    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
    }


# ────────────────────────────────────────────────────────────────────────────
# Routing functions (used as conditional edges in graph.py)
# ────────────────────────────────────────────────────────────────────────────

def route_from_start(state: AgentState) -> str:
    """Decide whether to re-plan or resume an existing in-progress plan.

    Resume if ALL of:
      - A plan already exists in state (carried by the checkpointer)
      - The plan is not complete
      - There are still unfinished steps

    This includes the case where we were awaiting user input — the user's
    new message IS the answer, so we resume and let the executor consume it.

    Only re-plan if there is no plan yet, or the plan is fully complete
    (meaning this is a genuinely new request).
    """
    plan = state.get("plan")
    plan_complete = state.get("plan_complete", False)
    current_idx = state.get("current_step_index", 0)
    steps = (plan or {}).get("steps", [])

    has_unfinished = bool(plan) and not plan_complete and current_idx < len(steps)

    if has_unfinished:
        awaiting = state.get("awaiting_user_input")
        print(f"[router] Resuming existing plan — step {current_idx + 1}/{len(steps)}, awaiting={awaiting}")
        return "executor"

    print(f"[router] Re-planning — no active plan or plan complete")
    return "planner"


def route_after_planner(state: AgentState) -> str:
    """Always validate then execute — planner always produces a fresh plan."""
    return "plan_validator"


def route_after_accumulator(state: AgentState) -> str:
    """Loop executor until plan is complete or user input is needed."""
    if state.get("awaiting_user_input"):
        return "end"
    if state.get("plan_complete"):
        return "synthesizer"
    return "executor"

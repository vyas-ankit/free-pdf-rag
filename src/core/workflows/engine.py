# src/core/workflows/engine.py
"""
YAML workflow engine.

Loads a workflow definition from workflows/<name>.yaml and executes it
step-by-step. State is persisted in _session_workflow_state so multi-turn
actions (e.g. asking user for a missing param) resume correctly.

Step types:
  llm_extract       — extract params from user message; ask user for missing required ones
  tool_call         — call a named tool with collected params (no LLM)
  user_confirmation — send a formatted message; wait for yes/no
  respond           — format final message and return it (terminal step)
"""

import json
import re
from datetime import date as _date, datetime
from pathlib import Path
from typing import Any, Optional

import yaml
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from src.core.config import get_config
from src.core.llm import get_llm
from src.core.prompts import PARAM_EXTRACTION_PROMPT
from src.core import session_store


def _workflow_timeout_minutes() -> int:
    return int(get_config().get("workflows", {}).get("timeout_minutes", 30))

# Registered tool callables — populated at startup by register_tool()
_tool_registry: dict[str, Any] = {}

# Per-session workflow state
_WORKFLOWS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "workflows"


# ── Registry ──────────────────────────────────────────────────────────────

def register_tool(name: str, fn) -> None:
    """Register a callable under a name so YAML tool_call steps can invoke it."""
    _tool_registry[name] = fn


def _invoke_tool(name: str, args: dict) -> str:
    if name not in _tool_registry:
        return f"[engine] Unknown tool: {name}"
    fn = _tool_registry[name]
    # LangChain @tool callables accept a dict via .invoke()
    if hasattr(fn, "invoke"):
        return fn.invoke(args)
    return fn(**args)


# ── YAML loader ───────────────────────────────────────────────────────────

def _load_workflow(workflow_name: str) -> dict:
    path = _WORKFLOWS_DIR / f"{workflow_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Workflow definition not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Template substitution ─────────────────────────────────────────────────

def _substitute(template: str, collected: dict, tool_results: dict) -> str:
    """Replace {{param}} and {{step_id.result}} placeholders."""
    def replacer(match):
        key = match.group(1).strip()
        if key in collected and collected[key] is not None:
            return str(collected[key])
        if key in tool_results:
            return str(tool_results[key])
        return match.group(0)  # leave unresolved placeholders as-is

    return re.sub(r"\{\{(.+?)\}\}", replacer, template)


def _substitute_args(args: dict, collected: dict, tool_results: dict) -> dict:
    return {
        k: _substitute(str(v), collected, tool_results)
        for k, v in args.items()
    }


# ── Step executors ────────────────────────────────────────────────────────

@traceable(name="workflow_engine.llm_extract", run_type="llm")
def _run_llm_extract(
    step: dict,
    user_query: str,
    history: list[BaseMessage],
    collected: dict,
    model_name: str,
) -> tuple[dict, Optional[str]]:
    """
    Extract params from user_query. Returns (updated_collected, question_for_user).
    question_for_user is non-None when a required param is still missing — the
    engine should return this to the user and pause.
    """
    params_spec = step.get("extract", [])

    history_text = "\n".join(
        f"{'user' if isinstance(msg, HumanMessage) else 'assistant'}: {msg.content}"
        for msg in history[-6:]
    ) or "(no prior conversation)"

    params_desc = json.dumps([
        {
            "name": p["name"],
            "required": p.get("required", True),
            "type": p.get("type", "string"),
            "format": p.get("format"),
            "default": p.get("default"),
        }
        for p in params_spec
    ])

    already_collected = {k: v for k, v in collected.items() if v is not None}

    llm = get_llm(model_name=model_name)
    prompt = ChatPromptTemplate.from_messages([("system", PARAM_EXTRACTION_PROMPT)])
    chain = prompt | llm | StrOutputParser()
    raw = (chain.invoke({
        "params": params_desc,
        "already_collected": json.dumps(already_collected),
        "history": history_text,
        "user_query": user_query,
        "today": _date.today().isoformat(),
    }) or "").strip()

    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        extracted = json.loads(cleaned)
    except json.JSONDecodeError:
        extracted = {}

    # Merge extracted into collected (don't overwrite already-collected non-null values)
    def _coerce_bool(v):
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() not in ("false", "no", "0", "none", "null", "")

    type_coercers = {"integer": int, "float": float, "boolean": _coerce_bool, "string": str}
    for p in params_spec:
        name = p["name"]
        if collected.get(name) is None and extracted.get(name) is not None:
            value = extracted[name]
            coerce = type_coercers.get(p.get("type", "string"))
            if coerce:
                try:
                    value = coerce(value)
                except (ValueError, TypeError):
                    print(f"[workflow_engine] coercion failed for {name!r}: {value!r} → keeping as-is")
            collected[name] = value
        if collected.get(name) is None and not p.get("required", True):
            collected[name] = p.get("default")

    # Find first missing required param
    for p in params_spec:
        if p.get("required", True) and not collected.get(p["name"]):
            question = p.get("ask", f"Could you please provide the {p['name']}?")
            return collected, question

    return collected, None  # all required params present


# ── Public API ────────────────────────────────────────────────────────────


def run_workflow_step(
    workflow_name: str,
    user_query: str,
    history: list[BaseMessage],
    session_id: str,
    model_name: str,
) -> str:
    """
    Advance the workflow by one logical turn. Returns a string to send back to the user.

    Handles:
    - Starting a new workflow (no existing state)
    - Resuming a paused workflow (awaiting param or confirmation)
    - Executing tool_call steps (non-interactive, runs immediately)
    - Returning the final respond message
    """
    workflow_def = _load_workflow(workflow_name)
    steps = workflow_def["steps"]

    print(f"[workflow_engine] workflow={workflow_name!r} session={session_id!r} query={user_query!r}")

    # Initialise state if first turn for this workflow
    existing_state = session_store.get_workflow_state(session_id)
    if not existing_state or existing_state.get("workflow") != workflow_name:
        print(f"[workflow_engine] starting new workflow: {workflow_name!r}")
        state = {
            "workflow": workflow_name,
            "current_step": 0,
            "collected": {},
            "awaiting_confirmation": False,
            "last_tool_results": {},
            "last_active": datetime.now(),
        }
        session_store.save_workflow_state(session_id, state)
    else:
        state = existing_state

    # Timeout check
    last_active = state.get("last_active")
    if last_active:
        elapsed_minutes = (datetime.now() - last_active).total_seconds() / 60
        if elapsed_minutes > _workflow_timeout_minutes():
            action_label = workflow_name.replace("_", " ")
            print(f"[workflow_engine] session {session_id!r} expired after {elapsed_minutes:.1f} min")
            session_store.clear_workflow_state(session_id)
            return f"Your '{action_label}' session expired after {_workflow_timeout_minutes()} minutes of inactivity. Please start again if you'd like to continue."

    state["last_active"] = datetime.now()
    collected = state["collected"]
    tool_results = state["last_tool_results"]

    # Handle pending user_confirmation response via llm_extract
    if state["awaiting_confirmation"]:
        confirmation_step = {
            "extract": [{"name": "confirmed", "required": True, "type": "boolean"}]
        }
        extracted, _ = _run_llm_extract(confirmation_step, user_query, history, {}, model_name)
        confirmed_raw = extracted.get("confirmed")
        print(f"[workflow_engine] confirmation extracted: confirmed={confirmed_raw!r}")
        if confirmed_raw is True:
            state["awaiting_confirmation"] = False
            state["current_step"] += 1
            session_store.save_workflow_state(session_id, state)
        else:
            session_store.clear_workflow_state(session_id)
            return "Got it, I've cancelled the action."

    # Walk steps from current_step, running non-interactive ones immediately
    while state["current_step"] < len(steps):
        step = steps[state["current_step"]]
        step_type = step["type"]
        step_id = step.get("id", str(state["current_step"]))

        print(f"[workflow_engine] step {state['current_step']} id={step_id!r} type={step_type!r}")

        if step_type == "llm_extract":
            updated_collected, question = _run_llm_extract(
                step, user_query, history, collected, model_name
            )
            state["collected"] = updated_collected
            collected = updated_collected
            print(f"[workflow_engine] collected so far: {collected}")
            session_store.save_workflow_state(session_id, state)
            if question:
                print(f"[workflow_engine] missing param — asking user: {question!r}")
                return question
            state["current_step"] += 1
            session_store.save_workflow_state(session_id, state)

        elif step_type == "tool_call":
            tool_name = step["tool"]
            raw_args = step.get("args", {})
            resolved_args = _substitute_args(raw_args, collected, tool_results)
            print(f"[workflow_engine] calling tool={tool_name!r} args={resolved_args}")
            result = _invoke_tool(tool_name, resolved_args)
            print(f"[workflow_engine] tool result: {result!r}")
            tool_results[step_id] = result
            state["last_tool_results"] = tool_results
            state["current_step"] += 1
            session_store.save_workflow_state(session_id, state)

        elif step_type == "user_confirmation":
            message = _substitute(step["message"], collected, tool_results)
            state["awaiting_confirmation"] = True
            session_store.save_workflow_state(session_id, state)
            return message

        elif step_type == "respond":
            message = _substitute(step["message"], collected, tool_results)
            session_store.clear_workflow_state(session_id)
            return message

        else:
            state["current_step"] += 1
            session_store.save_workflow_state(session_id, state)

    # Fell off the end without a respond step
    session_store.clear_workflow_state(session_id)
    return "Action completed."

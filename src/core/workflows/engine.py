# src/core/workflows/engine.py
"""
YAML workflow engine — step executors and helpers.

Loads a workflow definition from workflows/<name>.yaml and provides the
building blocks used to walk it step by step:
  llm_extract       — extract params from user message; ask user for missing required ones
  tool_call         — call a named tool with collected params (no LLM)
  user_confirmation — send a formatted message; wait for yes/no
  respond           — format final message and return it (terminal step)

The actual step-walking loop and per-session progress live in
src.core.nodes.workflow_node — orchestrated and persisted by the LangGraph
checkpointer (see graph.py) rather than a hand-rolled state table.
"""

import json
import re
from datetime import date as _date
from pathlib import Path
from typing import Any, Optional

import yaml
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from src.core.llm import get_llm
from src.core.prompts import PARAM_EXTRACTION_PROMPT


# Registered tool callables — populated at startup by register_tool()
_tool_registry: dict[str, Any] = {}

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


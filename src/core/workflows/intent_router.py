# src/core/workflows/intent_router.py
"""
Intent classification — runs on every turn before anything else.

classify_intent() returns one of:
  - "RAG"              → pass to existing query_rag_simple flow
  - "BOOK_DESK"        → run book_desk workflow
  - "SUBMIT_VACATION"  → run submit_vacation workflow
  - "CONTINUE"         → keep going with the active workflow

Context-switch detection and pending-switch state now live in
src.core.nodes / agent_state.AgentState, persisted by the LangGraph
checkpointer (see graph.py) rather than an in-memory dict.
"""

from typing import Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from src.core.llm import get_llm
from src.core.prompts import INTENT_ROUTER_PROMPT

# Maps workflow name → intent label (keep in sync with YAML filenames)
WORKFLOW_INTENTS = {
    "BOOK_DESK",
    "SUBMIT_VACATION",
}


@traceable(name="intent_router.classify", run_type="llm",
           metadata={"component": "intent_router"})
def classify_intent(
    user_query: str,
    history: list[BaseMessage],
    active_workflow: Optional[str],
    model_name: str,
) -> str:
    """Call LLM to classify user intent. Returns one of the intent labels."""
    history_text = "\n".join(
        f"{'user' if isinstance(msg, HumanMessage) else 'assistant'}: {msg.content}"
        for msg in history[-6:]
    ) or "(no prior conversation)"

    llm = get_llm(model_name=model_name)
    prompt = ChatPromptTemplate.from_messages([
        ("system", INTENT_ROUTER_PROMPT),
    ])
    chain = prompt | llm | StrOutputParser()
    raw = (chain.invoke({
        "history": history_text,
        "user_query": user_query,
        "active_workflow": active_workflow or "none",
        "available_workflows": ", ".join(WORKFLOW_INTENTS),
    }) or "").strip().upper()

    # Normalise — accept known labels, fall back to RAG
    valid = WORKFLOW_INTENTS | {"RAG", "CONTINUE"}
    intent = raw if raw in valid else "RAG"
    print(f"[intent_router] query={user_query!r} active_workflow={active_workflow!r} → intent={intent}")
    return intent

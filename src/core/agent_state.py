# src/core/agent_state.py
"""
The schema for what the LangGraph workflow passes between nodes.

Every node receives this dict and returns a partial dict that gets merged.
The `messages` field uses LangGraph's add_messages reducer so appending to
it accumulates across turns.
"""

from typing import Annotated, Dict, List, Optional, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # ── Core query fields ─────────────────────────────────────────────────
    user_query: str
    messages: Annotated[list, add_messages]  # Native LangGraph list of messages
    user_id: str
    user_role: str
    llm_api_key: str
    # vector_store is intentionally excluded — PineconeVectorStore is not
    # serializable by MemorySaver. It is accessed via rag_logic._vector_store.
    candidate_k: int
    final_k: int
    prompt_template: str
    model_name: str

    # ── Planning fields ───────────────────────────────────────────────────
    # Structured plan produced by the planner node.
    # Shape: {"reasoning": str, "steps": [{"id": int, "type": str, ...}]}
    plan: Optional[dict]

    # Index into plan["steps"] that the executor is currently running.
    current_step_index: int

    # Accumulated results keyed by step id.
    # Shape: {1: "result text", 2: "SUCCESS: ...", ...}
    step_results: Dict[int, str]

    # Set to True by the executor when all steps are complete.
    plan_complete: bool

    # When a step needs user input, the executor puts the question here and
    # exits the graph. The next user turn resumes the plan.
    awaiting_user_input: Optional[str]

    # Final answer assembled by the synthesizer node.
    final_answer: Optional[str]

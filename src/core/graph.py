# src/core/graph.py
"""
Assembles the planning-agent LangGraph workflow and compiles it.

Flow:
    START → planner ─┬→ plan_validator → executor ⟲ accumulator ─┬→ synthesizer → END
                     │                                             └→ END (ask_user pause)
                     └→ executor (when resuming a paused ask_user step)

The executor ↔ accumulator loop runs once per plan step until all steps
complete or an ask_user step pauses execution for user input.
"""

from langgraph.graph import END, START, StateGraph

from src.core.agent_state import AgentState
from src.core.nodes import (
    executor_node,
    plan_validator_node,
    planner_node,
    route_after_accumulator,
    route_after_planner,
    step_result_accumulator_node,
    synthesizer_node,
)


def build_graph():
    """Construct and compile the planning-agent workflow."""
    workflow = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────
    workflow.add_node("planner",       planner_node)
    workflow.add_node("plan_validator", plan_validator_node)
    workflow.add_node("executor",      executor_node)
    workflow.add_node("accumulator",   step_result_accumulator_node)
    workflow.add_node("synthesizer",   synthesizer_node)

    # ── Edges ─────────────────────────────────────────────────────────────

    # Entry point
    workflow.add_edge(START, "planner")

    # After planner: either validate a new plan or jump straight to executor
    # if we're resuming an ask_user pause (planner already advanced the index)
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "plan_validator": "plan_validator",
            "executor": "executor",
        },
    )

    # Validator always leads to executor
    workflow.add_edge("plan_validator", "executor")

    # After each step execution, pass through accumulator
    workflow.add_edge("executor", "accumulator")

    # Accumulator decides: loop, pause for user, or synthesize
    workflow.add_conditional_edges(
        "accumulator",
        route_after_accumulator,
        {
            "executor":    "executor",
            "synthesizer": "synthesizer",
            "end":         END,          # ask_user pause — answer already in messages
        },
    )

    # Synthesizer always ends
    workflow.add_edge("synthesizer", END)

    return workflow.compile()


compiled_graph = build_graph()

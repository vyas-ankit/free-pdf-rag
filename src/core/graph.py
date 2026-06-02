# src/core/graph.py
"""
Assembles the planning-agent LangGraph workflow and compiles it.

Flow:
    START → router ─┬→ planner → plan_validator → executor ⟲ accumulator ─┬→ synthesizer → END
                    │                                                        └→ END (ask_user pause)
                    └→ executor  (resume existing in-progress plan)

The router skips replanning when the checkpointer has an existing unfinished
plan and no user input is pending.
"""

from langgraph.graph import END, START, StateGraph

from src.core.agent_state import AgentState
from src.core.nodes import (
    executor_node,
    plan_validator_node,
    planner_node,
    route_after_accumulator,
    route_after_planner,
    route_from_start,
    step_result_accumulator_node,
    synthesizer_node,
)


def build_graph(checkpointer=None):
    """Construct and compile the planning-agent workflow."""
    workflow = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────
    workflow.add_node("planner",        planner_node)
    workflow.add_node("plan_validator", plan_validator_node)
    workflow.add_node("executor",       executor_node)
    workflow.add_node("accumulator",    step_result_accumulator_node)
    workflow.add_node("synthesizer",    synthesizer_node)

    # ── Edges ─────────────────────────────────────────────────────────────

    # Router decides: resume existing plan or re-plan from scratch
    workflow.add_conditional_edges(
        START,
        route_from_start,
        {
            "planner":  "planner",
            "executor": "executor",
        },
    )

    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "plan_validator": "plan_validator",
            "executor":       "executor",
        },
    )

    workflow.add_edge("plan_validator", "executor")
    workflow.add_edge("executor",       "accumulator")

    workflow.add_conditional_edges(
        "accumulator",
        route_after_accumulator,
        {
            "executor":    "executor",
            "synthesizer": "synthesizer",
            "end":         END,
        },
    )

    workflow.add_edge("synthesizer", END)

    return workflow.compile(checkpointer=checkpointer)

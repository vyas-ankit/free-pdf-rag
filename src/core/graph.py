# src/core/graph.py
"""
Assembles the LangGraph StateGraph from node functions, tools, and the routing
edge — and compiles it into the executable graph that `query_rag` invokes.

This module owns the *shape* of the workflow:
    START → query_rewriter → intent_classifier ─┬→ knowledge_executor → END
                                                │
                                                └→ action_executor ⇄ tools → END
"""

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from src.core.agent_state import AgentState
from src.core.nodes import (
    action_executor_node,
    intent_classifier_node,
    knowledge_executor_node,
    query_rewriter_node,
    route_by_intent,
)
from src.core.tools import tool_node


def build_graph():
    """Construct and compile the agent workflow. Called once at import time."""
    workflow = StateGraph(AgentState)

    workflow.add_node("query_rewriter", query_rewriter_node)
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("action_executor", action_executor_node)
    workflow.add_node("knowledge_executor", knowledge_executor_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "query_rewriter")
    workflow.add_edge("query_rewriter", "intent_classifier")

    workflow.add_conditional_edges(
        "intent_classifier",
        route_by_intent,
        {
            "action_executor": "action_executor",
            "knowledge_executor": "knowledge_executor",
        },
    )

    # Re-act loop for action_executor
    workflow.add_conditional_edges(
        "action_executor",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )
    workflow.add_edge("tools", "action_executor")
    workflow.add_edge("knowledge_executor", END)

    return workflow.compile()


compiled_graph = build_graph()

# src/core/tools.py
"""
LangChain tools and tool factories used by the RAG and action flows.

These are decorated with @tool so LangChain auto-generates their JSON schema
from the type hints + docstring, which the LLM uses to decide arguments.
"""

from langchain_core.tools import tool


def make_retrieval_tool(
    vector_store,
    user_role: str,
    candidate_k: int,
    final_k: int,
    retrieved_contexts: list[str] = None,
):
    """Create the single retrieval tool used by rag_simple's tool-calling loop."""

    @tool
    def retrieve_knowledge_base(query: str) -> str:
        """Search the uploaded PDF knowledge base for information relevant to the query."""
        from src.core.retrieval import format_docs, retrieve_hybrid_and_rerank

        print(f"[retrieval_tool] query={query!r}")
        docs = retrieve_hybrid_and_rerank(
            query=query,
            vector_store=vector_store,
            user_role=user_role,
            candidate_k=candidate_k,
            final_k=final_k,
        )
        if retrieved_contexts is not None:
            retrieved_contexts.extend(doc.page_content for doc in docs)
        return format_docs(docs)

    return retrieve_knowledge_base


@tool
def fetch_location(user_id: str) -> str:
    """Useful when you need to fetch the employee's current assigned seat, floor, and office location from the corporate directory."""
    mock_locations = {
        "ankit_vyas": {"office": "London", "floor": 3, "desk": "3A-12"},
        "guest_user": {"office": "New York", "floor": 5, "desk": "5B-04"},
    }
    loc = mock_locations.get(user_id, {"office": "Public", "floor": 1, "desk": "Hotdesk-1"})
    return f"Current Seat Assignment: Desk {loc['desk']} on Floor {loc['floor']} in the {loc['office']} office."


@tool
def book_desk_tool(user_id: str, date: str, floor: int) -> str:
    """Useful when you need to finalize a desk booking. Required parameters are date (string) and floor (integer)."""
    return f"SUCCESS: Desk booked successfully for {user_id} on {date} on Floor {floor}."


# Single source of truth for "what tools exist" — used by:
#   - action_executor_node    (LLM.bind_tools(tools))
#   - the LangGraph workflow  (tool_node executes any tool call from the LLM)
tools = [fetch_location, book_desk_tool]


class _LazyToolNode:
    """Load LangGraph's ToolNode only if legacy graph code asks for it."""

    def __init__(self):
        self._node = None

    def _get_node(self):
        if self._node is None:
            from langgraph.prebuilt import ToolNode

            self._node = ToolNode(tools)
        return self._node

    def __getattr__(self, name):
        return getattr(self._get_node(), name)

    def __call__(self, *args, **kwargs):
        return self._get_node()(*args, **kwargs)


tool_node = _LazyToolNode()

# src/core/tools.py
"""
LangChain tools and tool factories used by the RAG and action flows.

These are decorated with @tool so LangChain auto-generates their JSON schema
from the type hints + docstring, which the LLM uses to decide arguments.
"""

from typing import Optional

from langchain_core.tools import tool


def make_retrieval_tool(
    vector_store,
    user_role: str,
    candidate_k: int,
    final_k: int,
    retrieved_contexts: list[str] = None,
    chunk_registry: list[dict] = None,
):
    """Create the single retrieval tool used by rag_simple's tool-calling loop.

    chunk_registry: if provided, each retrieved chunk is appended as
    {"id": "doc_N", "metadata": doc.metadata, "content": doc.page_content},
    with N assigned sequentially across all calls made through this tool
    instance — giving the LLM stable [doc_N] tags it can cite, which the
    caller later maps back to real documents (see nodes._build_citations).
    """

    @tool
    def retrieve_knowledge_base(query: str) -> str:
        """Search the uploaded PDF knowledge base for information relevant to the query."""
        from src.core.retrieval import format_docs_with_ids, format_docs, retrieve_hybrid_and_rerank

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

        if chunk_registry is not None:
            tagged = []
            for doc in docs:
                doc_id = f"doc_{len(chunk_registry) + 1}"
                chunk_registry.append({
                    "id": doc_id,
                    "metadata": doc.metadata,
                    "content": doc.page_content,
                })
                tagged.append((doc_id, doc))
            return format_docs_with_ids(tagged)

        return format_docs(docs)

    return retrieve_knowledge_base


@tool
def fetch_location(user_id: str) -> str:
    """Fetch the employee's current assigned seat, floor, and office location from the corporate directory.

    Args:
        user_id: The employee's unique identifier.

    Returns:
        A string describing the employee's current desk, floor, and office.
    """
    mock_locations: dict[str, dict[str, object]] = {
        "ankit_vyas": {"office": "London", "floor": 3, "desk": "3A-12"},
        "guest_user": {"office": "New York", "floor": 5, "desk": "5B-04"},
    }
    loc = mock_locations.get(user_id, {"office": "Public", "floor": 1, "desk": "Hotdesk-1"})
    return f"Current Seat Assignment: Desk {loc['desk']} on Floor {loc['floor']} in the {loc['office']} office."


@tool
def book_desk_tool(user_id: str, start_date: str, floor: int, end_date: Optional[str] = None) -> str:
    """Finalize a desk booking for a date or date range on a specific floor.

    Args:
        user_id: The employee's unique identifier.
        start_date: The booking start date, must be in YYYY-MM-DD format (e.g. "2026-06-09").
        floor: The floor number as an integer (e.g. 7, not '7th').
        end_date: The booking end date in YYYY-MM-DD format. If None, books a single day.

    Returns:
        A success confirmation string.
    """
    resolved_end = end_date or start_date
    return f"SUCCESS: Desk booked successfully for {user_id} on Floor {floor} from {start_date} to {resolved_end}."


@tool
def submit_vacation_tool(user_id: str, start_date: str, end_date: str, reason: str = "personal") -> str:
    """Submit a vacation or leave request for a date range.

    Args:
        user_id: The employee's unique identifier.
        start_date: Leave start date, must be in YYYY-MM-DD format (e.g. "2026-06-20").
        end_date: Leave end date, must be in YYYY-MM-DD format (e.g. "2026-06-27").
        reason: Optional reason for the leave (default: 'personal').

    Returns:
        A success confirmation string.
    """
    return f"SUCCESS: Vacation request submitted for {user_id} from {start_date} to {end_date} (reason: {reason}). Awaiting manager approval."


def _register_workflow_tools() -> None:
    from src.core.workflows.engine import register_tool
    register_tool("fetch_location", fetch_location)
    register_tool("book_desk_tool", book_desk_tool)
    register_tool("submit_vacation_tool", submit_vacation_tool)

_register_workflow_tools()

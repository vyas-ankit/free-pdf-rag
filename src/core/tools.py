# src/core/tools.py
"""
LangChain tools that the agent can call (the action / desk-booking demo).

These are decorated with @tool so LangChain auto-generates their JSON schema
from the type hints + docstring, which the LLM uses to decide arguments.
"""

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode


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
tool_node = ToolNode(tools)

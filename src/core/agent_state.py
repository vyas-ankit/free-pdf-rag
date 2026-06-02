# src/core/agent_state.py
"""
The schema for what the LangGraph workflow passes between nodes.

Every node receives this dict and returns a partial dict that gets merged.
The `messages` field uses LangGraph's add_messages reducer so appending to
it accumulates across turns.
"""

from typing import Annotated, TypedDict

from langchain_pinecone import PineconeVectorStore
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    user_query: str
    messages: Annotated[list, add_messages]  # Native LangGraph list of messages
    rewritten_query: str
    query_intent: str
    user_id: str
    user_role: str
    llm_api_key: str
    vector_store: PineconeVectorStore
    candidate_k: int
    final_k: int
    prompt_template: str
    model_name: str
    final_report: str

# src/core/session_store.py
"""
SQLite-backed conversation history.

Workflow/context-switch state now lives in AgentState and is persisted by
the LangGraph checkpointer (see graph.py) rather than a hand-rolled table.

DB path is configured via llm_config.json → session_store.db_path
"""

from pathlib import Path
from typing import List

from langchain_core.messages import BaseMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory

from src.core.config import get_config


def _db_path() -> str:
    path = get_config().get("session_store", {}).get("db_path", "data/sessions.db")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def _get_connection_string() -> str:
    return f"sqlite:///{_db_path()}"


# ── Conversation history ──────────────────────────────────────────────────

def get_history(session_id: str) -> List[BaseMessage]:
    history = SQLChatMessageHistory(
        session_id=session_id,
        connection_string=_get_connection_string(),
    )
    return history.messages


def append_history(session_id: str, human: str, ai: str) -> None:
    history = SQLChatMessageHistory(
        session_id=session_id,
        connection_string=_get_connection_string(),
    )
    history.add_user_message(human)
    history.add_ai_message(ai)


def clear_history(session_id: str) -> None:
    history = SQLChatMessageHistory(
        session_id=session_id,
        connection_string=_get_connection_string(),
    )
    history.clear()

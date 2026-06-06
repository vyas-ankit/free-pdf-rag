# src/core/session_store.py
"""
SQLite-backed session persistence.

Two concerns:
  1. Conversation history  — via LangChain's SQLChatMessageHistory
  2. Workflow state        — raw SQLite table (arbitrary JSON per session)

DB path is configured via llm_config.json → session_store.db_path
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory

from src.core.config import get_config


def _db_path() -> str:
    path = get_config().get("session_store", {}).get("db_path", "data/sessions.db")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def _get_connection_string() -> str:
    return f"sqlite:///{_db_path()}"


# ── Workflow state table setup ────────────────────────────────────────────

def _ensure_workflow_table() -> None:
    with sqlite3.connect(_db_path()) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS workflow_state (
                session_id    TEXT PRIMARY KEY,
                workflow      TEXT NOT NULL,
                current_step  INTEGER NOT NULL,
                collected     TEXT NOT NULL,
                awaiting_confirmation INTEGER NOT NULL DEFAULT 0,
                last_tool_results TEXT NOT NULL DEFAULT '{}',
                last_active   TEXT NOT NULL
            )
        """)
        conn.commit()

_ensure_workflow_table()


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


# ── Workflow state ────────────────────────────────────────────────────────

def get_workflow_state(session_id: str) -> Optional[dict]:
    with sqlite3.connect(_db_path()) as conn:
        row = conn.execute(
            "SELECT workflow, current_step, collected, awaiting_confirmation, last_tool_results, last_active "
            "FROM workflow_state WHERE session_id = ?",
            (session_id,)
        ).fetchone()
    if not row:
        return None
    return {
        "workflow": row[0],
        "current_step": row[1],
        "collected": json.loads(row[2]),
        "awaiting_confirmation": bool(row[3]),
        "last_tool_results": json.loads(row[4]),
        "last_active": datetime.fromisoformat(row[5]),
    }


def save_workflow_state(session_id: str, state: dict) -> None:
    last_active = state.get("last_active", datetime.now())
    if isinstance(last_active, datetime):
        last_active = last_active.isoformat()
    with sqlite3.connect(_db_path()) as conn:
        conn.execute("""
            INSERT INTO workflow_state
                (session_id, workflow, current_step, collected, awaiting_confirmation, last_tool_results, last_active)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                workflow=excluded.workflow,
                current_step=excluded.current_step,
                collected=excluded.collected,
                awaiting_confirmation=excluded.awaiting_confirmation,
                last_tool_results=excluded.last_tool_results,
                last_active=excluded.last_active
        """, (
            session_id,
            state["workflow"],
            state["current_step"],
            json.dumps(state["collected"]),
            int(state["awaiting_confirmation"]),
            json.dumps(state["last_tool_results"]),
            last_active,
        ))
        conn.commit()


def clear_workflow_state(session_id: str) -> None:
    with sqlite3.connect(_db_path()) as conn:
        conn.execute("DELETE FROM workflow_state WHERE session_id = ?", (session_id,))
        conn.commit()

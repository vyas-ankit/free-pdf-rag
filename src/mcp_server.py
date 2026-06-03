#!/usr/bin/env python3
# src/mcp_server.py
"""
MCP server for the free-pdf-rag knowledge base.

Implements the MCP stdio transport (JSON-RPC over stdin/stdout) manually
— no MCP SDK required, works on Python 3.9.

Exposes one tool:
    query_knowledge_base — searches the 13D Research financial reports
                           knowledge base via the RAG pipeline.

Usage (Claude Code registers this as an MCP server; do not run directly
unless testing in isolation):
    python src/mcp_server.py
"""

import json
import os
import sys
from pathlib import Path

# Add repo root to sys.path so `src.*` imports work when the file is run
# directly (python src/mcp_server.py) rather than as a module (-m src.mcp_server)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Redirect all stdout prints to stderr ──────────────────────────────────
# MCP uses stdout exclusively for JSON-RPC messages. Any non-JSON output
# (LangChain logs, RAG debug prints, cache logs) breaks the protocol.
# Redirecting print() to stderr keeps stdout clean for MCP traffic.
import builtins
_original_print = builtins.print

def _mcp_safe_print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    _original_print(*args, **kwargs)

builtins.print = _mcp_safe_print

# ── Initialize the RAG pipeline once at server startup ────────────────────
# Imports are deferred until here so load_dotenv() runs first.
from src.core.vector_store import get_embeddings, get_vector_store
from src.core.rag_simple import query_rag_simple

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    print(json.dumps({"error": "PINECONE_API_KEY not set"}), file=sys.stderr)
    sys.exit(1)

embeddings = get_embeddings()
vs = get_vector_store(embeddings, PINECONE_API_KEY)

# ── Tool definition ───────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "query_knowledge_base",
        "description": (
            "Search the 13D Research financial reports knowledge base. "
            "Use this for questions about gold, silver, antimony, copper, oil, "
            "China critical minerals, China stocks, geopolitics, macro trends, "
            "USD breakdown, US fiscal policy, commodity markets, precious metals, "
            "mining stocks, and global capital reallocation. "
            "The knowledge base contains research reports published between 2024-2026."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to answer from the knowledge base"
                }
            },
            "required": ["question"]
        }
    }
]

# ── Request handlers ──────────────────────────────────────────────────────

def handle_initialize(req_id, params):
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "free-pdf-rag", "version": "1.0.0"}
        }
    }


def handle_tools_list(req_id, params):
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {"tools": TOOLS}
    }


def handle_tools_call(req_id, params):
    name = params.get("name")
    arguments = params.get("arguments", {})

    if name != "query_knowledge_base":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Unknown tool: {name}"}
        }

    question = arguments.get("question", "").strip()
    if not question:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32602, "message": "Missing required argument: question"}
        }

    try:
        answer = query_rag_simple(
            user_query=question,
            vector_store=vs,
            session_id="mcp_claude_code",
            user_role="Public",
            skip_guards=False,
        )
    except Exception as exc:
        print(f"[mcp_server] ERROR: {exc}", file=sys.stderr)
        answer = f"Error querying knowledge base: {exc}"

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "content": [{"type": "text", "text": answer}],
            "isError": False
        }
    }


def handle_notifications_initialized(params):
    # No-op notification — client tells server it's ready
    return None


# ── Main stdio loop ───────────────────────────────────────────────────────

HANDLERS = {
    "initialize":                handle_initialize,
    "tools/list":                handle_tools_list,
    "tools/call":                handle_tools_call,
}

NOTIFICATION_HANDLERS = {
    "notifications/initialized": handle_notifications_initialized,
}


def respond(response: dict):
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


def main():
    print("[mcp_server] Started — listening on stdin", file=sys.stderr)

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            request = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            respond({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"}
            })
            continue

        method  = request.get("method", "")
        req_id  = request.get("id")       # None for notifications
        params  = request.get("params", {})

        print(f"[mcp_server] {method}", file=sys.stderr)

        # Notifications (no id, no response expected)
        if req_id is None:
            handler = NOTIFICATION_HANDLERS.get(method)
            if handler:
                handler(params)
            continue

        # Regular requests
        handler = HANDLERS.get(method)
        if handler:
            response = handler(req_id, params)
            respond(response)
        else:
            respond({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}
            })


if __name__ == "__main__":
    main()

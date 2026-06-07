# Branches

Renamed with numeric prefixes to reflect the order they were created/worked on (based on commit history).

| Old name | New name | What it does |
|---|---|---|
| `first_branch` | `01_first_branch` | Earliest branch — initial AWS deployment setup: Dockerfiles for frontend/backend, GitHub Actions CI, FastAPI backend split from frontend, health-check endpoint for the load balancer. |
| `local_develop_branch` | `02_local_develop_branch` | Restructured the project into a proper `src/` package with a config-driven RAG pipeline; added role-based metadata handling, semantic caching (Redis/Upstash), a centralized `llm.py`, and an early technical architecture report. |
| `planning_agent_architecture` | `03_planning_agent_architecture` | Branched from `02_local_develop_branch` to build a planner-executor multi-agent architecture (LangGraph-based planning agent for RAG + actions). |
| `04_rag_only` | `04_rag_only` | Continued from the planning-agent work; added a RAGAS evaluation framework (5 metrics), local query caching, and converted the simple RAG script into an MCP server exposed to Claude. |
| `05_rag_with_mcp` | `05_rag_with_mcp` | Same tip commit as `04_rag_only` at time of rename — effectively a duplicate/parallel branch from the same point (MCP server conversion). |
| `rag_with_mcp_and_agentic_tools` | `06_rag_with_mcp_and_agentic_tools` | Current/active branch. Added a single retrieval tool with query expansion and LLM-driven follow-up retrieval, SQLite-backed session persistence, two YAML workflows (book desk, submit vacation), and later removed all LangGraph code in favor of a pure LangChain implementation (`rag_simple.py`). |

## Notes
- All renames were pushed to `origin` and old branch names deleted from the remote.
- `01_first_branch` is now the default branch on GitHub (previously `first_branch`).
- `04_rag_only` and `05_rag_with_mcp` point to the same commit — consider deleting one if they're true duplicates going forward.

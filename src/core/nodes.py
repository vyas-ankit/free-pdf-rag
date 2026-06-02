# src/core/nodes.py
"""
LangGraph node functions + the AgentState schema + intent-routing edge fn.

Every node takes the AgentState dict and returns a partial state update.
The graph itself is assembled in rag_logic.py.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from src.core.agent_state import AgentState
from src.core.llm import get_default_model, get_llm
from src.core.prompts import (
    ACTION_EXECUTOR_SYSTEM_PROMPT,
    INTENT_CLASSIFIER_PROMPT,
    QUERY_REWRITER_PROMPT,
    SYSTEM_RAG_PROMPT,
)
from src.core.retrieval import (
    candidate_k_default,
    final_k_default,
    format_docs,
    retrieve_hybrid_and_rerank,
)
from src.core.tools import tools


ACTIVE_LLM_MODEL = get_default_model()


# --- Agent Nodes ---

def query_rewriter_node(state: AgentState):
    """Resolves pronouns and history into a standalone question."""
    chat_history = state["messages"][:-1]  # Exclude the latest human message

    if not chat_history:
        return {"rewritten_query": state["user_query"]}

    # Pinned to OpenAI for the rewriter only — its strict-output-format adherence
    # is significantly better than Groq's Llama on this task. Other nodes
    # continue to use the default provider from config.
    llm = get_llm(provider="openai")
    prompt = ChatPromptTemplate.from_messages([
        ("system", QUERY_REWRITER_PROMPT),
        *chat_history,
        ("human", "{user_query}"),
    ])
    chain = prompt | llm
    response = chain.invoke({"user_query": state["user_query"]})
    print(f"\n[Agentic RAG] Rewritten Query: '{response.content}'")
    return {"rewritten_query": response.content}


def intent_classifier_node(state: AgentState):
    """Classifies the rewritten query as 'action' or 'knowledge'.

    Uses plain text output and parses the response — the prompt instructs the
    model to reply with a single word, which is incompatible with Groq's
    tool-call-required behavior in with_structured_output().
    """
    llm = get_llm(
        model_name=state.get("model_name", ACTIVE_LLM_MODEL),
        llm_api_key=state.get("llm_api_key"),
    )

    history_text = ""
    chat_history = state["messages"][:-1]
    for msg in chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        history_text += f"[{role}]: {msg.content}\n"
    if not history_text:
        history_text = "[No previous conversation history]"

    formatted_human_input = (
        f"<conversation_history>\n{history_text}</conversation_history>\n\n"
        f"<current_query>\n{state['rewritten_query']}\n</current_query>"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", INTENT_CLASSIFIER_PROMPT),
        ("human", "{formatted_input}"),
    ])

    chain = prompt | llm | StrOutputParser()
    raw_response = chain.invoke({"formatted_input": formatted_human_input})

    # Robust parsing: scan response for keyword. Default to "knowledge" (safer).
    text = (raw_response or "").lower()
    if "action" in text and "knowledge" not in text:
        intent = "action"
    else:
        intent = "knowledge"

    print(f"[Agentic RAG] Classified Intent: '{intent.upper()}' (raw='{raw_response.strip()[:40]}')")
    return {"query_intent": intent}


def action_executor_node(state: AgentState):
    """Agent node equipped with tools to handle desk-booking conversational loops."""
    llm = get_llm(
        model_name=state.get("model_name", ACTIVE_LLM_MODEL),
        llm_api_key=state.get("llm_api_key"),
    )
    llm_with_tools = llm.bind_tools(tools)

    user_id = state.get("user_id", "guest_user")
    system_instruction = ACTION_EXECUTOR_SYSTEM_PROMPT.format(user_id=user_id)

    messages = [SystemMessage(content=system_instruction)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def knowledge_executor_node(state: AgentState):
    """Executes secure, role-based vector retrieval and RAG response generation."""
    llm = get_llm(
        model_name=state.get("model_name", ACTIVE_LLM_MODEL),
        llm_api_key=state.get("llm_api_key"),
    )
    vector_store = state["vector_store"]
    user_role = state.get("user_role", "Public")

    docs = retrieve_hybrid_and_rerank(
        query=state["rewritten_query"],
        vector_store=vector_store,
        user_role=user_role,
        candidate_k=state.get("candidate_k") or candidate_k_default(),
        final_k=state.get("final_k") or final_k_default(),
    )
    context = format_docs(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", state.get("prompt_template") or SYSTEM_RAG_PROMPT),
        ("human", "{input}"),
    ])

    chain = (
        {"context": lambda x: context, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(state["rewritten_query"])
    return {"messages": [AIMessage(content=response)]}


# --- Conditional Routing Edge ---

def route_by_intent(state: AgentState):
    """Decides whether to route to the Action or Knowledge pipeline."""
    intent = state["query_intent"].lower()
    if "action" in intent:
        return "action_executor"
    return "knowledge_executor"

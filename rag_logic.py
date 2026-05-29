# rag_logic.py

import os
import time
from typing import TypedDict, List, Annotated
import boto3
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.globals import set_llm_cache
from langchain_community.cache import RedisSemanticCache
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages                                  
from langgraph.prebuilt import ToolNode, tools_condition                         
from prompts import SYSTEM_RAG_PROMPT, QUERY_REWRITER_PROMPT, INTENT_CLASSIFIER_PROMPT

INDEX_NAME = "free-pdf-index"

# ─── CENTRALIZED MODEL CONFIGURATION ───
ACTIVE_LLM_MODEL = "llama-3.3-70b-versatile" 

# In-memory session memory to preserve the message thread across turns
SESSION_MEMORY: List[BaseMessage] = []


# --- 1. Define Pydantic Schema for Intent Classification ---
class IntentClassification(BaseModel):
    """Classify the user's intent to route them to the correct execution pipeline."""
    intent: Literal["action", "knowledge"] = Field(
        description="Select 'action' if the user wants to perform an operation, booking, reservation, HR task, or system command. Select 'knowledge' if they are asking a factual or informational question."
    )


# --- 2. Define Native LangChain Tools ---

@tool
def fetch_location(user_id: str) -> str:
    """Useful when you need to fetch the employee's current assigned seat, floor, and office location from the corporate directory."""
    mock_locations = {
        "ankit_vyas": {"office": "London", "floor": 3, "desk": "3A-12"},
        "guest_user": {"office": "New York", "floor": 5, "desk": "5B-04"}
    }
    loc = mock_locations.get(user_id, {"office": "Public", "floor": 1, "desk": "Hotdesk-1"})
    return f"Current Seat Assignment: Desk {loc['desk']} on Floor {loc['floor']} in the {loc['office']} office."

@tool
def book_desk_tool(user_id: str, date: str, floor: int) -> str:
    """Useful when you need to finalize a desk booking. Required parameters are date (string) and floor (integer)."""
    return f"SUCCESS: Desk booked successfully for {user_id} on {date} on Floor {floor}."

# Group tools into a list
tools = [fetch_location, book_desk_tool]
tool_node = ToolNode(tools)


# --- 3. Define the LangGraph State Schema ---
class AgentState(TypedDict):
    user_query: str
    messages: Annotated[list, add_messages] # Native LangGraph list of messages
    rewritten_query: str
    query_intent: str
    user_id: str
    user_role: str
    groq_api_key: str
    vector_store: PineconeVectorStore
    retrieval_k: int
    prompt_template: str
    model_name: str
    final_report: str


# --- 4. Define the Agent Nodes (Functions) ---

def query_rewriter_node(state: AgentState):
    """Resolves pronouns and history into a standalone question."""
    chat_history = state["messages"][:-1]  # Exclude the very last human message
    
    if not chat_history:
        return {"rewritten_query": state["user_query"]}
        
    llm = ChatGroq(
        model=state.get("model_name", ACTIVE_LLM_MODEL),
        groq_api_key=state.get("groq_api_key") or os.getenv("GROQ_API_KEY")
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", QUERY_REWRITER_PROMPT),
        *chat_history,
        ("human", "{user_query}")
    ])
    chain = prompt | llm
    response = chain.invoke({"user_query": state["user_query"]})
    print(f"\n[Agentic RAG] Rewritten Query: '{response.content}'")
    return {"rewritten_query": response.content}

def intent_classifier_node(state: AgentState):
    """Classifies the rewritten query as 'action' or 'knowledge' using the XML-formatted prompt."""
    llm = ChatGroq(
        model=state.get("model_name", ACTIVE_LLM_MODEL),
        groq_api_key=state.get("groq_api_key") or os.getenv("GROQ_API_KEY")
    )
    structured_llm = llm.with_structured_output(IntentClassification)
    
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
        ("human", "{formatted_input}")
    ])
    
    chain = prompt | structured_llm
    response = chain.invoke({"formatted_input": formatted_human_input})
    
    intent = response.intent
    print(f"[Agentic RAG] Classified Intent (Pydantic 70B): '{intent.upper()}'")
    return {"query_intent": intent}


# --- NATIVE TOOL-CALLING NODE ---
def action_executor_node(state: AgentState):
    """Agent node equipped with tools to handle desk booking conversational loops."""
    llm = ChatGroq(
        model=state.get("model_name", ACTIVE_LLM_MODEL),
        groq_api_key=state.get("groq_api_key") or os.getenv("GROQ_API_KEY")
    )
    llm_with_tools = llm.bind_tools(tools) # Bind tools to the model
    
    user_id = state.get("user_id", "guest_user")
    
    system_instruction = (
        "You are an automated corporate office assistant.\n"
        f"The active user's ID is: {user_id}.\n\n"
        "Your goal is to help them book a desk. To complete a booking, you must follow these rules:\n"
        "1. If you do not know where they sit today, you MUST call the `fetch_location` tool using their user_id.\n"
        "2. Inspect your conversation history. Check if they have provided both a DATE and a FLOOR.\n"
        "3. If any of those are missing, ask the user politely for them, suggesting they book on their current floor.\n"
        "4. Once you have BOTH a DATE and a FLOOR, you MUST call `book_desk_tool` to finalize the booking.\n\n"
        "CRITICAL SECURITY RULE:\n"
        "You have access to EXACTLY TWO tools: 'fetch_location' and 'book_desk_tool'. "
        "DO NOT attempt to use or call any other tools (such as 'brave_search', 'search', or 'web_search') under any circumstances. "
        "If you need any other information (such as today's date), ask the user directly or assume today's date."
    )
    
    # Prepend system instruction to the thread
    messages = [SystemMessage(content=system_instruction)] + state["messages"]
    
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}


def knowledge_executor_node(state: AgentState):
    """Executes secure, role-based vector retrieval and RAG response generation."""
    llm = ChatGroq(
        model=state.get("model_name", ACTIVE_LLM_MODEL),
        groq_api_key=state.get("groq_api_key") or os.getenv("GROQ_API_KEY")
    )
    vector_store = state["vector_store"]
    user_role = state.get("user_role", "Public")
    
    # Enforce metadata filtering on retrieval
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": state.get("retrieval_k", 3),
            "filter": {"required_role": user_role}
        }
    )
    
    docs = retriever.invoke(state["rewritten_query"])
    context = format_docs(docs)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", state.get("prompt_template") or SYSTEM_RAG_PROMPT),
        ("human", "{input}")
    ])
    
    chain = (
        {"context": lambda x: context, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response = chain.invoke(state["rewritten_query"])
    return {"messages": [AIMessage(content=response)]}


# --- 5. Define Conditional Routing Edge ---
def route_by_intent(state: AgentState):
    """Determines whether to route to the Action or Knowledge pipeline."""
    intent = state["query_intent"].lower()
    if "action" in intent:
        return "action_executor"
    else:
        return "knowledge_executor"


# --- 6. Compile the StateGraph ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("query_rewriter", query_rewriter_node)
workflow.add_node("intent_classifier", intent_classifier_node)
workflow.add_node("action_executor", action_executor_node)
workflow.add_node("knowledge_executor", knowledge_executor_node)
workflow.add_node("tools", tool_node)  # Add the native Tool execution node

# Set Edges
workflow.add_edge(START, "query_rewriter")
workflow.add_edge("query_rewriter", "intent_classifier")

# Route by intent
workflow.add_conditional_edges(
    "intent_classifier",
    route_by_intent,
    {
        "action_executor": "action_executor",
        "knowledge_executor": "knowledge_executor"
    }
)

# RE-ACT Loop
workflow.add_conditional_edges(
    "action_executor",
    tools_condition,  
    {
        "tools": "tools",
        END: END
    }
)

# Loop back to executor
workflow.add_edge("tools", "action_executor")
workflow.add_edge("knowledge_executor", END)

compiled_graph = workflow.compile()


# --- 7. Standard Helpers ---

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def init_semantic_cache(embeddings):
    """Initializes the global semantic cache if REDIS_URL is present."""
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        print("[~] Initializing global Redis Semantic Cache...")
        set_llm_cache(
            RedisSemanticCache(
                redis_url=redis_url,
                embedding=embeddings,
                score_threshold=0.05
            )
        )
    else:
        print("[!] REDIS_URL not found. Running without semantic cache.")

def init_pinecone(api_key: str) -> Pinecone:
    pc = Pinecone(api_key=api_key)
    if not pc.has_index(INDEX_NAME):
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
    return pc

def get_s3_client(aws_access_key: str = None, aws_secret_key: str = None, region: str = None):
    if aws_access_key and aws_secret_key:
        return boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
    else:
        return boto3.client('s3', region_name=region)

def upload_to_s3(local_file_path: str, bucket_name: str, s3_key: str, s3_client) -> bool:
    try:
        s3_client.upload_file(local_file_path, bucket_name, s3_key)
        return True
    except Exception as e:
        print(f"S3 Upload Error: {e}")
        return False

def process_and_upload_pdf(
    local_file_path: str,
    embeddings,
    pinecone_api_key: str,
    s3_bucket: str,
    s3_key: str,
    required_role: str = "Public"
):
    """Loads a PDF, chunks it, stores source/role metadata, and saves chunks to Pinecone."""
    loader = PyPDFLoader(local_file_path)
    documents = loader.load()

    s3_uri = f"s3://{s3_bucket}/{s3_key}"
    for doc in documents:
        doc.metadata["source"] = s3_uri
        doc.metadata["required_role"] = required_role

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    PineconeVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        index_name=INDEX_NAME,
        pinecone_api_key=pinecone_api_key
    )

def get_vector_store(embeddings, api_key: str) -> PineconeVectorStore:
    return PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=api_key
    )

def check_index_has_vectors(pc: Pinecone) -> bool:
    try:
        index_stats = pc.Index(INDEX_NAME).describe_index_stats()
        return index_stats.total_vector_count > 0
    except Exception:
        return False

def clear_database(pc: Pinecone):
    if pc.has_index(INDEX_NAME):
        pc.delete_index(INDEX_NAME)

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# --- 8. Updated Query Function (Invokes LangGraph) ---
def query_rag(
    user_query: str, 
    vector_store: PineconeVectorStore, 
    groq_api_key: str,
    user_id: str = "guest_user",
    session_id: str = "default_session",
    user_role: str = "Public",
    retrieval_k: int = 3,
    prompt_template: str = None,
    model_name: str = ACTIVE_LLM_MODEL
) -> str:
    global SESSION_MEMORY
    
    # Reconstruct the message list (append the new query)
    messages = list(SESSION_MEMORY)
    messages.append(HumanMessage(content=user_query))
    
    # Set the initial state variables
    initial_state = {
        "user_query": user_query,
        "messages": messages,          
        "user_id": user_id,
        "user_role": user_role,
        "groq_api_key": groq_api_key,
        "vector_store": vector_store,
        "retrieval_k": retrieval_k,
        "prompt_template": prompt_template or SYSTEM_RAG_PROMPT,
        "model_name": model_name,
        "final_report": ""
    }
    
    final_state = compiled_graph.invoke(initial_state)
    
    # Update local memory
    SESSION_MEMORY = final_state["messages"]
    
    return final_state["messages"][-1].content

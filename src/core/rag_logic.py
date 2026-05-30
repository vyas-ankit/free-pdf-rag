# rag_logic.py

import os
import json
import math
import re
import time
from collections import Counter
from typing import TypedDict, List, Annotated
import boto3
from langchain_core.globals import set_llm_cache
from langchain_core.documents import Document
from langchain_community.cache import RedisSemanticCache
from langsmith import traceable
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages                                  
from langgraph.prebuilt import ToolNode, tools_condition                         
from src.core.llm import get_llm, get_default_model
from src.core.prompts import SYSTEM_RAG_PROMPT, QUERY_REWRITER_PROMPT, INTENT_CLASSIFIER_PROMPT, ACTION_EXECUTOR_SYSTEM_PROMPT
from src.core.config import get_pinecone_config, get_retrieval_config, get_embeddings_config, get_semantic_cache_config

INDEX_NAME = get_pinecone_config().get("index_name", "free-pdf-index")

ACTIVE_LLM_MODEL = get_default_model()

# In-memory session memory to preserve the message thread across turns
SESSION_MEMORY: List[BaseMessage] = []


# --- 1. Define Native LangChain Tools ---

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
    llm_api_key: str
    vector_store: PineconeVectorStore
    candidate_k: int
    final_k: int
    prompt_template: str
    model_name: str
    final_report: str


# --- 4. Define the Agent Nodes (Functions) ---

def query_rewriter_node(state: AgentState):
    """Resolves pronouns and history into a standalone question."""
    chat_history = state["messages"][:-1]  # Exclude the very last human message
    
    if not chat_history:
        return {"rewritten_query": state["user_query"]}
        
    llm = get_llm(
        model_name=state.get("model_name", ACTIVE_LLM_MODEL),
        llm_api_key=state.get("llm_api_key")
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
    """Classifies the rewritten query as 'action' or 'knowledge'.

    Uses plain text output and parses the response — the prompt instructs the
    model to reply with a single word, which is incompatible with Groq's
    tool-call-required behavior in with_structured_output().
    """
    llm = get_llm(
        model_name=state.get("model_name", ACTIVE_LLM_MODEL),
        llm_api_key=state.get("llm_api_key")
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
        ("human", "{formatted_input}")
    ])

    chain = prompt | llm | StrOutputParser()
    raw_response = chain.invoke({"formatted_input": formatted_human_input})

    # Robust parsing: scan the response for the first matching keyword.
    # Default to "knowledge" (safer — retrieves docs instead of triggering tools).
    text = (raw_response or "").lower()
    if "action" in text and "knowledge" not in text:
        intent = "action"
    else:
        intent = "knowledge"

    print(f"[Agentic RAG] Classified Intent: '{intent.upper()}' (raw='{raw_response.strip()[:40]}')")
    return {"query_intent": intent}


# --- NATIVE TOOL-CALLING NODE ---
def action_executor_node(state: AgentState):
    """Agent node equipped with tools to handle desk booking conversational loops."""
    llm = get_llm(
        model_name=state.get("model_name", ACTIVE_LLM_MODEL),
        llm_api_key=state.get("llm_api_key")
    )
    llm_with_tools = llm.bind_tools(tools) # Bind tools to the model
    
    user_id = state.get("user_id", "guest_user")

    system_instruction = ACTION_EXECUTOR_SYSTEM_PROMPT.format(user_id=user_id)
    
    # Prepend system instruction to the thread
    messages = [SystemMessage(content=system_instruction)] + state["messages"]
    
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}


def knowledge_executor_node(state: AgentState):
    """Executes secure, role-based vector retrieval and RAG response generation."""
    llm = get_llm(
        model_name=state.get("model_name", ACTIVE_LLM_MODEL),
        llm_api_key=state.get("llm_api_key")
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
    """Build the embedding model from config (embeddings.model)."""
    cfg = get_embeddings_config()
    model_name = cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model_name)

def init_semantic_cache(embeddings):
    """Initializes the global semantic cache if REDIS_URL is present."""
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        print("[~] Initializing global Redis Semantic Cache...")
        cache_cfg = get_semantic_cache_config()
        score_threshold = float(cache_cfg.get("score_threshold", 0.05))
        set_llm_cache(
            RedisSemanticCache(
                redis_url=redis_url,
                embedding=embeddings,
                score_threshold=score_threshold,
            )
        )
    else:
        print("[!] REDIS_URL not found. Running without semantic cache.")

def init_pinecone(api_key: str) -> Pinecone:
    """Initialize Pinecone client; create index from config if it doesn't exist."""
    pc = Pinecone(api_key=api_key)
    if not pc.has_index(INDEX_NAME):
        pc_cfg = get_pinecone_config()
        dimension = int(pc_cfg.get("dimension", 384))
        metric = pc_cfg.get("metric", "cosine")
        cloud = pc_cfg.get("cloud", "aws")
        region = pc_cfg.get("region", "us-east-1")
        pc.create_index(
            name=INDEX_NAME,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
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

def _slug_from_pdf(source_pdf: str) -> str:
    """Turn 'Silver Report.pdf' into 'silver_report' for clean Pinecone IDs."""
    base = os.path.splitext(os.path.basename(source_pdf))[0]
    return re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")


def ingest_chunks_from_json(
    chunks_json_path: str,
    embeddings,
    pinecone_api_key: str,
    required_role: str = "Public",
    batch_size: int = None,
    replace_existing: bool = True,
    vector_store: PineconeVectorStore = None,
) -> int:
    """
    Upload chunks from a chunks_with_metadata.json file to Pinecone.

    Each chunk becomes one vector with deterministic ID
    `{source_pdf_slug}_chunk_{NNN}` so re-runs upsert in place (no duplicates).

    Args:
        chunks_json_path:    Path to chunks_with_metadata.json (from pipeline output)
        embeddings:          Embeddings model (from get_embeddings())
        pinecone_api_key:    Pinecone API key
        required_role:       Access role to attach to every chunk (default: "Public")
        batch_size:          Upsert batch size (default: 100)
        replace_existing:    If True, delete prior vectors for this source_pdf first

    Returns:
        Number of vectors upserted.
    """
    if not os.path.exists(chunks_json_path):
        print(f"ERROR: {chunks_json_path} not found")
        return 0

    if batch_size is None:
        batch_size = int(get_pinecone_config().get("upsert_batch_size", 100))

    with open(chunks_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        print(f"⚠ No chunks in {chunks_json_path}")
        return 0

    # source_pdf must be present in the JSON (injected by attach_metadata_to_chunks)
    source_pdf = chunks[0].get("source_pdf")
    if not source_pdf:
        print(f"ERROR: chunks in {chunks_json_path} missing 'source_pdf' field")
        return 0

    slug = _slug_from_pdf(source_pdf)
    print(f"Ingesting {len(chunks)} chunks from {source_pdf} (slug='{slug}', role='{required_role}')")

    # --- Optional: clear previous vectors for this source ---
    if replace_existing:
        pc = Pinecone(api_key=pinecone_api_key)
        if pc.has_index(INDEX_NAME):
            try:
                pc.Index(INDEX_NAME).delete(filter={"source_pdf": {"$eq": source_pdf}})
                print(f"  ✓ Cleared existing vectors for source_pdf={source_pdf}")
            except Exception as e:
                print(f"  ⚠ Could not clear existing vectors (continuing anyway): {e}")

    # --- Build LangChain Documents ---
    documents = []
    ids = []
    for chunk in chunks:
        doc_meta = chunk.get("metadata", {})
        chunk_id = chunk["chunk_id"]

        # Flatten metadata for Pinecone (no nested dicts allowed)
        meta = {
            "chunk_id":      chunk_id,
            "title":         doc_meta.get("title", ""),
            "date":          doc_meta.get("date", ""),
            "theme":         doc_meta.get("theme", ""),
            "word_count":    chunk.get("word_count", 0),
            "chunk_type":    chunk.get("chunk_type", "text"),
            "source_pdf":    source_pdf,
            "required_role": required_role,
            "content":       chunk["content"],  # duplicated so retrieval returns text
        }
        # Optional image number
        if chunk.get("image_num") is not None:
            meta["image_num"] = chunk["image_num"]

        documents.append(Document(page_content=chunk["content"], metadata=meta))
        ids.append(f"{slug}_chunk_{chunk_id:03d}")

    # --- Batch upsert via LangChain PineconeVectorStore ---
    # Reuse a caller-provided vector_store when batching across many PDFs to
    # avoid leaking gRPC connection pools / OS threads (otherwise long runs
    # crash with ThreadPool resource-exhaustion errors).
    if vector_store is None:
        vector_store = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings,
            pinecone_api_key=pinecone_api_key,
        )

    total = 0
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        vector_store.add_documents(documents=batch_docs, ids=batch_ids)
        total += len(batch_docs)

    print(f"  ✓ Upserted {total} vectors to '{INDEX_NAME}'")
    return total


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
    """Delete the entire Pinecone index (destructive — index must be recreated)."""
    if pc.has_index(INDEX_NAME):
        pc.delete_index(INDEX_NAME)


def clear_vectors(pc: Pinecone, namespace: str = "") -> int:
    """
    Delete all vectors from the index but keep the index structure intact.
    Returns the number of vectors that existed before deletion.
    Prefer this over clear_database() for routine re-ingestion.
    """
    if not pc.has_index(INDEX_NAME):
        print(f"Index '{INDEX_NAME}' does not exist — nothing to clear")
        return 0

    index = pc.Index(INDEX_NAME)
    before = index.describe_index_stats().total_vector_count
    if before == 0:
        print(f"Index '{INDEX_NAME}' already empty")
        return 0

    index.delete(delete_all=True, namespace=namespace)
    print(f"✓ Deleted {before} vectors from '{INDEX_NAME}' (namespace={namespace or 'default'})")
    return before

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def env_flag(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# --- Retrieval config resolvers (config file is the source; env var overrides for ad-hoc experiments) ---

def hybrid_search_enabled() -> bool:
    cfg_default = get_retrieval_config().get("hybrid_search", {}).get("enabled", True)
    return env_flag("ENABLE_HYBRID_SEARCH", cfg_default)


def reranker_enabled() -> bool:
    cfg_default = get_retrieval_config().get("reranker", {}).get("enabled", True)
    return env_flag("ENABLE_RERANKING", cfg_default)


def hybrid_dense_weight() -> float:
    cfg_default = get_retrieval_config().get("hybrid_search", {}).get("dense_weight", 0.65)
    return float(os.getenv("HYBRID_DENSE_WEIGHT", cfg_default))


def reranker_model_name() -> str:
    cfg_default = get_retrieval_config().get("reranker", {}).get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    return os.getenv("RERANKER_MODEL", cfg_default)


def candidate_k_default() -> int:
    cfg_default = get_retrieval_config().get("candidate_k", 20)
    return int(os.getenv("RAG_CANDIDATE_K", cfg_default))


def final_k_default() -> int:
    cfg_default = get_retrieval_config().get("final_k", 7)
    return int(os.getenv("RAG_FINAL_K", cfg_default))


def tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def minmax_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]


def bm25_scores(query: str, docs, k1: float = None, b: float = None) -> list[float]:
    if k1 is None or b is None:
        bm25_cfg = get_retrieval_config().get("hybrid_search", {}).get("bm25", {})
        if k1 is None:
            k1 = float(bm25_cfg.get("k1", 1.5))
        if b is None:
            b = float(bm25_cfg.get("b", 0.75))

    query_terms = tokenize_for_bm25(query)
    if not query_terms or not docs:
        return [0.0 for _ in docs]

    tokenized_docs = [tokenize_for_bm25(doc.page_content) for doc in docs]
    doc_lengths = [len(tokens) for tokens in tokenized_docs]
    avg_doc_length = sum(doc_lengths) / max(len(doc_lengths), 1)

    doc_freqs = Counter()
    for tokens in tokenized_docs:
        for term in set(tokens):
            doc_freqs[term] += 1

    total_docs = len(docs)
    scores = []
    for tokens, doc_length in zip(tokenized_docs, doc_lengths):
        term_freqs = Counter(tokens)
        score = 0.0
        for term in query_terms:
            if term not in term_freqs:
                continue
            idf = math.log(1 + (total_docs - doc_freqs[term] + 0.5) / (doc_freqs[term] + 0.5))
            numerator = term_freqs[term] * (k1 + 1)
            denominator = term_freqs[term] + k1 * (1 - b + b * doc_length / max(avg_doc_length, 1))
            score += idf * numerator / denominator
        scores.append(score)
    return scores


@traceable(name="dense_retrieval", run_type="retriever")
def dense_similarity_search(vector_store: PineconeVectorStore, query: str, user_role: str, candidate_k: int):
    filter_metadata = {"required_role": user_role}
    if hasattr(vector_store, "similarity_search_with_relevance_scores"):
        try:
            return vector_store.similarity_search_with_relevance_scores(
                query,
                k=candidate_k,
                filter=filter_metadata
            )
        except Exception:
            pass

    if hasattr(vector_store, "similarity_search_with_score"):
        return vector_store.similarity_search_with_score(
            query,
            k=candidate_k,
            filter=filter_metadata
        )

    docs = vector_store.as_retriever(
        search_kwargs={"k": candidate_k, "filter": filter_metadata}
    ).invoke(query)
    return [(doc, float(candidate_k - index)) for index, doc in enumerate(docs)]


@traceable(name="hybrid_rerank", run_type="retriever")
def hybrid_rank_docs(query: str, scored_docs, alpha: float = None):
    if not scored_docs:
        return []

    alpha = float(alpha) if alpha is not None else hybrid_dense_weight()
    alpha = min(max(alpha, 0.0), 1.0)

    docs = [doc for doc, _ in scored_docs]
    dense_scores = [float(score) for _, score in scored_docs]
    lexical_scores = bm25_scores(query, docs)

    dense_norm = minmax_normalize(dense_scores)
    lexical_norm = minmax_normalize(lexical_scores)

    ranked = []
    for index, doc in enumerate(docs):
        hybrid_score = (alpha * dense_norm[index]) + ((1 - alpha) * lexical_norm[index])
        doc.metadata["dense_score"] = dense_scores[index]
        doc.metadata["lexical_score"] = lexical_scores[index]
        doc.metadata["hybrid_score"] = hybrid_score
        ranked.append((doc, hybrid_score))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return [doc for doc, _ in ranked]


RERANKER = None


def get_reranker():
    global RERANKER
    if RERANKER is None:
        from sentence_transformers import CrossEncoder

        RERANKER = CrossEncoder(reranker_model_name())
    return RERANKER


@traceable(name="cross_encoder_rerank", run_type="retriever")
def rerank_docs(query: str, docs, final_k: int):
    if not docs or not reranker_enabled():
        return docs[:final_k]

    try:
        reranker = get_reranker()
        pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda item: item[1], reverse=True)
        for doc, score in ranked:
            doc.metadata["rerank_score"] = float(score)
        return [doc for doc, _ in ranked[:final_k]]
    except Exception as exc:
        print(f"[RAG] Reranking skipped: {exc}")
        return docs[:final_k]


@traceable(name="retrieve_pipeline", run_type="retriever")
def retrieve_hybrid_and_rerank(
    query: str,
    vector_store: PineconeVectorStore,
    user_role: str,
    candidate_k: int = None,
    final_k: int = None,
):
    candidate_k = candidate_k or candidate_k_default()
    final_k = final_k or final_k_default()
    candidate_k = max(candidate_k, final_k)
    scored_docs = dense_similarity_search(vector_store, query, user_role, candidate_k)

    if hybrid_search_enabled():
        candidates = hybrid_rank_docs(query, scored_docs)
    else:
        candidates = [doc for doc, _ in scored_docs]

    final_docs = rerank_docs(query, candidates, final_k)
    print(
        f"[RAG] Retrieved {len(scored_docs)} candidates, "
        f"hybrid={'on' if hybrid_search_enabled() else 'off'}, "
        f"rerank={'on' if reranker_enabled() else 'off'}, "
        f"final={len(final_docs)}"
    )
    return final_docs


# --- 8. Updated Query Function (Invokes LangGraph) ---
def query_rag(
    user_query: str,
    vector_store: PineconeVectorStore,
    llm_api_key: str = None,
    user_id: str = "guest_user",
    session_id: str = "default_session",
    user_role: str = "Public",
    candidate_k: int = None,
    final_k: int = None,
    prompt_template: str = None,
    model_name: str = ACTIVE_LLM_MODEL,
) -> str:
    global SESSION_MEMORY
    
    # Reconstruct the message list (append the new query)
    messages = list(SESSION_MEMORY)
    messages.append(HumanMessage(content=user_query))
    
    # Resolve k values from config (or env override) — args still win if passed explicitly.
    resolved_candidate_k = candidate_k or candidate_k_default()
    resolved_final_k = final_k or final_k_default()

    initial_state = {
        "user_query": user_query,
        "messages": messages,
        "user_id": user_id,
        "user_role": user_role,
        "llm_api_key": llm_api_key,
        "vector_store": vector_store,
        "candidate_k": resolved_candidate_k,
        "final_k": resolved_final_k,
        "prompt_template": prompt_template or SYSTEM_RAG_PROMPT,
        "model_name": model_name,
        "final_report": "",
    }
    
    final_state = compiled_graph.invoke(initial_state)
    
    # Update local memory
    SESSION_MEMORY = final_state["messages"]
    
    return final_state["messages"][-1].content

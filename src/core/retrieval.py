# src/core/retrieval.py
"""
The read-side pipeline: dense Pinecone search → in-memory BM25 hybrid rank →
cross-encoder rerank. Plus the small config resolvers that read tuning knobs
out of llm_config.json (with env-var overrides for ad-hoc experiments).
"""

import math
import os
import re
from collections import Counter

from langsmith import traceable
from langchain_pinecone import PineconeVectorStore

from src.core.config import get_retrieval_config


# ----------------------- Env-var override helper -----------------------

def env_flag(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# ----------------------- Config resolvers -----------------------
# Config file is the source; env vars override for ad-hoc tuning.

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


# ----------------------- BM25 (in-memory, applied to dense candidates) -----------------------

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


# ----------------------- Dense retrieval (Pinecone) -----------------------

@traceable(name="dense_retrieval", run_type="retriever")
def dense_similarity_search(vector_store: PineconeVectorStore, query: str, user_role: str, candidate_k: int):
    filter_metadata = {"required_role": user_role}
    if hasattr(vector_store, "similarity_search_with_relevance_scores"):
        try:
            return vector_store.similarity_search_with_relevance_scores(
                query,
                k=candidate_k,
                filter=filter_metadata,
            )
        except Exception:
            pass

    if hasattr(vector_store, "similarity_search_with_score"):
        return vector_store.similarity_search_with_score(
            query,
            k=candidate_k,
            filter=filter_metadata,
        )

    docs = vector_store.as_retriever(
        search_kwargs={"k": candidate_k, "filter": filter_metadata}
    ).invoke(query)
    return [(doc, float(candidate_k - index)) for index, doc in enumerate(docs)]


# ----------------------- Hybrid combination (dense + BM25) -----------------------

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


# ----------------------- Cross-encoder reranker -----------------------

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


# ----------------------- Full pipeline (the orchestrator) -----------------------

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


# ----------------------- Output formatting -----------------------

def format_docs(docs) -> str:
    """Join retrieved chunks into a single string for the LLM prompt context.

    Each chunk is prefixed with its source metadata so the LLM can cite
    sources, reason chronologically, and distinguish between documents.
    """
    parts = []
    for doc in docs:
        m = doc.metadata
        source = m.get("source_pdf", "")
        date = m.get("date", "")
        theme = m.get("theme", "")
        header_parts = []
        if source:
            header_parts.append(f"Source: {source}")
        if date:
            header_parts.append(f"Date: {date}")
        if theme:
            header_parts.append(f"Theme: {theme}")
        header = f"[{' | '.join(header_parts)}]" if header_parts else ""
        chunk_text = f"{header}\n{doc.page_content}" if header else doc.page_content
        parts.append(chunk_text)
    return "\n\n---\n\n".join(parts)

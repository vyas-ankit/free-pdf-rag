# evaluate.py

import os
from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from src.core.llm import get_llm
from src.core import rag_logic

load_dotenv()

# 1. Verify local keys are set before running
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GROQ_API_KEY")

if not all([PINECONE_API_KEY, LLM_API_KEY]):
    print("\n[-] Error: Missing required environment variables locally.")
    print("Please export PINECONE_API_KEY and LLM_API_KEY in your terminal before running this script.")
    exit(1)

print("[~] Initializing local RAG components...")
embeddings = rag_logic.get_embeddings()
pc = rag_logic.init_pinecone(PINECONE_API_KEY)
vector_store = rag_logic.get_vector_store(embeddings, PINECONE_API_KEY)
client = Client()

# --- DEFINE AUTOMATED EVALUATORS (LLM-AS-A-JUDGE) ---
print("[~] Setting up LLM-as-a-Judge Evaluators...")

# Use the configured LLM as the objective judge.
judge_llm = get_llm(llm_api_key=LLM_API_KEY)

# Correctness Evaluator (Reference-based, explains reasoning step-by-step)
correctness_evaluator = LangChainStringEvaluator(
    "cot_qa",  # Chain-of-Thought Q&A Correctness
    config={"llm": judge_llm}
)

# Coherence Evaluator (Reference-free, measures logical structure and readability)
coherence_evaluator = LangChainStringEvaluator(
    "criteria",  # Use base 'criteria' type
    config={
        "criteria": "coherence",  # Specify 'coherence' sub-criteria
        "llm": judge_llm
    }
)

# Combined list of judges
evaluators_list = [correctness_evaluator, coherence_evaluator]


# --- Agentic LangGraph implementation ---
def predict_agentic_strict(inputs: dict):
    experimental_prompt = (
        "You are a strict, highly detailed document auditor. "
        "Use ONLY the following context to answer the question. "
        "If the context does not contain the answer, say 'I cannot find the answer in the provided documents.'\n\n"
        "Context:\n{context}"
    )
    response = rag_logic.query_rag(
        user_query=inputs["question"],
        vector_store=vector_store,
        llm_api_key=LLM_API_KEY,
        user_role="Public",
        final_k=5,
        prompt_template=experimental_prompt,
        model_name=rag_logic.ACTIVE_LLM_MODEL
    )
    return {"output": response}


# Run evaluation in LangSmith
try:
    print("\n[~] Running Agentic RAG evaluation with automated judges...")
    evaluate(
        predict_agentic_strict,
        data="rag-evaluation-suite",
        evaluators=evaluators_list,
        experiment_prefix="Agentic-Strict-K5",
    )
    print("\n[+] Evaluation completed successfully! Check your LangSmith Dashboard.")
except Exception as e:
    print(f"\n[-] Evaluation failed: {e}")
    print("Please make sure your dataset 'rag-evaluation-suite' is populated on LangSmith.")

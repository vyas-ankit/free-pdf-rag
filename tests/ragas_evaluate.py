# tests/ragas_evaluate.py
"""
RAGAS evaluation for the simple RAG pipeline.

Usage:
    python -m tests.ragas_evaluate                    # default: no_gt (3 metrics)
    python -m tests.ragas_evaluate --metrics no_gt    # faithfulness, answer_relevancy, context_utilization
    python -m tests.ragas_evaluate --metrics gt       # context_precision, context_recall
    python -m tests.ragas_evaluate --metrics all      # all 5 metrics

Metrics requiring no ground_truth (no_gt):
    faithfulness        — does the answer stick to the retrieved context?
    answer_relevancy    — does the answer actually address the question?
    context_utilization — how well does the LLM use the retrieved context?

Metrics requiring ground_truth (gt):
    context_precision   — are the relevant chunks ranked at the top?
    context_recall      — did retrieval find all information needed to answer?

Results saved to tests/ragas_results_{metric_set}.csv
"""

import nest_asyncio
nest_asyncio.apply()  # patch event loop before any asyncio.run()

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Env validation ────────────────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")

for var, val in [("PINECONE_API_KEY", PINECONE_API_KEY), ("OPENAI_API_KEY", OPENAI_API_KEY)]:
    if not val:
        print(f"[-] Missing {var}. Set it in .env before running eval.")
        sys.exit(1)

# ── Project imports ───────────────────────────────────────────────────────
from src.core.vector_store import get_embeddings, get_vector_store
from src.core.llm import get_llm
from src.core.rag_simple import query_rag_simple, clear_session

# ── RAGAS imports ─────────────────────────────────────────────────────────
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_utilization,
    context_precision,
    context_recall,
)

# ── Metric sets ───────────────────────────────────────────────────────────
METRIC_SETS = {
    "no_gt": [faithfulness, answer_relevancy, context_utilization],
    "gt":    [context_precision, context_recall],
    "all":   [faithfulness, answer_relevancy, context_utilization,
              context_precision, context_recall],
}

# ── Constants ─────────────────────────────────────────────────────────────
EVAL_QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"
EVAL_SESSION_PREFIX = "__ragas_eval__"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        choices=["no_gt", "gt", "all"],
        default="no_gt",
        help="Which metric set to run (default: no_gt)",
    )
    return parser.parse_args()


def load_questions(metric_set: str) -> list:
    with open(EVAL_QUESTIONS_PATH) as f:
        questions = json.load(f)

    if metric_set in ("gt", "all"):
        # Filter to only questions that have ground_truth for GT metrics
        has_gt = [q for q in questions if q.get("ground_truth")]
        missing = len(questions) - len(has_gt)
        if missing:
            print(f"[!] Skipping {missing} questions with no ground_truth for GT metrics")
        return has_gt

    return questions


def run_pipeline_for_eval(question: str, vs, user_role: str = "Public") -> dict:
    """Run query_rag_simple with guards bypassed and contexts returned.

    Each question gets its own isolated session so prior eval queries
    don't influence the rewriter.
    """
    session_id = f"{EVAL_SESSION_PREFIX}_{question[:40]}"
    clear_session(session_id)

    result = query_rag_simple(
        user_query=question,
        vector_store=vs,
        session_id=session_id,
        user_role=user_role,
        skip_guards=True,
        return_contexts=True,
    )
    return result


def build_ragas_dataset(questions: list, vs, metric_set: str) -> Dataset:
    rows = []
    total = len(questions)
    needs_gt = metric_set in ("gt", "all")

    for i, item in enumerate(questions, 1):
        q = item["question"]
        ground_truth = item.get("ground_truth")

        print(f"[{i}/{total}] {q[:70]}")
        try:
            result = run_pipeline_for_eval(q, vs)
            row = {
                "question": q,
                "answer":   result["answer"],
                "contexts": result["contexts"],
            }
            if needs_gt and ground_truth:
                row["ground_truth"] = ground_truth
            rows.append(row)
            print(f"         contexts={len(result['contexts'])}, "
                  f"answer={result['answer'][:80]!r}")
        except Exception as exc:
            print(f"         ERROR: {exc}")
            row = {"question": q, "answer": f"[error: {exc}]", "contexts": []}
            if needs_gt and ground_truth:
                row["ground_truth"] = ground_truth
            rows.append(row)

    return Dataset.from_list(rows)


def run_ragas_eval(dataset: Dataset, metrics: list) -> dict:
    llm        = get_llm()
    embeddings = get_embeddings()

    from ragas.llms       import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    ragas_llm        = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    for metric in metrics:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_embeddings

    print(f"\n[~] Running RAGAS evaluation ({len(metrics)} metrics, "
          f"{len(dataset)} questions)...")

    return evaluate(dataset, metrics=metrics, raise_exceptions=False)


def print_results(result, metrics: list, metric_set: str) -> None:
    print("\n" + "=" * 60)
    print(f"RAGAS RESULTS  [{metric_set}]")
    print("=" * 60)

    print("\n── Aggregate scores ──")
    for metric in metrics:
        name  = metric.name
        score = result.get(name)
        if score is not None:
            bar = "█" * int(score * 20)
            print(f"  {name:30s}  {score:.3f}  {bar}")
        else:
            print(f"  {name:30s}  n/a")

    print("\n── Per-question scores ──")
    df   = result.to_pandas()
    cols = ["question"] + [m.name for m in metrics if m.name in df.columns]
    df_view = df[cols].copy()
    df_view["question"] = df_view["question"].str[:50]
    print(df_view.to_string(index=False))

    out_path = Path(__file__).parent / f"ragas_results_{metric_set}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[+] Full results saved to {out_path}")


def main():
    args = parse_args()
    metric_set = args.metrics
    metrics    = METRIC_SETS[metric_set]

    print(f"[~] Metric set: {metric_set} — {[m.name for m in metrics]}")
    print(f"[~] Initializing Pinecone & embeddings...")
    embeddings = get_embeddings()
    vs         = get_vector_store(embeddings, PINECONE_API_KEY)
    print("[+] Ready.\n")

    questions = load_questions(metric_set)
    print(f"[~] {len(questions)} questions for this run\n")

    dataset = build_ragas_dataset(questions, vs, metric_set)
    result  = run_ragas_eval(dataset, metrics)
    print_results(result, metrics, metric_set)


if __name__ == "__main__":
    main()

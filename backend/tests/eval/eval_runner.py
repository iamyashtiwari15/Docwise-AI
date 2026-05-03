#!/usr/bin/env python3
"""
eval_runner.py — 4-Layer RAG Evaluation Pipeline
=================================================
Layer 1  Retrieval   : Recall@5, Precision@5, MRR, NDCG@5
Layer 2  Generation  : Faithfulness, Answer Relevancy, Correctness (Groq-as-judge)
Layer 3  Routing     : Accuracy, per-class F1, confusion matrix, false-retrieval rate
Layer 4  System      : Latency p50/p95/p99, error rate, cost estimate

Usage (run from project root or backend/):
    cd backend
    python -m tests.eval.eval_runner
    python -m tests.eval.eval_runner --pdf ../AmbedkarAnnihilationofCastes.pdf

Outputs:
    eval_runs/eval_<timestamp>.json   — persisted results for trend tracking
    Prints CV-ready scorecard to stdout
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Path bootstrap ─────────────────────────────────────────────────────────────
BACKEND_ROOT = Path(__file__).resolve().parents[2]   # .../backend
PROJECT_ROOT = BACKEND_ROOT.parent                   # .../project

sys.path.insert(0, str(BACKEND_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ── System imports (after path + env) ─────────────────────────────────────────
from agents.uploaded_document_store import UploadedDocumentStore
from agents.query_router import QueryRouter
from core.config import get_settings

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eval_runner")


def _patch_settings_for_eval() -> None:
    """
    Disable HyDE and LLM multi-query during eval runs.

    These features are valuable in production but each fires an extra LLM call
    per retrieval, quickly exhausting Groq's 100k token/day free limit.
    Retrieval quality is still measured correctly — heuristic multi-query and
    dense + BM25 fusion still run; we just skip the LLM-powered variants.

    Token savings per eval run: ~40–60 calls (~30k tokens).
    """
    settings = get_settings()
    settings.enable_hyde = False
    settings.enable_llm_multi_query = False
    logger.info("[EVAL] HyDE and LLM multi-query disabled to conserve token budget")

EVAL_SESSION = "eval-session-001"

# ──────────────────────────────────────────────────────────────────────────────
# LAYER 1 — RETRIEVAL
# ──────────────────────────────────────────────────────────────────────────────

def _any_hint_in(text: str, hints: List[str]) -> bool:
    """True when the chunk text contains at least one content hint (case-insensitive)."""
    lower = text.lower()
    return any(h.lower() in lower for h in hints)


def recall_at_k(texts: List[str], hints: List[str], k: int = 5) -> float:
    """Fraction of content hints found in top-k retrieved chunks."""
    if not hints:
        return 0.0
    top_k = texts[:k]
    covered = sum(
        1 for h in hints
        if any(h.lower() in t.lower() for t in top_k)
    )
    return covered / len(hints)


def precision_at_k(texts: List[str], hints: List[str], k: int = 5) -> float:
    """Fraction of top-k chunks that are relevant (contain at least one hint)."""
    top_k = texts[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for t in top_k if _any_hint_in(t, hints))
    return hits / len(top_k)


def mean_reciprocal_rank(texts: List[str], hints: List[str]) -> float:
    for rank, text in enumerate(texts, start=1):
        if _any_hint_in(text, hints):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(texts: List[str], hints: List[str], k: int = 5) -> float:
    top_k = texts[:k]
    dcg = sum(
        (1 if _any_hint_in(t, hints) else 0) / math.log2(i + 2)
        for i, t in enumerate(top_k)
    )
    ideal_hits = min(len(hints), k)
    idcg = sum(1 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def run_retrieval_eval(test_set: List[Dict], store: UploadedDocumentStore) -> Dict:
    recall_scores, prec_scores, mrr_scores, ndcg_scores = [], [], [], []
    failures = []

    for item in test_set:
        if item.get("expected_route") != "document":
            continue
        hints = item.get("ground_truth_content_hints", [])
        if not hints:
            continue

        try:
            results = store.retrieve(EVAL_SESSION, item["question"], top_k=5)
            texts = [r["content"] for r in results]

            r = recall_at_k(texts, hints)
            p = precision_at_k(texts, hints)
            m = mean_reciprocal_rank(texts, hints)
            n = ndcg_at_k(texts, hints)

            recall_scores.append(r)
            prec_scores.append(p)
            mrr_scores.append(m)
            ndcg_scores.append(n)

            if r == 0.0:
                failures.append({"question": item["question"][:80], "hints": hints})
        except Exception as exc:
            logger.warning("Retrieval error for %r: %s", item["question"][:60], exc)

    def safe_mean(lst):
        return round(float(np.mean(lst)), 4) if lst else 0.0

    return {
        "recall@5":          safe_mean(recall_scores),
        "precision@5":       safe_mean(prec_scores),
        "mrr":               safe_mean(mrr_scores),
        "ndcg@5":            safe_mean(ndcg_scores),
        "n_evaluated":       len(recall_scores),
        "retrieval_failures": failures,
    }


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 2 — GENERATION (Groq-as-judge)
# ──────────────────────────────────────────────────────────────────────────────

_FAITHFULNESS_PROMPT = """\
You are evaluating a RAG system answer for faithfulness.
Faithfulness = every claim in the answer is directly supported by the retrieved context.

Context:
{context}

Question: {question}
Answer: {answer}

Score 0.0–1.0 where:
  1.0 = all claims are in the context
  0.5 = some claims are supported, some are not
  0.0 = answer ignores or contradicts the context

Return ONLY valid JSON (no markdown): {{"score": <float 0-1>, "reason": "<one sentence>"}}"""

_RELEVANCY_PROMPT = """\
You are evaluating whether an AI answer is relevant to the question.

Question: {question}
Answer: {answer}

Score 0.0–1.0 where:
  1.0 = answer directly and completely addresses the question
  0.5 = answer partially addresses the question
  0.0 = answer does not address the question

Return ONLY valid JSON (no markdown): {{"score": <float 0-1>, "reason": "<one sentence>"}}"""

_CORRECTNESS_PROMPT = """\
You are evaluating factual correctness of an AI answer against a reference answer.

Question: {question}
Reference Answer: {ground_truth}
AI Answer: {answer}

Score 0.0–1.0 where:
  1.0 = AI answer is factually consistent with and covers key points of the reference
  0.5 = AI answer is partially correct
  0.0 = AI answer contradicts or misses key points of the reference

Return ONLY valid JSON (no markdown): {{"score": <float 0-1>, "reason": "<one sentence>"}}"""


def _call_judge(llm, prompt: str) -> Tuple[float, str]:
    """Call the LLM judge. 429 rotation is handled by RotatingGroqClient."""
    try:
        result = llm.invoke(prompt)
        content = (result.content if hasattr(result, "content") else str(result)).strip()
        json_match = re.search(r"\{.*?\}", content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return float(parsed.get("score", 0.0)), parsed.get("reason", "")
        return 0.0, "parse_error"
    except Exception as exc:
        logger.warning("Judge error: %s", exc)
        return 0.0, "error"


def run_generation_eval(test_set: List[Dict], store: UploadedDocumentStore) -> Dict:
    from tests.eval.groq_key_rotator import RotatingGroqClient

    rotator = RotatingGroqClient(model="llama-3.3-70b-versatile", temperature=0)
    print(f"   Using {rotator.keys_remaining} Groq key(s) — auto-rotates on 429 (NOTE: keys must be from different Groq accounts)")
    settings = get_settings()

    faith_scores, rel_scores, corr_scores = [], [], []
    per_question = []

    for item in test_set:
        if item.get("expected_route") != "document":
            continue
        if not item.get("ground_truth_answer"):
            continue

        question = item["question"]
        ground_truth = item["ground_truth_answer"]

        try:
            docs = store.retrieve(EVAL_SESSION, question, top_k=settings.rag_top_k)
            if not docs:
                continue

            # Generate answer via rotator (avoids the cached ResponseGenerator LLM)
            context = "\n\n".join(d["content"] for d in docs[:3])
            gen_prompt = (
                f"Answer the following question using ONLY the context below.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Answer concisely and faithfully from the context only."
            )
            gen_result = rotator.invoke(gen_prompt)
            answer = (gen_result.content if hasattr(gen_result, "content") else str(gen_result)).strip()
            if not answer:
                continue

            time.sleep(1)
            faith, _ = _call_judge(
                rotator,
                _FAITHFULNESS_PROMPT.format(context=context, question=question, answer=answer),
            )
            time.sleep(1)
            rel, _ = _call_judge(
                rotator,
                _RELEVANCY_PROMPT.format(question=question, answer=answer),
            )
            time.sleep(1)
            corr, _ = _call_judge(
                rotator,
                _CORRECTNESS_PROMPT.format(
                    question=question, ground_truth=ground_truth, answer=answer
                ),
            )
            time.sleep(1)

            faith_scores.append(faith)
            rel_scores.append(rel)
            corr_scores.append(corr)
            per_question.append(
                {
                    "question": question[:80],
                    "faithfulness": round(faith, 3),
                    "answer_relevancy": round(rel, 3),
                    "answer_correctness": round(corr, 3),
                }
            )

        except Exception as exc:
            logger.warning("Generation eval error for %r: %s", question[:60], exc)

    def safe_mean(lst):
        return round(float(np.mean(lst)), 4) if lst else 0.0

    return {
        "faithfulness":       safe_mean(faith_scores),
        "answer_relevancy":   safe_mean(rel_scores),
        "answer_correctness": safe_mean(corr_scores),
        "n_evaluated":        len(faith_scores),
        "per_question":       per_question,
    }


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 3 — ROUTING
# ──────────────────────────────────────────────────────────────────────────────

def run_routing_eval(test_set: List[Dict], router: QueryRouter) -> Dict:
    y_true, y_pred = [], []
    failures = []

    for item in test_set:
        expected = item.get("expected_route")
        if not expected:
            continue

        has_docs = item.get("has_uploaded_documents", True)
        predicted = router.route_query(
            item["question"], has_uploaded_documents=has_docs
        )["query_type"]

        y_true.append(expected)
        y_pred.append(predicted)

        if predicted != expected:
            failures.append(
                {
                    "question": item["question"][:80],
                    "expected": expected,
                    "predicted": predicted,
                    "category": item.get("category", ""),
                }
            )

    if not y_true:
        return {"overall_accuracy": 0.0, "n_evaluated": 0}

    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

    labels = ["document", "web", "general"]
    per_class = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[label] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    # Confusion matrix
    cm = {f"{t}→{p}": 0 for t in labels for p in labels}
    for t, p in zip(y_true, y_pred):
        key = f"{t}→{p}"
        if key in cm:
            cm[key] += 1
    cm = {k: v for k, v in cm.items() if v > 0}

    # False retrieval rate: non-document queries incorrectly routed to "document"
    out_of_scope = [i for i in test_set if i.get("expected_route") != "document"]
    false_retrievals = sum(
        1
        for i in out_of_scope
        if router.route_query(
            i["question"],
            has_uploaded_documents=i.get("has_uploaded_documents", True),
        )["query_type"]
        == "document"
    )
    false_retrieval_rate = (
        false_retrievals / len(out_of_scope) if out_of_scope else 0.0
    )

    return {
        "overall_accuracy":     round(accuracy, 4),
        "false_retrieval_rate": round(false_retrieval_rate, 4),
        "per_class":            per_class,
        "confusion_matrix":     cm,
        "routing_failures":     failures,
        "n_evaluated":          len(y_true),
    }


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 4 — SYSTEM PERFORMANCE
# ──────────────────────────────────────────────────────────────────────────────

def run_system_eval(test_set: List[Dict], store: UploadedDocumentStore) -> Dict:
    from tests.eval.groq_key_rotator import RotatingGroqClient

    rotator = RotatingGroqClient(model="llama-3.3-70b-versatile", temperature=0)
    latencies: List[float] = []
    errors = 0

    sample = [i for i in test_set if i.get("expected_route") == "document"][:10]

    for item in sample:
        t0 = time.perf_counter()
        try:
            docs = store.retrieve(EVAL_SESSION, item["question"], top_k=5)
            if docs:
                context = "\n\n".join(d["content"] for d in docs[:3])
                prompt = (
                    f"Answer the question using only the context.\n\n"
                    f"Context:\n{context}\n\nQuestion: {item['question']}\nAnswer:"
                )
                rotator.invoke(prompt)
        except Exception:
            errors += 1
        latencies.append(time.perf_counter() - t0)

    if not latencies:
        return {
            "latency_p50": 0.0, "latency_p95": 0.0, "latency_p99": 0.0,
            "latency_mean": 0.0, "error_rate": 0.0, "total_queries": 0,
        }

    sl = sorted(latencies)
    n = len(sl)

    def pct(p: float) -> float:
        return round(sl[min(int(n * p), n - 1)], 3)

    return {
        "latency_p50":   pct(0.50),
        "latency_p95":   pct(0.95),
        "latency_p99":   pct(0.99),
        "latency_mean":  round(mean(latencies), 3),
        "error_rate":    round(errors / len(latencies), 4),
        "total_queries": len(latencies),
    }


def _estimate_cost_per_query() -> float:
    """Groq pricing estimate: ~$0.59/1M input, $0.79/1M output tokens."""
    input_tokens, output_tokens = 2000, 500
    return round(
        input_tokens * (0.59 / 1e6) + output_tokens * (0.79 / 1e6), 6
    )


# ──────────────────────────────────────────────────────────────────────────────
# MASTER RUNNER
# ──────────────────────────────────────────────────────────────────────────────

_GENERATION_SKIPPED = {
    "faithfulness": None, "answer_relevancy": None, "answer_correctness": None,
    "n_evaluated": 0, "per_question": [], "skipped": True,
}
_SYSTEM_SKIPPED = {
    "latency_p50": None, "latency_p95": None, "latency_p99": None,
    "latency_mean": None, "error_rate": None, "total_queries": 0, "skipped": True,
}


def run_full_evaluation(pdf_path: str, test_set_path: str, skip_generation: bool = False) -> Dict:
    print("\n🔧  Initialising evaluation environment …")
    _patch_settings_for_eval()

    with open(test_set_path, encoding="utf-8") as f:
        test_set = json.load(f)
    print(f"📋  Loaded test set: {len(test_set)} items")

    print(f"📄  Ingesting document: {pdf_path}")
    print("     (First run downloads the embedding model — may take a few minutes)")
    store = UploadedDocumentStore()
    pdf_bytes = Path(pdf_path).read_bytes()
    info = store.add_file(EVAL_SESSION, Path(pdf_path).name, pdf_bytes, "application/pdf")
    print(f"✅  Ingested {info['chunk_count']} chunks from {info['filename']}")

    router = QueryRouter()

    print("\n── Layer 1 · Retrieval  [no LLM] ───────────────────────────")
    retrieval = run_retrieval_eval(test_set, store)
    print(f"   Evaluated {retrieval['n_evaluated']} document queries")

    if skip_generation:
        print("\n── Layer 2 · Generation  [SKIPPED — run without --skip-generation when tokens reset]")
        generation = _GENERATION_SKIPPED
    else:
        print("\n── Layer 2 · Generation (Groq-as-judge) ────────────────────")
        print("   Calling Groq 4× per question (generate + 3 judges) — may take 2–3 min …")
        generation = run_generation_eval(test_set, store)
        print(f"   Evaluated {generation['n_evaluated']} questions")

    print("\n── Layer 3 · Routing  [no LLM] ─────────────────────────────")
    routing = run_routing_eval(test_set, router)
    print(f"   Evaluated {routing['n_evaluated']} queries")

    if skip_generation:
        print("\n── Layer 4 · System  [SKIPPED — requires LLM] ──────────────")
        system = _SYSTEM_SKIPPED
    else:
        print("\n── Layer 4 · System performance ────────────────────────────")
        system = run_system_eval(test_set, store)
        print(f"   Sampled {system['total_queries']} end-to-end queries for latency")

    return {
        "timestamp":          datetime.now().isoformat(),
        "test_set_size":      len(test_set),
        "document_ingested":  Path(pdf_path).name,
        "chunk_count":        info["chunk_count"],
        "retrieval":          retrieval,
        "generation":         generation,
        "routing":            routing,
        "system":             system,
        "cost_per_query_usd": _estimate_cost_per_query(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _ok(val: float, threshold: float, lower_is_better: bool = False) -> str:
    passed = val <= threshold if lower_is_better else val >= threshold
    return "✅" if passed else "⚠️ "


def print_scorecard(results: Dict) -> None:
    r  = results["retrieval"]
    g  = results["generation"]
    ro = results["routing"]
    s  = results["system"]

    print("\n" + "=" * 62)
    print("          RAG SYSTEM EVALUATION SCORECARD")
    print(f"          {results['timestamp'][:10]}  ·  {results['document_ingested']}")
    print("=" * 62)

    print(f"\n📥  RETRIEVAL   (n={r['n_evaluated']})")
    print(f"    Recall@5    : {r['recall@5']:.3f}  {_ok(r['recall@5'], 0.80)}  target >0.80")
    print(f"    Precision@5 : {r['precision@5']:.3f}  {_ok(r['precision@5'], 0.70)}  target >0.70")
    print(f"    MRR         : {r['mrr']:.3f}  {_ok(r['mrr'], 0.75)}  target >0.75")
    print(f"    NDCG@5      : {r['ndcg@5']:.3f}  {_ok(r['ndcg@5'], 0.78)}  target >0.78")
    if r.get("retrieval_failures"):
        print(f"    Zero-recall queries: {len(r['retrieval_failures'])}")

    if g.get("skipped"):
        print(f"\n✍️   GENERATION  — skipped (token limit). Run without --skip-generation to evaluate.")
    else:
        print(f"\n✍️   GENERATION  (n={g['n_evaluated']})")
        print(f"    Faithfulness     : {g['faithfulness']:.3f}  {_ok(g['faithfulness'], 0.85)}  target >0.85")
        print(f"    Answer Relevancy : {g['answer_relevancy']:.3f}  {_ok(g['answer_relevancy'], 0.85)}  target >0.85")
        print(f"    Answer Correct.  : {g['answer_correctness']:.3f}  {_ok(g['answer_correctness'], 0.80)}  target >0.80")

    print(f"\n🔀  ROUTING    (n={ro['n_evaluated']})")
    print(f"    Overall Accuracy  : {ro['overall_accuracy']:.3f}  {_ok(ro['overall_accuracy'], 0.90)}  target >0.90")
    print(f"    False Retrieval   : {ro['false_retrieval_rate']:.3f}  {_ok(ro['false_retrieval_rate'], 0.10, True)}  target <0.10")
    if ro.get("per_class"):
        for cls, m in ro["per_class"].items():
            print(f"    {cls:<12} F1 : {m['f1']:.3f}")
    if ro.get("routing_failures"):
        print(f"    Routing failures  : {len(ro['routing_failures'])}")
        for f in ro["routing_failures"][:3]:
            print(f"      ❌ [{f['expected']}→{f['predicted']}] {f['question'][:55]}")

    if s.get("skipped"):
        print(f"\n⚡  SYSTEM     — skipped (token limit). Run without --skip-generation to evaluate.")
    else:
        print(f"\n⚡  SYSTEM     (n={s['total_queries']})")
        print(f"    Latency p50  : {s['latency_p50']}s  {_ok(2.0 - s['latency_p50'], 0)}  target <2s")
        print(f"    Latency p95  : {s['latency_p95']}s  {_ok(4.0 - s['latency_p95'], 0)}  target <4s")
        print(f"    Error Rate   : {s['error_rate']:.1%}  {_ok(s['error_rate'], 0.02, True)}  target <2%")
        print(f"    Cost/query   : ${results['cost_per_query_usd']:.5f}  (Groq estimate)")

    print("\n" + "=" * 62)
    print("📄  CV-READY SUMMARY")
    print("-" * 62)
    faith_str = f"{g['faithfulness']:.2f}" if not g.get("skipped") else "pending"
    rel_str   = f"{g['answer_relevancy']:.2f}" if not g.get("skipped") else "pending"
    corr_str  = f"{g['answer_correctness']:.2f}" if not g.get("skipped") else "pending"
    p95_str   = f"{s['latency_p95']}s" if not s.get("skipped") else "pending"
    err_str   = f"{s['error_rate']:.1%}" if not s.get("skipped") else "pending"
    print(
        f"Built and evaluated a multi-agent RAG system (PDF + web + general chat)\n"
        f"with automated 4-layer evaluation pipeline:\n"
        f"  · Faithfulness: {faith_str}  |  Answer relevancy: {rel_str}  |  Correctness: {corr_str}\n"
        f"  · Retrieval Recall@5: {r['recall@5']:.2f}  |  MRR: {r['mrr']:.2f}  |  NDCG@5: {r['ndcg@5']:.2f}\n"
        f"  · Intent router: {ro['overall_accuracy']:.0%} accuracy across document / web / general routes\n"
        f"  · p95 latency: {p95_str}  |  error rate: {err_str}  |  cost/query: ${results['cost_per_query_usd']:.5f}"
    )
    print("=" * 62)


def save_results(results: Dict, run_dir: str) -> str:
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    ts = results["timestamp"].replace(":", "-").replace(".", "-")
    path = f"{run_dir}/eval_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    return path


def compare_runs(run_dir: str, last_n: int = 5) -> None:
    """Load last N eval runs and print metric trends."""
    files = sorted(Path(run_dir).glob("eval_*.json"))[-last_n:]
    if not files:
        print("No eval runs found.")
        return

    rows = []
    for path in files:
        with open(path, encoding="utf-8") as fp:
            res = json.load(fp)
        rows.append(
            {
                "date":         res["timestamp"][:10],
                "faithfulness": res["generation"].get("faithfulness"),
                "recall@5":     res["retrieval"].get("recall@5"),
                "router_acc":   res["routing"].get("overall_accuracy"),
                "p95_lat":      res["system"].get("latency_p95"),
            }
        )

    header = f"{'date':<12} {'faith':>7} {'rec@5':>7} {'rte_acc':>8} {'p95_lat':>8}"
    print("\nMetric trends (last", len(rows), "runs):")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['date']:<12} "
            f"{row['faithfulness'] or 'N/A':>7} "
            f"{row['recall@5'] or 'N/A':>7} "
            f"{row['router_acc'] or 'N/A':>8} "
            f"{str(row['p95_lat']) + 's' or 'N/A':>8}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="4-Layer RAG Evaluation Runner")
    parser.add_argument(
        "--pdf",
        default=str(PROJECT_ROOT / "AmbedkarAnnihilationofCastes.pdf"),
        help="Path to the evaluation document (PDF)",
    )
    parser.add_argument(
        "--test-set",
        default=str(Path(__file__).parent / "test_set.json"),
        help="Path to test_set.json",
    )
    parser.add_argument(
        "--output-dir",
        default=str(BACKEND_ROOT / "eval_runs"),
        help="Directory to save JSON results",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Print metric trends from past eval runs and exit",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip Layer 2 (generation) and Layer 4 (system) — zero LLM calls, safe to run right now",
    )
    args = parser.parse_args()

    if args.history:
        compare_runs(args.output_dir)
        return

    if not Path(args.pdf).exists():
        print(f"❌  PDF not found: {args.pdf}")
        sys.exit(1)

    if args.skip_generation:
        print("⚡  --skip-generation mode: Layer 1 (Retrieval) + Layer 3 (Routing) only. No LLM calls.")

    results = run_full_evaluation(args.pdf, args.test_set, skip_generation=args.skip_generation)
    print_scorecard(results)
    saved = save_results(results, args.output_dir)
    print(f"\n💾  Results saved → {saved}")


if __name__ == "__main__":
    main()

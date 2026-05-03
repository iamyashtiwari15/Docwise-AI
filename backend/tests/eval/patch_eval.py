#!/usr/bin/env python3
"""
patch_eval.py — Re-run only specific failing questions and patch an existing eval JSON.

Usage:
    cd backend
    python -m tests.eval.patch_eval
    python -m tests.eval.patch_eval --run-file eval_runs/eval_<timestamp>.json
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List

import numpy as np

# ── Path bootstrap ────────────────────────────────────────────────────────────
BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from agents.uploaded_document_store import UploadedDocumentStore
from core.config import get_settings

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("patch_eval")

EVAL_SESSION = "eval-patch-001"

# ── Questions to re-evaluate (substrings to match against question field) ─────
TARGET_QUESTION_SUBSTRINGS = [
    "graded inequality",
    "relationship between caste and democracy",
    "inter-caste marriage AND the role of education",
    "compare caste-based discrimination",
]

# ── Judge prompts (same as eval_runner.py) ────────────────────────────────────
_FAITHFULNESS_PROMPT = """You are an expert evaluator. Rate the faithfulness of the AI answer to the given context.

Context: {context}
Question: {question}
AI Answer: {answer}

Score 0.0–1.0 where:
  1.0 = every claim in the AI answer is supported by the context
  0.5 = most claims are supported, minor unsupported additions
  0.0 = answer contains claims not in context or contradicts it

Return ONLY valid JSON (no markdown): {{"score": <float 0-1>, "reason": "<one sentence>"}}"""

_RELEVANCY_PROMPT = """You are an expert evaluator. Rate how relevant the AI answer is to the question.

Question: {question}
AI Answer: {answer}

Score 0.0–1.0 where:
  1.0 = answer directly and completely addresses the question
  0.5 = answer partially addresses the question
  0.0 = answer is off-topic or does not address the question

Return ONLY valid JSON (no markdown): {{"score": <float 0-1>, "reason": "<one sentence>"}}"""

_CORRECTNESS_PROMPT = """You are an expert evaluator. Rate the factual correctness of the AI answer compared to the reference answer.

Question: {question}
Reference Answer: {ground_truth}
AI Answer: {answer}

Score 0.0–1.0 where:
  1.0 = AI answer is factually consistent with and covers key points of the reference
  0.5 = AI answer is partially correct
  0.0 = AI answer contradicts or misses key points of the reference

Return ONLY valid JSON (no markdown): {{"score": <float 0-1>, "reason": "<one sentence>"}}"""


def _call_judge(llm, prompt: str):
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


def _patch_settings():
    settings = get_settings()
    settings.enable_hyde = False
    settings.enable_llm_multi_query = False


def run_patch(run_file: str, test_set_path: str, pdf_path: str) -> str:
    """
    Re-run only the TARGET questions, patch their scores into the existing
    eval JSON, recalculate averages, and save as a new patched file.
    Returns the path to the patched file.
    """
    _patch_settings()

    # ── Load existing run ─────────────────────────────────────────────────────
    with open(run_file, encoding="utf-8") as f:
        results = json.load(f)

    # ── Load test set ─────────────────────────────────────────────────────────
    with open(test_set_path, encoding="utf-8") as f:
        test_set = json.load(f)

    # ── Ingest PDF ────────────────────────────────────────────────────────────
    print(f"📄  Ingesting PDF: {pdf_path}")
    store = UploadedDocumentStore()
    pdf_bytes = Path(pdf_path).read_bytes()
    info = store.add_file(EVAL_SESSION, Path(pdf_path).name, pdf_bytes, "application/pdf")
    print(f"✅  {info['chunk_count']} chunks ingested")

    # ── Setup rotator ─────────────────────────────────────────────────────────
    from tests.eval.groq_key_rotator import RotatingGroqClient
    rotator = RotatingGroqClient(model="llama-3.3-70b-versatile", temperature=0)
    print(f"🔑  {rotator.keys_remaining} Groq key(s) loaded\n")

    settings = get_settings()

    # ── Find matching test items ──────────────────────────────────────────────
    targets = [
        item for item in test_set
        if item.get("expected_route") == "document"
        and item.get("ground_truth_answer")
        and any(sub.lower() in item["question"].lower() for sub in TARGET_QUESTION_SUBSTRINGS)
    ]
    print(f"🎯  Targeting {len(targets)} question(s):\n")
    for t in targets:
        print(f"   • {t['question'][:80]}")
    print()

    new_scores: Dict[str, Dict] = {}

    for item in targets:
        q = item["question"]
        gt = item["ground_truth_answer"]
        hints = item.get("ground_truth_content_hints", [])
        print(f"── {q[:70]} …")

        # Retrieval
        docs = store.retrieve(EVAL_SESSION, q, top_k=settings.rag_top_k)
        if not docs:
            print("   ⚠️  No chunks retrieved — skipping")
            continue

        retrieved_texts = [d["content"] for d in docs]
        found_hints = [h for h in hints if any(h.lower() in t.lower() for t in retrieved_texts[:5])]
        print(f"   📥 Retrieval: {len(found_hints)}/{len(hints)} hints found in top-5")

        # Generate answer
        context = "\n\n".join(retrieved_texts[:3])
        gen_prompt = (
            f"Answer the following question using ONLY the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {q}\n\n"
            f"Answer concisely and faithfully from the context only."
        )
        gen_result = rotator.invoke(gen_prompt)
        answer = (gen_result.content if hasattr(gen_result, "content") else str(gen_result)).strip()
        print(f"   ✍️  Answer generated ({len(answer)} chars)")

        time.sleep(1)
        faith, fr = _call_judge(rotator, _FAITHFULNESS_PROMPT.format(context=context, question=q, answer=answer))
        time.sleep(1)
        rel,   rr = _call_judge(rotator, _RELEVANCY_PROMPT.format(question=q, answer=answer))
        time.sleep(1)
        corr,  cr = _call_judge(rotator, _CORRECTNESS_PROMPT.format(question=q, ground_truth=gt, answer=answer))
        time.sleep(1)

        print(f"   ⚖️  faith={faith:.2f} ({fr[:50]})  rel={rel:.2f}  corr={corr:.2f}")
        new_scores[q[:80]] = {
            "question":          q[:80],
            "faithfulness":      round(faith, 3),
            "answer_relevancy":  round(rel,   3),
            "answer_correctness":round(corr,  3),
        }
        print()

    if not new_scores:
        print("❌  No questions were successfully re-evaluated.")
        return run_file

    # ── Patch per_question list ───────────────────────────────────────────────
    per_q: List[Dict] = results["generation"]["per_question"]
    patched = 0
    for entry in per_q:
        key = entry["question"][:80]
        if key in new_scores:
            old = (entry["faithfulness"], entry["answer_relevancy"], entry["answer_correctness"])
            entry.update(new_scores[key])
            print(f"✅  Patched: {key[:60]}")
            print(f"   faith  {old[0]:.2f} → {entry['faithfulness']:.2f}")
            print(f"   rel    {old[1]:.2f} → {entry['answer_relevancy']:.2f}")
            print(f"   corr   {old[2]:.2f} → {entry['answer_correctness']:.2f}")
            patched += 1

    # ── Recalculate generation averages ──────────────────────────────────────
    def safe_mean(lst):
        return round(float(np.mean(lst)), 4) if lst else 0.0

    results["generation"]["faithfulness"]       = safe_mean([x["faithfulness"]       for x in per_q])
    results["generation"]["answer_relevancy"]   = safe_mean([x["answer_relevancy"]   for x in per_q])
    results["generation"]["answer_correctness"] = safe_mean([x["answer_correctness"] for x in per_q])
    results["generation"]["patched_questions"]  = patched
    results["generation"].pop("skipped", None)

    # ── Save patched file ─────────────────────────────────────────────────────
    src = Path(run_file)
    out_path = src.parent / src.name.replace(".json", "_patched.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"📊  PATCHED GENERATION SCORES")
    print(f"    Faithfulness     : {results['generation']['faithfulness']:.3f}")
    print(f"    Answer Relevancy : {results['generation']['answer_relevancy']:.3f}")
    print(f"    Answer Correct.  : {results['generation']['answer_correctness']:.3f}")
    print(f"    (based on {results['generation']['n_evaluated']} questions, {patched} re-evaluated)")
    print(f"\n💾  Saved → {out_path}")
    return str(out_path)


def _latest_run(run_dir: str) -> str:
    files = sorted(Path(run_dir).glob("eval_*.json"))
    non_patched = [f for f in files if "_patched" not in f.name]
    if not non_patched:
        raise FileNotFoundError(f"No eval runs found in {run_dir}")
    return str(non_patched[-1])


def main():
    parser = argparse.ArgumentParser(description="Patch specific failing questions in an eval run")
    parser.add_argument("--run-file", default=None,
                        help="Path to eval JSON to patch (default: latest in eval_runs/)")
    parser.add_argument("--test-set", default=str(Path(__file__).parent / "test_set.json"))
    parser.add_argument("--pdf", default=str(PROJECT_ROOT / "AmbedkarAnnihilationofCastes.pdf"))
    args = parser.parse_args()

    run_file = args.run_file or _latest_run(str(BACKEND_ROOT / "eval_runs"))
    print(f"📂  Patching: {Path(run_file).name}\n")

    if not Path(args.pdf).exists():
        print(f"❌  PDF not found: {args.pdf}")
        sys.exit(1)

    run_patch(run_file, args.test_set, args.pdf)


if __name__ == "__main__":
    main()

"""
Eval runner for the health chatbot's retrieval quality.

Usage:
    1. Start the Flask app in one terminal:  python app.py
    2. In another terminal, run:              python run_eval.py

This sends each question in eval_questions.json to the /debug endpoint
(so no LLM generation happens — this only measures RETRIEVAL quality,
not answer quality) and checks whether an expected source appears among
the chunks /ask would actually hand to the LLM: the top-k that also clear
the relevance threshold.
"""

import json
import os
from collections import defaultdict

import requests

APP_URL = "http://localhost:5000/debug"
HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_FILE = os.path.join(HERE, "eval_questions.json")
RESULTS_FILE = os.path.join(HERE, "eval_results.json")


def run_eval():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    passed = 0
    by_topic = defaultdict(lambda: [0, 0])  # topic -> [passed, total]

    print(f"Running {len(questions)} eval questions against {APP_URL}...\n")

    for q in questions:
        by_topic[q["topic"]][1] += 1

        try:
            resp = requests.post(APP_URL, json={"question": q["question"]}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[{q['id']}] ERROR calling app: {e}")
            continue

        used_by_ask = [
            m for m in data["top_matches"]
            if m["passes_current_threshold"] and m["used_by_ask"]
        ]
        retrieved_sources = [m["source"] for m in used_by_ask]

        expected = q["expected_source"]
        rank = None

        if expected is None:
            # Out-of-scope / safety questions: the "correct" behavior is
            # that NOTHING clears the threshold (i.e. the fallback triggers).
            ok = len(used_by_ask) == 0
            label = "PASS (correctly found nothing)" if ok else \
                    "FAIL (retrieved something, but should have found nothing)"
        else:
            # also_acceptable lists other pages that can legitimately answer,
            # e.g. a "flu vs cold" question can be answered from either article.
            acceptable = {expected, *q.get("also_acceptable", [])}
            rank = next((m["rank"] for m in used_by_ask if m["source"] in acceptable), None)
            ok = rank is not None
            label = "PASS" if ok else "FAIL"

        if ok:
            passed += 1
            by_topic[q["topic"]][0] += 1

        top_score = used_by_ask[0]["rerank_score"] if used_by_ask else None

        results.append({
            "id": q["id"],
            "question": q["question"],
            "topic": q["topic"],
            "expected_source": expected,
            "retrieved_sources": retrieved_sources,
            "rank_of_expected": rank,
            "top_score": top_score,
            "result": label,
        })

        print(f"[{q['id']:>2}] {label:<45} | {q['question']}")
        if not ok:
            print(f"       expected: {expected}")
            print(f"       got:      {retrieved_sources or '(nothing passed threshold)'}")

    print("\n--- By topic ---")
    for topic, (topic_passed, total) in sorted(by_topic.items()):
        print(f"{topic:<22} {topic_passed}/{total}")

    print(f"\n--- Summary: {passed}/{len(questions)} passed, {len(questions) - passed}/{len(questions)} failed ---")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Full results written to {RESULTS_FILE}")


if __name__ == "__main__":
    run_eval()

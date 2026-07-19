"""
Eval runner for the health chatbot's retrieval quality.

Usage:
    1. Start the Flask app in one terminal:  python health_chatbot.py
    2. In another terminal, run:              python run_eval.py

This sends each question in eval_questions.json to the /debug endpoint
(so no LLM generation happens — this only measures RETRIEVAL quality,
not answer quality) and checks whether the expected source appears
among the results that clear the current relevance threshold.
"""

import json
import requests

APP_URL = "http://localhost:5000/debug"
EVAL_FILE = "eval_questions.json"


def run_eval():
    with open(EVAL_FILE, "r") as f:
        questions = json.load(f)

    results = []
    passed = 0
    failed = 0

    print(f"Running {len(questions)} eval questions against {APP_URL}...\n")

    for q in questions:
        try:
            resp = requests.post(APP_URL, json={"question": q["question"]}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[{q['id']}] ERROR calling app: {e}")
            failed += 1
            continue

        matches_above_threshold = [
            m for m in data["top_matches"] if m["passes_current_threshold"]
        ]
        retrieved_sources = {m["source"] for m in matches_above_threshold}

        expected = q["expected_source"]

        if expected is None:
            # Out-of-scope / safety questions: the "correct" behavior is
            # that NOTHING clears the threshold (i.e. the fallback triggers).
            ok = len(matches_above_threshold) == 0
            label = "PASS (correctly found nothing)" if ok else \
                    "FAIL (retrieved something, but should have found nothing)"
        else:
            ok = expected in retrieved_sources
            label = "PASS" if ok else "FAIL"

        if ok:
            passed += 1
        else:
            failed += 1

        top_score = matches_above_threshold[0]["score"] if matches_above_threshold else None

        results.append({
            "id": q["id"],
            "question": q["question"],
            "topic": q["topic"],
            "expected_source": expected,
            "retrieved_sources": list(retrieved_sources),
            "top_score": top_score,
            "result": label,
        })

        print(f"[{q['id']:>2}] {label:<45} | {q['question']}")
        if not ok:
            print(f"       expected: {expected}")
            print(f"       got:      {list(retrieved_sources) or '(nothing passed threshold)'}")

    print(f"\n--- Summary: {passed}/{len(questions)} passed, {failed}/{len(questions)} failed ---")

    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Full results written to eval_results.json")


if __name__ == "__main__":
    run_eval()
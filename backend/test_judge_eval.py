"""Tests for judge_eval.py that run offline with no API key. python -m unittest test_judge_eval"""

import json
import unittest

import anthropic
import httpx2 as httpx

import judge_eval as j


def verdict(overall="tie", **dimensions):
    base = {"reasoning": "r", "faithfulness": "tie", "relevance": "tie",
            "completeness": "tie", "safety": "tie", "overall": overall}
    return {**base, **dimensions}


def run(answers):
    return {"answers": {k: {"answer": a, "question": "q", "context": "c"} for k, a in answers.items()}}


def rubric(claims=("supported",), relevance="addresses", completeness="complete", safety="safe", abstained=False):
    return {"claims": [{"claim": "c", "verdict": v, "evidence": ""} for v in claims],
            "abstained": abstained, "relevance": relevance, "relevance_reason": "",
            "completeness": completeness, "missing_points": [], "safety": safety, "safety_issues": []}


class OrderMapping(unittest.TestCase):
    def test_verdict_maps_back_through_display_order(self):
        self.assertEqual(j.to_pipeline("1", "ab"), "a")
        self.assertEqual(j.to_pipeline("1", "ba"), "b")
        self.assertEqual(j.to_pipeline("2", "ba"), "a")
        self.assertEqual(j.to_pipeline("tie", "ba"), "tie")

    def test_same_answer_winning_in_both_orders_is_a_real_win(self):
        # B is shown second in "ab" (verdict "2") and first in "ba" (verdict "1").
        summary = j.summarize_comparison(
            run({"1": "same size"}), run({"1": "same size!"}),
            {"1": {"ab": verdict("2"), "ba": verdict("1")}})
        self.assertEqual(summary["tallies"]["overall"]["b"], 1)
        self.assertEqual(summary["tallies"]["overall"]["flip"], 0)

    def test_always_picking_first_position_is_a_flip_not_a_win(self):
        summary = j.summarize_comparison(
            run({"1": "same size"}), run({"1": "same size!"}),
            {"1": {"ab": verdict("1"), "ba": verdict("1")}})
        overall = summary["tallies"]["overall"]
        self.assertEqual((overall["a"], overall["b"], overall["flip"]), (0, 0, 1))
        self.assertEqual(summary["position"]["first_shown_wins"], 2)


class BiasChecks(unittest.TestCase):
    def test_position_bias_is_flagged(self):
        verdicts = {str(i): {"ab": verdict("1"), "ba": verdict("1")} for i in range(10)}
        answers = {str(i): "x" for i in range(10)}
        summary = j.summarize_comparison(run(answers), run(answers), verdicts)
        self.assertEqual(summary["position"]["first_shown_wins"], 20)
        self.assertLess(summary["position"]["p_value"], 0.05)

    def test_longer_answer_wins_are_counted_either_way_round(self):
        long_text = "word " * 50
        summary = j.summarize_comparison(
            run({"1": long_text, "2": "short"}), run({"1": "short", "2": long_text}),
            {"1": {"ab": verdict("1"), "ba": verdict("2")},  # A wins, A is longer
             "2": {"ab": verdict("1"), "ba": verdict("2")}})  # A wins, A is shorter
        self.assertEqual(summary["length"]["decided_with_length_gap"], 2)
        self.assertEqual(summary["length"]["longer_wins"], 1)

    def test_similar_lengths_are_reported_separately(self):
        summary = j.summarize_comparison(
            run({"1": "one two three four five"}), run({"1": "one two three four six"}),
            {"1": {"ab": verdict("2"), "ba": verdict("1")}})
        self.assertEqual(summary["length"]["similar_length"]["b"], 1)
        self.assertEqual(summary["length"]["decided_with_length_gap"], 0)

    def test_identical_pairs_tie_without_counting_as_decided(self):
        summary = j.summarize_comparison(run({"1": "x"}), run({"1": "x"}), {"1": {"identical": True}})
        self.assertEqual(summary["tallies"]["overall"]["tie"], 1)
        self.assertEqual(summary["identical_pairs"], 1)
        self.assertEqual(summary["position"]["decided_judgements"], 0)


class SignTest(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(j.sign_test(0, 0), 1.0)
        self.assertEqual(j.sign_test(5, 5), 1.0)
        self.assertAlmostEqual(j.sign_test(10, 0), 2 / 1024)
        self.assertAlmostEqual(j.sign_test(0, 10), 2 / 1024)
        self.assertAlmostEqual(j.sign_test(8, 2), 2 * (1 + 10 + 45) / 1024)


class Scoring(unittest.TestCase):
    def test_faithfulness_is_share_of_supported_claims(self):
        self.assertEqual(j.faithfulness_of(rubric(("supported", "supported", "contradicted", "not_in_context"))), 0.5)

    def test_no_claims_means_no_faithfulness_score_not_a_perfect_one(self):
        self.assertIsNone(j.faithfulness_of(rubric(claims=(), abstained=True)))
        summary = j.summarize_scores({"1": rubric(claims=(), abstained=True), "2": rubric(("contradicted",))})
        self.assertEqual(summary["faithfulness"], 0.0)
        self.assertEqual(summary["abstained"], 1)

    def test_problem_answers_are_listed_by_dimension(self):
        summary = j.summarize_scores({
            "2": rubric(("not_in_context",)), "10": rubric(("contradicted",), safety="unsafe"),
            "3": rubric(safety="concern")})
        self.assertEqual(summary["with_unsupported_claims"], ["2", "10"])
        self.assertEqual(summary["with_contradicted_claims"], ["10"])
        self.assertEqual(summary["unsafe"], ["10"])
        self.assertEqual(summary["safety_concern"], ["3"])

    def test_not_applicable_completeness_is_left_out_of_the_average(self):
        summary = j.summarize_scores({"1": rubric(completeness="not_applicable"), "2": rubric(completeness="partial")})
        self.assertEqual(summary["completeness"], 0.5)


class Independence(unittest.TestCase):
    def test_same_family_judge_is_refused(self):
        with self.assertRaises(SystemExit):
            j.check_independence("claude-opus-5-5", [{"label": "x", "generator": "claude-haiku-5-5"}])
        with self.assertRaises(SystemExit):
            j.check_independence("gemma-3-27b", [{"label": "x", "generator": "gemma-3-1b-it-q4_0.gguf"}])

    def test_different_family_judge_is_allowed(self):
        j.check_independence("claude-opus-5-5", [{"label": "x", "generator": "gemma-3-1b-it-q4_0.gguf"}])


def fake_judge(handler):
    client = anthropic.Anthropic(api_key="test", max_retries=0,
                                 http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return j.Judge(client=client)


def message(text, stop_reason="end_turn", model=j.JUDGE_MODEL):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": text}], "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": 1000, "output_tokens": 500}}


class JudgeRequests(unittest.TestCase):
    """Checks what the installed SDK sends and parses, using a fake HTTP server."""

    def test_request_shape_and_parsing(self):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            seen["beta"] = request.headers.get("anthropic-beta", "")
            return httpx.Response(200, json=message(json.dumps(rubric())))

        judge = fake_judge(handler)
        result, served_by = judge.ask(j.SCORE_SYSTEM, "<question>q</question>", j.RubricScore)

        body = seen["body"]
        self.assertEqual(body["model"], "claude-opus-5-5")
        self.assertEqual(body["fallbacks"], "default")
        self.assertIn(j.FALLBACK_BETA, seen["beta"])
        self.assertEqual(body["output_config"]["effort"], "high")
        self.assertEqual(body["output_config"]["format"]["type"], "json_schema")
        self.assertIn("claims", body["output_config"]["format"]["schema"]["properties"])
        self.assertNotIn("temperature", body)
        self.assertEqual(result["safety"], "safe")
        self.assertEqual(served_by, j.JUDGE_MODEL)
        self.assertAlmostEqual(judge.cost(), (1000 * 4 + 500 * 20) / 1_000_000)

    def test_refusal_is_an_error_rather_than_a_verdict(self):
        judge = fake_judge(lambda r: httpx.Response(200, json=message("", stop_reason="refusal")))
        with self.assertRaises(ValueError):
            judge.ask(j.SCORE_SYSTEM, "x", j.RubricScore)

    def test_truncated_output_is_an_error_rather_than_a_verdict(self):
        judge = fake_judge(lambda r: httpx.Response(200, json=message('{"claims": [', stop_reason="max_tokens")))
        with self.assertRaises(ValueError):
            judge.ask(j.SCORE_SYSTEM, "x", j.RubricScore)


class StaleVerdicts(unittest.TestCase):
    def test_fingerprint_changes_with_the_answer_or_the_context(self):
        base = j.fingerprint({"answer": "a", "context": "c"})
        self.assertNotEqual(base, j.fingerprint({"answer": "a2", "context": "c"}))
        self.assertNotEqual(base, j.fingerprint({"answer": "a", "context": "c2"}))
        self.assertEqual(base, j.fingerprint({"answer": "a", "context": "c"}))


class Prompts(unittest.TestCase):
    def test_compare_shows_each_answer_with_its_own_context(self):
        text = j.compare_message("q", {"answer": "A1", "context": "C1"}, {"answer": "A2", "context": "C2"})
        self.assertLess(text.index("C1"), text.index("A1"))
        self.assertLess(text.index("A1"), text.index("C2"))
        self.assertLess(text.index("C2"), text.index("A2"))

    def test_compare_prompt_does_not_reveal_which_pipeline_is_which(self):
        text = j.compare_message("q", {"answer": "A1", "context": "C1", "sources": ["S"]},
                                 {"answer": "A2", "context": "C2", "sources": ["S"]})
        for word in ("rerank", "hybrid", "gemma", "pipeline"):
            self.assertNotIn(word, text.lower())


if __name__ == "__main__":
    unittest.main()

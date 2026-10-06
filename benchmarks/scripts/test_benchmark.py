#!/usr/bin/env python3
"""Offline streaming, token-count, TTFT and near-1M needle regressions."""

import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).with_name("benchmark.py")
SPEC = importlib.util.spec_from_file_location("v41_benchmark", SCRIPT)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class Response:
    status = 200

    def __init__(self, payload):
        self.content = io.BytesIO(payload)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.content.close()

    def __iter__(self):
        return iter(self.content)

    def read(self, size=-1):
        return self.content.read(size)


def stream(prompt_tokens, text="A measured answer.", include_usage=True, include_done=True):
    lines = [
        {"model": BENCH.MODEL, "choices": [{"delta": {"content": ""}}]},
        {"model": BENCH.MODEL, "choices": [{"delta": {"content": text}}]},
    ]
    if include_usage:
        lines.append({
            "model": BENCH.MODEL, "choices": [],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 10},
        })
    result = b"".join(b"data: " + json.dumps(line).encode() + b"\n\n" for line in lines)
    if include_done:
        result += b"data: [DONE]\n\n"
    return Response(result)


class V41BenchmarkTests(unittest.TestCase):
    def test_streamed_ttft_decode_and_prefill_are_finite(self):
        with patch.object(BENCH.urllib.request, "urlopen", return_value=stream(8000)):
            with patch.object(BENCH.time, "perf_counter", side_effect=[10.0, 10.25, 11.0]):
                result = BENCH.measure_one(
                    "http://example.test:8889", BENCH.MODEL, "example", 8000, 8192, 32, 60
                )
        self.assertEqual(result["prompt_tokens"], 8000)
        self.assertEqual(result["ttft_seconds"], 0.25)
        self.assertEqual(result["decode_tokens_per_second"], 12.0)
        self.assertEqual(result["effective_prefill_tokens_per_second"], 32000.0)

    def test_stream_without_actual_usage_or_done_is_not_a_success(self):
        for include_usage, include_done in ((False, True), (True, False)):
            with self.subTest(include_usage=include_usage, include_done=include_done):
                response = stream(8000, include_usage=include_usage, include_done=include_done)
                with patch.object(BENCH.urllib.request, "urlopen", return_value=response):
                    with patch.object(BENCH.time, "perf_counter", side_effect=[10.0, 10.25, 11.0]):
                        with self.assertRaisesRegex(BENCH.BenchmarkError, "without a generated token"):
                            BENCH.measure_one(
                                "http://example.test:8889", BENCH.MODEL, "example", 8000, 8192, 32, 60
                            )

    def test_tokenizer_response_must_report_a_real_count(self):
        response = Response(json.dumps({
            "count": 5, "max_model_len": BENCH.MAX_CONTEXT_TOKENS, "tokens": [1, 2]
        }).encode())
        with patch.object(BENCH.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(BENCH.BenchmarkError, "disagrees"):
                BENCH.tokenize_count("http://example.test", BENCH.MODEL, "example", 60)

    def test_calibrates_actual_near_million_needle_and_checks_answer(self):
        def count(_url, _model, prompt, _timeout):
            return 2 * prompt.count("token") + 42, BENCH.MAX_CONTEXT_TOKENS

        with patch.object(BENCH, "tokenize_count", side_effect=count):
            prompt, actual, maximum = BENCH.calibrate_prompt(
                "http://example.test", BENCH.MODEL, BENCH.MAX_CONTEXT_TOKENS,
                64, "prose", 0.4, 60,
            )
        self.assertEqual(maximum, BENCH.MAX_CONTEXT_TOKENS)
        self.assertGreater(actual, int(maximum * 0.95))
        self.assertIn(BENCH.NEEDLE_CODE, prompt)
        with patch.object(BENCH.urllib.request, "urlopen",
                          return_value=stream(actual, BENCH.NEEDLE_CODE)):
            with patch.object(BENCH.time, "perf_counter", side_effect=[10.0, 12.0, 13.0]):
                result = BENCH.measure_one(
                    "http://example.test", BENCH.MODEL, prompt, actual,
                    BENCH.MAX_CONTEXT_TOKENS, 64, 60, needle_position=0.4,
                )
        self.assertTrue(result["needle_retrieved"])
        self.assertEqual(result["needle_code"], BENCH.NEEDLE_CODE)
        self.assertIn(BENCH.NEEDLE_CODE, result["needle_response_excerpt"])
        with patch.object(BENCH.urllib.request, "urlopen",
                          return_value=stream(actual, "guessed wrong")):
            with patch.object(BENCH.time, "perf_counter", side_effect=[10.0, 12.0, 13.0]):
                with self.assertRaisesRegex(BENCH.BenchmarkError, "did not retrieve"):
                    BENCH.measure_one(
                        "http://example.test", BENCH.MODEL, prompt, actual,
                        BENCH.MAX_CONTEXT_TOKENS, 64, 60, needle_position=0.4,
                    )

    def test_actual_usage_mismatch_rejects_approximated_context(self):
        with patch.object(BENCH.urllib.request, "urlopen", return_value=stream(4000)):
            with patch.object(BENCH.time, "perf_counter", side_effect=[10.0, 10.25, 11.0]):
                with self.assertRaisesRegex(BENCH.BenchmarkError, "disagrees"):
                    BENCH.measure_one(
                        "http://example.test", BENCH.MODEL, "example", 8000, 8192, 32, 60
                    )

    def test_metric_position_labels_remain_distinct(self):
        metrics = BENCH.summarize_metrics([
            "# HELP vllm:spec_acceptance accepted",
            'vllm:spec_acceptance{position="1"} 7',
            'vllm:spec_acceptance{position="2"} 2',
        ])
        self.assertEqual(len(metrics), 2)
        self.assertEqual(metrics['vllm:spec_acceptance{position="1"}'], 7.0)

    def test_near_million_report_contains_measured_values_and_metrics_deltas(self):
        with tempfile.TemporaryDirectory(prefix="v41-benchmark-test-") as directory:
            report_path = Path(directory) / "measured.json"
            result = {
                "status": 200, "context_budget_tokens": BENCH.MAX_CONTEXT_TOKENS,
                "tokenizer_prompt_tokens": 1048100, "prompt_tokens": 1048100,
                "completion_tokens": 10, "total_seconds": 4.0, "ttft_seconds": 2.0,
                "decode_tokens_per_second": 4.5, "effective_prefill_tokens_per_second": 524050.0,
                "needle_retrieved": True, "needle_code": BENCH.NEEDLE_CODE,
                "needle_response_excerpt": BENCH.NEEDLE_CODE,
            }
            argv = [
                "benchmark-v41-deepseek.py", "--context-token", "1048576",
                "--max-tokens", "128", "--needle-position", "0.4",
                "--samples", "1", "--prefill-only", "--json-out", str(report_path),
            ]
            with patch.object(BENCH.sys, "argv", argv):
                with patch.object(BENCH, "calibrate_prompt",
                                  return_value=("measured prompt", 1048100, BENCH.MAX_CONTEXT_TOKENS)):
                    with patch.object(BENCH, "measure_one", return_value=result):
                        with patch.object(BENCH, "fetch_metrics_lines", side_effect=[
                            ['vllm:spec_acceptance{position="1"} 7'],
                            ['vllm:spec_acceptance{position="1"} 9'],
                        ]):
                            with patch("builtins.print"):
                                self.assertEqual(BENCH.main(), 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["results"]["context_1048576"]["attempts"][0]["prompt_tokens"], 1048100)
            self.assertEqual(report["results"]["context_1048576"]["median_ttft_seconds"], 2.0)
            self.assertEqual(report["metrics_deltas"]['vllm:spec_acceptance{position="1"}'], 2.0)
            self.assertEqual(report["needle_position"], 0.4)

    def test_near_million_needle_rejects_a_shorter_context_budget(self):
        with patch.object(BENCH.sys, "argv", [
            "benchmark-v41-deepseek.py", "--context-token", "524288",
            "--needle-position", "0.4", "--json-out", "unused.json",
        ]):
            with self.assertRaisesRegex(BENCH.BenchmarkError, "actual 1M context"):
                BENCH.main()


if __name__ == "__main__":
    unittest.main()

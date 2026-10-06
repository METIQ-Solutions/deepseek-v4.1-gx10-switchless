#!/usr/bin/env python3
"""Token-calibrated, streaming DeepSeek V4.1 TP4 benchmark (stdlib-only).

Targets the cluster head API directly. Streamed output measures actual TTFT,
the decode interval, and effective prefill throughput including queue time. The
optional near-1M needle checks both the server-reported prompt tokens and the
answer itself. A measurement report is not production acceptance by itself;
see docs/benchmarks.md.
"""

import argparse
import concurrent.futures
import json
import math
import shlex
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

MODEL = "deepseek-v4.1-flash"
DEFAULT_CONCURRENCY = [1, 2, 4, 8]
DEFAULT_CONTEXT_TOKENS = [8192, 32768, 131072, 524288]
MAX_CONTEXT_TOKENS = 1048576
NEEDLE_CODE = "V41-NEEDLE-7E2B3D"
TASKS = {
    "coding": "After reading the records, write a short Python function returning the sum of two integers.",
    "prose": "After reading the records, summarize their repeated pattern in one sentence.",
    "reasoning": "After reading the records, explain briefly why 23 plus 19 equals 42.",
}


class BenchmarkError(Exception):
    pass


def _token_like_word(i: int) -> str:
    return f"token{i % 1000:03d}"


def build_prompt(word_count: int, task: str = "prose", needle_position: float | None = None) -> str:
    if word_count < 1 or task not in TASKS:
        raise BenchmarkError("a positive word count and known benchmark task are required")
    if needle_position is not None and not 0 < needle_position < 1:
        raise BenchmarkError("needle position must lie strictly inside the context")
    pieces = [_token_like_word(i) for i in range(word_count)]
    if needle_position is not None:
        pieces.insert(
            round(word_count * needle_position),
            f"ARCHIVE FACT: The one-time retrieval code is {NEEDLE_CODE}.",
        )
        question = f"Find the ARCHIVE FACT above. Respond with its exact one-time retrieval code."
    else:
        question = TASKS[task]
    return " ".join(pieces) + "\n" + question


def request_json(base_url: str, path: str, payload: dict, timeout: int):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                raise BenchmarkError(f"{path} returned HTTP {resp.status}")
            document = json.loads(resp.read(64 * 1024 * 1024 + 1))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"{path} failed: {error}") from error
    if not isinstance(document, dict):
        raise BenchmarkError(f"{path} did not return a JSON object")
    return document


def tokenize_count(base_url: str, model: str, prompt: str, timeout: int) -> tuple[int, int]:
    document = request_json(
        base_url,
        "/tokenize",
        {"model": model, "messages": [{"role": "user", "content": prompt}]},
        timeout,
    )
    count, maximum = document.get("count"), document.get("max_model_len")
    if type(count) is not int or count < 1 or type(maximum) is not int or maximum < 1:
        raise BenchmarkError("/tokenize did not report actual prompt count and model context")
    token_ids = document.get("tokens")
    if token_ids is not None and (not isinstance(token_ids, list) or len(token_ids) != count):
        raise BenchmarkError("/tokenize count disagrees with returned token IDs")
    return count, maximum


def calibrate_prompt(
    base_url: str,
    model: str,
    context_budget: int,
    max_tokens: int,
    task: str,
    needle_position: float | None,
    timeout: int,
) -> tuple[str, int, int]:
    if context_budget < 512 or max_tokens < 1 or context_budget > MAX_CONTEXT_TOKENS:
        raise BenchmarkError("context budget or completion length is outside the approved model target")
    target = context_budget - max_tokens - 64
    if target < 256:
        raise BenchmarkError("context must leave room for completion and chat framing")
    first, model_max = tokenize_count(base_url, model, build_prompt(32, task), timeout)
    second, other_max = tokenize_count(base_url, model, build_prompt(512, task), timeout)
    if other_max != model_max or model_max < context_budget or second <= first:
        raise BenchmarkError("tokenizer and model context limits are inconsistent")
    tokens_per_word = (second - first) / 480
    estimate = max(32, round(32 + (target - first) / tokens_per_word))
    tolerance = max(32, round(context_budget * 0.002))
    for _ in range(6):
        prompt = build_prompt(estimate, task, needle_position)
        observed, current_max = tokenize_count(base_url, model, prompt, timeout)
        if current_max != model_max:
            raise BenchmarkError("model context limit changed during token calibration")
        if abs(observed - target) <= tolerance and observed + max_tokens <= model_max:
            return prompt, observed, model_max
        estimate = max(32, round(estimate + (target - observed) / tokens_per_word))
    raise BenchmarkError(f"cannot calibrate an actual {context_budget}-token context safely")


def measure_one(
    base_url: str, model: str, prompt: str, expected_tokens: int,
    context_budget: int, max_tokens: int, timeout: int,
    needle_position: float | None = None,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token = None
    usage = None
    done = False
    content = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status != 200:
                raise BenchmarkError(f"chat completion returned HTTP {response.status}")
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                value = line[5:].strip()
                if value == "[DONE]":
                    done = True
                    break
                event = json.loads(value)
                if not isinstance(event, dict) or event.get("error"):
                    raise BenchmarkError(f"streaming completion error: {str(event)[:500]}")
                if event.get("model") not in (None, model):
                    raise BenchmarkError("streaming response served a different model")
                if event.get("usage") is not None:
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    text = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                    if text and first_token is None:
                        first_token = time.perf_counter()
                    if delta.get("content"):
                        content.append(delta["content"])
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, UnicodeError) as error:
        raise BenchmarkError(f"streaming chat completion failed: {error}") from error
    finished = time.perf_counter()
    if not done or first_token is None or not isinstance(usage, dict):
        raise BenchmarkError("stream ended without a generated token, [DONE], or token usage")
    prompt_tokens, completion_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if type(prompt_tokens) is not int or type(completion_tokens) is not int or completion_tokens < 1:
        raise BenchmarkError("stream did not report positive prompt and completion token counts")
    tolerance = max(32, round(context_budget * 0.02))
    if abs(prompt_tokens - expected_tokens) > tolerance or prompt_tokens + max_tokens > MAX_CONTEXT_TOKENS:
        raise BenchmarkError("actual inferred prompt length disagrees with calibration or 1M bound")
    if needle_position is not None and (
        context_budget != MAX_CONTEXT_TOKENS
        or prompt_tokens < math.floor(MAX_CONTEXT_TOKENS * 0.95)
        or NEEDLE_CODE not in "".join(content)
    ):
        raise BenchmarkError("near-1M needle inference did not retrieve the required code")
    ttft = first_token - started
    elapsed = finished - started
    decoding = finished - first_token
    if ttft <= 0 or decoding <= 0:
        raise BenchmarkError("stream timestamps are not strictly ordered")
    return {
        "status": 200,
        "context_budget_tokens": context_budget,
        "tokenizer_prompt_tokens": expected_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_seconds": round(elapsed, 3),
        "ttft_seconds": round(ttft, 3),
        "decode_tokens_per_second": round((completion_tokens - 1) / decoding, 3)
        if completion_tokens > 1 else None,
        "effective_prefill_tokens_per_second": round(prompt_tokens / ttft, 3),
        "needle_retrieved": needle_position is not None,
        "needle_code": NEEDLE_CODE if needle_position is not None else None,
        "needle_response_excerpt": "".join(content)[:160] if needle_position is not None else None,
    }


def fetch_metrics_lines(base_url: str) -> list[str]:
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=30) as resp:
            return resp.read().decode("utf-8", "replace").splitlines()
    except (urllib.error.URLError, TimeoutError) as error:
        raise BenchmarkError(f"cannot read V4.1 live /metrics: {error}") from error


def summarize_metrics(lines: list[str]) -> dict:
    metrics = {}
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{", 1)[0].strip()
        try:
            value = float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue
        wanted = ("draft", "acceptance", "speculate", "num_speculative", "e2e_latency", "prefill")
        if any(w in name for w in wanted):
            metrics[line.rsplit(" ", 1)[0]] = value
    return metrics


def run_memory_cmd(cmd: str) -> str | None:
    if not cmd:
        return None
    try:
        argv = shlex.split(cmd)
        if not argv:
            raise BenchmarkError("memory capture command is empty")
        return subprocess.check_output(argv, text=True, timeout=60).strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, ValueError) as error:
        raise BenchmarkError(f"memory capture command failed: {error}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8889", help="cluster head base URL")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--json-out", required=True, help="path to write the JSON report")
    parser.add_argument("--context-token", type=int, default=None, help="override total context budget (e.g. 1048576)")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--samples", type=int, default=3, help="repeat count per point")
    parser.add_argument("--concurrency", default="1,2,4,8", help="comma-separated concurrency levels")
    parser.add_argument(
        "--concurrency-context-token",
        type=int,
        default=32768,
        help="context budget for independent concurrent requests",
    )
    parser.add_argument("--task", choices=tuple(TASKS), default="prose")
    parser.add_argument("--needle-position", type=float, default=None, help="fractional needle position; requires --context-token 1048576")
    parser.add_argument("--timeout", type=int, default=7200, help="per-request timeout in seconds")
    parser.add_argument("--prefill-only", action="store_true", help="only run prefill/context points (C1)")
    parser.add_argument("--memory-cmd", default="", help="optional local command whose output is recorded; per-node collection is done by the operator, not over SSH from here")
    args = parser.parse_args()

    base_url = args.endpoint.rstrip("/")
    context_tokens = [args.context_token] if args.context_token else DEFAULT_CONTEXT_TOKENS
    concurrency = [int(x) for x in args.concurrency.split(",") if x.strip()]
    if (
        args.samples < 1
        or args.max_tokens < 1
        or args.timeout < 1
        or args.concurrency_context_token < 512
        or args.concurrency_context_token > MAX_CONTEXT_TOKENS
    ):
        raise BenchmarkError("samples, completion length, and timeout must all be positive")
    if not concurrency or any(count < 1 or count > 8 for count in concurrency):
        raise BenchmarkError("concurrency must contain only positive C1–C8 levels")
    if args.needle_position is not None and context_tokens != [MAX_CONTEXT_TOKENS]:
        raise BenchmarkError("needle-style acceptance must exercise the actual 1M context budget")

    report = {
        "endpoint": base_url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "samples": args.samples,
        "concurrency": concurrency,
        "concurrency_context_token": args.concurrency_context_token,
        "context_tokens": context_tokens,
        "task": args.task,
        "needle_position": args.needle_position,
        "methodology": "streaming TTFT; decode=(completion_tokens-1)/(finish-first); effective prefill=actual prompt tokens/TTFT including queue and transport",
        "results": {},
    }
    report["metrics_before"] = summarize_metrics(fetch_metrics_lines(base_url))

    for ctx in context_tokens:
        prompt, actual_tokens, _ = calibrate_prompt(
            base_url, args.model, ctx, args.max_tokens, args.task,
            args.needle_position, args.timeout,
        )
        points = [
            measure_one(
                base_url, args.model, prompt, actual_tokens, ctx,
                args.max_tokens, args.timeout, args.needle_position,
            )
            for _ in range(args.samples)
        ]
        report["results"][f"context_{ctx}"] = {
            "attempts": points,
            "median_ttft_seconds": round(statistics.median(p["ttft_seconds"] for p in points), 3),
            "median_effective_prefill_tokens_per_second": round(
                statistics.median(p["effective_prefill_tokens_per_second"] for p in points), 3
            ),
            "median_decode_tokens_per_second": round(
                statistics.median(p["decode_tokens_per_second"] for p in points if p["decode_tokens_per_second"] is not None), 3
            ) if any(p["decode_tokens_per_second"] is not None for p in points) else None,
            "median_total_seconds": round(statistics.median(p["total_seconds"] for p in points), 3),
        }

    if not args.prefill_only:
        concurrency_ctx = args.concurrency_context_token
        concurrency_prompt, tokens, _ = calibrate_prompt(
            base_url, args.model, concurrency_ctx, args.max_tokens, args.task, None, args.timeout
        )
        for c in concurrency:
            with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
                workers = list(
                    pool.map(lambda _: measure_one(
                        base_url, args.model, concurrency_prompt, tokens, concurrency_ctx,
                        args.max_tokens, args.timeout
                    ), range(c))
                )
            report["results"][f"concurrency_c{c}"] = {
                "context_tokens": concurrency_ctx,
                "attempts": workers,
                "median_decode_tokens_per_second": round(
                    statistics.median(p["decode_tokens_per_second"] for p in workers if p["decode_tokens_per_second"] is not None), 3
                ) if any(p["decode_tokens_per_second"] is not None for p in workers) else None,
            }

    report["metrics_after"] = summarize_metrics(fetch_metrics_lines(base_url))
    report["metrics_deltas"] = {
        name: round(value - report["metrics_before"][name], 6)
        for name, value in report["metrics_after"].items()
        if name in report["metrics_before"]
    }
    report["memory_capture"] = run_memory_cmd(args.memory_cmd)

    with open(args.json_out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (BenchmarkError, urllib.error.URLError, OSError, ValueError, ZeroDivisionError) as error:
        print(f"FAIL V4.1 benchmark: {error}", file=sys.stderr)
        sys.exit(1)

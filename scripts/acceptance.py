#!/usr/bin/env python3
"""Functional acceptance for the served DeepSeek V4.1 Flash deployment.

Stdlib only. Qualifies the cluster head API directly, with no downstream
gateway in the path, so a failure here is a failure of the model deployment
itself.

Verifies, in order:
  * served model identity via /v1/models;
  * that a bounded chat completion really generates at least one token and
    returns the expected marker;
  * optional tool/function calling (the serving command enables auto tool
    choice);
  * optional reasoning-effort levels, each of which must be accepted.

Usage:
  python3 acceptance.py --endpoint http://HEAD:8889 [--model NAME]
                        [--with-tool-call] [--with-reasoning-levels none,low,high,max]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

MARKER = "V41_SERVED_OK"


def request_json(base_url: str, path: str, payload: dict | None = None, timeout: int = 300):
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(f"{base_url}{path}", data=data, headers=headers,
                                     method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        try:
            document = json.load(error)
        except (json.JSONDecodeError, UnicodeDecodeError):
            document = None
        return error.code, document


def fail(message: str) -> None:
    print(f"FAIL {message}", file=sys.stderr)
    raise SystemExit(1)


SPECULATIVE_METRIC_HINTS = ("draft", "acceptance", "accept", "speculate", "num_speculative",
                            "num_accepted")


def speculative_counters(base_url: str, timeout: int) -> dict[str, float]:
    """Return the speculative-decoding counters currently exported by /metrics.

    Names are discovered rather than assumed, because the exact series differ
    between vLLM revisions. A deployment with DSpark disabled exports none of
    them, which is exactly the condition this check exists to detect.
    """
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=timeout) as response:
            lines = response.read().decode("utf-8", "replace").splitlines()
    except (urllib.error.URLError, TimeoutError) as error:
        fail(f"cannot read /metrics for the DSpark check: {error}")
    counters: dict[str, float] = {}
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        sample = line.rsplit(" ", 1)[0]
        name = sample.split("{", 1)[0].strip()
        if not any(hint in name for hint in SPECULATIVE_METRIC_HINTS):
            continue
        try:
            value = float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue
        if "total" in name or "count" in name:
            counters[sample] = value
    return counters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="cluster head base URL, e.g. http://10.20.0.11:8889")
    parser.add_argument("--model", default="deepseek-v4.1-flash")
    parser.add_argument("--with-tool-call", action="store_true")
    parser.add_argument(
        "--with-speculative",
        action="store_true",
        help="prove DSpark speculative decoding is actually running (counters move)",
    )
    parser.add_argument("--with-reasoning-levels", default="")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    endpoint = args.endpoint.rstrip("/")
    model = args.model

    status, document = request_json(endpoint, "/v1/models", timeout=args.timeout)
    if status != 200 or document is None:
        fail(f"model discovery returned {status}")
    model_ids = [item.get("id") for item in document.get("data", [])]
    if model not in model_ids:
        fail(f"served model {model} missing from /v1/models: {model_ids}")
    print(f"PASS model identity ({model} served; ids={model_ids})")

    status, document = request_json(endpoint, "/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": f"Reply with exactly {MARKER}."}],
        "temperature": 0,
        "max_tokens": 64,
    }, timeout=args.timeout)
    if status != 200 or document is None:
        fail(f"generation returned {status}")
    usage = document.get("usage") or {}
    if int(usage.get("completion_tokens", 0)) < 1:
        fail("generation did not report at least one completion token")
    choices = document.get("choices") or []
    content = (choices[0].get("message") or {}).get("content") or "" if choices else ""
    if MARKER not in content:
        fail("generation content did not contain the expected marker")
    print(f"PASS generation (completion_tokens={usage.get('completion_tokens')} "
          f"response_model={document.get('model')})")

    if args.with_tool_call:
        status, document = request_json(endpoint, "/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": "What is the weather in Paris? Call get_weather."}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }],
            "max_tokens": 128,
        }, timeout=args.timeout)
        if status != 200 or document is None:
            fail(f"tool-call smoke returned {status}")
        message = (document.get("choices") or [{}])[0].get("message") or {}
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            fail("tool-call smoke did not emit a tool call")
        print(f"PASS tool call ({tool_calls[0].get('function', {}).get('name')})")

    if args.with_speculative:
        # DSpark operation: require that the speculative-decode counters exist
        # and are moving, rather than trusting the configuration. A run with the
        # feature silently disabled still generates fine, so it must be proven.
        before = speculative_counters(endpoint, args.timeout)
        if not before:
            fail("no speculative-decoding metrics exported; DSpark may not be active")
        status, document = request_json(endpoint, "/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": "Write two short sentences about the sea."}],
            "temperature": 0,
            "max_tokens": 256,
        }, timeout=args.timeout)
        if status != 200 or document is None:
            fail(f"speculative smoke completion returned {status}")
        after = speculative_counters(endpoint, args.timeout)
        moved = {
            name: after.get(name, 0.0) - before.get(name, 0.0)
            for name in set(before) | set(after)
        }
        advanced = [name for name, delta in moved.items() if delta > 0]
        if not advanced:
            fail(f"speculative-decode counters did not advance: {moved}")
        print(f"PASS DSpark active ({len(advanced)} counters advanced, e.g. {sorted(advanced)[:2]})")

    for level in [item.strip() for item in args.with_reasoning_levels.split(",") if item.strip()]:
        status, document = request_json(endpoint, "/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": "Think step by step and answer briefly."}],
            "reasoning_effort": level,
            "max_tokens": 128,
        }, timeout=args.timeout)
        if status != 200 or document is None:
            fail(f"reasoning level '{level}' returned {status}")
        print(f"PASS reasoning level '{level}'")

    print("ACCEPTANCE_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

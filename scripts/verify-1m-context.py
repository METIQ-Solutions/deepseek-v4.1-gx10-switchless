#!/usr/bin/env python3
"""Hard near-1M context acceptance: a real 1,048,576-token request must work.

A model that merely *starts* with --max-model-len 1048576 has not demonstrated
anything. This driver creates a prompt that the model's own tokenizer measures
at 1,048,576 tokens (less the completion reserve), hides a unique retrieval code
inside the measured context at three positions, and requires the served model to
return that code every time.

It asserts, from the measured report rather than from the request:

  * the server still reports a 1,048,576-token context budget;
  * the actual inferred prompt is at least 95% of that budget;
  * the retrieval code came back in the completion.

This is deliberately a long-running check: a near-1M prefill takes many minutes
on four GB10 nodes, so the per-request timeout defaults to two hours.

Usage:
  python3 verify-1m-context.py --endpoint http://HEAD:8889 [--out DIR]
                              [--positions 0.25,0.5,0.75] [--max-tokens 128]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

MAX_CONTEXT_TOKENS = 1_048_576
NEEDLE_CODE = "V41-NEEDLE-7E2B3D"
MINIMUM_FRACTION = 0.95


def run_position(benchmark: Path, endpoint: str, model: str, position: float,
                 max_tokens: int, timeout: int, out_dir: Path) -> dict:
    report_path = out_dir / f"needle-{position:.2f}.json"
    command = [
        sys.executable, str(benchmark),
        "--endpoint", endpoint,
        "--model", model,
        "--context-token", str(MAX_CONTEXT_TOKENS),
        "--max-tokens", str(max_tokens),
        "--needle-position", str(position),
        "--samples", "1",
        "--prefill-only",
        "--timeout", str(timeout),
        "--json-out", str(report_path),
    ]
    print(f"--- needle at {position:.0%} of the context", flush=True)
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout + 600)
    if completed.returncode != 0:
        raise SystemExit(
            f"FAIL near-1M request at position {position} failed:\n{completed.stderr.strip()}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    attempt = report["results"][f"context_{MAX_CONTEXT_TOKENS}"]["attempts"][0]

    if attempt["context_budget_tokens"] != MAX_CONTEXT_TOKENS:
        raise SystemExit(f"FAIL position {position}: context budget was {attempt['context_budget_tokens']}")
    minimum = math.floor(MAX_CONTEXT_TOKENS * MINIMUM_FRACTION)
    if attempt["prompt_tokens"] < minimum:
        raise SystemExit(
            f"FAIL position {position}: only {attempt['prompt_tokens']} prompt tokens, need >= {minimum}"
        )
    if not attempt.get("needle_retrieved"):
        raise SystemExit(f"FAIL position {position}: the report does not record a retrieved needle")
    if attempt.get("needle_code") != NEEDLE_CODE:
        raise SystemExit(f"FAIL position {position}: reported needle code {attempt.get('needle_code')!r}")
    excerpt = attempt.get("needle_response_excerpt") or ""
    if NEEDLE_CODE not in excerpt:
        raise SystemExit(f"FAIL position {position}: the completion did not contain the retrieval code")
    return attempt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="cluster head base URL")
    parser.add_argument("--model", default="deepseek-v4.1-flash")
    parser.add_argument("--out", default=".", help="directory for the JSON reports")
    parser.add_argument("--positions", default="0.25,0.5,0.75")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=7200, help="per-request timeout in seconds")
    parser.add_argument(
        "--benchmark",
        default=str(Path(__file__).resolve().parent.parent / "benchmarks" / "scripts" / "benchmark.py"),
        help="path to benchmark.py",
    )
    args = parser.parse_args()

    benchmark = Path(args.benchmark)
    if not benchmark.is_file():
        raise SystemExit(f"FAIL benchmark driver not found: {benchmark}")

    positions = [float(item) for item in args.positions.split(",") if item.strip()]
    if not positions:
        raise SystemExit("FAIL at least one needle position is required")
    for position in positions:
        if not 0 < position < 1:
            raise SystemExit(f"FAIL needle position must be strictly inside the context: {position}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    attempts = []
    for position in positions:
        attempts.append((position, run_position(
            benchmark, args.endpoint.rstrip("/"), args.model, position,
            args.max_tokens, args.timeout, out_dir,
        )))

    print()
    print(f"{'position':>8} {'prompt_tokens':>14} {'ttft_s':>10} {'prefill_tok/s':>14}  code")
    for position, attempt in attempts:
        print(
            f"{position:>8.0%} {attempt['prompt_tokens']:>14} {attempt['ttft_seconds']:>10.1f} "
            f"{attempt['effective_prefill_tokens_per_second']:>14.1f}  {attempt['needle_code']}"
        )
    print()
    print(
        f"NEAR_1M_PASS positions={len(attempts)} context={MAX_CONTEXT_TOKENS} "
        f"minimum_prompt_tokens={math.floor(MAX_CONTEXT_TOKENS * MINIMUM_FRACTION)}"
    )
    print("note: effective prefill throughput includes HTTP, queueing and transport time")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Benchmarks and measurements

## Tooling

`benchmarks/scripts/benchmark.py` (stdlib only) drives the cluster head API
directly and measures:

* **TTFT** — first *generated* token, measured from request start to the first
  stream delta that carries content or reasoning content;
* **decode tokens/s** — `(completion_tokens - 1) / (finish - first_token)`, so
  the prefill is excluded rather than averaged into the rate;
* **effective prefill tokens/s** — `actual_prompt_tokens / TTFT`. This is
  **not** pure GPU prefill speed: it includes HTTP, queueing and transport time;
* **concurrency C1/C2/C4/C8** at a calibrated context;
* DSpark speculative metrics from `/metrics`, including per-position labels, and
  their deltas across the run.

Prompt lengths are calibrated against the server's own `/tokenize` endpoint
rather than assumed from repeated words, and each streamed completion must
report `usage` consistent with the calibration — a run that over- or
under-shoots the target context is rejected rather than reported.

```sh
python3 benchmarks/scripts/benchmark.py --endpoint http://HEAD:8889 \
  --task prose --samples 3 --json-out /tmp/prose.json
python3 benchmarks/scripts/benchmark.py --endpoint http://HEAD:8889 \
  --task coding --samples 3 --json-out /tmp/coding.json
python3 benchmarks/scripts/benchmark.py --endpoint http://HEAD:8889 \
  --task reasoning --samples 3 --json-out /tmp/reasoning.json
```

Offline regression tests for the driver itself (no server needed):

```sh
python3 -m unittest benchmarks.scripts.test_benchmark -v
# or, from benchmarks/scripts:
python3 -m unittest test_benchmark -v
```

## The hard near-1M test

`scripts/verify-1m-context.py` is the acceptance gate, not a benchmark. It
inserts a unique retrieval code inside a prompt the tokenizer measures at
1,048,576 tokens (less the completion reserve), at three positions, and requires
the model to return the code. It rejects a report where the context budget is
not exactly 1,048,576 or where the measured prompt is below 95% of it.

```sh
python3 scripts/verify-1m-context.py --endpoint http://HEAD:8889 --out ./reports
```

A near-1M prefill takes many minutes, so this is expected to be slow. A pass
here means the deployment genuinely serves its advertised context; a model that
only *starts* with `--max-model-len 1048576` proves nothing.

## Measured results

See [`../benchmarks/results/v4.1-gx10-tp4-switchless.md`](../benchmarks/results/v4.1-gx10-tp4-switchless.md)
for our measurements, the exact configuration they were taken with, and the
caveats. In short:

| Point | Prompt tokens | TTFT | Effective prefill | Decode |
| --- | --- | --- | --- | --- |
| 128K | 130,752 | 68.454 s | ≈1,910 tok/s | ≈35.24 tok/s |
| near-1M | 1,048,283 | 949.635 s | ≈1,103.88 tok/s | — |

The 8K/32K/512K points and the C1–C8 decode table are produced by the commands
above; they are not reproduced here because we do not have measured values for
them to publish.

## How to compare against published numbers

Different deployments of the same model differ in context length, KV cache
dtype, Engram backing store, speculative decoding settings and GPU clock state,
so a throughput comparison is only meaningful with all of those held constant.

* **Independent four-GB10 reference** (aidendle94, "Serving
  DeepSeek-V4.1-Flash on 4x DGX Spark", 2026-09): TP4, DCP4, DSpark k=5, 8K
  prefill chunks, 6 GiB KV per rank, 500K context — decode
  **90.7 / 63.5 / 31.3 tok/s** (counting/code/prose, fp8 Engram on NVMe) and
  87.3 / 66.4 / 27.1 tok/s (NVFP4 Engram over NFS). Needle retrieval at 32K and
  128K. This is a different context and different KV/Engram configuration from
  ours; it is a sanity band, not a like-for-like result.
* **Our near-1M cross-check**: an independent run of the same model with the
  same flags retrieved a needle at 993,435 prompt tokens with TTFT 982.4 s
  (≈1,011 tok/s) and measured ≈1,590 tok/s at a 48K prompt. Our 1,103.88 tok/s
  at 1,048,283 tokens sits in the same band.

## Measurement discipline that matters

These were learned the hard way and apply to any set of numbers you take with
this tooling:

1. **Check the GPUs are not latched in their slow state.** GB10 parts can drop
   into a slow clock state that moves results by up to ~1.5×. Measure clocks
   immediately before and after every block; discard a block from a latched
   node. Back-to-back counting runs have been observed at 62.2 then 92.2 tok/s
   on the same node.
2. **Record host memory as well as throughput.** Track `MemAvailable` and cgroup
   usage per rank; a run whose memory headroom collapsed is not comparable.
3. **Never claim an "independent prefill" block without checking prefix-cache
   metrics.** `vllm:prefix_cache_hits_total` must be unchanged across the block,
   otherwise a reused prefix, not the scheduler, produced the result.
4. **Effective prefill is not GPU prefill.** Always state that it includes
   HTTP, queueing and transport time.
5. **Repeat after a restart and after a warm long-running period.** A single
   cold run hides degradation.

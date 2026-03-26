# vLLM Metrics Reference

This file explains the Prometheus metrics commonly printed by VLMBench when you use:

```bash
python3 main.py bench --enable-prometheus-metrics ...
```

VLMBench compares snapshots before and after a run and reports metric differences (deltas), so values represent work done during your benchmark window.

## Cache Efficiency Metrics

### `vllm:prefix_cache_queries_total`

- Type: `counter`
- Meaning: number of queried prefix-cache tokens.
- Use it to estimate how often prompts attempted cache lookup.

### `vllm:prefix_cache_hits_total`

- Type: `counter`
- Meaning: number of prefix tokens served from cache.
- Higher is generally better for repeated/shared-prefix workloads.

## Token Volume Metrics

### `vllm:prompt_tokens_total`

- Type: `counter`
- Meaning: total prefill (input) tokens processed.
- Tracks input-side workload size.

### `vllm:generation_tokens_total`

- Type: `counter`
- Meaning: total generated output tokens.
- Tracks decode-side workload size.

## Latency Metrics

### `vllm:time_to_first_token_seconds`

- Type: `histogram`
- Meaning: time to first token (TTFT).
- Lower TTFT improves perceived responsiveness.

### `vllm:inter_token_latency_seconds`

- Type: `histogram`
- Meaning: per-token latency during generation.
- Lower values indicate faster token streaming.

### `vllm:request_time_per_output_token_seconds`

- Type: `histogram`
- Meaning: per-request average time per generated token.
- Useful for comparing decode efficiency across runs.

### `vllm:e2e_request_latency_seconds`

- Type: `histogram`
- Meaning: full end-to-end request latency.
- Includes all request phases.

### `vllm:request_inference_time_seconds`

- Type: `histogram`
- Meaning: time spent in RUNNING/inference phase.

### `vllm:request_prefill_time_seconds`

- Type: `histogram`
- Meaning: time spent in PREFILL phase.

### `vllm:request_decode_time_seconds`

- Type: `histogram`
- Meaning: time spent in DECODE phase.

## KV-Compute Metrics

### `vllm:request_prefill_kv_computed_tokens`

- Type: `histogram`
- Meaning: newly computed KV tokens during prefill (excluding cache hits).
- Lower values for repeated-prefix workloads usually indicate better cache reuse.

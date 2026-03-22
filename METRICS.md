# vLLM Metrics

These metrics will be fetched from vLLM Prometheus exporter.

```txt
1.
# HELP vllm:prefix_cache_queries_total Prefix cache queries, in terms of number of queried tokens.
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total 413.0

2.
# HELP vllm:prefix_cache_hits_total Prefix cache hits, in terms of number of cached tokens.
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total 96.0

3.
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total 413.0

4.
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total 10752.0

5.
# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_count 21.0
vllm:time_to_first_token_seconds_sum 0.5709736347198486

6.
# HELP vllm:inter_token_latency_seconds Histogram of inter-token latency in seconds.
# TYPE vllm:inter_token_latency_seconds histogram
vllm:inter_token_latency_seconds_count 10731.0
vllm:inter_token_latency_seconds_sum 62.844789976486936

7.
# HELP vllm:request_time_per_output_token_seconds Histogram of time_per_output_token_seconds per request.
# TYPE vllm:request_time_per_output_token_seconds histogram
vllm:request_time_per_output_token_seconds_count 21.0
vllm:request_time_per_output_token_seconds_sum 0.12298393341778265

8.
# HELP vllm:e2e_request_latency_seconds Histogram of e2e request latency in seconds.
# TYPE vllm:e2e_request_latency_seconds histogram
vllm:e2e_request_latency_seconds_count 21.0
vllm:e2e_request_latency_seconds_sum 63.41587209701538

9.
# HELP vllm:request_inference_time_seconds Histogram of time spent in RUNNING phase for request.
# TYPE vllm:request_inference_time_seconds histogram
vllm:request_inference_time_seconds_count 21.0
vllm:request_inference_time_seconds_sum 63.351715261349455

10.
# HELP vllm:request_prefill_time_seconds Histogram of time spent in PREFILL phase for request.
# TYPE vllm:request_prefill_time_seconds histogram
vllm:request_prefill_time_seconds_count 21.0
vllm:request_prefill_time_seconds_sum 0.5069252848625183

11.
# HELP vllm:request_decode_time_seconds Histogram of time spent in DECODE phase for request.
# TYPE vllm:request_decode_time_seconds histogram
vllm:request_decode_time_seconds_count 21.0
vllm:request_decode_time_seconds_sum 62.844789976486936

12.
# HELP vllm:request_prefill_kv_computed_tokens Histogram of new KV tokens computed during prefill (excluding cached tokens).
# TYPE vllm:request_prefill_kv_computed_tokens histogram
vllm:request_prefill_kv_computed_tokens_count 21.0
vllm:request_prefill_kv_computed_tokens_sum 317.0
```

# VLMBench

A scalable benchmarking framework for evaluating LLM inference performance via the OpenAI-compatible API. It is specifically designed for testing vLLM instances, supporting workloads from small micro-benchmarks (latency, token throughput) to large-scale stress tests (high concurrency, multi-GPU scaling). The system enables configurable experiments and detailed metric collection to analyze performance, scalability, and stability under different deployment conditions.

## Prereqs

* A running vLLM instance accessible via HTTP with OpenAI API available.
* Python 3.10+

## Install

```bash
./setup.sh
```

## Usage

```bash
# List available benchmarks
python main.py --list

# Run benchmarks against a vLLM endpoint
python main.py [--endpoint URL] [--model MODEL] [--data-dir DIR] [--clients N] benchmark1 [benchmark2 ...]

# Run warmup plugin only
python main.py --warmup --total-kv-tokens 8388608
```

### Options

* `--endpoint URL` — vLLM endpoint (default: `http://127.0.0.1:8080`)
* `--model MODEL` — Model name (auto-detected from endpoint if omitted)
* `--data-dir DIR` — Dataset cache directory (default: `./data`)
* `--stop-after N` — Stop after processing N entries (for quick testing; default: 0, meaning no limit)
* `--clients N` — Number of concurrent client workers (default: `1`)
* `--truncate` - Truncate input requests based on the maximum model len
* `--warmup` — Run warmup plugin before benchmarks (or standalone)
* `--total-kv-tokens N` — Total KV cache tokens (required with `--warmup`)
* Warmup estimates concurrency from `max_model_len` and `--total-kv-tokens` (no metrics polling)

### Examples

```bash
# Run with defaults (localhost:8080, auto-detect model)
python main.py narrativeqa humaneval

# Specify endpoint and model
python main.py --endpoint http://127.0.0.1:8080 --model facebook/opt-125m alpaca triviaqa

# Custom data directory
python main.py --data-dir /tmp/datasets narrativeqa

# Run with 10 concurrent clients
python main.py --clients 10 narrativeqa

# Warmup then run benchmarks
python main.py --warmup --total-kv-tokens 8388608 narrativeqa

# Warmup only
python main.py --warmup --total-kv-tokens 8388608
```

## Files

```txt
.
├── main.py              # Benchmark runner (CLI entry point)
├── benchmarks/          # Benchmark task implementations
├── dataloaders/         # Dataset loading utilities
├── src/                 # Core benchmark base classes
└── tasks/               # Task definitions
```

## Authors

* Alexander "Sasha" Joukov (<alexander.joukov@stonybrook.edu>)
* Amir Zadeh (<anajafizadeh@cs.stonybrook.edu>)

File Systems & Storage Lab @ Stony Brook University, 2026

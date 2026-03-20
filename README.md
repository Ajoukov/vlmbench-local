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
python3 main.py bench --list

# Run benchmarks against a vLLM endpoint
python3 main.py bench [--endpoint URL] [--model MODEL] [--data-dir DIR] [--clients N] benchmark1 [benchmark2 ...]

# List available plugins
python3 main.py plugin --list

# Run simulator plugin
python3 main.py plugin simulator --total-kv-tokens 8388608

# Show plugin-specific help
python3 main.py plugin simulator --help
```

### Options

* `--endpoint URL` — vLLM endpoint (default: `http://127.0.0.1:8080`)
* `--model MODEL` — Model name (auto-detected from endpoint if omitted)
* `--data-dir DIR` — Dataset cache directory (default: `./data`)

Bench command options (`python3 main.py bench ...`):

* `--list` — List available benchmarks
* `--stop-after N` — Stop after processing N entries (for quick testing; default: `0` means no limit)
* `--clients N` — Number of concurrent client workers (default: `1`)
* `--truncate` - Truncate input requests based on the maximum model len
* `--random-populate` - Sample benchmark entries randomly instead of sequentially
* `--seed N` - RNG seed for deterministic sampling in `--random-populate` mode

Plugin command options (`python3 main.py plugin ...`):

* `--list` — List available plugins
* `PLUGIN_NAME` — Plugin to run (currently: `simulator`)
* Each plugin defines its own arguments and help under its subcommand
* Use `python3 main.py plugin PLUGIN_NAME --help` to see plugin-specific options

### Examples

```bash
# Run with defaults (localhost:8080, auto-detect model)
python3 main.py bench narrativeqa humaneval

# Specify endpoint and model
python3 main.py bench --endpoint http://127.0.0.1:8080 --model facebook/opt-125m alpaca triviaqa

# Custom data directory
python3 main.py bench --data-dir /tmp/datasets narrativeqa

# Run with 10 concurrent clients
python3 main.py bench --clients 10 narrativeqa

# Deterministic random sampling across clients
python3 main.py bench --clients 100 --random-populate --seed 42 alpaca

# List plugins
python3 main.py plugin --list

# Simulator plugin
python3 main.py plugin simulator --total-kv-tokens 8388608

# Simulator plugin with task/source tuning
python3 main.py plugin simulator --total-kv-tokens 8388608 --source-type squad --task qa --utilization-perc 90

# Plugin-specific help
python3 main.py plugin simulator --help
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

# VLMBench

VLMBench is a command-line benchmarking framework for OpenAI-compatible vLLM instances.

It helps you run repeatable workloads, compare performance across configurations, and collect latency, throughput, and optional Prometheus metric deltas.

## Documentation Index

- [CLI Reference](CLI_REFERENCE.md): complete command, flag, and example guide.
- [Metrics Reference](METRICS.md): Prometheus metric definitions, interpretation, and tips.

## What You Can Do

- Run benchmark suites against a vLLM instance.
- Control concurrency and request population behavior.
- Truncate payloads to fit model context length.
- Run plugins for readiness checks and KV-cache simulation.
- Optionally collect Prometheus metrics before/after runs.

## Requirements

- Python 3.10+
- A reachable vLLM server that exposes OpenAI-compatible endpoints

## Installation

From the repository root:

```bash
bash scripts/setup.sh
```

You can also install dependencies manually with:

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# 1) List available benchmarks
python3 main.py bench --list

# 2) Run a benchmark with defaults
python3 main.py bench wmt16

# 3) List available plugins
python3 main.py plugin --list

# 4) Run readiness checks
python3 main.py plugin readiness --retrys 5
```

## Main CLI Commands

### Benchmark Command

```bash
python3 main.py bench [OPTIONS] BENCHMARK [BENCHMARK ...]
```

Commonly used options:

- `--endpoint`: vLLM base endpoint (also comes from `DEFAULT_ENDPOINT`)
- `--model`: model name; auto-detected from `/v1/models` if omitted
- `--data-dir`: dataset/cache directory (also comes from `DEFAULT_DATA_DIR`)
- `--clients`: number of concurrent clients
- `--truncate`: trim oversized payloads to fit detected model context
- `--stop-after`: stop after N entries per benchmark
- `--random-populate`: sample benchmark entries randomly
- `--seed`: deterministic sampling seed for random mode
- `--random-batch-size`: buffered random sampling batch size
- `--enable-prometheus-metrics`: collect and print Prometheus metric deltas

### Plugin Command

```bash
python3 main.py plugin [--list]
python3 main.py plugin PLUGIN_NAME [PLUGIN_OPTIONS]
```

Built-in plugins:

- `readiness`: checks endpoint health repeatedly
- `simulator`: synthetic KV-cache prefix-sharing workload

Use plugin help for full plugin arguments:

```bash
python3 main.py plugin readiness --help
python3 main.py plugin simulator --help
```

For a full CLI parameter guide with examples, see [CLI_REFERENCE.md](CLI_REFERENCE.md).

## Available Benchmarks

Current benchmark registry includes:

- `local_alpaca`
- `local_longbench_qmsum`
- `local_narrativeqa`
- `local_sharegpt`
- `wmt16`

List dynamically from your current checkout:

```bash
python3 main.py bench --list
```

## Example Workflows

### 1) Baseline single-client run

```bash
python3 main.py bench --endpoint http://127.0.0.1:8080 wmt16
```

### 2) Multi-client benchmark run

```bash
python3 main.py bench --clients 16 local_alpaca local_narrativeqa
```

### 3) Deterministic random population

```bash
python3 main.py bench \
    --clients 32 \
    --random-populate \
    --seed 42 \
    --random-batch-size 200 \
    local_sharegpt
```

### 4) Truncation + Prometheus metrics

```bash
python3 main.py bench \
    --truncate \
    --enable-prometheus-metrics \
    local_longbench_qmsum
```

### 5) KV-cache simulator

```bash
python3 main.py plugin simulator \
    --total-kv-tokens 8388608 \
    --source-type wikitext \
    --prefix-length-perc 70 \
    --n-runs 3 \
    --utilization-perc 90
```

## Environment Variables

- `DEFAULT_ENDPOINT`: overrides default endpoint
- `DEFAULT_DATA_DIR`: overrides default data/cache directory
- `REQUEST_TIMEOUT`: default per-request timeout used by benchmark runners
- `HF_HOME`: if set, it overrides `--data-dir` for cache location

## Project Layout

```txt
.
├── main.py                # CLI entry point
├── benchmarks/            # Benchmark definitions and registry
├── dataloaders/           # Dataset loader utilities
├── plugins/               # Plugin system (readiness, simulator)
├── src/                   # Core orchestration, runner, tokens, utils
├── tasks/                 # Task templates/types
├── METRICS.md             # Metric reference and interpretation notes
└── CLI_REFERENCE.md       # Complete CLI flag reference
```

## Notes

- If `--model` is omitted, VLMBench reads the first model from `/v1/models`.
- `--truncate` requires model max length detection from `/v1/models`.
- For plugin help forwarding, this works as expected:

```bash
python3 main.py plugin --help simulator
python3 main.py plugin simulator --help
```

## Authors

- Alexander "Sasha" Joukov (<alexander.joukov@stonybrook.edu>)
- Amir Zadeh (<anajafizadeh@cs.stonybrook.edu>)

File Systems & Storage Lab, Stony Brook University (2024-2026)

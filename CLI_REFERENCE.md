# VLMBench CLI Reference

This document lists all major CLI parameters and flags with practical examples.

## Command Overview

```bash
python3 main.py <command> [options]
```

Commands:

- `bench`: run one or more benchmarks
- `plugin`: run plugin tools (`readiness`, `simulator`)

## Global/Common Options

These are available for `bench` and plugin subcommands because they share common parser parents.

| Flag                          | Type   | Default                                         | Description                                                                      |
|-------------------------------|--------|-------------------------------------------------|----------------------------------------------------------------------------------|
| `--endpoint`                  | string | `http://127.0.0.1:8080` (or `DEFAULT_ENDPOINT`) | Base URL of the vLLM OpenAI-compatible server.                                   |
| `--model`                     | string | auto-detect                                     | Model id. If omitted, VLMBench queries `/v1/models` and uses the first model id. |
| `--data-dir`                  | path   | `llm-datasets` (or `DEFAULT_DATA_DIR`)          | Dataset/cache directory. If `HF_HOME` is set, it overrides this internally.      |
| `--enable-prometheus-metrics` | flag   | off                                             | Collects Prometheus counters/histograms before and after run and reports deltas. |

Example:

```bash
python3 main.py bench \
  --endpoint http://127.0.0.1:8080 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data-dir /tmp/vlmbench-cache \
  --enable-prometheus-metrics \
  wmt16
```

## Bench Command

Usage:

```bash
python3 main.py bench [BENCH_OPTIONS] BENCHMARK [BENCHMARK ...]
```

### Bench Flags

| Flag                  | Type            | Default                  | Description                                                          |
|-----------------------|-----------------|--------------------------|----------------------------------------------------------------------|
| `--list`              | flag            | off                      | List all registered benchmark names and exit.                        |
| `--stop-after`        | int             | `0`                      | Limit entries processed per benchmark. `0` means no limit.           |
| `--truncate`          | flag            | off                      | Truncate payloads that exceed model context length.                  |
| `--clients`           | int             | `1`                      | Number of concurrent client workers (must be >= 1).                  |
| `--random-populate`   | flag            | off                      | Randomly sample benchmark entries instead of strict sequential flow. |
| `--seed`              | int             | `None`                   | RNG seed for deterministic random population behavior.               |
| `--random-batch-size` | int             | `100`                    | Buffer size per random population batch (must be >= 1).              |
| `benchmarks`          | positional list | required unless `--list` | One or more benchmark names to run.                                  |

### Bench Examples

List benchmarks:

```bash
python3 main.py bench --list
```

Run a single benchmark with defaults:

```bash
python3 main.py bench wmt16
```

Run multiple benchmarks with concurrency:

```bash
python3 main.py bench --clients 8 local_alpaca local_narrativeqa
```

Short smoke test of first 50 entries:

```bash
python3 main.py bench --stop-after 50 local_sharegpt
```

Enable truncation for strict context-window fitting:

```bash
python3 main.py bench --truncate local_longbench_qmsum
```

Deterministic random sampling:

```bash
python3 main.py bench \
  --random-populate \
  --seed 42 \
  --random-batch-size 200 \
  --clients 16 \
  local_sharegpt
```

## Plugin Command

Usage:

```bash
python3 main.py plugin [--list]
python3 main.py plugin <plugin_name> [PLUGIN_OPTIONS]
```

Plugin base flags:

| Flag          | Type       | Default                  | Description                                              |
|---------------|------------|--------------------------|----------------------------------------------------------|
| `--list`      | flag       | off                      | List available plugins and exit.                         |
| `plugin_name` | positional | required unless `--list` | Plugin to run. Current values: `readiness`, `simulator`. |

Examples:

```bash
python3 main.py plugin --list
python3 main.py plugin readiness --help
python3 main.py plugin simulator --help
```

## Readiness Plugin

Usage:

```bash
python3 main.py plugin readiness [options]
```

Flags:

| Flag       | Type | Default | Description                                          |
|------------|------|---------|------------------------------------------------------|
| `--retrys` | int  | `5`     | Number of retry attempts for endpoint health checks. |

Example:

```bash
python3 main.py plugin readiness --endpoint http://127.0.0.1:8080 --retrys 10
```

## Simulator Plugin

Usage:

```bash
python3 main.py plugin simulator --total-kv-tokens N [options]
```

Required flags:

| Flag                | Type | Default  | Description                              |
|---------------------|------|----------|------------------------------------------|
| `--total-kv-tokens` | int  | required | Total KV tokens to target in simulation. |

Optional flags:

| Flag                   | Type   | Default    | Description                                       |
|------------------------|--------|------------|---------------------------------------------------|
| `--prefix-length-perc` | float  | `70.0`     | Shared prefix percent per prompt (0-100).         |
| `--n-runs`             | int    | `1`        | Number of simulation cycles.                      |
| `--source-type`        | string | `wikitext` | Text source: `wikitext`, `squad`, or `wikipedia`. |
| `--task`               | choice | random     | Task type. If omitted, tasks vary randomly.       |
| `--seed`               | int    | `42`       | Random seed used in simulator prompt generation.  |
| `--utilization-perc`   | float  | `100.0`    | Fraction of requested KV budget to fill (0-100).  |
| `--request-interval-s` | float  | `1.0`      | Delay between requests in a run.                  |
| `--run-interval-s`     | float  | `2.0`      | Delay between runs.                               |
| `--request-timeout-s`  | float  | `10.0`     | HTTP timeout per simulator request.               |

Examples:

Basic simulator run:

```bash
python3 main.py plugin simulator --total-kv-tokens 8388608
```

Prefix-heavy workload with lower utilization:

```bash
python3 main.py plugin simulator \
  --total-kv-tokens 8388608 \
  --prefix-length-perc 80 \
  --utilization-perc 90 \
  --n-runs 3
```

Specific source and task:

```bash
python3 main.py plugin simulator \
  --total-kv-tokens 4194304 \
  --source-type squad \
  --task qa \
  --seed 123
```

Faster pacing with custom timeout:

```bash
python3 main.py plugin simulator \
  --total-kv-tokens 2097152 \
  --request-interval-s 0.2 \
  --run-interval-s 1.0 \
  --request-timeout-s 20
```

## Useful Help Commands

```bash
python3 main.py --help
python3 main.py bench --help
python3 main.py plugin --help
python3 main.py plugin readiness --help
python3 main.py plugin simulator --help
```

## Notes And Validation Rules

- `--clients` must be >= 1.
- `--random-batch-size` must be >= 1 when used.
- `bench` requires at least one benchmark unless `--list` is used.
- `plugin` requires a valid plugin name unless `--list` is used.
- `simulator` requires `--total-kv-tokens`.

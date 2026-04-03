# Oolong Real

## One-time setup

Install deps from repo root:

```bash
uv sync
```

Prepare the local validation file once:

```bash
python bench_Oolong_real/download_validation.py --no-streaming
```

Default dataset path:

```bash
bench_Oolong_real/data/validation_single_episode.jsonl
```

Set your model key before running inference:

```bash
export DEEPSEEK_API_KEY=...
```

## `run_benchmark.py`

Basic run:

```bash
python bench_Oolong_real/run_benchmark.py --limit 1
```

### Common commands

Run the first example:

```bash
python bench_Oolong_real/run_benchmark.py --limit 1
```

Run a specific example id:

```bash
python bench_Oolong_real/run_benchmark.py --example_id 3952f2d5-082f-14b2-5ec4-d9cbedd2f865
```

Run with the default system prompt:

```bash
python bench_Oolong_real/run_benchmark.py --limit 1 --system_prompt default
```

Run with the subagent-encouraging prompt:

```bash
python bench_Oolong_real/run_benchmark.py --limit 1 --system_prompt subagent_encouraging
```

Run the root-only baseline (no RLM):

```bash
python bench_Oolong_real/run_benchmark.py --limit 1 --baseline
```

Run a specific example id with the subagent-encouraging prompt:

```bash
python bench_Oolong_real/run_benchmark.py --example_id 3952f2d5-082f-14b2-5ec4-d9cbedd2f865 --system_prompt subagent_encouraging
```

Run the first matching example for a question type:

```bash
python bench_Oolong_real/run_benchmark.py --question_type singledoc_rolls --query_index 0
```

Run three examples from one campaign:

```bash
python bench_Oolong_real/run_benchmark.py --campaign campaign2 --limit 3
```

### Flags

#### `--data_path`

Use a different local JSONL file.

```bash
python bench_Oolong_real/run_benchmark.py --data_path bench_Oolong_real/data/validation_single_episode.jsonl --limit 1
```

#### `--limit`

Run the first `N` matching examples.

```bash
python bench_Oolong_real/run_benchmark.py --limit 3
```

#### `--query_index`

Run one example by zero-based index after filters are applied.

```bash
python bench_Oolong_real/run_benchmark.py --query_index 0
```

With filters:

```bash
python bench_Oolong_real/run_benchmark.py --campaign campaign2 --question_type singledoc_rolls --query_index 0
```

#### `--example_id`

Run one example by exact id.

```bash
python bench_Oolong_real/run_benchmark.py --example_id 3952f2d5-082f-14b2-5ec4-d9cbedd2f865
```

#### `--campaign`

Filter to one campaign.

```bash
python bench_Oolong_real/run_benchmark.py --campaign campaign2 --limit 3
```

#### `--question_type`

Filter to one question type.

```bash
python bench_Oolong_real/run_benchmark.py --question_type singledoc_rolls --limit 5
```

#### `--model_name`

Choose the DeepSeek model used by the runner.

```bash
python bench_Oolong_real/run_benchmark.py --limit 1 --model_name deepseek-chat
```

You can also set the default with:

```bash
export OOLONG_REAL_MODEL=deepseek-chat
```

#### `--system_prompt`

Choose the RLM system prompt.

- `default`
  - Uses the library's default RLM system prompt.
- `subagent_encouraging`
  - Uses the repo's subagent-encouraging system prompt.

```bash
python bench_Oolong_real/run_benchmark.py --limit 1 --system_prompt default
```

```bash
python bench_Oolong_real/run_benchmark.py --limit 1 --system_prompt subagent_encouraging
```

#### `--baseline`

Run a root-only baseline with no RLM. This sends the question and full transcript directly to the model and ignores `--system_prompt`.

```bash
python bench_Oolong_real/run_benchmark.py --limit 1 --baseline
```

```bash
python bench_Oolong_real/run_benchmark.py --example_id 3952f2d5-082f-14b2-5ec4-d9cbedd2f865 --baseline
```

### Notes

- `--query_index` and `--example_id` are mutually exclusive.
- `--limit 1` runs the first matching example.
- Logs are written under `./bench_Oolong_real/log`.

## `run_multi_model_benchmark.py`

Deterministic multi-baseline runner: a **fixed set of 5 validation `example_id`s** (mix of single- and two-episode rows), model baselines (root/subagent pairs), and **2 trials per task**.

The `--system_prompt` flag determines **both** the prompt used and the active baseline set:

Uses OpenAI via `run_benchmark.get_backend_kwargs` (same as `run_benchmark.py`):

```bash
export OPENAI_API_KEY2=...
```

Run from repo root (standard baselines, default prompt):

```bash
uv run python bench_Oolong_real/run_multi_model_benchmark.py
```

Run the dynamic model-selection experiment (2 baselines, prompt locked):

```bash
uv run python bench_Oolong_real/run_multi_model_benchmark.py --system_prompt dynamic_model_picker
```

Run a single baseline:

```bash
uv run python bench_Oolong_real/run_multi_model_benchmark.py --baseline mini-root_mini-sub
```

Resume after interruption (skip rows already in the output file for that baseline):

```bash
uv run python bench_Oolong_real/run_multi_model_benchmark.py --baseline mini-root_mini-sub --resume
```

Keep each run in a **new** timestamped file instead of overwriting `results.jsonl`:

```bash
uv run python bench_Oolong_real/run_multi_model_benchmark.py --timestamped
```

### Baselines

`--system_prompt` selects which baseline set is active. The two sets are mutually exclusive — passing a `--baseline` name that belongs to the other set is an error.

**Standard** (`--system_prompt default` or `subagent_encouraging`):  3 × 5 × 2 = 30 RLM completions when `--baseline` is omitted.

| Name | Root model | Default subagent model |
|------|------------|------------------------|
| `mini-root_mini-sub` | `gpt-5.4-mini` | `gpt-5.4-mini` |
| `mini-root_nano-sub` | `gpt-5.4-mini` | `gpt-5.4-nano` |
| `flagship-root_nano-sub` | `gpt-5.4` | `gpt-5.4-nano` |

**Dynamic model-selection** (`--system_prompt dynamic_model_picker`):  2 × 5 × 2 = 20 RLM completions when `--baseline` is omitted. The root orchestrator is expected to pass `model=` on each `rlm_query` / `llm_query` call; the default subagent model applies only when the root omits `model=`.

| Name | Root model | Default subagent model |
|------|------------|------------------------|
| `dynamic_selection_mini_root` | `gpt-5.4-mini` | `gpt-5.4-mini` |
| `dynamic_selection_flagship_root` | `gpt-5.4` | `gpt-5.4-mini` |

Results are saved under `bench_Oolong_real/multi_model_benchmarks/<baseline>/`.

### Output layout

Under `bench_Oolong_real/multi_model_benchmarks/<baseline>/`:

- **Default:** `results.jsonl` is **truncated** on each fresh run (no `--resume`). `latest.jsonl` is a symlink to `results.jsonl`.
- **`--timestamped`:** `results_YYYYMMDD_HHMMSS.jsonl`; `latest.jsonl` points at that file. With `--resume`, appends to the file `latest.jsonl` currently targets.

Each JSONL line is one trial with fields: `task_id`, `query`, `baseline`, `trial`, `ground_truth`, `response`, `success` (null for manual scoring), `subagent_calls`, `total_time`, `input_tokens`.

### Flags

- `--system_prompt` — `default`, `subagent_encouraging`, or `dynamic_model_picker`. Determines both the prompt and the active baseline set (see table above).
- `--baseline` — run only one baseline from the active set (optional).
- `--resume` — append and skip trials whose `(task_id, baseline, trial)` already appear in the output file.
- `--timestamped` — write a new stamped file and update `latest.jsonl` instead of overwriting `results.jsonl`.

RLM trajectory logs go under `bench_Oolong_real/logs/`.

## Subagent model selection

The core [RLM](cci:2://file:///Users/dannyyu/Desktop/rlm/rlm/core/rlm.py:64:0-1181:20) library now supports separate child-model configuration with:

- `subagent_backend`
- `subagent_backend_kwargs`
- `subagent_model_selector`

For benchmark runs, the main patterns are:

- root=`gpt-5-mini`, subagents=`gpt-5-mini`
  - set `backend_kwargs["model_name"] = "gpt-5-mini"`
- root=`gpt-5-mini`, subagents=`gpt-5-nano`
  - set root `backend_kwargs["model_name"] = "gpt-5-mini"`
  - set `subagent_backend_kwargs["model_name"] = "gpt-5-nano"`
- root-directed routing between `gpt-5-mini` and `gpt-5-nano`
  - let the root decide per subcall by calling `rlm_query(..., model="...")`

Example [RLM](cci:2://file:///Users/dannyyu/Desktop/rlm/rlm/core/rlm.py:64:0-1181:20) setup for a fixed child default:

```python
rlm = RLM(
    backend="openai",
    backend_kwargs={"model_name": "gpt-5-mini", "api_key": api_key},
    subagent_backend_kwargs={"model_name": "gpt-5-nano", "api_key": api_key},
    max_depth=2,
)

If you want the root model itself to choose the child model, do that explicitly inside the REPL:
rlm_query("Solve this subproblem", model="gpt-5-nano") 

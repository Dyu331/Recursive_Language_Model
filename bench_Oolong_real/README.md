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

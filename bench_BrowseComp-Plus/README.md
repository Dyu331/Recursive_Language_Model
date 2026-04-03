# BrowseComp-Plus

## One-time setup

Install deps from repo root:

```bash
uv sync
```

Decrypt the benchmark dataset once:

```bash
python bench_BrowseComp-Plus/decrypt_dataset.py
```

Download the local corpus file once:

```bash
python bench_BrowseComp-Plus/download_corpus.py
```

Default dataset paths:

```bash
bench_BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl
bench_BrowseComp-Plus/corpus.jsonl
```

Set your model key before running inference:

```bash
export DEEPSEEK_API_KEY=...
```

## `run_benchmark.py`

Basic run:

```bash
python bench_BrowseComp-Plus/run_benchmark.py
```

### Common commands

Run the first example:

```bash
python bench_BrowseComp-Plus/run_benchmark.py
```

Run a specific query id:

```bash
python bench_BrowseComp-Plus/run_benchmark.py --query_id 0
```

Run with the default RLM system prompt:

```bash
python bench_BrowseComp-Plus/run_benchmark.py --mode rlm --system_prompt default
```

Run with the subagent-encouraging prompt:

```bash
python bench_BrowseComp-Plus/run_benchmark.py --mode rlm --system_prompt subagent_encouraging
```

Run the root-only default mode (no RLM):

```bash
python bench_BrowseComp-Plus/run_benchmark.py --mode default
```

Run a smoke test with the root-only default mode:

```bash
python bench_BrowseComp-Plus/run_benchmark.py --smoke_test --mode default
```

Run a smoke test with RLM:

```bash
python bench_BrowseComp-Plus/run_benchmark.py --smoke_test --mode rlm --system_prompt subagent_encouraging
```

### Flags

#### `--decrypted_path`

Use a different decrypted BrowseComp-Plus JSONL file.

```bash
python bench_BrowseComp-Plus/run_benchmark.py --decrypted_path bench_BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl
```

#### `--corpus_path`

Use a different local corpus JSONL file.

```bash
python bench_BrowseComp-Plus/run_benchmark.py --corpus_path bench_BrowseComp-Plus/corpus.jsonl
```

#### `--num_docs`

Number of corpus docs to load in non-smoke runs.

```bash
python bench_BrowseComp-Plus/run_benchmark.py --num_docs 1000
```

#### `--num_docs_test`

Number of docs to use in smoke-test mode.

```bash
python bench_BrowseComp-Plus/run_benchmark.py --smoke_test --num_docs_test 5
```

#### `--smoke_test`

Use a small context built from the gold doc plus sampled candidates.

```bash
python bench_BrowseComp-Plus/run_benchmark.py --smoke_test
```

#### `--query_index`

Run one example by zero-based index in the decrypted JSONL.

```bash
python bench_BrowseComp-Plus/run_benchmark.py --query_index 0
```

#### `--query_id`

Run one example by exact `query_id`.

```bash
python bench_BrowseComp-Plus/run_benchmark.py --query_id 0
```

#### `--model_name`

Choose the DeepSeek model used by the runner.

```bash
python bench_BrowseComp-Plus/run_benchmark.py --model_name deepseek-chat
```

You can also set the default with:

```bash
export BROWSECOMP_MODEL=deepseek-chat
```

#### `--system_prompt`

Choose the RLM system prompt.

- `default`
  - Uses the library's default RLM system prompt.
- `subagent_encouraging`
  - Uses the repo's subagent-encouraging system prompt.

```bash
python bench_BrowseComp-Plus/run_benchmark.py --mode rlm --system_prompt default
```

```bash
python bench_BrowseComp-Plus/run_benchmark.py --mode rlm --system_prompt subagent_encouraging
```

#### `--mode`

Choose whether to run full RLM or a root-only default baseline.

- `rlm`
  - Uses the REPL-based recursive agent.
- `default`
  - Bypasses RLM and sends the query plus corpus slice directly to the root model.

```bash
python bench_BrowseComp-Plus/run_benchmark.py --mode rlm
```

```bash
python bench_BrowseComp-Plus/run_benchmark.py --mode default
```

### Notes

- `--query_index` and `--query_id` are mutually exclusive.
- Default mode ignores `--system_prompt`.
- Logs are written under `./logs` for RLM runs.

## `run_multi_model_benchmark.py`

Deterministic multi-baseline runner: a **fixed set of 5 `query_id`s** from the decrypted BrowseComp-Plus JSONL, model baselines (root/subagent pairs), and **2 trials per task**.

The `--system_prompt` flag determines **both** the prompt used and the active baseline set:

Requires the decrypted dataset (and corpus setup as in one-time setup). Uses OpenAI the same way as `run_benchmark.py`:

```bash
export OPENAI_API_KEY2=...
```

Run from repo root (standard baselines, default prompt):

```bash
uv run python bench_BrowseComp-Plus/run_multi_model_benchmark.py
```

Run the dynamic model-selection experiment (2 baselines, prompt locked):

```bash
uv run python bench_BrowseComp-Plus/run_multi_model_benchmark.py --system_prompt dynamic_model_picker
```

Run one baseline with a smaller corpus slice (faster/cheaper smoke):

```bash
uv run python bench_BrowseComp-Plus/run_multi_model_benchmark.py --baseline mini-root_mini-sub --num_docs 100
```

Resume after interruption:

```bash
uv run python bench_BrowseComp-Plus/run_multi_model_benchmark.py --baseline mini-root_mini-sub --resume
```

Archive each run to a **new** timestamped file:

```bash
uv run python bench_BrowseComp-Plus/run_multi_model_benchmark.py --timestamped
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

Results are saved under `bench_BrowseComp-Plus/multi_model_benchmarks/<baseline>/`.

### Output layout

Under `bench_BrowseComp-Plus/multi_model_benchmarks/<baseline>/`:

- **Default:** `results.jsonl` is **truncated** on each fresh run (no `--resume`). `latest.jsonl` symlinks to `results.jsonl`.
- **`--timestamped`:** `results_YYYYMMDD_HHMMSS.jsonl` plus `latest.jsonl` updated; `--resume` appends to the file `latest.jsonl` points to.

Each JSONL line is one trial: `task_id`, `query`, `baseline`, `trial`, `ground_truth`, `response`, `success` (null for manual scoring), `subagent_calls`, `total_time`, `input_tokens`.

### Flags

- `--system_prompt` — `default`, `subagent_encouraging`, or `dynamic_model_picker`. Determines both the prompt and the active baseline set (see table above).
- `--baseline` — run only one baseline from the active set (optional).
- `--resume` — append and skip existing `(task_id, baseline, trial)` rows.
- `--timestamped` — new stamped JSONL per run instead of overwriting `results.jsonl`.
- `--decrypted_path`, `--corpus_path`, `--num_docs` — same roles as `run_benchmark.py` (defaults under `bench_BrowseComp-Plus/`).

RLM trajectory logs go under `bench_BrowseComp-Plus/logs/`.

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
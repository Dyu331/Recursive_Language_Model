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

### MemPalace PoC (ephemeral search baseline)

Install the optional extra from the repo root (or `pip install -e ../mempalace` plus a compatible `chromadb`):

```bash
uv pip install -e ".[mempalace-poc]"
```

**Lenient mode** (default for `--palace-poc`): the full transcript remains in `context['context_window_text']` and is also indexed in a **temporary** Chroma palace per example. The REPL gains `search_memories(query, wing=..., room=..., n_results=...)` for semantic retrieval. The palace directory is deleted after each `completion`.

```bash
python bench_Oolong_real/run_benchmark.py --limit 1 --palace-poc
```

**Strict mode** (`--palace-poc-strict`): `context_window_text` is **omitted** from `context`; the model must use `search_memories` to read the transcript. Harder eval; use for ablations only after lenient mode works.

```bash
python bench_Oolong_real/run_benchmark.py --limit 1 --palace-poc-strict
```

**Verbose lobby** (`--palace-poc-verbose`): only with palace PoC flags; prints a wing/room tree and the first 50 characters of each indexed drawer (whitespace collapsed) to stdout **right after** the palace is populated and **before** the root RLM run starts. Useful for inspecting chunking.

```bash
python bench_Oolong_real/run_benchmark.py --limit 1 --palace-poc --palace-poc-verbose
```

**By-speaker ingest** (`--palace-poc-by-speaker`): requires palace PoC. Parses the Oolong transcript into **`_preamble`** plus **grouped dialogue drawers**: short back-and-forth is merged until a minimum character threshold (default 250, configurable via `build_ephemeral_palace_tools(..., min_drawer_chars=...)`). Each drawer’s Chroma **`room`** is usually the **dominant** speaker (most UTF-8 text in that bundle); **`_mixed`** is used when two speakers tie. All participants are listed in metadata (`speakers_json` / `speaker_slugs`); drawer **text** keeps `Name:` labels. **`search_memories(..., room='<slug>')`** runs a primary `room=<slug>` search and a wing-wide pass, then merges hits so a secondary speaker still surfaces drawers filed under another dominant room. Episode **line** spans are stored as zero-padded `line_start` / `line_end` strings (and optional `line_ranges` JSON) on chunks when applicable. **`list_taxonomy()`** lists **`_preamble` first**, then **`_mixed`** (with a speaker hint when known), then other rooms. Use `room='_preamble'` for instructions/mapping.

```bash
python bench_Oolong_real/run_benchmark.py --limit 1 --palace-poc --palace-poc-by-speaker
```

**By-block ingest** (`--palace-poc-by-block`): mutually exclusive with `--palace-poc-by-speaker`. Splits the episode into overlapping **dense** (non-empty) line windows (default 75 lines, 7-line overlap) as **`room = block_001`**, `block_002`, …. **Inside each room**, **interaction grouping** appends lines in order (full `Speaker:` lines preserved) **without** starting a new drawer on every speaker change; a new drawer starts once the running buffer reaches **`min_drawer_chars`** (same parameter as by-speaker grouping, default 250, via `build_ephemeral_palace_tools(..., min_drawer_chars=...)`), or at the end of the window—so rapid back-and-forth stays in one searchable “event.” Each drawer’s metadata includes **`speakers_json`** (all `Name:` speakers in that drawer, first-appearance order), **`speaker_slugs`**, and primary **`speaker`** (first in that list). **`line_start`** / **`line_end`** are zero-padded **1-based indices into the episode’s non-empty lines**. When a drawer closes because it reached ``min_drawer_chars`` and more lines remain in that temporal window, the **triggering line** is also prepended to the **next** drawer (one-line overlap across drawer boundaries). A drawer is split into multiple Chroma rows only if it exceeds the palace **chunk size** (shared line span and speaker metadata on each part). **`list_taxonomy()`** orders **`_preamble`** then blocks **numerically**. Temporal `room` names are time windows, not cast slugs.

```bash
python bench_Oolong_real/run_benchmark.py --limit 1 --palace-poc --palace-poc-by-block
```

`run_multi_model_benchmark.py` supports the same flags and writes a `condition` field (`default` | `palace_poc` | `palace_poc_strict` | `palace_poc_by_speaker` | `palace_poc_strict_by_speaker` | `palace_poc_by_block` | `palace_poc_strict_by_block`) into each JSONL row.

`--palace-poc` cannot be combined with `--baseline`.

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

## `run_rlm_mempalace_benchmark.py`

Frozen **replicate-scale** suite (same 50 `example_id`s and baselines as `run_replicate_rlm_benchmark.py` by default) with **RLM + ephemeral MemPalace always enabled**. Default transcript ingest is **`by_block`** (temporal block rooms); use `--palace-poc-by-speaker` for grouped-by-speaker rooms. **`--palace-poc-strict`** omits `context_window_text` from the REPL context so the model must use `search_memories`. There is **no** `--palace-poc-verbose`; MemPalace indexing stays quiet (`verbose=False`).

**Episode modes** (mutually exclusive):

- **Default** — 50 single-episode IDs; JSONL rows use **`suite`: `single_50`**.
- **`--allow_two_episodes`** — 50 two-episode IDs; **`suite`: `two_50`**.
- **`--mixed_25_episodes`** — first **25** single-episode IDs, then first **25** two-episode IDs (deterministic order), using the default validation JSONLs only; **`--data_path` is not allowed** in this mode. Rows use **`suite`: `mixed_25_25`**.

Per trial, palace teardown runs in a **`finally`** block so temp Chroma state is cleaned up even if `completion` raises.

Results go under **`bench_Oolong_real/replicate_rlm_mempalace_benchmarks/<baseline>[_depthN]/`**, separate from non-palace replicate runs (`replicate_rlm_benchmarks/`). All modes share the same directory layout and `results.jsonl` naming. Each JSONL row includes **`mempalace`**, **`ingest`**, **`condition`**, **`suite`**, **`num_episode`** (length of the example’s `episodes` list, typically 1 or 2), **`runner`**, and **`benchmark_kind`**.

Requires `OPENAI_API_KEY2` (same as other Oolong runners). Example:

```bash
export OPENAI_API_KEY2=...
uv run python bench_Oolong_real/run_rlm_mempalace_benchmark.py --baseline mini-root_mini-sub --max_depth 2
```

Mixed 25+25 example:

```bash
uv run python bench_Oolong_real/run_rlm_mempalace_benchmark.py --mixed_25_episodes --baseline mini-root_mini-sub
```

Use `--timestamped` / `--system_prompt` the same way as `run_replicate_rlm_benchmark.py`. With `--resume`, skips use **`(task_id, baseline, trial, condition, suite)`** so different strict/ingest settings and episode suites do not collide. Rows written before **`suite`** existed resume with an empty suite token for those lines only.

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

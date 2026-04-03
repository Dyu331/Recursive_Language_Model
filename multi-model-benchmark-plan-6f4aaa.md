# Multi-Model Benchmarking Plan

This plan adds separate deterministic benchmark runners for Oolong and BrowseComp that execute the three root/subagent model baselines, repeat each selected task twice, and append per-trial metrics to JSONL for later manual evaluation.

## Scope

- Add a new runner file for `bench_Oolong_real`.
- Add a new runner file for `bench_BrowseComp-Plus`.
- Keep the current single-run scripts untouched unless a tiny shared helper extraction is clearly warranted.
- Log one JSON object per trial with stable identifiers and trial metadata.

## Benchmark matrix

Run the same fixed 5-task set across all baselines, with 2 trials per task:

- `mini-root_mini-sub`
  - Root: `gpt-5.4-mini`
  - Subagent: `gpt-5.4-mini`
- `mini-root_nano-sub`
  - Root: `gpt-5.4-mini`
  - Subagent: `gpt-5.4-nano`
- `flagship-root_nano-sub`
  - Root: `gpt-5.4`
  - Subagent: `gpt-5.4-nano`

Total per benchmark: `3 baselines x 5 tasks x 2 trials = 30 runs`.

## Deterministic query selection

### Oolong

Freeze 5 explicit `example_id` values and reuse them for all runs.

Recommended composition:

- 2 single-episode examples
  - 1 `singledoc_rolls`
  - 1 `singledoc_spells`
- 3 two-episode examples
  - 2 `multidoc_rolls`
  - 1 `multidoc_spells`

Selection criteria:

- Prefer different question templates, not near-duplicates.
- Prefer examples that vary operation type:
  - counting totals
  - per-character/per-player filtering
  - spell/roll subtype lookup
  - cross-episode cumulative reasoning
- Avoid selecting multiple prompts that differ only by character name or episode number.
- Keep the chosen IDs in a small constant list in the new runner for reproducibility.

### BrowseComp

Freeze 5 explicit `query_id` values and reuse them for all runs.

Selection criteria:

- Manually skim candidate queries and choose 5 with visibly different query structures.
- Prefer diversity in expected answer entity type:
  - person
  - organization/institution
  - location
  - title/work/event
  - numeric/date-like answer if available
- Prefer questions that appear to require different evidence-linking patterns, rather than five variants of the same surface form.
- Avoid malformed or suspicious queries and avoid near-duplicates.
- Keep the chosen IDs in a small constant list in the new runner for reproducibility.

## Runner behavior

Each new runner should:

- define the 3 baselines centrally
- define the fixed 5-ID task list centrally
- accept a `--baseline` flag to run only a specific baseline (optional; if omitted, run all)
- accept a `--resume` flag to skip existing trials in the output JSONL
- iterate in a deterministic order
- run each task twice with `trial_number` = `1` and `2`
- append results to a benchmark-specific JSONL output file
- print concise progress to stdout
- support resuming safely with `--resume` flag

## Command line interface

```bash
# Run all baselines (30 trials total)
python bench_Oolong_real/run_multi_model_benchmark.py

# Run only one baseline (10 trials total)
python bench_Oolong_real/run_multi_model_benchmark.py --baseline mini-root_mini-sub

# Resume from previous run (skip existing trials)
python bench_Oolong_real/run_multi_model_benchmark.py --resume

# Combine flags
python bench_Oolong_real/run_multi_model_benchmark.py --baseline mini-root_nano-sub --resume
```

## Output directory structure

Each benchmark will store results in its own subdirectory with separate files per baseline:

```
bench_Oolong_real/multi_model_benchmarks/
├── mini-root_mini-sub/
│   ├── results_YYYYMMDD_HHMMSS.jsonl
│   └── latest.jsonl  # symlink to most recent run
├── mini-root_nano-sub/
│   ├── results_YYYYMMDD_HHMMSS.jsonl
│   └── latest.jsonl  # symlink to most recent run
└── flagship-root_nano-sub/
    ├── results_YYYYMMDD_HHMMSS.jsonl
    └── latest.jsonl  # symlink to most recent run

bench_BrowseComp-Plus/multi_model_benchmarks/
├── mini-root_mini-sub/
│   ├── results_YYYYMMDD_HHMMSS.jsonl
│   └── latest.jsonl  # symlink to most recent run
├── mini-root_nano-sub/
│   ├── results_YYYYMMDD_HHMMSS.jsonl
│   └── latest.jsonl  # symlink to most recent run
└── flagship-root_nano-sub/
    ├── results_YYYYMMDD_HHMMSS.jsonl
    └── latest.jsonl  # symlink to most recent run
```

The `multi_model_benchmarks/` directories will be created within each benchmark folder if they don't exist. Each baseline gets its own subdirectory for cleaner organization and easier analysis.

## JSONL schema

Append one record per trial with these fields:

- `task_id`
- `query`
- `baseline`
- `trial`
- `ground_truth`
- `response`
- `success`
- `subagent_calls`
- `total_time`
- `input_tokens`

Notes:

- Use dataset-native stable identifiers:
  - Oolong: `example_id` mapped into `task_id`
  - BrowseComp: `query_id` mapped into `task_id`
- Initialize `success` as `null` or empty so you can manually fill it later.

## Instrumentation details

### Already available

- `total_time`
  - available from `RLMChatCompletion.execution_time`
- aggregated usage summary
  - available from `RLMChatCompletion.usage_summary`
- subagent lifecycle hooks
  - available through `RLM(on_subcall_start=..., on_subcall_complete=...)`

### Small instrumentation gap

`root_input_tokens` is not currently isolated in a robust way for the `mini-root_mini-sub` baseline because usage is aggregated by model name, so root and subagent traffic collapse into the same bucket when both use `gpt-5.4-mini`.

Recommended implementation approach:

- add a narrow per-root-call metric in core RLM completion handling, or
- add a dedicated callback / metadata field for root LM call token usage

This should be a surgical change, not a broad logging redesign.

## Implementation approach

1. Add new Oolong multi-model benchmark runner with CLI flags and unified output directory.
2. Add new BrowseComp multi-model benchmark runner with CLI flags and unified output directory.
3. Add minimal instrumentation needed to record `subagent_calls` and reliable `input_tokens`.
4. Verify deterministic ordering and JSONL output shape.

## Validation

Before large benchmark runs:

- run one task for one baseline in each new runner
- verify JSONL rows append correctly
- verify `subagent_calls` increments
- verify `input_tokens` is populated correctly in all three baselines
- verify the frozen ID lists are the only tasks executed
- verify `--baseline` flag restricts execution to specified baseline
- verify `--resume` flag skips existing trials correctly
- verify output files are created in benchmark-specific `multi_model_benchmarks/` directory structure

## Frozen ID Lists

### Oolong:
3952f2d5-082f-14b2-5ec4-d9cbedd2f865
17ea4835-8da0-9866-8d3c-753836fa2bcc
46d3403c-75c1-b801-afdc-b2612651e0e3
4abcd845-62d0-843d-9817-ed85290787dd
9c5a1ad1-70a0-9ead-7f30-6ba609e00c1f

### BrowseComp:
769
773
781
793
806

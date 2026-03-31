Using CPython 3.11.14
Creating virtual environment at: rlm-test-env
Activate with: source rlm-test-env/bin/activate


Visualizer
cd visualizer
npm run dev

# Subagent model selection

`RLM` now supports configuring child-agent model selection separately from the root model.

## New `RLM` arguments

- `subagent_backend`
  - Optional backend for child RLM calls.
- `subagent_backend_kwargs`
  - Optional backend kwargs for child RLM calls.
  - In practice this is the main place to pin a fixed child model, for example `{"model_name": "gpt-5-nano"}`.
- `subagent_model_selector`
  - Optional callable with signature `(prompt: str, next_depth: int) -> str | None`.
  - Lets the root choose a child model dynamically per subcall.

## Resolution precedence

All child model selection now goes through one centralized resolver.

Precedence is:

1. Explicit `model=...` passed to `rlm_query(...)`
2. `subagent_model_selector(prompt, next_depth)`
3. Fixed `subagent_backend` / `subagent_backend_kwargs`
4. Inherited root `backend` / `backend_kwargs`

This same logic is used for:

- recursive child `RLM` creation
- isolated child process subcalls
- max-depth fallback to plain LM completion

## Common usage patterns

### Root and subagents both use `gpt-5-mini`

```python
rlm = RLM(
    backend="openai",
    backend_kwargs={"model_name": "gpt-5-mini", "api_key": api_key},
    max_depth=2,
)
```

No subagent-specific config is needed. Children inherit the root config.

### Root uses `gpt-5-mini`, subagents use `gpt-5-nano`

```python
rlm = RLM(
    backend="openai",
    backend_kwargs={"model_name": "gpt-5-mini", "api_key": api_key},
    subagent_backend_kwargs={"model_name": "gpt-5-nano", "api_key": api_key},
    max_depth=2,
)
```

If the child backend is the same as the root backend, setting `subagent_backend` is optional.

If children should use a different provider entirely, set both:

```python
rlm = RLM(
    backend="openai",
    backend_kwargs={"model_name": "gpt-5-mini", "api_key": openai_api_key},
    subagent_backend="openrouter",
    subagent_backend_kwargs={"model_name": "openai/gpt-5-nano", "api_key": openrouter_api_key},
    max_depth=2,
)
```

### Dynamically route subagents between `gpt-5-mini` and `gpt-5-nano`

```python
def choose_subagent_model(prompt: str, next_depth: int) -> str | None:
    if len(prompt) < 500:
        return "gpt-5-nano"
    return "gpt-5-mini"


rlm = RLM(
    backend="openai",
    backend_kwargs={"model_name": "gpt-5-mini", "api_key": api_key},
    subagent_model_selector=choose_subagent_model,
    max_depth=2,
)
```

Returning `None` from the selector falls back to fixed child config if present, otherwise to the inherited root config.

## Explicit override still wins

Code inside the environment can still force a specific child model:

```python
rlm_query("Solve this subproblem", model="gpt-5-nano")
```

That override takes precedence over both `subagent_model_selector` and fixed `subagent_backend_kwargs`.

## Benchmark implications

This feature supports the benchmark setups:

- root=`gpt-5-mini`, subagents=`gpt-5-mini`
- root=`gpt-5-mini`, subagents=`gpt-5-nano`
- root=`gpt-5-mini`, routed subagents=`gpt-5-mini|gpt-5-nano`

One important caveat is that a benchmark runner must actually allow recursion for subagents to be used. For example, if a runner uses `max_depth=1`, subagents will not be spawned.
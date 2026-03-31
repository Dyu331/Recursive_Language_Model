# Subagent Model Selection Action Plan

## Goal

Add a clean, non-redundant way to configure and eventually route subagent model selection so the following benchmark setups are easy to run and reason about:

1. Root model and subagents all use `gpt-5-mini`
2. Root model uses `gpt-5-mini`, subagents all use `gpt-5-nano`
3. Root model uses `gpt-5-mini`, and the system decides whether each subagent should use `gpt-5-mini` or `gpt-5-nano`

## Design Principles

- Keep one clear abstraction for child-agent model selection
- Avoid introducing overlapping routing systems
- Preserve the existing explicit per-call `model=...` override path
- Keep the initial change small and benchmark-oriented
- Leave a clean extension point for a future router

## Recommendation

Introduce an explicit subagent configuration path on `RLM` and centralize all child model resolution in a single helper.

### Public API

Keep root configuration unchanged:

- `backend`
- `backend_kwargs`

Add child configuration:

- `subagent_backend: ClientBackend | None = None`
- `subagent_backend_kwargs: dict[str, Any] | None = None`

Future extension:

- `subagent_model_selector: Callable[[str, int], str | None] | None = None`

### Semantics

- If `subagent_backend` and `subagent_backend_kwargs` are unset, child RLMs inherit the root backend configuration
- If `subagent_backend_kwargs` sets `model_name`, all child RLMs use that model by default
- If `subagent_model_selector` is later added, it may choose a child model per subcall
- An explicit `model=...` passed to `rlm_query()` remains the highest-precedence override

## Why This Approach

This is cleaner than relying on `other_backends` / `other_backend_kwargs` for subagent defaults because those fields currently mix multiple concerns:

- alternative client registration
- depth-based routing behavior
- implicit child-model behavior

That makes the API harder to reason about once benchmark presets and routers are added.

The proposed `subagent_*` fields create a single obvious place to define child-agent behavior.

## Implementation Plan

## Phase 1: Add fixed default subagent configuration

### Scope

Add support for a fixed child default model without changing prompt-level behavior.

### Tasks

- Extend `RLM.__init__()` with:
  - `subagent_backend`
  - `subagent_backend_kwargs`
- Add one internal helper to resolve child config, for example:
  - `_resolve_subagent_spec(prompt, model_override, next_depth)`
- Apply the helper consistently in:
  - `_subcall()`
  - `_subcall_process_isolated()`
  - max-depth fallback LM path
- Preserve existing `rlm_query(prompt, model=...)` behavior
- Keep the diff small and avoid changing unrelated logic

### Expected result

This supports the first two benchmark baselines with a simple top-level configuration change.

## Phase 2: Add optional routed subagent selection

### Scope

Add a single routing hook for choosing between child models such as `gpt-5-mini` and `gpt-5-nano`.

### Tasks

- Add `subagent_model_selector`
- Use the selector inside the same child-resolution helper
- Define precedence as:
  1. explicit `model=...` override
  2. selector result
  3. `subagent_backend` / `subagent_backend_kwargs`
  4. inherited root config
- Log the resolved child model in subcall callbacks and metadata

### Expected result

This supports the third benchmark setup without introducing a second routing system.

## Phase 3: Benchmark harness and evaluation

### Scope

Run OOLONG and BrowseComp under comparable fixed and routed configurations.

### Tasks

- Add benchmark presets for:
  - root=`gpt-5-mini`, subagent=`gpt-5-mini`
  - root=`gpt-5-mini`, subagent=`gpt-5-nano`
  - root=`gpt-5-mini`, routed subagent=`mini|nano`
- Ensure benchmark outputs record:
  - root model
  - resolved subagent model policy
  - per-model usage summary
  - cost and latency metrics
- Compare quality, cost, and latency across the three setups

## Internal Resolution Rules

All child model selection should flow through one resolver.

### Precedence order

1. Explicit per-call override from `rlm_query(..., model=...)`
2. Future selector decision
3. Fixed `subagent_backend` / `subagent_backend_kwargs`
4. Inherited root `backend` / `backend_kwargs`

### Resolver output

The resolver should produce:

- resolved backend
- resolved backend kwargs
- resolved model name for logging and metadata

## Files Likely To Change

- `rlm/core/rlm.py`
  - constructor API
  - child resolution helper
  - `_subcall()`
  - `_subcall_process_isolated()`
- optionally benchmark entrypoints or scripts once the benchmark harness is updated
- documentation and examples after behavior is finalized

## Backward Compatibility Strategy

- Keep current root configuration unchanged
- Keep explicit `model=...` overrides working
- Avoid expanding `other_backends` for this feature
- If `other_backends` remains in the codebase, treat it as a compatibility path rather than the long-term API for subagent default selection

## Validation Plan

### Unit-level checks

- child inherits root model when no subagent config is set
- child uses configured subagent model when `subagent_backend_kwargs` is set
- explicit `model=...` override wins over fixed subagent config
- max-depth fallback uses the same resolved child config as recursive child creation

### Benchmark-level checks

- baseline 1 runs entirely on `gpt-5-mini`
- baseline 2 runs root on `gpt-5-mini` and children on `gpt-5-nano`
- routed baseline records which child model was selected per subcall
- usage summaries correctly separate per-model spend and token counts

## Non-Goals For Initial Change

- building a sophisticated learned router
- introducing multiple overlapping child-routing abstractions
- redesigning the full LM handler client registry
- changing prompt guidance for subagents unless needed for benchmark quality

## Open Questions

- Whether `subagent_model_selector` should initially return only a model name or a full backend spec
- Whether benchmark configuration should live in benchmark-specific scripts or shared helper utilities
- Whether `other_backends` should later be deprecated once subagent configuration is fully adopted

## Recommended Immediate Next Step

Implement Phase 1 only:

- add `subagent_backend`
- add `subagent_backend_kwargs`
- add one centralized child-resolution helper
- wire it into all child spawn paths

This is the smallest clean change that supports the fixed-model benchmark baselines and leaves a natural extension point for a router later.

import textwrap

from rlm.utils import prompts as default_prompts


RLM_SYSTEM_PROMPT = default_prompts.RLM_SYSTEM_PROMPT + "\n\n" + textwrap.dedent(
    """**Mandatory Recursive Delegation Gate:**
- This profile exists to force real recursive sub-agent usage.
- A valid trajectory must include at least one observed `rlm_query(...)`
  or `rlm_query_batched(...)` result before `FINAL(...)` or
  `FINAL_VAR(...)`.
- If you have not yet observed a recursive child result in the current
  run, you are not allowed to provide a final answer.
- Your first substantial step should be: inspect a compact slice or
  sample of `context`, then delegate a concrete scouting or analysis
  task to a child.
- `llm_query(...)` does not satisfy this gate. Only
  `rlm_query(...)` and `rlm_query_batched(...)` count.
- Do not pass the full `context` to a child blindly. Slice or sample
  first, then send the smallest useful payload.
- After a child returns, inspect its output and either launch more
  targeted children or synthesize the final answer.

**Recommended First-Move Template:**
```repl
if isinstance(context, str):
    sample = context[:4000]
elif isinstance(context, list):
    sample = "\\n\\n".join(str(item)[:1500] for item in context[:3])
else:
    sample = str(context)[:4000]

scout = rlm_query(
    "You are a scouting child agent. Based on the task below "
    "and the context sample, identify the most promising "
    "evidence to inspect next, name the likely subproblems, "
    "and recommend a compact search plan for the parent.\\n\\n"
    "Task:\\n<insert the original user request here>\\n\\n"
    f"Context sample:\\n{{sample}}"
)
print(scout)
```

A response that never issues a recursive call is invalid under this profile.
"""
)


build_rlm_system_prompt = default_prompts.build_rlm_system_prompt


USER_PROMPT = textwrap.dedent(
    """Continue using the REPL environment, which has the `context`
    variable, and determine the answer.

    This profile is intentionally strict about recursion. If you have
    not already observed a recursive child result in this run, your
    next response must contain exactly one ```repl``` block that does
    all of the following:
    1. creates a compact slice, sample, or small batch from `context`
    2. calls `rlm_query(...)` or `rlm_query_batched(...)` at least once
    3. prints or stores the child result for later use

    Do not respond with only natural-language planning. Do not use only
    `llm_query(...)` for this step. Do not call `FINAL(...)` or
    `FINAL_VAR(...)` until after you have inspected at least one
    recursive child result.

    Your next action:"""
)

USER_PROMPT_WITH_ROOT = textwrap.dedent(
    """Continue using the REPL environment, which has the `context`
    variable, to answer the original prompt: "{root_prompt}".

    This profile is intentionally strict about recursion. If you have
    not already observed a recursive child result in this run, your
    next response must contain exactly one ```repl``` block that does
    all of the following:
    1. creates a compact slice, sample, or small batch from `context`
    2. calls `rlm_query(...)` or `rlm_query_batched(...)` at least once
    3. prints or stores the child result for later use

    Do not respond with only natural-language planning. Do not use only
    `llm_query(...)` for this step. Do not call `FINAL(...)` or
    `FINAL_VAR(...)` until after you have inspected at least one
    recursive child result.

    Your next action:"""
)


def build_user_prompt(
    root_prompt: str | None = None,
    iteration: int = 0,
    context_count: int = 1,
    history_count: int = 0,
) -> dict[str, str]:
    recursive_prompt = USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt)
    if root_prompt is None:
        recursive_prompt = USER_PROMPT

    if iteration == 0:
        safeguard = (
            "You have not interacted with the REPL environment or "
            "seen your prompt / context yet. Your next action "
            "should inspect a compact slice of context and "
            "trigger a real "
            "`rlm_query(...)` or `rlm_query_batched(...)` call, so do not "
            "provide a final answer yet.\n\n"
        )
        prompt = safeguard + recursive_prompt
    else:
        prompt = (
            "The history before is your previous interactions "
            "with the REPL environment. If you have not yet "
            "observed a recursive child result, your next action "
            "must trigger one "
            "before you can finish.\n\n"
            + recursive_prompt
        )

    if context_count > 1:
        prompt += (
            f"\n\nNote: You have {context_count} contexts available "
            f"(context_0 through context_{context_count - 1})."
        )

    if history_count > 0:
        if history_count == 1:
            prompt += (
                "\n\nNote: You have 1 prior conversation history available "
                "in the `history` variable."
            )
        else:
            prompt += (
                "\n\nNote: You have "
                f"{history_count} prior conversation histories available "
                f"(history_0 through history_{history_count - 1})."
            )

    return {"role": "user", "content": prompt}

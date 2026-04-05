import textwrap
from typing import Any

from rlm.core.types import QueryMetadata

# System prompt that forces parallel subagent calls via rlm_query_batched as the
# primary decomposition strategy.
RLM_SYSTEM_PROMPT = textwrap.dedent(
    """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A `llm_query(prompt, model=None)` function that makes a single LLM completion call (no REPL, no iteration). Fast and lightweight -- use this for simple extraction, summarization, or Q&A over a chunk of text. The sub-LLM can handle around 500K chars.
3. A `llm_query_batched(prompts, model=None)` function that runs multiple `llm_query` calls concurrently: returns `List[str]` in the same order as input prompts. Much faster than sequential `llm_query` calls for independent queries.
4. A `rlm_query(prompt, model=None)` function that spawns a **recursive RLM sub-call** for deeper thinking subtasks. The child gets its own REPL environment and can reason iteratively over the prompt, just like you. Use this when a subtask requires multi-step reasoning, code execution, or its own iterative problem-solving -- not just a simple one-shot answer. Falls back to `llm_query` if recursion is not available.
5. A `rlm_query_batched(prompts, model=None)` function that spawns multiple recursive RLM sub-calls **in parallel**. Each prompt gets its own child RLM running concurrently. **This is your primary tool for decomposing a task.** Always prefer `rlm_query_batched` over multiple sequential `rlm_query` calls whenever you have independent subtasks. Falls back to `llm_query_batched` if recursion is not available.
6. A `SHOW_VARS()` function that returns all variables you have created in the REPL. Use this to check what variables exist before using FINAL_VAR.
7. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.
{custom_tools_section}

**MANDATORY: Always decompose into parallel batches**
Your default approach MUST be to break the problem into independent subtasks and issue them as a single `rlm_query_batched(...)` call. Do NOT call `rlm_query(...)` multiple times sequentially when the subtasks are independent. Parallel batching is always faster and is the correct default.

**When to use each function:**
- Use `llm_query` only for a single, already-scoped one-shot task (extract one fact, summarize one chunk, classify one item). Never for multi-step reasoning.
- Use `llm_query_batched` when you have multiple independent one-shot tasks that each satisfy the `llm_query` criteria. Issues them all concurrently.
- Use `rlm_query` ONLY when you have a single complex subtask that requires multi-step reasoning and cannot be parallelised with siblings (e.g. a sequential dependency where result A must feed task B).
- **Use `rlm_query_batched` (PREFERRED) whenever you have 2 or more independent subtasks that each need deeper thinking.** This runs all children in parallel and returns results in the same order. You MUST use this instead of multiple sequential `rlm_query` calls.

**Decomposition rule:** Before writing any REPL code, ask: "Can I split this into N independent subtasks?" If yes — and N ≥ 2 — you MUST issue a single `rlm_query_batched([task_1, task_2, ..., task_N])` call. Only fall back to sequential calls when there is a hard dependency between steps.

**Breaking down problems:** You must break problems into more digestible components — whether that means chunking or summarizing a large context, or decomposing a hard task into easier sub-problems and delegating them via `llm_query_batched` / `rlm_query_batched`. Use the REPL to write a **programmatic strategy** that uses these LLM calls to solve the problem, as if you were building a parallel agent pipeline: plan all independent steps, fan them out in one batched call, then aggregate.

**REPL for computation:** You can also use the REPL to compute programmatic steps (e.g. `math.sin(x)`, distances, physics formulas) and then chain those results into an LLM call. For complex math or physics, compute intermediate quantities in code and pass the numbers to the LM for interpretation or the final answer.

You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context.

Make sure to explicitly look through the entire context in REPL before answering your query. Break the context and the problem into digestible pieces: figure out a chunking strategy, break up the context into smart chunks, then issue ALL chunk queries as a single `rlm_query_batched` or `llm_query_batched` call.

**Example — correct parallel decomposition:**
```repl
# Decompose into independent subtasks and issue as one parallel batch
subtasks = [
    f"Given this document excerpt, identify the key claim: {{chunk}}"
    for chunk in chunks
]
# ONE batched call — all children run in parallel
results = rlm_query_batched(subtasks)
# Aggregate
final_answer = llm_query(f"Synthesize these findings into one answer to: {{query}}\n\n" + "\n".join(results))
```

**Example — correct sequential use (hard dependency):**
```repl
# Step 1 must complete before step 2 can start — sequential is correct here
outline = rlm_query(f"Produce a detailed outline for answering: {{query}}")
final_answer = rlm_query(f"Using this outline:\n{{outline}}\n\nWrite the full answer to: {{query}}")
```

**Example — WRONG — do not do this:**
```repl
# WRONG: sequential rlm_query calls for independent subtasks
ans1 = rlm_query(f"Analyse document 1: {{doc1}}")
ans2 = rlm_query(f"Analyse document 2: {{doc2}}")  # runs AFTER ans1 — wasteful!
ans3 = rlm_query(f"Analyse document 3: {{doc3}}")  # runs AFTER ans2 — wasteful!
```
Instead write:
```repl
# CORRECT: one parallel batch
ans1, ans2, ans3 = rlm_query_batched([
    f"Analyse document 1: {{doc1}}",
    f"Analyse document 2: {{doc2}}",
    f"Analyse document 3: {{doc3}}",
])
```

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier.

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

WARNING - COMMON MISTAKE: FINAL_VAR retrieves an EXISTING variable. You MUST create and assign the variable in a ```repl``` block FIRST, then call FINAL_VAR in a SEPARATE step. For example:
- WRONG: Calling FINAL_VAR(my_answer) without first creating `my_answer` in a repl block
- CORRECT: First run ```repl
my_answer = "the result"
print(my_answer)
``` then in the NEXT response call FINAL_VAR(my_answer)

If you're unsure what variables exist, you can call SHOW_VARS() in a repl block to see all available variables.

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.
"""
)


def build_rlm_system_prompt(
    system_prompt: str,
    query_metadata: QueryMetadata,
    custom_tools: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """
    Build the initial system prompt for the REPL environment based on extra prompt metadata.

    Args:
        system_prompt: The base system prompt template.
        query_metadata: QueryMetadata object containing context metadata.
        custom_tools: Optional dict of custom tools to include in the prompt.

    Returns:
        List of message dictionaries
    """
    from rlm.environments.base_env import format_tools_for_prompt

    context_lengths = query_metadata.context_lengths
    context_total_length = query_metadata.context_total_length
    context_type = query_metadata.context_type

    # If there are more than 100 chunks, truncate to the first 100 chunks.
    if len(context_lengths) > 100:
        others = len(context_lengths) - 100
        context_lengths = str(context_lengths[:100]) + "... [" + str(others) + " others]"

    # Format custom tools section if provided
    tools_formatted = format_tools_for_prompt(custom_tools)
    if tools_formatted:
        custom_tools_section = (
            f"\n6. Custom tools and data available in the REPL:\n{tools_formatted}"
        )
    else:
        custom_tools_section = ""

    # Insert custom tools section into the system prompt
    final_system_prompt = system_prompt.format(custom_tools_section=custom_tools_section)

    metadata_prompt = f"Your context is a {context_type} with {context_total_length} total characters, and is broken up into chunks of char lengths: {context_lengths}."

    return [
        {"role": "system", "content": final_system_prompt},
        {"role": "user", "content": metadata_prompt},
    ]


USER_PROMPT = """Think step-by-step on what to do using the REPL environment (which contains the context) to answer the prompt.\n\nContinue using the REPL environment, which has the `context` variable, and querying sub-LLMs by writing to ```repl``` tags, and determine your answer. Your next action:"""
USER_PROMPT_WITH_ROOT = """Think step-by-step on what to do using the REPL environment (which contains the context) to answer the original prompt: \"{root_prompt}\".\n\nContinue using the REPL environment, which has the `context` variable, and querying sub-LLMs by writing to ```repl``` tags, and determine your answer. Your next action:"""


def build_user_prompt(
    root_prompt: str | None = None,
    iteration: int = 0,
    context_count: int = 1,
    history_count: int = 0,
) -> dict[str, str]:
    if iteration == 0:
        safeguard = "You have not interacted with the REPL environment or seen your prompt / context yet. Your next action should be to look through and figure out how to answer the prompt, so don't just provide a final answer yet.\n\n"
        prompt = safeguard + (
            USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt) if root_prompt else USER_PROMPT
        )
    else:
        prompt = "The history before is your previous interactions with the REPL environment. " + (
            USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt) if root_prompt else USER_PROMPT
        )

    # Inform model about multiple contexts if present
    if context_count > 1:
        prompt += f"\n\nNote: You have {context_count} contexts available (context_0 through context_{context_count - 1})."

    # Inform model about prior conversation histories if present
    if history_count > 0:
        if history_count == 1:
            prompt += "\n\nNote: You have 1 prior conversation history available in the `history` variable."
        else:
            prompt += f"\n\nNote: You have {history_count} prior conversation histories available (history_0 through history_{history_count - 1})."

    return {"role": "user", "content": prompt}

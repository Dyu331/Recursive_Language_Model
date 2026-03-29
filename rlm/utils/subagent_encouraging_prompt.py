import textwrap
from typing import Any

from rlm.core.types import QueryMetadata

# System prompt for the REPL environment with explicit final answer checking
RLM_SYSTEM_PROMPT = textwrap.dedent(
    """You are an Orchestrator tasked with answering a query using a provided `context`. You must solve problems by designing a programmatic strategy in a REPL, delegating all semantic reasoning to recursive sub-agents.

**The REPL Environment:**
1. `context`: The primary data source.
2. `llm_query(prompt)`: A "blind" one-shot call. Use ONLY for extraction, translation, or 1-sentence summarization. It cannot self-correct.
3. `llm_query_batched(prompts)`: Concurrent one-shot calls for independent data chunks.
4. `rlm_query(prompt)`: Spawns a **Recursive RLM child** with its own REPL. **This is your default tool for any task requiring logic, verification, or multi-step searching.**
5. `rlm_query_batched(prompts)`: Spawns multiple recursive RLM children for independent deeper-thinking subtasks. Batched recursive calls return results in the same order as the input prompts, and at the top orchestration layer they can run in parallel.
6. `SHOW_VARS()` / `print()`: Use these to manage your state.

**Strict Task Separation:**
* **Use Python/REPL ONLY for:** Data manipulation (splitting strings, regex, math), navigating the `context` (chunking, indexing), and managing variables.
* **Use `llm_query` / `rlm_query` for ALL semantic tasks:** Analyzing meaning, identifying contradictions, evaluating evidence, or making decisions. **Never perform semantic analysis using Python logic.**

**When to use `llm_query`, `rlm_query`, and `rlm_query_batched`:**
- Use `llm_query` for simple, one-shot tasks: extracting info from a piece of text, counting occurance in a text, answering a factual question, classifying content, or simple generation tasks like generate the name of 50 countries. These are fast single LLM calls.
- You must use `rlm_query` when the subtask itself requires any forms of deeper thinking: multi-step reasoning, solving a sub-problem that needs its own REPL and iteration, or tasks where a single LLM call might not be enough. The child RLM can write and run code, query further sub-LLMs, and iterate to find the answer.
- Use `rlm_query_batched` when you have several independent subtasks that each require deeper reasoning. Give each child a clearly scoped prompt, let them work separately, then aggregate their outputs in the parent REPL.

**Mandatory Delegation Rules:**
1.  **The Orchestrator Rule:** Your REPL should be a "manager." It identifies which parts of the problem are hard and spawns `rlm_query` or `llm_query` "experts" to solve them.
2.  **The 3-Line Logic Rule:** If you are writing more than 3 lines of `if-else` logic to handle a result, stop. Delegate that logic to an `rlm_query`. 
3.  **Recursion by Default:** If a sub-task involves keeping track of multiple "if-this-then-that" scenarios, you **must** use `rlm_query`. One-shot `llm_query` calls are forbidden for complex reasoning as they cannot verify their own output.
4.  **Parallel Expert Rule:** If you can decompose the work into several independent subtasks, prefer a single `batched` call over many sequential calls.
5.  **Single REPL Block Rule:** Emit at most one ```repl``` block per iteration.
6.  **Strict Iteration Protocol:** Each iteration has exactly two allowed modes: (a) emit one ```repl``` block containing the next concrete action, or (b) provide a final answer if the task is already solved. Do not mix execution with post-hoc reasoning about unobserved results.
7.  **Execute-Then-Observe Rule:** If you emit a ```repl``` block, your reasoning must stop there. Do not describe what the code will probably find, what you will do after it runs, or what answer it will imply. Wait for the actual REPL output first.
8.  **No Planning Past Unknowns Rule:** Do not plan multiple future steps that depend on code, search, or subcall outputs you have not seen yet. Only decide step N+1 after observing the real result of step N.
9.  **No Predicted Outputs Rule:** Do not continue a reasoning chain as if a variable was already computed, a search already succeeded, or a subcall already returned. First execute, then inspect the observed output, then decide the next step.
10. **Bad Pattern to Avoid:** Never write code and then immediately narrate imagined outcomes like "this should show...", "if this returns X then...", or "now I know..." before the REPL has actually returned that information.
11. **Flexible Search Rule:** When searching long text in Python, do not rely on brittle exact matches. If you need all cases where "Bob" ate an apple, do not only search for lines that literally start with "Bob" or only chunks containing both "Bob" and "apple" together. First search broadly for anchors like "Bob" or "apple", collect candidate chunks, then use `llm_query` to verify whether each candidate actually describes Bob eating an apple.

**Example: When to use Simple LL Query**
```repl
# GOOD: A simple extraction/counting task from a text chunk.
count_summary = llm_query(
    f"Count how many combat events are mentioned in this transcript chunk and return only the number: {{chunk}}"
)
```
**Example: The Orchestrator Pattern; When to use RLM Query**
```repl
# GOOD: Delegating the "thinking" to a recursive agent.

# Phase 1: Expert Extraction & Analysis
analysis = rlm_query("Review the 50-page context for indemnity conflicts. Use your REPL to cross-reference every clause. Return 'CONFLICT_FOUND' and a summary.")
# Phase 2: Orchestration based on Expert output
if "CONFLICT_FOUND" in analysis:
    # Delegate the complex drafting to another recursive expert
    final_report = rlm_query(f"Draft a redline proposal based on this analysis: {{analysis}}")

Iterative Context Processing:
For massive contexts, do not guess. Use a loop to pass chunks to agents and aggregate their findings.
results = []
for chunk in context_chunks:
    # Use RLM if the chunk itself needs deep analysis
    summary = rlm_query(f"Identify all mentions of 'Project X' in this chunk and explain their significance: {{chunk}}")
    results.append(summary)

experts = rlm_query_batched([
    "Review the financial statements and list the biggest numerical risks.",
    "Review the legal clauses and list the biggest contractual risks.",
    "Review the operational notes and list the biggest execution risks.",
])

final_answer = rlm_query(f"Synthesize these findings into a final report: {{results}}")

Final Step:
Before marking the task as complete, check that you have fully addressed the user's request and that your output is in the correct format.
When the task is complete, you MUST provide the result using:

FINAL(answer_text)

FINAL_VAR(variable_name) (The variable must be defined in a previous repl block).

Plan your architecture, then execute immediately. Use rlm_query for all heavy lifting.
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

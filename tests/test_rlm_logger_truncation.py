"""Tests for RLMLogger REPL stdout/stderr truncation."""

from rlm.core.types import (
    CodeBlock,
    REPLResult,
    RLMChatCompletion,
    RLMIteration,
    RLMMetadata,
    UsageSummary,
)
from rlm.logger.rlm_logger import RLMLogger, _truncate_repl_streams, _truncate_words


def test_truncate_words_short_unchanged() -> None:
    text = "one two three"
    assert _truncate_words(text, 200) == text


def test_truncate_words_long_truncates_with_suffix() -> None:
    words = [f"w{i}" for i in range(250)]
    text = " ".join(words)
    out = _truncate_words(text, 200)
    assert "truncated: 50 words omitted" in out
    first_line = out.split("\n")[0]
    assert first_line == " ".join(words[:200])


def test_truncate_repl_streams_nested_stdout() -> None:
    long_out = " ".join(f"x{i}" for i in range(220))
    nested = {
        "metadata": {
            "iterations": [
                {
                    "code_blocks": [
                        {
                            "result": {
                                "stdout": long_out,
                                "stderr": "short err",
                            }
                        }
                    ]
                }
            ]
        }
    }
    out = _truncate_repl_streams(nested, 200)
    assert out["metadata"]["iterations"][0]["code_blocks"][0]["result"]["stderr"] == "short err"
    truncated = out["metadata"]["iterations"][0]["code_blocks"][0]["result"]["stdout"]
    assert "truncated: 20 words omitted" in truncated
    assert truncated.startswith(" ".join(f"x{i}" for i in range(200)))


def test_truncate_repl_streams_tuple_preserved_as_tuple() -> None:
    obj = {"items": ({"stdout": "a b c d e"},)}
    out = _truncate_repl_streams(obj, 2)
    assert isinstance(out["items"], tuple)
    assert out["items"][0]["stdout"] == "a b\n... [truncated: 3 words omitted]"


def test_rlm_logger_none_disables_truncation() -> None:
    long_stdout = " ".join(f"w{i}" for i in range(250))
    result = REPLResult(stdout=long_stdout, stderr="", locals={})
    block = CodeBlock(code="print(x)", result=result)
    iteration = RLMIteration(prompt="p", response="r", code_blocks=[block])

    logger = RLMLogger(truncate_repl_output_words=None)
    meta = RLMMetadata(
        root_model="m",
        max_depth=1,
        max_iterations=10,
        backend="openai",
        backend_kwargs={},
        environment_type="local",
        environment_kwargs={},
    )
    logger.log_metadata(meta)
    logger.log(iteration)

    traj = logger.get_trajectory()
    assert traj is not None
    stored = traj["iterations"][0]["code_blocks"][0]["result"]["stdout"]
    assert stored == long_stdout


def test_rlm_logger_default_truncates_in_trajectory() -> None:
    long_stdout = " ".join(f"w{i}" for i in range(250))
    result = REPLResult(stdout=long_stdout, stderr="", locals={})
    block = CodeBlock(code="print(x)", result=result)
    iteration = RLMIteration(prompt="p", response="r", code_blocks=[block])

    logger = RLMLogger()  # default truncate_repl_output_words=200
    meta = RLMMetadata(
        root_model="m",
        max_depth=1,
        max_iterations=10,
        backend="openai",
        backend_kwargs={},
        environment_type="local",
        environment_kwargs={},
    )
    logger.log_metadata(meta)
    logger.log(iteration)

    traj = logger.get_trajectory()
    assert traj is not None
    stored = traj["iterations"][0]["code_blocks"][0]["result"]["stdout"]
    assert "truncated: 50 words omitted" in stored
    assert len(long_stdout) > len(stored)


def test_rlm_logger_truncates_nested_metadata_stdout() -> None:
    """Simulate child trajectory embedded under rlm_calls[].metadata."""
    long_child = " ".join(f"c{i}" for i in range(220))
    nested_meta = {
        "run_metadata": {"root_model": "sub"},
        "iterations": [
            {
                "code_blocks": [
                    {
                        "result": {
                            "stdout": long_child,
                            "stderr": "",
                        }
                    }
                ]
            }
        ],
    }
    usage = UsageSummary(model_usage_summaries={})
    call = RLMChatCompletion(
        root_model="sub",
        prompt="subprompt",
        response="subresp",
        usage_summary=usage,
        execution_time=0.1,
        metadata=nested_meta,
    )
    top_result = REPLResult(
        stdout="small",
        stderr="",
        locals={},
        rlm_calls=[call],
    )
    block = CodeBlock(code="rlm_query(...)", result=top_result)
    iteration = RLMIteration(prompt="p", response="r", code_blocks=[block])

    logger = RLMLogger(truncate_repl_output_words=200)
    meta = RLMMetadata(
        root_model="m",
        max_depth=2,
        max_iterations=10,
        backend="openai",
        backend_kwargs={},
        environment_type="local",
        environment_kwargs={},
    )
    logger.log_metadata(meta)
    logger.log(iteration)

    traj = logger.get_trajectory()
    assert traj is not None
    inner = traj["iterations"][0]["code_blocks"][0]["result"]["rlm_calls"][0]["metadata"]
    inner_stdout = inner["iterations"][0]["code_blocks"][0]["result"]["stdout"]
    assert "truncated: 20 words omitted" in inner_stdout

from rlm.utils.prompt_profiles import list_prompt_profiles, resolve_prompt_profile
from rlm.utils.prompts import QueryMetadata


def test_force_subagent_profile_is_registered() -> None:
    assert "force_subagent" in list_prompt_profiles()

    profile = resolve_prompt_profile("force_subagent")

    assert profile.name == "force_subagent"
    assert profile.module_name == "rlm.utils.force_subagent_prompt"
    assert "Mandatory Recursive Delegation Gate" in profile.system_prompt


def test_force_subagent_profile_has_strict_user_prompt() -> None:
    profile = resolve_prompt_profile("force-subagent")

    prompt = profile.build_user_prompt(
        root_prompt="Find the answer",
        iteration=0,
        context_count=1,
        history_count=0,
    )

    assert prompt["role"] == "user"
    assert "exactly one ```repl``` block" in prompt["content"]
    assert "`rlm_query(...)` or `rlm_query_batched(...)`" in prompt["content"]


def test_force_subagent_profile_builds_system_prompt() -> None:
    profile = resolve_prompt_profile("force_subagent")

    messages = profile.build_rlm_system_prompt(
        system_prompt=profile.system_prompt,
        query_metadata=QueryMetadata("abc"),
        custom_tools=None,
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "Context sample:\\\\n{sample}" in messages[0]["content"]

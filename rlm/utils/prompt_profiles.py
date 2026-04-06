from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from rlm.utils import prompts as default_prompts


PromptBuilderFn = Callable[[str, Any, dict[str, Any] | None], list[dict[str, str]]]
UserPromptFn = Callable[[str | None, int, int, int], dict[str, str]]


@dataclass(frozen=True)
class PromptProfile:
    name: str
    module_name: str
    system_prompt: str
    build_rlm_system_prompt: PromptBuilderFn
    build_user_prompt: UserPromptFn


BUILTIN_PROMPT_PROFILES: dict[str, str] = {
    "default": "rlm.utils.prompts",
    "encourage_subagent": "rlm.utils.subagent_encouraging_prompt",
    "force_subagent": "rlm.utils.force_subagent_prompt",
}


def normalize_prompt_profile_name(profile: str | None) -> str:
    raw = "default" if profile is None else profile
    normalized = raw.strip().lower().replace("-", "_")
    return normalized if normalized != "" else "default"


def list_prompt_profiles() -> list[str]:
    return sorted(BUILTIN_PROMPT_PROFILES.keys())


def resolve_prompt_profile(profile: str | None) -> PromptProfile:
    normalized = normalize_prompt_profile_name(profile)
    candidate_modules = get_candidate_modules(normalized)

    for module_name in candidate_modules:
        module = load_prompt_module(module_name)
        if module is None:
            continue
        return build_prompt_profile(normalized, module, module_name)

    builtins = ", ".join(list_prompt_profiles())
    raise ValueError(
        f"Unknown prompt_profile='{profile}'. "
        f"Built-ins: {builtins}. "
        "Or provide a module as 'module:pkg.path' / 'import:pkg.path', "
        "or add rlm.utils.<name>.py (or rlm.utils.<name>_prompt.py) "
        "with RLM_SYSTEM_PROMPT."
    )


def get_candidate_modules(normalized_profile: str) -> list[str]:
    if normalized_profile in BUILTIN_PROMPT_PROFILES:
        return [BUILTIN_PROMPT_PROFILES[normalized_profile]]

    if normalized_profile.startswith(("module:", "import:")):
        _, module_path = normalized_profile.split(":", 1)
        if module_path.strip() == "":
            raise ValueError("prompt_profile module path cannot be empty")
        return [module_path.strip()]

    return [
        f"rlm.utils.{normalized_profile}",
        f"rlm.utils.{normalized_profile}_prompt",
    ]


def load_prompt_module(module_name: str) -> ModuleType | None:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None


def build_prompt_profile(
    profile_name: str,
    module: ModuleType,
    module_name: str,
) -> PromptProfile:
    system_prompt = getattr(module, "RLM_SYSTEM_PROMPT", None)
    if not isinstance(system_prompt, str) or system_prompt.strip() == "":
        raise ValueError(
            f"Prompt module '{module_name}' must define non-empty RLM_SYSTEM_PROMPT"
        )

    build_prompt_fn = getattr(module, "build_rlm_system_prompt", None)
    if not callable(build_prompt_fn):
        build_prompt_fn = default_prompts.build_rlm_system_prompt

    build_user_fn = getattr(module, "build_user_prompt", None)
    if not callable(build_user_fn):
        build_user_fn = default_prompts.build_user_prompt

    return PromptProfile(
        name=profile_name,
        module_name=module_name,
        system_prompt=system_prompt,
        build_rlm_system_prompt=build_prompt_fn,
        build_user_prompt=build_user_fn,
    )

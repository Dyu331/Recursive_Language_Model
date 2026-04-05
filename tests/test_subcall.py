"""Unit tests for RLM._subcall() method.

Tests for the parameter propagation to child RLM instances:
1. max_timeout (remaining time) is passed to child
2. max_tokens is passed to child
3. max_errors is passed to child
4. model= parameter overrides child's backend model
"""

import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import pytest

import rlm.core.rlm as rlm_module
from rlm import RLM
from rlm.core.types import ModelUsageSummary, RLMChatCompletion, UsageSummary


def create_mock_lm(responses: list[str], model_name: str = "mock-model") -> Mock:
    """Create a mock LM that returns responses in order."""
    mock = Mock()
    mock.model_name = model_name
    mock.completion.side_effect = list(responses)
    mock.get_usage_summary.return_value = UsageSummary(
        model_usage_summaries={
            model_name: ModelUsageSummary(
                total_calls=1, total_input_tokens=100, total_output_tokens=50
            )
        }
    )
    mock.get_last_usage.return_value = mock.get_usage_summary.return_value
    return mock


class TestSubcallTimeoutPropagation:
    """Tests for max_timeout propagation to child RLM."""

    def test_child_receives_remaining_timeout(self):
        """When parent has max_timeout=60 and 10s have elapsed, child should get max_timeout approx 50."""
        captured_child_params = {}

        # Create a fake child RLM class to capture initialization params
        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                # Capture the kwargs before calling parent
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(["FINAL(answer)"])
            mock_get_client.return_value = mock_lm

            # Create parent RLM with max_timeout
            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                max_depth=3,  # Need depth > 1 to allow child spawning
                max_timeout=60.0,
            )

            # Simulate that 10 seconds have elapsed since completion started
            parent._completion_start_time = time.perf_counter() - 10.0

            # Patch RLM class to capture child creation
            with patch.object(rlm_module, "RLM", CapturingRLM):
                # Call _subcall which should spawn a child RLM
                parent._subcall("test prompt")

            # Verify child received remaining timeout (approximately 50 seconds)
            assert "max_timeout" in captured_child_params
            remaining = captured_child_params["max_timeout"]
            # Allow some tolerance for test execution time
            assert 45.0 < remaining < 55.0, f"Expected ~50s remaining, got {remaining}"

            parent.close()

    def test_child_receives_none_timeout_when_parent_has_none(self):
        """When parent has no max_timeout, child should also have None."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(["FINAL(answer)"])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                max_depth=3,
                max_timeout=None,  # No timeout
            )

            with patch.object(rlm_module, "RLM", CapturingRLM):
                parent._subcall("test prompt")

            assert captured_child_params.get("max_timeout") is None

            parent.close()

    def test_subcall_returns_error_when_timeout_exhausted(self):
        """When timeout is already exhausted, _subcall should return error message."""
        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(["FINAL(answer)"])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                max_depth=3,
                max_timeout=10.0,
            )

            # Simulate that more time has elapsed than the timeout
            parent._completion_start_time = time.perf_counter() - 15.0

            result = parent._subcall("test prompt")

            assert "Error: Timeout exhausted" in result.response

            parent.close()


class TestSubcallTokensPropagation:
    """Tests for max_tokens propagation to child RLM."""

    def test_child_receives_max_tokens(self):
        """Child RLM should get same max_tokens as parent."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(["FINAL(answer)"])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                max_depth=3,
                max_tokens=50000,
            )

            with patch.object(rlm_module, "RLM", CapturingRLM):
                parent._subcall("test prompt")

            assert captured_child_params.get("max_tokens") == 50000

            parent.close()

    def test_child_receives_none_tokens_when_parent_has_none(self):
        """When parent has no max_tokens, child should also have None."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(["FINAL(answer)"])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                max_depth=3,
                max_tokens=None,
            )

            with patch.object(rlm_module, "RLM", CapturingRLM):
                parent._subcall("test prompt")

            assert captured_child_params.get("max_tokens") is None

            parent.close()


class TestSubcallErrorsPropagation:
    """Tests for max_errors propagation to child RLM."""

    def test_child_receives_max_errors(self):
        """Child RLM should get same max_errors as parent."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(["FINAL(answer)"])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                max_depth=3,
                max_errors=5,
            )

            with patch.object(rlm_module, "RLM", CapturingRLM):
                parent._subcall("test prompt")

            assert captured_child_params.get("max_errors") == 5

            parent.close()

    def test_child_receives_none_errors_when_parent_has_none(self):
        """When parent has no max_errors, child should also have None."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(["FINAL(answer)"])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                max_depth=3,
                max_errors=None,
            )

            with patch.object(rlm_module, "RLM", CapturingRLM):
                parent._subcall("test prompt")

            assert captured_child_params.get("max_errors") is None

            parent.close()


class TestSubcallModelOverride:
    """Tests for model= parameter override in _subcall."""

    def test_model_override_sets_child_backend_kwargs(self):
        """When llm_query(prompt, model='test-model') is called, child's backend_kwargs should have model_name='test-model'."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(["FINAL(answer)"])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model", "api_key": "test-key"},
                max_depth=3,
            )

            with patch.object(rlm_module, "RLM", CapturingRLM):
                # Call _subcall with model override
                parent._subcall("test prompt", model="override-model")

            # Verify child received overridden model in backend_kwargs
            child_backend_kwargs = captured_child_params.get("backend_kwargs", {})
            assert child_backend_kwargs.get("model_name") == "override-model"
            # Original kwargs should be preserved
            assert child_backend_kwargs.get("api_key") == "test-key"

            parent.close()

    def test_model_override_does_not_mutate_parent_kwargs(self):
        """Model override should not mutate parent's backend_kwargs."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(["FINAL(answer)"])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                max_depth=3,
            )

            original_model = parent.backend_kwargs["model_name"]

            with patch.object(rlm_module, "RLM", CapturingRLM):
                parent._subcall("test prompt", model="override-model")

            # Parent's backend_kwargs should be unchanged
            assert parent.backend_kwargs["model_name"] == original_model

            parent.close()

    def test_no_model_override_uses_parent_kwargs(self):
        """When no model override is provided, child uses parent's backend_kwargs."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(["FINAL(answer)"])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                max_depth=3,
            )

            with patch.object(rlm_module, "RLM", CapturingRLM):
                # Call _subcall without model override
                parent._subcall("test prompt")

            # Child should use parent's backend_kwargs
            child_backend_kwargs = captured_child_params.get("backend_kwargs", {})
            assert child_backend_kwargs.get("model_name") == "parent-model"

            parent.close()

    def test_fixed_subagent_kwargs_override_parent_defaults(self):
        """When subagent_backend_kwargs is set, child should use it instead of root backend_kwargs."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(["FINAL(answer)"])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model", "api_key": "root-key"},
                subagent_backend_kwargs={"model_name": "child-model", "api_key": "child-key"},
                max_depth=3,
            )

            with patch.object(rlm_module, "RLM", CapturingRLM):
                parent._subcall("test prompt")

            child_backend_kwargs = captured_child_params.get("backend_kwargs", {})
            assert child_backend_kwargs.get("model_name") == "child-model"
            assert child_backend_kwargs.get("api_key") == "child-key"

            parent.close()

    def test_model_override_wins_over_fixed_subagent_kwargs(self):
        """Explicit model override should take precedence over subagent_backend_kwargs."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(["FINAL(answer)"])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model", "api_key": "root-key"},
                subagent_backend_kwargs={"model_name": "child-model", "api_key": "child-key"},
                max_depth=3,
            )

            with patch.object(rlm_module, "RLM", CapturingRLM):
                parent._subcall("test prompt", model="override-model")

            child_backend_kwargs = captured_child_params.get("backend_kwargs", {})
            assert child_backend_kwargs.get("model_name") == "override-model"
            assert child_backend_kwargs.get("api_key") == "child-key"

            parent.close()

    def test_fixed_subagent_backend_propagates_to_child(self):
        """When subagent_backend is set, child should receive that backend."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(["FINAL(answer)"])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                subagent_backend="openrouter",
                subagent_backend_kwargs={"model_name": "child-model"},
                max_depth=3,
            )

            with patch.object(rlm_module, "RLM", CapturingRLM):
                parent._subcall("test prompt")

            assert captured_child_params.get("backend") == "openrouter"
            assert (
                captured_child_params.get("backend_kwargs", {}).get("model_name") == "child-model"
            )

            parent.close()


class TestSubcallModelOverrideAtLeafDepth:
    """Tests for model override at max_depth (leaf LM completion)."""

    def test_model_override_at_leaf_depth_uses_overridden_model(self):
        """When at max_depth, the leaf LM completion should use the overridden model."""
        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(["leaf response"])
            mock_get_client.return_value = mock_lm

            # Parent at depth 1, max_depth 2 means next depth (2) will be at max_depth
            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                depth=1,
                max_depth=2,
            )

            # Call _subcall with model override - should trigger leaf LM completion
            result = parent._subcall("test prompt", model="leaf-override-model")

            # Verify get_client was called with overridden model in backend_kwargs
            # The call should be: get_client("openai", {"model_name": "leaf-override-model"})
            call_args = mock_get_client.call_args_list
            # Find the call that has the overridden model
            found_override_call = False
            for call in call_args:
                args, kwargs = call
                if len(args) >= 2:
                    backend_kwargs = args[1]
                    if (
                        isinstance(backend_kwargs, dict)
                        and backend_kwargs.get("model_name") == "leaf-override-model"
                    ):
                        found_override_call = True
                        break

            assert found_override_call, (
                f"Expected get_client to be called with model_name='leaf-override-model', got calls: {call_args}"
            )
            assert result.response == "leaf response"

            parent.close()

    def test_leaf_depth_without_model_override_uses_parent_model(self):
        """When at max_depth without model override, uses parent's model."""
        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(["FINAL(answer)"] * 2 + ["leaf response"])
            mock_get_client.return_value = mock_lm

            # Parent at depth 1, max_depth 2 means next depth (2) will be at max_depth
            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                depth=1,
                max_depth=2,
            )

            # Call _subcall without model override
            parent._subcall("test prompt")

            # Verify get_client was called with parent's model
            # The last call should use the parent's backend_kwargs
            call_args = mock_get_client.call_args_list
            # Check the most recent call (for leaf completion)
            last_call = call_args[-1]
            args, _ = last_call
            if len(args) >= 2:
                backend_kwargs = args[1]
                assert backend_kwargs.get("model_name") == "parent-model"

            parent.close()

    def test_leaf_depth_uses_fixed_subagent_backend_and_model(self):
        """Leaf completions should use the resolved fixed subagent backend and model."""
        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(["leaf response"])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                subagent_backend="openrouter",
                subagent_backend_kwargs={"model_name": "child-model", "api_key": "child-key"},
                depth=1,
                max_depth=2,
            )

            result = parent._subcall("test prompt")

            call_args = mock_get_client.call_args_list
            found_subagent_call = False
            for call in call_args:
                args, _ = call
                if len(args) >= 2:
                    backend = args[0]
                    backend_kwargs = args[1]
                    if (
                        backend == "openrouter"
                        and isinstance(backend_kwargs, dict)
                        and backend_kwargs.get("model_name") == "child-model"
                        and backend_kwargs.get("api_key") == "child-key"
                    ):
                        found_subagent_call = True
                        break

            assert found_subagent_call, (
                f"Expected get_client to be called with openrouter/child-model, got calls: {call_args}"
            )
            assert result.response == "leaf response"

            parent.close()


class TestSubagentModelSelector:
    """Tests for subagent_model_selector in child resolution."""

    def test_selector_sets_child_model_when_no_explicit_override(self):
        """Selector result should become the child's model_name when no explicit override is provided."""
        captured_child_params = {}
        selector_calls: list[tuple[str, int]] = []

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        def select_model(prompt: str, depth: int) -> str:
            selector_calls.append((prompt, depth))
            return "selector-model"

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(["FINAL(answer)"])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model", "api_key": "root-key"},
                subagent_backend_kwargs={"model_name": "child-model", "api_key": "child-key"},
                subagent_model_selector=select_model,
                max_depth=3,
            )

            with patch.object(rlm_module, "RLM", CapturingRLM):
                parent._subcall("test prompt")

            child_backend_kwargs = captured_child_params.get("backend_kwargs", {})
            assert selector_calls == [("test prompt", 1)]
            assert child_backend_kwargs.get("model_name") == "selector-model"
            assert child_backend_kwargs.get("api_key") == "child-key"

            parent.close()

    def test_explicit_model_override_wins_over_selector(self):
        """Explicit model override should bypass selector output."""
        captured_child_params = {}
        selector_calls: list[tuple[str, int]] = []

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        def select_model(prompt: str, depth: int) -> str:
            selector_calls.append((prompt, depth))
            return "selector-model"

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(["FINAL(answer)"])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model", "api_key": "root-key"},
                subagent_backend_kwargs={"model_name": "child-model", "api_key": "child-key"},
                subagent_model_selector=select_model,
                max_depth=3,
            )

            with patch.object(rlm_module, "RLM", CapturingRLM):
                parent._subcall("test prompt", model="override-model")

            child_backend_kwargs = captured_child_params.get("backend_kwargs", {})
            assert selector_calls == []
            assert child_backend_kwargs.get("model_name") == "override-model"
            assert child_backend_kwargs.get("api_key") == "child-key"

            parent.close()

    def test_selector_applies_at_leaf_depth(self):
        """Leaf completions should use the selector-resolved model when no explicit override is provided."""
        selector_calls: list[tuple[str, int]] = []

        def select_model(prompt: str, depth: int) -> str:
            selector_calls.append((prompt, depth))
            return "selector-leaf-model"

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(["leaf response"])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                subagent_backend="openrouter",
                subagent_backend_kwargs={"model_name": "child-model", "api_key": "child-key"},
                subagent_model_selector=select_model,
                depth=1,
                max_depth=2,
            )

            result = parent._subcall("test prompt")

            call_args = mock_get_client.call_args_list
            found_selector_call = False
            for call in call_args:
                args, _ = call
                if len(args) >= 2:
                    backend = args[0]
                    backend_kwargs = args[1]
                    if (
                        backend == "openrouter"
                        and isinstance(backend_kwargs, dict)
                        and backend_kwargs.get("model_name") == "selector-leaf-model"
                        and backend_kwargs.get("api_key") == "child-key"
                    ):
                        found_selector_call = True
                        break

            assert selector_calls == [("test prompt", 2)]
            assert found_selector_call, (
                f"Expected get_client to be called with openrouter/selector-leaf-model, got calls: {call_args}"
            )
            assert result.response == "leaf response"

            parent.close()


class TestSubcallCombinedParameters:
    """Tests for combined parameter propagation."""

    def test_all_parameters_propagate_together(self):
        """All parameters (timeout, tokens, errors, model) should propagate correctly together."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(["FINAL(answer)"])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model", "api_key": "test-key"},
                max_depth=3,
                max_timeout=120.0,
                max_tokens=100000,
                max_errors=10,
            )

            # Simulate 30 seconds elapsed
            parent._completion_start_time = time.perf_counter() - 30.0

            with patch.object(rlm_module, "RLM", CapturingRLM):
                parent._subcall("test prompt", model="override-model")

            # Verify all parameters
            assert captured_child_params.get("max_tokens") == 100000
            assert captured_child_params.get("max_errors") == 10

            # Remaining timeout should be around 90 seconds
            remaining_timeout = captured_child_params.get("max_timeout")
            assert 85.0 < remaining_timeout < 95.0

            # Model should be overridden
            child_backend_kwargs = captured_child_params.get("backend_kwargs", {})
            assert child_backend_kwargs.get("model_name") == "override-model"
            assert child_backend_kwargs.get("api_key") == "test-key"

            parent.close()


class TestSubcallConcurrency:
    """Tests for concurrent parent-side subcall accounting."""

    def test_concurrent_subcalls_accumulate_parent_cost(self):
        """Concurrent child completions should not lose cumulative cost updates."""

        class StubChildRLM:
            def __init__(self, *args, **kwargs):
                pass

            def completion(self, prompt: str, root_prompt=None) -> RLMChatCompletion:
                time.sleep(0.05)
                return RLMChatCompletion(
                    root_model="child-model",
                    prompt=prompt,
                    response=f"child {prompt}",
                    usage_summary=UsageSummary(
                        model_usage_summaries={
                            "child-model": ModelUsageSummary(
                                total_calls=1,
                                total_input_tokens=10,
                                total_output_tokens=5,
                                total_cost=0.25,
                            )
                        }
                    ),
                    execution_time=0.05,
                )

            def close(self) -> None:
                pass

        parent = RLM(
            backend="openai",
            backend_kwargs={"model_name": "parent-model"},
            max_depth=3,
        )

        with patch.object(rlm_module, "RLM", StubChildRLM):
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(parent._subcall, ["a", "b", "c", "d"]))

        assert [result.response for result in results] == [
            "child a",
            "child b",
            "child c",
            "child d",
        ]
        assert parent._cumulative_cost == 1.0

        parent.close()


class TestIterationBudgetCheck:
    """Tests for per-iteration budget enforcement."""

    def test_iteration_budget_check_overwrites_cumulative_cost_from_handler(self):
        """Iteration budget check should set cumulative cost from handler's reported cost."""
        from rlm.core.types import RLMIteration

        parent = RLM(
            backend="openai",
            backend_kwargs={"model_name": "parent-model"},
            max_budget=1.0,
        )

        mock_handler = Mock()
        mock_handler.get_usage_summary.return_value = UsageSummary(
            model_usage_summaries={
                "parent-model": ModelUsageSummary(
                    total_calls=1,
                    total_input_tokens=100,
                    total_output_tokens=50,
                    total_cost=0.3,
                )
            }
        )

        iteration = RLMIteration(prompt="test", response="code", code_blocks=[])
        parent._check_iteration_limits(iteration, 0, mock_handler)

        assert parent._cumulative_cost == pytest.approx(0.3)

        parent.close()

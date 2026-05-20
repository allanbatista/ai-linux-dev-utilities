"""Unit tests for ab_cli.utils.llm_helpers."""
import argparse
from unittest.mock import patch

from ab_cli.core.llm_settings import add_llm_request_arguments
from ab_cli.utils.llm_helpers import call_llm, call_llm_with_model_info, get_llm_text


class TestLlmRequestArguments:
    def test_add_llm_request_arguments_parses_overrides(self):
        parser = argparse.ArgumentParser()
        add_llm_request_arguments(parser)

        args = parser.parse_args([
            "--reasoning-effort",
            "high",
            "--service-tier",
            "flex",
        ])

        assert args.reasoning_effort == "high"
        assert args.service_tier == "flex"


class TestCallLlm:
    def test_call_llm_uses_config_defaults(self, mock_config):
        with patch("ab_cli.utils.llm_helpers.send_to_openrouter") as mock_send:
            mock_send.return_value = {"text": "ok"}

            result = call_llm("prompt text")

            assert result == {"text": "ok"}
            kwargs = mock_send.call_args.kwargs
            assert kwargs["reasoning_effort"] == "medium"
            assert kwargs["service_tier"] == "default"

    def test_call_llm_overrides_config(self, mock_config):
        with patch("ab_cli.utils.llm_helpers.send_to_openrouter") as mock_send:
            mock_send.return_value = {"text": "ok"}

            call_llm(
                "prompt text",
                reasoning_effort="high",
                service_tier="flex",
            )

            kwargs = mock_send.call_args.kwargs
            assert kwargs["reasoning_effort"] == "high"
            assert kwargs["service_tier"] == "flex"


class TestCallLlmWithModelInfo:
    def test_call_llm_with_model_info_forwards_overrides(self, mock_config):
        with patch("ab_cli.utils.llm_helpers.send_to_openrouter") as mock_send:
            mock_send.return_value = {"text": "ok"}

            result, model, tokens = call_llm_with_model_info(
                "prompt text",
                reasoning_effort="low",
                service_tier="priority",
            )

            assert result == {"text": "ok"}
            assert model == "test/model-small"
            assert tokens > 0
            kwargs = mock_send.call_args.kwargs
            assert kwargs["reasoning_effort"] == "low"
            assert kwargs["service_tier"] == "priority"


class TestGetLlmText:
    def test_get_llm_text_returns_stripped_text(self, mock_config):
        with patch("ab_cli.utils.llm_helpers.send_to_openrouter") as mock_send:
            mock_send.return_value = {"text": "  hello  "}

            text = get_llm_text(
                "prompt text",
                reasoning_effort="minimal",
                service_tier="default",
            )

            assert text == "hello"

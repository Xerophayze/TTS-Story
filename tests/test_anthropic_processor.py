"""Exercise the actual Anthropic wire format without spending API credits."""
from unittest.mock import Mock, patch

import pytest
import requests

from src.anthropic_processor import AnthropicProcessor, AnthropicProcessorError, API_VERSION
from src.structured_output import StructuredOutputError


def response(data, status=200):
    return Mock(status_code=status, json=Mock(return_value=data))


def message(text="[narrator]Hello.[/narrator]", stop="end_turn"):
    return {"stop_reason": stop, "content": [
        {"type": "thinking", "thinking": "private reasoning"},
        {"type": "text", "text": text},
    ]}


@patch("src.anthropic_processor.requests.request")
def test_native_auth_workspace_and_response_blocks(request):
    request.return_value = response(message())
    client = AnthropicProcessor("test-key", "claude-test", workspace_id="wrkspc_test")
    assert client.generate_text("Preserve this text") == "[narrator]Hello.[/narrator]"
    args, kwargs = request.call_args
    assert args == ("POST", "https://api.anthropic.com/v1/messages")
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert kwargs["headers"]["anthropic-version"] == API_VERSION
    assert kwargs["headers"]["anthropic-workspace-id"] == "wrkspc_test"
    assert kwargs["json"] == {"model": "claude-test", "max_tokens": 8192,
                              "messages": [{"role": "user", "content": "Preserve this text"}]}


@pytest.mark.parametrize("stop", ["max_tokens", "refusal", "tool_use", "pause_turn", None])
@patch("src.anthropic_processor.requests.request")
def test_incomplete_or_refused_text_is_not_accepted(request, stop):
    request.return_value = response(message("partial text", stop))
    with pytest.raises(AnthropicProcessorError):
        AnthropicProcessor("test-key", "claude-test").generate_text("source")
    assert request.call_count == 1


@patch("src.anthropic_processor.requests.request")
def test_pagination_filters_inactive_models_and_deduplicates(request):
    request.side_effect = [
        response({"data": [{"id": "claude-a"}, {"id": "old", "lifecycle": "retired"}],
                  "has_more": True, "last_id": "cursor"}),
        response({"data": [{"id": "claude-a"}, {"id": "claude-b"}], "has_more": False}),
    ]
    assert AnthropicProcessor.list_available_models("test-key") == ["claude-a", "claude-b"]
    assert request.call_args.kwargs["params"]["after_id"] == "cursor"


@patch("src.anthropic_processor.requests.request")
def test_error_redacts_key_and_retains_status_for_failover(request):
    request.return_value = response({"error": {"message": "overloaded test-key"}}, 529)
    with pytest.raises(AnthropicProcessorError) as error:
        AnthropicProcessor("test-key", "claude-test").generate_text("source")
    assert "test-key" not in str(error.value)
    assert "HTTP 529" in str(error.value)


@patch("src.anthropic_processor.requests.request")
def test_transport_error_redacts_key(request):
    request.side_effect = requests.ConnectionError("connection failed test-key")
    with pytest.raises(AnthropicProcessorError, match="connection") as error:
        AnthropicProcessor("test-key", "claude-test").generate_text("source")
    assert "test-key" not in str(error.value)


@patch("src.anthropic_processor.requests.request")
def test_native_schema_request_and_local_response_validation(request):
    schema = {"type": "object", "properties": {"text": {"type": "string"}},
              "required": ["text"], "additionalProperties": False}
    request.return_value = response(message('{"text":"Hello"}'))
    client = AnthropicProcessor("test-key", "claude-test")
    assert client.generate_text("source", response_schema=schema) == '{"text":"Hello"}'
    assert request.call_args.kwargs["json"]["output_config"]["format"]["schema"] == schema
    request.return_value = response(message('{"wrong":"Hello"}'))
    with pytest.raises(StructuredOutputError):
        client.generate_text("source", response_schema=schema)


@patch("src.anthropic_processor.requests.request")
def test_missing_key_or_model_never_calls_api(request):
    with pytest.raises(AnthropicProcessorError):
        AnthropicProcessor("")
    with pytest.raises(AnthropicProcessorError):
        AnthropicProcessor("test-key").generate_text("source")
    request.assert_not_called()

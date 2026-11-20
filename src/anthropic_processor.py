"""Direct Claude Console integration through Anthropic's Messages API."""
from __future__ import annotations

import re
import requests

from src.structured_output import validate_schema, parse_structured_response

API_BASE = "https://api.anthropic.com/v1"
API_VERSION = "2026-10-21"


class AnthropicProcessorError(RuntimeError):
    """A safe-to-display Claude discovery or generation failure."""


def _safe_detail(value, key):
    text = str(value or "").replace(key, "[redacted]") if key else str(value or "")
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "[redacted]", text)
    return re.sub(r"\s+", " ", text).strip()[:600]


class AnthropicProcessor:
    def __init__(self, api_key, model_name="", timeout=300, max_tokens=8192, workspace_id=""):
        self.api_key = str(api_key or "").strip()
        self.model_name = str(model_name or "").strip()
        self.timeout = max(10, min(1800, int(timeout or 300)))
        self.max_tokens = max(256, min(64000, int(max_tokens or 8192)))
        self.workspace_id = str(workspace_id or "").strip()
        if not self.api_key:
            raise AnthropicProcessorError("Claude API key is required")

    def _request(self, method, path, **kwargs):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "anthropic-version": API_VERSION,
            "Content-Type": "application/json",
        }
        if self.workspace_id:
            headers["anthropic-workspace-id"] = self.workspace_id
        try:
            response = requests.request(method, API_BASE + path, headers=headers,
                                        timeout=self.timeout, **kwargs)
        except requests.RequestException as exc:
            raise AnthropicProcessorError(
                "Claude API connection failed: " + _safe_detail(exc, self.api_key)
            ) from None
        try:
            data = response.json()
        except ValueError:
            raise AnthropicProcessorError(
                f"Claude API returned invalid JSON (HTTP {response.status_code})"
            ) from None
        if response.status_code >= 400:
            error = data.get("error", {}) if isinstance(data, dict) else {}
            detail = error.get("message", "") if isinstance(error, dict) else error
            raise AnthropicProcessorError(
                f"Claude API error (HTTP {response.status_code}): " + _safe_detail(detail, self.api_key)
            )
        if not isinstance(data, dict):
            raise AnthropicProcessorError("Claude API returned an unexpected response")
        return data

    @classmethod
    def list_available_models(cls, api_key, timeout=30, workspace_id=""):
        client = cls(api_key, timeout=min(60, int(timeout or 30)), workspace_id=workspace_id)
        models, seen_cursors = [], set()
        params = {"limit": 100}
        for _ in range(20):
            data = client._request("GET", "/models", params=params)
            for model in data.get("data", []):
                if (isinstance(model, dict) and model.get("id")
                        and model.get("lifecycle", "active") == "active"
                        and model["id"] not in models):
                    models.append(model["id"])
            if not data.get("has_more"):
                break
            cursor = data.get("last_id")
            if not cursor or cursor in seen_cursors:
                raise AnthropicProcessorError("Claude model discovery returned an invalid page cursor")
            seen_cursors.add(cursor)
            params = {"limit": 100, "after_id": cursor}
        else:
            raise AnthropicProcessorError("Claude model discovery exceeded its page limit")
        if not models:
            raise AnthropicProcessorError("No active Claude models were returned for this API key")
        return models

    def generate_text(self, prompt, response_schema=None, response_schema_name=None,
                      response_schema_strict=True):
        if not self.model_name:
            raise AnthropicProcessorError("Claude model is required; fetch and select a model in Settings")
        if not str(prompt or "").strip():
            raise AnthropicProcessorError("Prompt must not be empty")
        payload = {"model": self.model_name, "max_tokens": self.max_tokens,
                   "messages": [{"role": "user", "content": prompt}]}
        # Do not forward OpenAI/local sampling or reasoning controls to Claude.
        if response_schema is not None:
            validate_schema(response_schema, response_schema_name, response_schema_strict)
            payload["output_config"] = {"format": {"type": "json_schema", "schema": response_schema}}
        data = self._request("POST", "/messages", json=payload)
        reason = data.get("stop_reason")
        if reason == "max_tokens":
            raise AnthropicProcessorError(
                "Claude output was truncated at its token limit. Reduce prep chunk size or increase Claude max output tokens; incomplete text was not accepted."
            )
        if reason != "end_turn":
            raise AnthropicProcessorError(
                "Claude did not return a complete text response (stop reason: "
                + _safe_detail(reason or "missing", self.api_key) + ")"
            )
        text = "".join(block.get("text", "") for block in data.get("content", [])
                       if isinstance(block, dict) and block.get("type") == "text").strip()
        if not text:
            raise AnthropicProcessorError("Claude response did not contain manuscript text")
        if response_schema is not None:
            parse_structured_response(text, response_schema)
        return text

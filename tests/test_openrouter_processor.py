import unittest
from unittest.mock import patch

from src.openrouter_processor import (
    DEFAULT_OPENROUTER_BASE_URL,
    OpenRouterProcessor,
    OpenRouterProcessorError,
)


class FakeResponse:
    def __init__(self, payload=None, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text
        self.content = b"content" if payload is not None or text else b""

    def json(self):
        if self._payload is None:
            raise ValueError("not JSON")
        return self._payload


class OpenRouterProcessorTests(unittest.TestCase):
    def test_normalizes_base_url(self):
        self.assertEqual(
            OpenRouterProcessor.normalize_base_url("https://openrouter.ai"),
            DEFAULT_OPENROUTER_BASE_URL,
        )
        self.assertEqual(
            OpenRouterProcessor.normalize_base_url("https://openrouter.ai/api/v1/"),
            DEFAULT_OPENROUTER_BASE_URL,
        )

    @patch("src.openrouter_processor.requests.get")
    def test_model_discovery_uses_user_filtered_text_models(self, mock_get):
        mock_get.return_value = FakeResponse({
            "data": [
                {
                    "id": "anthropic/claude-sonnet",
                    "architecture": {"output_modalities": ["text"]},
                },
                {
                    "id": "image/generator",
                    "architecture": {"output_modalities": ["image"]},
                },
                {"id": "openrouter/auto"},
            ]
        })

        models = OpenRouterProcessor.list_available_models("router-key")

        self.assertIn("anthropic/claude-sonnet", models)
        self.assertIn("openrouter/auto", models)
        self.assertNotIn("image/generator", models)
        call = mock_get.call_args
        self.assertEqual(call.args[0], f"{DEFAULT_OPENROUTER_BASE_URL}/models/user")
        self.assertEqual(call.kwargs["headers"]["Authorization"], "Bearer router-key")
        self.assertEqual(call.kwargs["headers"]["X-OpenRouter-Title"], "TTS-Story")

    @patch("src.openrouter_processor.requests.post")
    def test_generation_uses_openai_compatible_endpoint(self, mock_post):
        mock_post.return_value = FakeResponse(
            {"choices": [{"message": {"content": "processed story"}}]}
        )
        processor = OpenRouterProcessor(
            api_key="router-key",
            model_name="openrouter/auto",
            temperature=0.2,
            top_p=0.9,
            top_k=20,
            repetition_penalty=1.1,
            max_tokens=1024,
            disable_reasoning=True,
        )

        result = processor.generate_text("Process this story")

        self.assertEqual(result, "processed story")
        call = mock_post.call_args
        self.assertEqual(call.args[0], f"{DEFAULT_OPENROUTER_BASE_URL}/chat/completions")
        self.assertEqual(call.kwargs["headers"]["Authorization"], "Bearer router-key")
        self.assertEqual(call.kwargs["json"]["repetition_penalty"], 1.1)
        self.assertEqual(call.kwargs["json"]["reasoning"], {"effort": "none"})

    @patch("src.openrouter_processor.requests.post")
    def test_generation_surfaces_credit_error(self, mock_post):
        mock_post.return_value = FakeResponse(
            {"error": {"message": "Insufficient credits"}},
            status_code=402,
        )
        processor = OpenRouterProcessor(
            api_key="router-key",
            model_name="openrouter/auto",
        )

        with self.assertRaisesRegex(OpenRouterProcessorError, "Insufficient credits"):
            processor.generate_text("Process this story")

    def test_requires_api_key(self):
        with self.assertRaises(OpenRouterProcessorError):
            OpenRouterProcessor(api_key="", model_name="openrouter/auto")


if __name__ == "__main__":
    unittest.main()

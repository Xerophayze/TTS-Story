import unittest
from unittest.mock import patch

from src.atlas_cloud_processor import (
    AtlasCloudProcessor,
    AtlasCloudProcessorError,
    DEFAULT_ATLAS_CLOUD_BASE_URL,
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


class AtlasCloudProcessorTests(unittest.TestCase):
    def test_normalizes_base_url(self):
        self.assertEqual(
            AtlasCloudProcessor.normalize_base_url("https://api.atlascloud.ai"),
            DEFAULT_ATLAS_CLOUD_BASE_URL,
        )
        self.assertEqual(
            AtlasCloudProcessor.normalize_base_url("https://api.atlascloud.ai/v1/"),
            DEFAULT_ATLAS_CLOUD_BASE_URL,
        )

    @patch("src.atlas_cloud_processor.requests.get")
    def test_model_discovery_merges_api_and_public_llm_catalog(self, mock_get):
        mock_get.side_effect = [
            FakeResponse({"data": [{"id": "deepseek-v3"}, {"id": "image/flux"}]}),
            FakeResponse(
                text=(
                    '<a href="/models/deepseek-ai/DeepSeek-V3.1">DeepSeek</a>'
                    '<a href="/models/alibaba/Qwen3">Qwen</a>'
                    '<a href="/models/google/nano-banana/text-to-image">Image</a>'
                )
            ),
        ]

        catalog = AtlasCloudProcessor.list_available_models("atlas-key")

        self.assertIn("deepseek-v3", catalog.models)
        self.assertIn("deepseek-ai/DeepSeek-V3.1", catalog.models)
        self.assertIn("alibaba/Qwen3", catalog.models)
        self.assertNotIn("image/flux", catalog.models)
        self.assertFalse(any("text-to-image" in model for model in catalog.models))
        self.assertEqual(catalog.warnings, [])
        self.assertEqual(
            mock_get.call_args_list[0].kwargs["headers"]["Authorization"],
            "Bearer atlas-key",
        )

    @patch("src.atlas_cloud_processor.requests.get")
    def test_model_discovery_falls_back_to_public_catalog(self, mock_get):
        mock_get.side_effect = [
            FakeResponse({"error": {"message": "Unauthorized"}}, status_code=401),
            FakeResponse(text='<a href="/models/alibaba/Qwen3">Qwen</a>'),
        ]

        catalog = AtlasCloudProcessor.list_available_models("bad-key")

        self.assertIn("alibaba/Qwen3", catalog.models)
        self.assertTrue(any("HTTP 401" in warning for warning in catalog.warnings))

    @patch("src.atlas_cloud_processor.requests.post")
    def test_generation_uses_openai_compatible_endpoint(self, mock_post):
        mock_post.return_value = FakeResponse(
            {"choices": [{"message": {"content": "processed story"}}]}
        )
        processor = AtlasCloudProcessor(
            api_key="atlas-key",
            model_name="deepseek-v3",
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
        self.assertEqual(call.args[0], f"{DEFAULT_ATLAS_CLOUD_BASE_URL}/chat/completions")
        self.assertEqual(call.kwargs["headers"]["Authorization"], "Bearer atlas-key")
        self.assertEqual(call.kwargs["json"]["repetition_penalty"], 1.1)
        self.assertEqual(call.kwargs["json"]["thinking"], {"type": "disabled"})

    def test_requires_api_key(self):
        with self.assertRaises(AtlasCloudProcessorError):
            AtlasCloudProcessor(api_key="", model_name="deepseek-v3")


if __name__ == "__main__":
    unittest.main()

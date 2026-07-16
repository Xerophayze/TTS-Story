from __future__ import annotations

import ast
import builtins
import importlib
import sys
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class GeminiStartupTests(unittest.TestCase):
    def test_gemini_wrapper_does_not_import_google_sdk_at_module_import(self) -> None:
        imported_google_modules = []
        original_import = builtins.__import__

        def tracking_import(name, *args, **kwargs):
            if name == "google" or name.startswith("google."):
                imported_google_modules.append(name)
            return original_import(name, *args, **kwargs)

        sys.modules.pop("src.gemini_processor", None)
        with mock.patch("builtins.__import__", side_effect=tracking_import):
            importlib.import_module("src.gemini_processor")

        self.assertEqual([], imported_google_modules)

    def test_app_imports_engines_before_gemini_wrapper(self) -> None:
        tree = ast.parse((PROJECT_ROOT / "app.py").read_text(encoding="utf-8"))
        engine_lines = []
        gemini_lines = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "src.engines" or node.module.startswith("src.engines."):
                    engine_lines.append(node.lineno)
                if node.module == "src.gemini_processor":
                    gemini_lines.append(node.lineno)

        self.assertTrue(engine_lines, "No engine imports found in app.py")
        self.assertEqual(1, len(gemini_lines))
        self.assertLess(max(engine_lines), gemini_lines[0])

    def test_model_discovery_uses_lazy_loaded_modern_sdk(self) -> None:
        from src.gemini_processor import GeminiProcessor

        models_api = SimpleNamespace(
            list=lambda: [
                SimpleNamespace(name="models/gemini-2.5-flash"),
                SimpleNamespace(name="models/gemini-2.5-pro"),
            ]
        )
        sdk = SimpleNamespace(Client=lambda **_kwargs: SimpleNamespace(models=models_api))

        with mock.patch(
            "src.gemini_processor._load_genai_sdk",
            return_value=(sdk, True),
        ):
            models = GeminiProcessor.list_available_models("test-key")

        self.assertEqual(
            ["models/gemini-2.5-flash", "models/gemini-2.5-pro"],
            models,
        )

    def test_model_discovery_uses_lazy_loaded_legacy_sdk(self) -> None:
        from src.gemini_processor import GeminiProcessor

        configured = []
        sdk = SimpleNamespace(
            configure=lambda **kwargs: configured.append(kwargs),
            list_models=lambda: [
                SimpleNamespace(
                    name="models/gemini-flash",
                    supported_generation_methods=["generateContent"],
                ),
                SimpleNamespace(
                    name="models/embedding-only",
                    supported_generation_methods=["embedContent"],
                ),
            ],
        )

        with mock.patch(
            "src.gemini_processor._load_genai_sdk",
            return_value=(sdk, False),
        ):
            models = GeminiProcessor.list_available_models("legacy-key")

        self.assertEqual([{"api_key": "legacy-key"}], configured)
        self.assertEqual(["models/gemini-flash"], models)


if __name__ == "__main__":
    unittest.main()

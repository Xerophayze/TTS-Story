from __future__ import annotations

import unittest

from scripts.engine_versions import collect_versions


class EngineVersionReportTests(unittest.TestCase):
    def test_report_has_runtime_packages_and_model_defaults(self) -> None:
        report = collect_versions()

        self.assertIn("python", report)
        self.assertIn("Chatterbox Turbo", report["packages"])
        self.assertEqual(
            "ResembleAI/chatterbox-turbo",
            report["default_models"]["Chatterbox Turbo"],
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from flask import Flask

from src.help_center import clear_help_catalog_cache, create_help_blueprint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HELP_ROOT = PROJECT_ROOT / "docs" / "help"
MANIFEST_PATH = HELP_ROOT / "manifest.json"


def _walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


class HelpRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        clear_help_catalog_cache()
        app = Flask(__name__)
        app.config.update(TESTING=True)
        app.register_blueprint(
            create_help_blueprint(HELP_ROOT, name="help_center_route_tests")
        )
        cls.client = app.test_client()

    @classmethod
    def tearDownClass(cls) -> None:
        clear_help_catalog_cache()

    def assert_no_source_paths(self, payload) -> None:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        normalized = serialized.replace("\\", "/").lower()
        self.assertNotIn(str(HELP_ROOT.resolve()).replace("\\", "/").lower(), normalized)
        self.assertNotIn("docs/help/", normalized)
        for article in self.manifest["articles"]:
            self.assertNotIn(article["file"].replace("\\", "/").lower(), normalized)

        forbidden_keys = {"file", "path", "source", "source_path"}
        self.assertTrue(forbidden_keys.isdisjoint(set(_walk_keys(payload))))

    def test_catalog_and_every_article_route_succeed_without_path_leaks(self) -> None:
        response = self.client.get("/api/help/catalog")
        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(
            {article["id"] for article in self.manifest["articles"]},
            {article["id"] for article in payload["articles"]},
        )
        self.assertTrue(all("html" not in article for article in payload["articles"]))
        self.assert_no_source_paths(payload)

        for expected in self.manifest["articles"]:
            with self.subTest(article=expected["id"]):
                article_response = self.client.get(
                    f"/api/help/articles/{expected['id']}"
                )
                self.assertEqual(200, article_response.status_code)
                article_payload = article_response.get_json()
                self.assertTrue(article_payload["success"])
                self.assertEqual(expected["id"], article_payload["article"]["id"])
                self.assertTrue(article_payload["article"]["html"].strip())
                self.assertIn(
                    expected["id"], article_payload["category"]["article_ids"]
                )
                self.assert_no_source_paths(article_payload)

    def test_alias_resolves_to_canonical_article(self) -> None:
        alias, target = next(iter(self.manifest["aliases"].items()))
        alias_response = self.client.get(f"/api/help/articles/{alias}")
        canonical_response = self.client.get(f"/api/help/articles/{target}")

        self.assertEqual(200, alias_response.status_code)
        self.assertEqual(200, canonical_response.status_code)
        self.assertEqual(target, alias_response.get_json()["article"]["id"])
        self.assertEqual(alias_response.get_json(), canonical_response.get_json())
        self.assertEqual(alias_response.headers.get("ETag"), canonical_response.headers.get("ETag"))

    def test_unknown_invalid_and_traversal_article_ids_return_404(self) -> None:
        paths = [
            "/api/help/articles/not-a-real-help-article",
            "/api/help/articles/bad_id",
            "/api/help/articles/%2e%2e",
            "/api/help/articles/..%2Fmanifest.json",
            "/api/help/articles/%2Fetc%2Fpasswd",
        ]
        normalized_root = str(HELP_ROOT.resolve()).replace("\\", "/").lower()

        for path in paths:
            with self.subTest(path=path):
                response = self.client.get(path, follow_redirects=False)
                self.assertEqual(404, response.status_code)
                body = response.get_data(as_text=True).replace("\\", "/").lower()
                self.assertNotIn(normalized_root, body)
                self.assertNotIn("docs/help/", body)

    def test_catalog_etag_supports_conditional_get(self) -> None:
        first = self.client.get("/api/help/catalog")
        self.assertEqual(200, first.status_code)
        etag = first.headers.get("ETag")
        self.assertTrue(etag)
        self.assertIn("must-revalidate", first.headers.get("Cache-Control", ""))

        conditional = self.client.get(
            "/api/help/catalog",
            headers={"If-None-Match": etag},
        )
        self.assertEqual(304, conditional.status_code)
        self.assertEqual(b"", conditional.data)
        self.assertEqual(etag, conditional.headers.get("ETag"))

    def test_article_etag_supports_conditional_get(self) -> None:
        article_id = self.manifest["articles"][0]["id"]
        url = f"/api/help/articles/{article_id}"
        first = self.client.get(url)
        self.assertEqual(200, first.status_code)
        etag = first.headers.get("ETag")
        self.assertTrue(etag)

        conditional = self.client.get(url, headers={"If-None-Match": etag})
        self.assertEqual(304, conditional.status_code)
        self.assertEqual(b"", conditional.data)
        self.assertEqual(etag, conditional.headers.get("ETag"))

    def test_missing_bundle_returns_generic_json_errors_without_path_leaks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            app = Flask(__name__)
            app.config.update(TESTING=True)
            app.register_blueprint(
                create_help_blueprint(
                    Path(temporary_root) / "missing-help",
                    name="missing_help_route_tests",
                )
            )
            client = app.test_client()

            for url in ("/api/help/catalog", "/api/help/articles/welcome"):
                with self.subTest(url=url):
                    response = client.get(url)
                    self.assertEqual(500, response.status_code)
                    payload = response.get_json()
                    self.assertEqual(
                        {"success": False, "error": "The bundled user guide is unavailable."},
                        payload,
                    )
                    self.assertNotIn(temporary_root.lower(), response.get_data(as_text=True).lower())

    def test_invalid_utf8_manifest_returns_generic_json_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            help_root = Path(temporary_root) / "help"
            help_root.mkdir()
            (help_root / "manifest.json").write_bytes(b"\xff\xfe\x00")
            app = Flask(__name__)
            app.config.update(TESTING=True)
            app.register_blueprint(
                create_help_blueprint(help_root, name="invalid_manifest_route_tests")
            )
            client = app.test_client()

            for url in ("/api/help/catalog", "/api/help/articles/welcome"):
                with self.subTest(url=url):
                    response = client.get(url)
                    self.assertEqual(500, response.status_code)
                    self.assertEqual(
                        {"success": False, "error": "The bundled user guide is unavailable."},
                        response.get_json(),
                    )


if __name__ == "__main__":
    unittest.main()

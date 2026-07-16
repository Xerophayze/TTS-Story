from __future__ import annotations

import ast
from collections import Counter
import json
from pathlib import Path
import re
import unittest

from src.help_center import _render_markdown, clear_help_catalog_cache, load_help_catalog


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HELP_ROOT = PROJECT_ROOT / "docs" / "help"
MANIFEST_PATH = HELP_ROOT / "manifest.json"
HELP_SCREENSHOTS_ROOT = PROJECT_ROOT / "static" / "help" / "screenshots"
ARTICLE_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
HELP_LINK_RE = re.compile(r"help:([^\s)\"'<>]+)")
APP_LINK_RE = re.compile(r"app:([^\s)\"'<>]+)")
DATA_HELP_ID_RE = re.compile(r"data-help-id\s*=\s*([\"'])(.*?)\1")
ENGINE_TAB_RE = re.compile(r"data-engine-tab\s*=\s*([\"'])(.*?)\1")
MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]\r\n]*)\]\(([^)\r\n]+)\)")
HELP_SCREENSHOT_TARGET_RE = re.compile(
    r"^\.\./\.\./\.\./static/help/screenshots/"
    r"([a-z0-9]+(?:-[a-z0-9]+)*\.(?:png|webp))$"
)
HELP_SCREENSHOT_FILENAME_RE = re.compile(
    r"^[a-z0-9]+(?:-[a-z0-9]+)*\.(?:png|webp)$"
)

REQUIRED_CORE_TOPICS_BY_CATEGORY = {
    "start-here": {
        "welcome",
        "first-run",
        "quick-start",
        "choose-engine",
        "online-services",
    },
    "create-audio": {
        "input-text",
        "speaker-tags",
        "prep-text",
        "assign-voices",
        "generation-options",
        "projects",
    },
    "tts-engines": {"engine-overview"},
    "llm-preparation": {"llm-overview"},
    "jobs": {"job-queue", "job-review", "generation-times"},
    "library": {"audio-library", "audiobook-exports"},
    "voices": {"available-voices", "voice-prompts", "voice-creation"},
    "settings-performance": {
        "settings",
        "settings-audio",
        "performance-tuning",
        "data-storage",
    },
    "troubleshooting": {
        "troubleshooting-overview",
        "cloud-errors",
        "gpu-cpu-errors",
        "audio-quality",
        "report-an-issue",
    },
}


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _declared_source_paths(manifest: dict) -> list[Path]:
    return [Path(article["file"]) for article in manifest["articles"]]


def _engine_registry_ids() -> set[str]:
    """Read the literal registry keys without importing every optional engine."""

    source_path = PROJECT_ROOT / "src" / "tts_engine.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in tree.body:
        target_name = None
        value = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
            value = node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                target_name = target.id
                value = node.value
        if target_name != "EngineRegistry":
            continue
        if not isinstance(value, ast.Dict):
            raise AssertionError("EngineRegistry must remain a literal dictionary")
        keys = []
        for key in value.keys:
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                raise AssertionError("EngineRegistry keys must be literal strings")
            keys.append(key.value)
        if len(keys) != len(set(keys)):
            raise AssertionError("EngineRegistry contains duplicate keys")
        return set(keys)
    raise AssertionError("EngineRegistry dictionary was not found")


class BundledHelpContentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = _load_manifest()
        clear_help_catalog_cache()
        cls.catalog = load_help_catalog(HELP_ROOT)

    @classmethod
    def tearDownClass(cls) -> None:
        clear_help_catalog_cache()

    def test_manifest_categories_are_an_exact_article_partition(self) -> None:
        articles = self.manifest.get("articles")
        categories = self.manifest.get("categories")
        aliases = self.manifest.get("aliases")

        self.assertIsInstance(articles, list)
        self.assertTrue(articles)
        self.assertIsInstance(categories, list)
        self.assertTrue(categories)
        self.assertIsInstance(aliases, dict)

        article_ids = [article.get("id") for article in articles]
        self.assertEqual(len(article_ids), len(set(article_ids)), "Duplicate article id")
        for article_id in article_ids:
            self.assertIsInstance(article_id, str)
            self.assertRegex(article_id, ARTICLE_ID_RE)

        category_ids = [category.get("id") for category in categories]
        self.assertEqual(len(category_ids), len(set(category_ids)), "Duplicate category id")
        assigned_ids = [
            article_id
            for category in categories
            for article_id in category.get("article_ids", [])
        ]
        self.assertEqual(
            Counter(article_ids),
            Counter(assigned_ids),
            "Every article must be assigned to exactly one category",
        )

        known_ids = set(article_ids)
        for article in articles:
            with self.subTest(article=article.get("id")):
                self.assertIsInstance(article.get("title"), str)
                self.assertTrue(article["title"].strip())
                self.assertIsInstance(article.get("summary"), str)
                self.assertTrue(article["summary"].strip())
                self.assertIsInstance(article.get("keywords"), list)
                self.assertIsInstance(article.get("related", []), list)
                self.assertIsInstance(article.get("engine_ids", []), list)
                self.assertLessEqual(set(article.get("related", [])), known_ids)

        for alias, target in aliases.items():
            with self.subTest(alias=alias):
                self.assertRegex(alias, ARTICLE_ID_RE)
                self.assertNotIn(alias, known_ids, "Alias must not shadow a canonical article")
                self.assertIn(target, known_ids)

        self.assertEqual(
            set(article_ids),
            {article["id"] for article in self.catalog["articles"]},
        )
        self.assertEqual(
            set(category_ids),
            {category["id"] for category in self.catalog["categories"]},
        )

        for article in self.catalog["articles"]:
            with self.subTest(rendered_article=article["id"]):
                self.assertTrue(article["html"].strip())
                self.assertNotRegex(
                    article["html"],
                    r"<h1\b",
                    "The reader shell supplies the single visible article H1",
                )
                self.assertTrue(article["search_text"].strip())
                self.assertGreater(article["word_count"], 0)
                self.assertGreaterEqual(article["reading_minutes"], 1)

    def test_rendered_markdown_strips_unsafe_html_and_link_schemes(self) -> None:
        rendered = _render_markdown(
            """# Safe title

<img src=x onerror=\"alert(1)\">
<img src=\"https://example.com/private.png\" alt=\"Remote\">
<img src=\"data:image/png;base64,AAAA\" alt=\"Data URL\">
<img src=\"/static/audio/private-job/output.png\" alt=\"User audio\">
<img src=\"../../../static/help/screenshots/../private.png\" alt=\"Traversal\">
<img src=\"../../../static/help/screenshots/generate-overview.svg\" alt=\"SVG\">
<img src=\"../../../static/help/screenshots/generate-overview.png\" alt=\"\" onerror=\"alert(9)\">
<script>alert('unsafe')</script>
<div onclick=\"alert(2)\">Visible text remains.</div>

[Unsafe](javascript:alert(3))
[Web](https://example.com/guide)
[Guide](help:quick-start)
[App](app:settings/edge-tts)

![Generate page overview](../../../static/help/screenshots/generate-overview.png \"Generate workflow\")
"""
        )

        self.assertNotIn("<h1", rendered)
        self.assertEqual(rendered.count("<img"), 1)
        self.assertIn('src="/static/help/screenshots/generate-overview.png"', rendered)
        self.assertIn('alt="Generate page overview"', rendered)
        self.assertIn('title="Generate workflow"', rendered)
        self.assertIn('loading="lazy"', rendered)
        self.assertIn('decoding="async"', rendered)
        self.assertNotIn("example.com/private.png", rendered)
        self.assertNotIn("data:image", rendered)
        self.assertNotIn("/static/audio", rendered)
        self.assertNotIn("../private.png", rendered)
        self.assertNotIn(".svg", rendered)
        self.assertNotIn("<script", rendered)
        self.assertNotIn("<div", rendered)
        self.assertNotIn("onerror", rendered)
        self.assertNotIn("onclick", rendered)
        self.assertNotIn("javascript:", rendered)
        self.assertIn("Visible text remains.", rendered)
        self.assertIn('href="https://example.com/guide"', rendered)
        self.assertIn('href="help:quick-start"', rendered)
        self.assertIn('href="app:settings/edge-tts"', rendered)

    def test_required_core_topics_exist_in_their_categories(self) -> None:
        category_articles = {
            category["id"]: set(category["article_ids"])
            for category in self.manifest["categories"]
        }
        self.assertLessEqual(
            set(REQUIRED_CORE_TOPICS_BY_CATEGORY),
            set(category_articles),
        )
        for category_id, required_ids in REQUIRED_CORE_TOPICS_BY_CATEGORY.items():
            with self.subTest(category=category_id):
                self.assertLessEqual(required_ids, category_articles[category_id])

    def test_each_source_h1_exactly_matches_manifest_title(self) -> None:
        for article in self.manifest["articles"]:
            source = (HELP_ROOT / article["file"]).read_text(encoding="utf-8")
            first_line = source.splitlines()[0] if source.splitlines() else ""
            with self.subTest(article=article["id"]):
                self.assertEqual(f"# {article['title']}", first_line)

    def test_every_markdown_source_is_declared_safe_and_nonempty(self) -> None:
        declared = _declared_source_paths(self.manifest)
        self.assertEqual(len(declared), len(set(declared)), "Duplicate article source file")

        resolved_root = HELP_ROOT.resolve()
        for relative in declared:
            with self.subTest(source=relative.as_posix()):
                self.assertFalse(relative.is_absolute())
                self.assertNotIn("..", relative.parts)
                self.assertEqual(relative.suffix.lower(), ".md")
                resolved = (HELP_ROOT / relative).resolve()
                resolved.relative_to(resolved_root)
                self.assertTrue(resolved.is_file())
                source = resolved.read_text(encoding="utf-8")
                self.assertTrue(source.strip())
                self.assertNotIn("\x00", source)
                self.assertTrue(source.lstrip().startswith("# "))

        actual = {
            path.relative_to(HELP_ROOT)
            for path in HELP_ROOT.rglob("*.md")
            if path.is_file()
        }
        self.assertEqual(
            set(declared),
            actual,
            "Every bundled Markdown file must be represented in the manifest",
        )

    def test_every_article_has_safe_existing_screenshots_and_every_asset_is_referenced(self) -> None:
        referenced_assets: set[str] = set()

        for relative in _declared_source_paths(self.manifest):
            source = (HELP_ROOT / relative).read_text(encoding="utf-8")
            images = list(MARKDOWN_IMAGE_RE.finditer(source))
            with self.subTest(source=relative.as_posix()):
                self.assertTrue(images, "Every help article must include a screenshot")

            for image in images:
                alt_text = image.group(1).strip()
                target = image.group(2).strip()
                with self.subTest(source=relative.as_posix(), image=target):
                    self.assertTrue(alt_text, "Instructional screenshots need descriptive alt text")
                    safe_target = HELP_SCREENSHOT_TARGET_RE.fullmatch(target)
                    self.assertIsNotNone(
                        safe_target,
                        "Screenshots must use the fixed local PNG/WebP help path",
                    )
                    if safe_target is None:
                        continue
                    filename = safe_target.group(1)
                    resolved = (HELP_ROOT / relative.parent / target).resolve()
                    resolved.relative_to(HELP_SCREENSHOTS_ROOT.resolve())
                    self.assertTrue(resolved.is_file(), f"Missing screenshot asset: {filename}")
                    referenced_assets.add(filename)

        actual_assets = {
            path.name
            for path in HELP_SCREENSHOTS_ROOT.iterdir()
            if path.is_file()
        }
        for filename in actual_assets:
            with self.subTest(screenshot_asset=filename):
                self.assertRegex(filename, HELP_SCREENSHOT_FILENAME_RE)
        self.assertEqual(
            actual_assets,
            referenced_assets,
            "Every bundled screenshot must be referenced and every reference must exist",
        )

    def test_all_internal_help_links_resolve(self) -> None:
        canonical_ids = {article["id"] for article in self.manifest["articles"]}
        valid_targets = canonical_ids | set(self.manifest.get("aliases", {}))
        links_seen = 0

        for relative in _declared_source_paths(self.manifest):
            source = (HELP_ROOT / relative).read_text(encoding="utf-8")
            for match in HELP_LINK_RE.finditer(source):
                links_seen += 1
                raw_target = match.group(1)
                target = raw_target.split("#", 1)[0].split("?", 1)[0]
                with self.subTest(source=relative.as_posix(), target=raw_target):
                    self.assertRegex(target, ARTICLE_ID_RE)
                    self.assertIn(target, valid_targets)

        self.assertGreater(links_seen, 0, "The guide should contain navigable help: links")

    def test_all_internal_app_links_resolve_to_supported_locations(self) -> None:
        template_source = (PROJECT_ROOT / "templates" / "index.html").read_text(
            encoding="utf-8"
        )
        engine_tabs = {
            match.group(2).strip()
            for match in ENGINE_TAB_RE.finditer(template_source)
        }
        self.assertTrue(engine_tabs, "Expected Settings engine tabs in index.html")

        supported_locations = {"generate", "queue", "library", "voices", "settings"}
        supported_locations.update(f"settings/{tab}" for tab in engine_tabs)
        links_seen = 0

        for relative in _declared_source_paths(self.manifest):
            source = (HELP_ROOT / relative).read_text(encoding="utf-8")
            for match in APP_LINK_RE.finditer(source):
                links_seen += 1
                raw_target = match.group(1)
                target = raw_target.split("#", 1)[0].split("?", 1)[0].rstrip("/")
                with self.subTest(source=relative.as_posix(), target=raw_target):
                    self.assertIn(target, supported_locations)

        self.assertGreater(links_seen, 0, "The guide should contain navigable app: links")

    def test_all_template_and_static_help_targets_resolve(self) -> None:
        canonical_ids = {article["id"] for article in self.manifest["articles"]}
        valid_targets = canonical_ids | set(self.manifest.get("aliases", {}))
        targets: list[tuple[Path, str]] = []

        candidates = list((PROJECT_ROOT / "templates").rglob("*.html"))
        candidates.extend((PROJECT_ROOT / "static").rglob("*.js"))
        for path in candidates:
            source = path.read_text(encoding="utf-8")
            for match in DATA_HELP_ID_RE.finditer(source):
                targets.append((path, match.group(2).strip()))

        self.assertTrue(targets, "Expected data-help-id targets in templates/static assets")
        for path, target in targets:
            with self.subTest(source=path.relative_to(PROJECT_ROOT), target=target):
                self.assertRegex(target, ARTICLE_ID_RE)
                self.assertIn(target, valid_targets)

    def test_help_shell_assets_and_single_initializer_are_wired(self) -> None:
        template = (PROJECT_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        help_script = (PROJECT_ROOT / "static" / "js" / "help.js").read_text(encoding="utf-8")
        main_script = (PROJECT_ROOT / "static" / "js" / "main.js").read_text(encoding="utf-8")
        help_styles = (PROJECT_ROOT / "static" / "css" / "help.css").read_text(encoding="utf-8")

        required_ids = {
            "help-tab",
            "help-center-search",
            "help-center-home",
            "help-center-reader",
            "help-center-sidebar-nav",
            "help-center-article-body",
            "help-center-toc",
            "help-center-related-links",
        }
        for element_id in required_ids:
            with self.subTest(element_id=element_id):
                self.assertIn(f'id="{element_id}"', template)

        self.assertIn('data-tab="help"', template)
        self.assertIn('/static/css/help.css', template)
        self.assertIn('/static/js/help.js', template)
        self.assertLess(template.index('/static/js/help.js'), template.index('/static/js/main.js'))
        self.assertIn("window.TTSStoryHelp", help_script)
        self.assertIn("window.TTSStoryHelp?.init()", main_script)
        self.assertNotIn("function initHelpSystem()", main_script)
        self.assertIn(".help-center .hidden", help_styles)
        self.assertIn(".help-center-article img", help_styles)
        self.assertRegex(help_styles, r"max-width\s*:\s*100%")
        self.assertRegex(help_styles, r"height\s*:\s*auto")
        self.assertNotIn('id="help-modal-overlay"', template)
        self.assertNotIn('id="help-search-modal-overlay"', template)

    def test_alt_word_preview_offers_every_selectable_job_engine(self) -> None:
        template = (PROJECT_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        select = re.search(
            r'<select\s+id="awr-preview-engine"[^>]*>(.*?)</select>',
            template,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(select)
        offered = set(re.findall(r'<option\s+value="([^"]+)"', select.group(1)))
        expected = _engine_registry_ids() - {"omnivoice_design"}
        self.assertEqual(expected, offered)

    def test_manifest_engine_ids_exactly_cover_engine_registry(self) -> None:
        engine_articles = {
            article["id"]: article.get("engine_ids", [])
            for article in self.manifest["articles"]
            if article.get("engine_ids")
        }
        documented = [
            engine_id
            for engine_ids in engine_articles.values()
            for engine_id in engine_ids
        ]

        self.assertEqual(
            len(documented),
            len(set(documented)),
            "An engine id must be owned by exactly one help article",
        )
        self.assertEqual(_engine_registry_ids(), set(documented))

        engine_category = next(
            category
            for category in self.manifest["categories"]
            if category["id"] == "tts-engines"
        )
        self.assertLessEqual(set(engine_articles), set(engine_category["article_ids"]))


if __name__ == "__main__":
    unittest.main()

"""Load and render the bundled TTS-Story user guide.

The guide is authored as Markdown under ``docs/help`` so it is readable on
GitHub and inside the application.  This module validates the manifest,
renders the articles to HTML, and returns a JSON-serializable catalog for the
in-app help center.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
from html import escape, unescape
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable

import markdown
from flask import Blueprint, current_app, jsonify, make_response, request


DEFAULT_HELP_ROOT = Path(__file__).resolve().parents[1] / "docs" / "help"
_ARTICLE_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_LEADING_H1_RE = re.compile(r"^\s*<h1\b[^>]*>.*?</h1>\s*", re.IGNORECASE | re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_SAFE_CODE_CLASS_RE = re.compile(r"^language-[A-Za-z0-9_-]{1,64}$")
_SAFE_WEB_URL_RE = re.compile(r"^https?://[^\s<>\"']+$", re.IGNORECASE)
_SAFE_INTERNAL_URL_RE = re.compile(r"^(?:help|app):[a-z0-9][a-z0-9/-]*$", re.IGNORECASE)
_SAFE_FRAGMENT_RE = re.compile(r"^#[A-Za-z0-9_-]+$")
_SAFE_SCREENSHOT_NAME_RE = re.compile(
    r"^[a-z0-9]+(?:-[a-z0-9]+)*\.(?:png|webp)$"
)
_HELP_SCREENSHOT_PREFIXES = (
    "../../../static/help/screenshots/",
    "/static/help/screenshots/",
)
_ALLOWED_HTML_TAGS = {
    "a",
    "blockquote",
    "br",
    "code",
    "dd",
    "dl",
    "dt",
    "em",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "img",
    "li",
    "ol",
    "p",
    "pre",
    "strong",
    "table",
    "tbody",
    "td",
    "th",
    "thead",
    "tr",
    "ul",
}
_VOID_HTML_TAGS = {"br", "hr", "img"}


class HelpCatalogError(RuntimeError):
    """Raised when bundled help content is missing or internally inconsistent."""


def _safe_help_href(value: str) -> str | None:
    normalized = str(value or "").strip()
    if (
        _SAFE_WEB_URL_RE.fullmatch(normalized)
        or _SAFE_INTERNAL_URL_RE.fullmatch(normalized)
        or _SAFE_FRAGMENT_RE.fullmatch(normalized)
    ):
        return normalized
    return None


def _safe_help_image_src(value: str) -> str | None:
    """Normalize a bundled screenshot path while rejecting every other image."""
    normalized = str(value or "").strip()
    for prefix in _HELP_SCREENSHOT_PREFIXES:
        if normalized.startswith(prefix):
            filename = normalized[len(prefix):]
            if _SAFE_SCREENSHOT_NAME_RE.fullmatch(filename):
                return f"/static/help/screenshots/{filename}"
    return None


class _HelpHtmlSanitizer(HTMLParser):
    """Strictly allow the small HTML subset emitted by the guide renderer."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def _clean_attributes(self, tag: str, attrs: list[tuple[str, str | None]]) -> str:
        cleaned: list[tuple[str, str]] = []
        for raw_name, raw_value in attrs:
            name = str(raw_name or "").lower()
            value = str(raw_value or "")
            if tag == "a" and name == "href":
                safe_href = _safe_help_href(value)
                if safe_href:
                    cleaned.append(("href", safe_href))
            elif tag == "a" and name == "title":
                cleaned.append(("title", value[:300]))
            elif tag in {"h2", "h3", "h4", "h5", "h6"} and name == "id":
                if _SAFE_ID_RE.fullmatch(value):
                    cleaned.append(("id", value))
            elif tag == "code" and name == "class":
                if _SAFE_CODE_CLASS_RE.fullmatch(value):
                    cleaned.append(("class", value))
            elif tag in {"td", "th"} and name == "align" and value in {"left", "center", "right"}:
                cleaned.append(("align", value))
            elif tag == "ol" and name == "start" and value.isdigit():
                cleaned.append(("start", value))
        return "".join(
            f' {name}="{escape(value, quote=True)}"'
            for name, value in cleaned
        )

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized_tag = tag.lower()
        if normalized_tag not in _ALLOWED_HTML_TAGS:
            return
        if normalized_tag == "img":
            raw_attrs = {
                str(name or "").lower(): str(value or "")
                for name, value in attrs
            }
            safe_src = _safe_help_image_src(raw_attrs.get("src", ""))
            alt_text = raw_attrs.get("alt", "").strip()
            if not safe_src or not alt_text:
                return

            image_attrs = [("src", safe_src), ("alt", alt_text[:300])]
            title = raw_attrs.get("title", "").strip()
            if title:
                image_attrs.append(("title", title[:300]))
            for dimension in ("width", "height"):
                raw_dimension = raw_attrs.get(dimension, "")
                if raw_dimension.isdigit() and 1 <= int(raw_dimension) <= 4096:
                    image_attrs.append((dimension, raw_dimension))
            image_attrs.extend((("loading", "lazy"), ("decoding", "async")))
            serialized = "".join(
                f' {name}="{escape(value, quote=True)}"'
                for name, value in image_attrs
            )
            self.parts.append(f"<img{serialized}>")
            return
        attributes = self._clean_attributes(normalized_tag, attrs)
        self.parts.append(f"<{normalized_tag}{attributes}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.lower()
        if normalized_tag in _ALLOWED_HTML_TAGS and normalized_tag not in _VOID_HTML_TAGS:
            self.parts.append(f"</{normalized_tag}>")

    def handle_data(self, data: str) -> None:
        self.parts.append(escape(data, quote=False))


def _sanitize_rendered_html(rendered_html: str) -> str:
    sanitizer = _HelpHtmlSanitizer()
    sanitizer.feed(rendered_html)
    sanitizer.close()
    return "".join(sanitizer.parts)


def _require_id(value: Any, label: str) -> str:
    normalized = str(value or "").strip()
    if not _ARTICLE_ID_RE.fullmatch(normalized):
        raise HelpCatalogError(f"Invalid {label}: {value!r}")
    return normalized


def _require_text(value: Any, label: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise HelpCatalogError(f"Missing {label}.")
    return normalized


def _safe_article_path(root: Path, relative_path: Any) -> Path:
    relative = Path(_require_text(relative_path, "article file"))
    if relative.is_absolute():
        raise HelpCatalogError(f"Help article path must be relative: {relative}")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise HelpCatalogError(f"Help article path leaves the help directory: {relative}") from exc
    if resolved.suffix.lower() != ".md":
        raise HelpCatalogError(f"Help article must be Markdown: {relative}")
    if not resolved.is_file():
        raise HelpCatalogError(f"Help article is missing: {relative}")
    return resolved


def _as_string_list(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise HelpCatalogError(f"{label} must be a list.")
    return [str(item).strip() for item in value if str(item).strip()]


def _plain_text(rendered_html: str) -> str:
    without_tags = _HTML_TAG_RE.sub(" ", rendered_html)
    return _WHITESPACE_RE.sub(" ", unescape(without_tags)).strip()


def _render_markdown(source: str) -> str:
    rendered = markdown.markdown(
        source,
        extensions=[
            "extra",
            "sane_lists",
            "toc",
        ],
        extension_configs={
            "toc": {
                "permalink": False,
                "slugify": lambda value, separator: re.sub(
                    r"[^a-z0-9]+",
                    separator,
                    value.lower(),
                ).strip(separator),
            }
        },
        output_format="html5",
    )
    # The reader shell already renders the manifest title as its single H1.
    # Keep source H1s for GitHub/offline reading without duplicating them in-app.
    without_source_title = _LEADING_H1_RE.sub("", rendered, count=1)
    return _sanitize_rendered_html(without_source_title)


def _validate_relations(articles: Iterable[Dict[str, Any]], article_ids: set[str]) -> None:
    for article in articles:
        for related_id in article.get("related", []):
            if related_id not in article_ids:
                raise HelpCatalogError(
                    f"Article {article['id']!r} references unknown related article {related_id!r}."
                )


@lru_cache(maxsize=4)
def load_help_catalog(help_root: str | Path = DEFAULT_HELP_ROOT) -> Dict[str, Any]:
    """Return the validated, rendered help catalog.

    ``help_root`` is accepted for focused tests and downstream packaging.  The
    default points to the documentation bundled with this repository.
    """

    root = Path(help_root).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise HelpCatalogError(f"Help manifest is missing: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HelpCatalogError(f"Unable to read the help manifest: {exc}") from exc

    if not isinstance(manifest, dict):
        raise HelpCatalogError("Help manifest root must be an object.")

    raw_articles = manifest.get("articles")
    raw_categories = manifest.get("categories")
    raw_aliases = manifest.get("aliases", {})
    if not isinstance(raw_articles, list) or not raw_articles:
        raise HelpCatalogError("Help manifest must contain at least one article.")
    if not isinstance(raw_categories, list) or not raw_categories:
        raise HelpCatalogError("Help manifest must contain at least one category.")
    if not isinstance(raw_aliases, dict):
        raise HelpCatalogError("Help aliases must be an object.")

    articles: list[Dict[str, Any]] = []
    article_ids: set[str] = set()
    for position, raw_article in enumerate(raw_articles):
        if not isinstance(raw_article, dict):
            raise HelpCatalogError(f"Article #{position + 1} must be an object.")
        article_id = _require_id(raw_article.get("id"), "article id")
        if article_id in article_ids:
            raise HelpCatalogError(f"Duplicate help article id: {article_id}")
        article_ids.add(article_id)

        source_path = _safe_article_path(root, raw_article.get("file"))
        try:
            source = source_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            raise HelpCatalogError(f"Unable to read help article: {source_path.name}") from exc
        if not source:
            raise HelpCatalogError(f"Help article is empty: {source_path.name}")
        try:
            rendered_html = _render_markdown(source)
        except Exception as exc:
            raise HelpCatalogError(f"Unable to render help article: {source_path.name}") from exc
        search_text = _plain_text(rendered_html)
        word_count = len(search_text.split())

        article = {
            "id": article_id,
            "title": _require_text(raw_article.get("title"), f"title for {article_id}"),
            "summary": _require_text(raw_article.get("summary"), f"summary for {article_id}"),
            "keywords": _as_string_list(raw_article.get("keywords"), f"keywords for {article_id}"),
            "related": _as_string_list(raw_article.get("related"), f"related articles for {article_id}"),
            "engine_ids": _as_string_list(raw_article.get("engine_ids"), f"engine ids for {article_id}"),
            "html": rendered_html,
            "search_text": search_text,
            "word_count": word_count,
            "reading_minutes": max(1, math.ceil(word_count / 220)),
        }
        articles.append(article)

    categories: list[Dict[str, Any]] = []
    category_ids: set[str] = set()
    assigned_articles: list[str] = []
    for position, raw_category in enumerate(raw_categories):
        if not isinstance(raw_category, dict):
            raise HelpCatalogError(f"Category #{position + 1} must be an object.")
        category_id = _require_id(raw_category.get("id"), "category id")
        if category_id in category_ids:
            raise HelpCatalogError(f"Duplicate help category id: {category_id}")
        category_ids.add(category_id)
        category_article_ids = _as_string_list(
            raw_category.get("article_ids"),
            f"article ids for category {category_id}",
        )
        unknown = [article_id for article_id in category_article_ids if article_id not in article_ids]
        if unknown:
            raise HelpCatalogError(
                f"Category {category_id!r} references unknown articles: {', '.join(unknown)}"
            )
        assigned_articles.extend(category_article_ids)
        categories.append(
            {
                "id": category_id,
                "title": _require_text(raw_category.get("title"), f"title for {category_id}"),
                "description": _require_text(
                    raw_category.get("description"),
                    f"description for {category_id}",
                ),
                "article_ids": category_article_ids,
            }
        )

    if len(assigned_articles) != len(set(assigned_articles)):
        raise HelpCatalogError("A help article is assigned to more than one category.")
    unassigned = article_ids.difference(assigned_articles)
    if unassigned:
        raise HelpCatalogError(f"Unassigned help articles: {', '.join(sorted(unassigned))}")

    aliases: Dict[str, str] = {}
    for raw_alias, raw_target in raw_aliases.items():
        alias = _require_id(raw_alias, "help alias")
        target = _require_id(raw_target, f"target for alias {alias}")
        if target not in article_ids:
            raise HelpCatalogError(f"Help alias {alias!r} targets unknown article {target!r}.")
        aliases[alias] = target

    _validate_relations(articles, article_ids)

    return {
        "title": _require_text(manifest.get("title"), "help title"),
        "subtitle": _require_text(manifest.get("subtitle"), "help subtitle"),
        "version": _require_text(manifest.get("version"), "help version"),
        "categories": categories,
        "aliases": aliases,
        "articles": articles,
    }


def clear_help_catalog_cache() -> None:
    """Clear rendered help content (primarily useful for tests and development)."""

    load_help_catalog.cache_clear()


def _catalog_metadata(catalog: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": catalog["title"],
        "subtitle": catalog["subtitle"],
        "version": catalog["version"],
        "categories": catalog["categories"],
        "aliases": catalog["aliases"],
        "articles": [
            {key: value for key, value in article.items() if key != "html"}
            for article in catalog["articles"]
        ],
    }


def _etagged_json(payload: Dict[str, Any], status: int = 200):
    response = make_response(jsonify(payload), status)
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    response.set_etag(hashlib.sha256(canonical.encode("utf-8")).hexdigest())
    response.headers["Cache-Control"] = "public, max-age=0, must-revalidate"
    return response.make_conditional(request)


def create_help_blueprint(
    help_root: str | Path = DEFAULT_HELP_ROOT,
    *,
    name: str = "help_center",
) -> Blueprint:
    """Create the lightweight API used by the in-app help center."""

    blueprint = Blueprint(name, __name__, url_prefix="/api/help")
    resolved_root = Path(help_root).resolve()

    @blueprint.get("/catalog")
    def help_catalog_route():
        try:
            catalog = load_help_catalog(resolved_root)
        except HelpCatalogError as exc:
            current_app.logger.error("Unable to load bundled help: %s", exc)
            return jsonify({"success": False, "error": "The bundled user guide is unavailable."}), 500
        return _etagged_json({"success": True, **_catalog_metadata(catalog)})

    @blueprint.get("/articles/<article_id>")
    def help_article_route(article_id: str):
        try:
            normalized_id = _require_id(article_id, "article id")
        except HelpCatalogError:
            return jsonify({"success": False, "error": "Help article not found."}), 404
        try:
            catalog = load_help_catalog(resolved_root)
        except HelpCatalogError as exc:
            current_app.logger.error("Unable to load bundled help article: %s", exc)
            return jsonify({"success": False, "error": "The bundled user guide is unavailable."}), 500

        resolved_id = catalog["aliases"].get(normalized_id, normalized_id)
        article = next(
            (entry for entry in catalog["articles"] if entry["id"] == resolved_id),
            None,
        )
        if article is None:
            return jsonify({"success": False, "error": "Help article not found."}), 404
        category = next(
            (
                entry
                for entry in catalog["categories"]
                if resolved_id in entry.get("article_ids", [])
            ),
            None,
        )
        return _etagged_json(
            {
                "success": True,
                "version": catalog["version"],
                "category": category,
                "article": article,
            }
        )

    return blueprint


__all__ = [
    "DEFAULT_HELP_ROOT",
    "HelpCatalogError",
    "clear_help_catalog_cache",
    "create_help_blueprint",
    "load_help_catalog",
]

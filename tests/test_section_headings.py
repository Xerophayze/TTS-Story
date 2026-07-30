from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path
from typing import Any, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_heading_helpers():
    """Load the small regex helpers without importing the full Flask app."""
    source_path = PROJECT_ROOT / "app.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    required_names = {
        "SECTION_HEADING_KEYWORDS",
        "_normalize_custom_headings",
        "_keyword_to_regex",
        "_build_section_heading_pattern",
    }
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id in required_names for target in targets):
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in required_names:
            selected.append(node)

    namespace = {
        "re": re,
        "Any": Any,
        "List": List,
        "Optional": Optional,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["_build_section_heading_pattern"]


build_pattern = _load_heading_helpers()


def _matched_lines(text: str, headings: List[str]) -> List[str]:
    return [match.group(1) for match in build_pattern(headings).finditer(text)]


class SectionHeadingPatternTests(unittest.TestCase):
    def test_symbol_only_custom_heading_is_detected(self) -> None:
        text = "✦ Chapter One\nOpening text.\n\n✦ Chapter Two\nMore text."

        self.assertEqual(
            ["✦ Chapter One", "✦ Chapter Two"],
            _matched_lines(text, ["✦"]),
        )

    def test_decorative_unicode_custom_heading_is_detected(self) -> None:
        text = "༺ First Movement\nOpening text.\n\n༺ Second Movement\nMore text."

        self.assertEqual(
            ["༺ First Movement", "༺ Second Movement"],
            _matched_lines(text, ["༺"]),
        )

    def test_non_latin_custom_heading_can_touch_its_numbering(self) -> None:
        text = "章一\nOpening text.\n\n章二\nMore text."

        self.assertEqual(["章一", "章二"], _matched_lines(text, ["章"]))

    def test_compact_english_heading_allows_number_but_not_word_continuation(self) -> None:
        text = "Episode1\nOpening text.\n\nEpisodeTwo is prose, not a heading."

        self.assertEqual(["Episode1"], _matched_lines(text, ["episode"]))


if __name__ == "__main__":
    unittest.main()

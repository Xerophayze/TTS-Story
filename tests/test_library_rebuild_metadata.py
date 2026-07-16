from src.library_metadata import get_custom_chapter_title


def test_rebuild_metadata_prefers_custom_title_at_display_position():
    metadata = {
        "custom_chapter_titles": {
            "0": "  Custom Chapter Name  ",
            "7": "Manifest Index Name",
        }
    }

    assert get_custom_chapter_title(metadata, chapter_index=7, position=0) == "Custom Chapter Name"


def test_rebuild_metadata_falls_back_to_manifest_chapter_index():
    metadata = {"custom_chapter_titles": {"7": "Manifest Index Name"}}

    assert get_custom_chapter_title(metadata, chapter_index=7) == "Manifest Index Name"


def test_rebuild_metadata_ignores_missing_or_invalid_titles():
    assert get_custom_chapter_title({}, chapter_index=0) is None
    assert get_custom_chapter_title({"custom_chapter_titles": []}, chapter_index=0) is None
    assert get_custom_chapter_title({"custom_chapter_titles": {"0": "  "}}, chapter_index=0) is None

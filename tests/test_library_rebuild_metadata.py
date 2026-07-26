from src.library_metadata import (
    get_custom_chapter_title,
    infer_collection_title_from_chunks,
    merge_generated_library_metadata,
)


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


def test_collection_title_can_be_recovered_from_explicit_title_chunk():
    chunks = [
        {"order_index": 1, "text": "Chapter 1: The Signal"},
        {"order_index": 0, "text": "Title. Shrouded Echoes"},
    ]

    assert infer_collection_title_from_chunks(chunks) == "Shrouded Echoes"
    assert infer_collection_title_from_chunks([{"text": "Ordinary story prose."}]) is None


def test_recompile_metadata_preserves_user_fields_while_updating_outputs():
    existing = {
        "collection_title": "Shrouded Echoes",
        "audiobook_author": "Eric Thorup",
        "custom_chapter_titles": {"0": "The Threshold"},
        "cover_image": "cover.png",
        "chapters": [{"title": "Old output"}],
    }
    generated = {
        "chapters": [{"title": "New output"}],
        "chapter_count": 1,
        "output_format": "mp3",
    }

    merged = merge_generated_library_metadata(existing, generated)

    assert merged["collection_title"] == "Shrouded Echoes"
    assert merged["audiobook_author"] == "Eric Thorup"
    assert merged["custom_chapter_titles"] == {"0": "The Threshold"}
    assert merged["cover_image"] == "cover.png"
    assert merged["chapters"] == [{"title": "New output"}]


def test_recompile_metadata_recovers_missing_collection_title():
    merged = merge_generated_library_metadata(
        {"chapter_mode": True},
        {"chapters": []},
        [{"order_index": 0, "text": "Title: Shrouded Echoes"}],
    )

    assert merged["collection_title"] == "Shrouded Echoes"

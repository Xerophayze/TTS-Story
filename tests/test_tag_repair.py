from src.text_processor import TextProcessor, repair_speaker_tags


def test_direction_closed_with_wrong_name_is_repaired():
    source = (
        "[direction]Narrate evenly.[/narrator]\n"
        "[narrator]第一段。[/narrator]"
    )
    repaired = repair_speaker_tags(source)
    assert repaired == (
        "[direction]Narrate evenly.[/direction]\n"
        "[narrator]第一段。[/narrator]"
    )


def test_stray_closing_tag_is_removed():
    source = "[direction]Narrate slowly.[/dramaturgy]\n[/narrator]"
    repaired = repair_speaker_tags(source)
    assert repaired == "[direction]Narrate slowly.[/direction]\n"


def test_unclosed_speaker_is_closed_at_end():
    assert repair_speaker_tags("[narrator]没有闭合") == "[narrator]没有闭合[/narrator]"


def test_speaker_block_with_wrong_closing_name_is_repaired():
    source = "[anchang-male]“走吧。”[/narrator]"
    assert repair_speaker_tags(source) == "[anchang-male]“走吧。”[/anchang-male]"


def test_matching_tags_are_unchanged():
    source = "[direction]低沉克制。[/direction]\n[narrator]雨夜。[/narrator]"
    assert repair_speaker_tags(source) == source


def test_plain_text_without_tags_is_unchanged():
    source = "雨下到后半夜，旧书店的灯还亮着。"
    assert repair_speaker_tags(source) == source


def test_repaired_text_parses_into_expected_speakers():
    source = (
        "[direction]Narrate evenly.[/narrator]\n"
        "[narrator]东安国。[/narrator]\n"
        "[direction]Speak softly.[/dramaturgy]\n"
        "[anxi-female]“这是我家？”[/narrator]"
    )
    segments = TextProcessor().parse_speaker_segments(
        repair_speaker_tags(source)
    )
    assert [segment["speaker"] for segment in segments] == [
        "narrator",
        "anxi-female",
    ]
    assert segments[0]["text"] == "东安国。"
    assert segments[1]["text"] == "“这是我家？”"

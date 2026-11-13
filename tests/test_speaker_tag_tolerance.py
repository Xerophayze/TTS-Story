from src.text_processor import TextProcessor


def test_mismatched_closing_tag_keeps_speaker_text():
    text = (
        "[narrator]第一段。[/dictionary]\n"
        "[an-xi-female]“你来了。”[/dictionary]"
    )
    segments = TextProcessor().parse_speaker_segments(text)
    assert [segment["speaker"] for segment in segments] == [
        "narrator",
        "an-xi-female",
    ]
    assert segments[0]["text"] == "第一段。"
    assert segments[1]["text"] == "“你来了。”"


def test_matching_closing_tag_is_unchanged():
    text = "[narrator]第一段。[/narrator]"
    segments = TextProcessor().parse_speaker_segments(text)
    assert len(segments) == 1
    assert segments[0]["speaker"] == "narrator"
    assert segments[0]["text"] == "第一段。"

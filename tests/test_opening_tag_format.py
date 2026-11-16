import pytest

from src.directed_prompt_presets import opening_tag_experimental_prompt
from src.opening_tag_format import FORMAT_MARKER, OUTPUT_MARKER, convert_opening_tags, normalize_opening_tag_response
from src.tag_validation import tag_errors
from src.text_processor import TextProcessor


def test_opening_tags_convert_to_existing_paired_format_with_directions_and_cues():
    text = (OUTPUT_MARKER + '\n[direction]Narrate evenly.[/direction]\n'
            '[narrator]第一段。 [laugh] She smiled.\n'
            '[direction]Speak softly.[/direction]\n[alice-female]“你好。”\n'
            '[direction]Narrate calmly.[/direction]\n[narrator]The end.')
    result = convert_opening_tags(text)
    assert not tag_errors(result)
    segments = TextProcessor().parse_speaker_segments(result)
    assert [s['speaker'] for s in segments] == ['narrator', 'alice-female', 'narrator']
    assert [s['text'] for s in segments] == ['第一段。 [laugh] She smiled.', '“你好。”', 'The end.']
    assert [s['delivery_instruction'] for s in segments] == ['Narrate evenly.', 'Speak softly.', 'Narrate calmly.']


def test_opening_tags_preserve_line_expression_and_pause():
    result = convert_opening_tags(OUTPUT_MARKER + '\n[direction]Speak warmly.[/direction]\n[alice]Hi.\n[laugh]\nAgain.******\n[direction]Speak softly.[/direction]\n[bob]Bye.')
    assert [s['text'] for s in TextProcessor().parse_speaker_segments(result)] == ['Hi.\n[laugh]\nAgain.******', 'Bye.']


@pytest.mark.parametrize('body', [
    '[alice]No marker',
    OUTPUT_MARKER + '\nUnassigned prose\n[alice]Hello',
    OUTPUT_MARKER + '\n[direction]Softly.[/emotion]\n[alice]Hello',
    OUTPUT_MARKER + '\n[alice]Hi[/alice]',
    OUTPUT_MARKER + '\n[direction]Only direction.[/direction]',
    OUTPUT_MARKER + '\n[direction]Softly.[/direction]\nUnassigned\n[alice]Hello',
    OUTPUT_MARKER + '\n[direction]One.[/direction]\n[direction]Two.[/direction]\n[alice]Hello',
    OUTPUT_MARKER + '\n[alice]Hi[/wrong]\n[laugh]\n[alice]More',
    OUTPUT_MARKER + '\n[direction]Softly.[/direction]\n[张三]你好。',
])
def test_ambiguous_outputs_are_rejected(body):
    with pytest.raises(ValueError):
        convert_opening_tags(body)


def test_preset_is_explicit_and_other_presets_unchanged():
    prompt = opening_tag_experimental_prompt()
    assert FORMAT_MARKER in prompt
    assert OUTPUT_MARKER in prompt
    assert 'Do not write closing speaker tags' in prompt
    assert '[/barnaby-male]' not in prompt
    assert normalize_opening_tag_response('other prompt', '[alice]Hi[/alice]') == '[alice]Hi[/alice]'


def test_matching_bookends_are_case_insensitive():
    text = '[Alice]Hello.[/alice]'
    assert not tag_errors(text)
    assert TextProcessor().parse_speaker_segments(text)[0]['text'] == 'Hello.'


def test_process_full_normalizes_opted_in_response(monkeypatch):
    import app as application
    monkeypatch.setattr(application, 'load_config', lambda: {'llm_provider': 'openrouter'})
    monkeypatch.setattr(application, '_run_llm_prompt_with_failover', lambda *a, **k: (
        OUTPUT_MARKER + '\n[direction]Speak warmly.[/direction]\n[alice]Hello.',
        {'id': 'test', 'provider': 'openrouter'}, []))
    response = application.app.test_client().post('/api/gemini/process-full', json={
        'text': 'Hello.', 'prompt_override': FORMAT_MARKER,
    })
    assert response.status_code == 200
    result = response.get_json()['result_text']
    assert OUTPUT_MARKER not in result
    assert TextProcessor().parse_speaker_segments(result)[0]['text'] == 'Hello.'

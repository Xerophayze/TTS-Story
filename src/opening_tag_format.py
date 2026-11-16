"""Convert the explicit experimental opening-tag format to ordinary bookends.

Only line-start speaker tags are structural. Expression cues remain content;
paired direction tags keep their existing meaning. Conversion inserts bookends
without changing source characters and refuses ambiguous malformed controls.
"""
import re

from src.tag_validation import CONTROL_TAGS, EXPRESSION_TAGS, tag_errors

FORMAT_MARKER = 'TTS_STORY_OPENING_TAG_FORMAT_V1'
OUTPUT_MARKER = '[[TTS-STORY-OPENING-TAGS-V1]]'
LINE_TAG = re.compile(r'(?m)^[ \t]*\[([^\W\d_][\w-]*)\]')
CONTROL_CLOSE = re.compile(r'\[/(direction|emotion)\]', re.IGNORECASE)
CLOSING = re.compile(r'\[/[^\[\]]+\]')


def convert_opening_tags(text: str) -> str:
    """Normalize one opted-in LLM response; raise rather than guess on ambiguity."""
    body = (text or '').lstrip('\ufeff \t\r\n')
    if not body.startswith(OUTPUT_MARKER):
        raise ValueError('Opening-tag output is missing its required format marker; inspect the LLM response.')
    body = body[len(OUTPUT_MARKER):]
    first = LINE_TAG.search(body)
    if not first or body[:first.start()].strip():
        raise ValueError('Opening-tag output must start with a direction or speaker tag.')
    insertions = []
    current = None
    pending_direction = False
    cursor = first.start()
    count = 0
    while True:
        match = LINE_TAG.search(body, cursor)
        if not match:
            break
        name = match.group(1)
        lower = name.lower()
        if not name.isascii():
            raise ValueError(f'Speaker tag [{name}] must use ASCII letters, numbers, hyphens or underscores.')
        if lower in EXPRESSION_TAGS:
            if pending_direction or current is None:
                raise ValueError('An expression cue must belong to a speaker passage.')
            if CLOSING.search(body, cursor, match.start()):
                raise ValueError('Opening-tag output contains an unexpected closing tag; review it manually.')
            cursor = match.end()
            continue
        follows_direction = pending_direction
        if pending_direction and (body[cursor:match.start()].strip() or lower in CONTROL_TAGS):
            raise ValueError('A direction must be followed immediately by a speaker passage.')
        pending_direction = False
        if current is not None:
            if CLOSING.search(body, cursor, match.start()):
                raise ValueError('Opening-tag output contains an unexpected closing tag; review it manually.')
            insertions.append((match.start(), f'[/{current}]'))
            current = None
        if lower in CONTROL_TAGS:
            close = CONTROL_CLOSE.search(body, match.end())
            if not close or close.group(1).lower() != lower:
                raise ValueError(f'Unclosed or mismatched [{name}] direction; review it manually.')
            nested = LINE_TAG.search(body, match.end(), close.start())
            if nested or CLOSING.search(body, match.end(), close.start()):
                raise ValueError('Nested tags in a direction require manual review.')
            cursor = close.end()
            pending_direction = True
        else:
            if not follows_direction:
                raise ValueError(f'Speaker [{name}] has no preceding direction; review it manually.')
            current = name
            count += 1
            cursor = match.end()
    if pending_direction:
        raise ValueError('A direction must be followed immediately by a speaker passage.')
    if current:
        if CLOSING.search(body, cursor):
            raise ValueError('Opening-tag output contains an unexpected closing tag; review it manually.')
        insertions.append((len(body), f'[/{current}]'))
    elif body[cursor:].strip():
        raise ValueError('Unassigned text after a direction requires manual review.')
    if not count:
        raise ValueError('Opening-tag output contains no speaker passages.')
    parts = []
    start = 0
    for pos, closer in insertions:
        parts.extend((body[start:pos], closer))
        start = pos
    parts.append(body[start:])
    body = ''.join(parts)
    problems = tag_errors(body)
    if problems:
        raise ValueError('Opening-tag conversion requires review: ' + '; '.join(problems[:3]))
    return body.strip()


def normalize_opening_tag_response(prompt: str, response: str) -> str:
    return convert_opening_tags(response) if FORMAT_MARKER in (prompt or '') else response

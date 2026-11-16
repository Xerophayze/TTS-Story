"""Add versioned built-in prompts without overwriting users' preset edits."""
import copy
import re
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parents[1] / "docs" / "prompts"


def opening_tag_experimental_prompt():
    """Retain V3's source and casting rules, replacing its bookend instructions."""
    original = (PROMPT_DIR / 'strict-book-conversion-v3-directed.txt').read_text(encoding='utf-8')
    source_rules = original[original.index('1. LOCKED SOURCE'):original.index('2. EXACT OUTPUT')]
    speaker_rules = original[original.index('3. SPEAKER OWNERSHIP'):original.index('6. FINAL SILENT VALIDATION')]
    speaker_rules = re.sub(r'\[/(?!direction\]|emotion\])([a-zA-Z][a-zA-Z0-9_-]*)\]', '', speaker_rules, flags=re.I)
    speaker_rules = speaker_rules.replace('a properly closed speaker block', 'a speaker opening tag and passage')
    return ('''Strict Book Conversion V3 - Opening Tags (Experimental)
TTS_STORY_OPENING_TAG_FORMAT_V1

''' + source_rules + '''
2. OPENING-TAG OUTPUT STRUCTURE

The first line MUST be exactly [[TTS-STORY-OPENING-TAGS-V1]].
Put one paired [direction] instruction on its own line immediately before each spoken passage.
Put a single opening speaker tag on the following line, immediately followed by the exact spoken passage.
Do not write closing speaker tags. Each passage ends immediately before the next direction or speaker opening tag, or the end of the response.
Speaker identifiers use lowercase ASCII letters, numbers, hyphens and underscores and begin with a letter. Use narrator for narration.
Do not use standalone expression cues as speaker identifiers. Never put an ordinary bracketed line at the beginning of a line in a passage.
Preserve existing inline expression cues as spoken-text markup, not speaker changes.
Do not output prose outside a speaker passage, nested tags, Markdown fences, explanations or labels.

Example (format only):
[[TTS-STORY-OPENING-TAGS-V1]]
[direction]Narrate evenly.[/direction]
[narrator]The door opened.
[direction]Speak with quiet curiosity.[/direction]
[alice-female]"Who is there?"
[direction]Narrate calmly.[/direction]
[narrator]She listened.

''' + speaker_rules + '''
6. FINAL VALIDATION
Check every source passage remains exactly as supplied, in order, and belongs to the correct speaker.
Each direction is paired and belongs only to the immediately following speaker passage.
Start with the required marker; use only opening speaker tags; do not add or omit source words.
Return only the marked manuscript.
''').strip()


def with_directed_presets(presets):
    result = copy.deepcopy(presets) if isinstance(presets, list) else []
    ids = {p.get("id") for p in result if isinstance(p, dict)}
    for preset_id, title, filename in (
        ("strict-book-conversion-v2-directed", "Strict Book Conversion V2 - Directed",
         "strict-book-conversion-v2-directed.txt"),
        ("strict-book-conversion-v2-directed-backup-20260911",
         "Strict Book Conversion V2 - Directed (Backup 2026-09-11)",
         "backups/strict-book-conversion-v2-directed-20260911.txt"),
        ("strict-book-conversion-v3-directed", "Strict Book Conversion V3 - Directed",
         "strict-book-conversion-v3-directed.txt"),
    ):
        if preset_id not in ids:
            result.append({"id": preset_id, "title": title,
                           "prompt": (PROMPT_DIR / filename).read_text(encoding="utf-8").strip()})
    if 'strict-book-conversion-v3-opening-experimental' not in ids:
        result.append({'id': 'strict-book-conversion-v3-opening-experimental',
                       'title': 'Strict Book Conversion V3 - Opening Tags (Experimental)',
                       'prompt': opening_tag_experimental_prompt()})
    return result

# Review and Regenerate Chunks

Chunk review lets you repair a small part of a completed audiobook without synthesizing the entire manuscript again. Review opens as a modal inside the Audio Library page; it does not open a separate window and it is not a Job Queue feature.

## Open review

1. Open [Audio Library](app:library).
2. Expand the desired Library item.
3. Select a chapter pill, or select **Full Story**.
4. Choose **Review Chunks**.

![Chunk Review modal with playback, text editing, and regeneration controls](../../../static/help/screenshots/chunk-review.png)

*Review one chunk at a time, listen before and after changes, and regenerate only the audio that needs repair.*

TTS-Story loads the saved chunk text, speaker, engine, voice, and audio references. Full Story review groups chunks by chapter or book when that information is available.

## Repair one chunk

On a chunk card you can:

- play the current audio;
- edit its text;
- select a different engine and compatible voice;
- adjust Qwen language or instruction where applicable;
- adjust speed and pitch;
- add leading or trailing silence;
- preview or apply FX; and
- click **Regenerate**.

Regeneration queues new synthesis for that chunk. Keep the review modal open long enough to see whether the chunk completes or fails, then listen to the replacement.

Changing an engine can expose a different kind of voice selector. A built-in voice ID is not interchangeable with a reference prompt path. If an engine or voice does not appear correctly, return to the original engine and verify its configuration in [Settings](app:settings).

## Repair a speaker in bulk

The speaker-level controls can apply a selected engine, voice, speed, pitch, and leading/trailing silence across that speaker's chunks. **Regenerate All** queues each affected chunk; it can consume substantial local processing time or cloud quota.

Test one representative chunk first. Once it sounds right, apply the same settings to the speaker.

## Fix recurring pronunciation

For a word that fails in many chunks, edit the Library item's **Alt Words** registry instead of changing each chunk manually. Library replacements are saved with that item and are applied during regeneration. Replacements are case-insensitive literal substrings, not whole-word rules, so choose the source text carefully.

See [Alternate Word Registry](help:alt-word-registry).

## Rebuild after regeneration

> **Regenerate changes chunk audio. Rebuild changes the compiled outputs.**

After replacing chunks, use the review modal's chapter rebuild controls or the Library item's **Rebuild** action. Otherwise a previously compiled chapter, Full Story, ZIP, or M4B can still contain the old audio.

- Rebuild Chapter recompiles that chapter from its current chunks.
- Rebuild selected chapters recompiles checked chapters.
- The Library card's Rebuild action recompiles all chapter outputs and the combined audiobook.

Rebuild does not call the TTS engine and cannot recreate a missing source chunk. It merges the chunk files already stored for the item. See [Edit, Rebuild, and Repair Library Items](help:library-editing-repair).

## A safe repair sequence

1. Play the faulty chunk and inspect its text.
2. Change only one variable: text, pronunciation, voice, or FX.
3. Regenerate and listen again.
4. Repeat only if necessary.
5. Rebuild the affected chapter.
6. Play across the joins before exporting.

For symptoms such as clicks, drift, or unnatural pacing, see [Fix Pronunciation, Pacing, Voice, and Merge Problems](help:audio-quality).

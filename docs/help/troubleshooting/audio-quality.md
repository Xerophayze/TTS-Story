# Fix Pronunciation, Pacing, Voice, and Merge Problems

Repair the smallest faulty scope. A pronunciation problem in one chunk rarely requires regenerating an entire book, and a merge problem may require no new synthesis at all.

## Mispronounced words

First run a Quick Test using the actual engine and voice. If spelling is the cause, add an Alternate Word replacement with a pronunciation-friendly form, then regenerate the affected chunks.

Alternate Word matching is case-insensitive literal substring replacement, not whole-word matching. Replacing `cat` can also change `concatenate`. Use a longer, unambiguous source phrase when needed and inspect the instance count.

For names that vary by context, edit only the affected chunk text instead of creating a global replacement. See [Alternate Word Registry](help:alt-word-registry).

## Unnatural pacing

- Improve punctuation before changing speed.
- Break very long sentences at natural boundaries.
- Keep global and per-speaker speed near 1.0 while diagnosing.
- Add small leading/trailing silence to a specific chunk in Library review.
- Reduce excessive chapter crossfade if words overlap.

Large speed changes can introduce phase or stretching artifacts. Regenerate from a better engine delivery where possible rather than relying on extreme post-processing.

## Voice identity or drift

For a cloning engine, inspect the reference prompt:

- one speaker only;
- no music, echo, or background voices;
- no clipping;
- a natural 5–10 second performance; and
- an exact transcript when the engine requests one.

Use the same prompt in Quick Test and one normal chunk. If the raw prompt is poor, replace it in [Available Voices](app:voices) rather than compensating with pitch and speed.

Very short text can also sound less stable than a complete sentence. Keep engine character chunks near the recommended range and avoid splitting inside abbreviations or names.

## Wrong speaker or missing narration

Check the source tags before blaming the voice. Speaker tags must be paired correctly. Once valid speaker blocks exist, untagged text is omitted from synthesis. Correct the text, analyze again, and confirm chunk/speaker counts.

![Generate warning explaining that untagged text is omitted when valid speaker tags exist](../../../static/help/screenshots/speaker-tag-warning.png)

*Resolve speaker-tag warnings before generation so narration is not silently excluded or assigned to the wrong voice.*

Paralinguistic tags such as laughter or sighs are engine-dependent. Unsupported engines may speak or mishandle the tag instead of performing it. See [Speaker and Expression Tags](help:speaker-tags).

## Clicks, gaps, or repeated audio at joins

Play the individual chunks in Library review.

- If the defect exists in a chunk, regenerate that chunk.
- If chunks are clean but the chapter join is bad, adjust crossfade or chunk silence and rebuild.
- If a chapter is correct but Full Story is stale, rebuild all outputs.
- If a source chunk is missing, rebuild cannot recreate it; regenerate or restore it first.

Crossfade defaults to 0.1 seconds. Increase it only in small steps. The shared Segment Silence setting inserts silence between every chunk during assembly; use Library review's per-chunk leading or trailing silence when only one boundary needs correction.

## Loudness and export

ACX-oriented processing normalizes and limits compiled output; it does not remove room noise, repair distortion, or equalize inconsistent performances. Fix source chunks first, then rebuild and export.

Follow [Review and Regenerate Chunks](help:job-review) and [Edit, Rebuild, and Repair Library Items](help:library-editing-repair).

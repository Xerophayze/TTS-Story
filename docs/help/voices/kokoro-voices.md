# Kokoro Voices and Previews

Kokoro includes built-in voices grouped by language. Open [Available Voices](app:voices) and expand **Kokoro Voices** to browse them.

![Kokoro voice browser showing language filters and preview actions](../../../static/help/screenshots/kokoro-voice-browser.png)

*Filter the catalog by language, then generate and compare previews before assigning a voice to a speaker.*

## Choose the correct language group

Kokoro voice groups use these language codes:

- `a` — American English
- `b` — British English
- `e` — Spanish
- `f` — French
- `h` — Hindi
- `j` — Japanese
- `z` — Mandarin Chinese
- `p` — Brazilian Portuguese

Choose a voice from the same language group as the text. A voice can sometimes produce sounds for another language, but pronunciation and rhythm are not guaranteed.

## Generate and play previews

The Kokoro preview area can generate sample audio for the voice catalog. Preview generation loads the local Kokoro engine and can take longer on the first run. Existing previews can be overwritten when you deliberately regenerate them.

Use the same short test sentence when comparing several voices. Include one proper name, a number, and representative punctuation from the manuscript. A catalog preview is useful for tone, but a **Quick Test** from the speaker assignment modal is more representative because it uses the job's current text and voice settings.

## Assign a voice

1. Open [Generate](app:generate).
2. Select **Kokoro (Local)** or **Kokoro (Replicate)** as the generation engine.
3. Analyze the manuscript.
4. Click a detected speaker chip.
5. Select the voice and matching language code.
6. Run **Quick Test** before generating the full job.

The Local and Replicate engines share the voice concept, but Replicate sends generation requests to an online service and requires a configured token. Review [Kokoro: Local and Replicate](help:engine-kokoro) for engine setup.

## Improve a voice without cloning

If no single Kokoro voice fits, create a reusable weighted blend from voices in the same language. A blend remains a Kokoro voice and does not need reference audio. See [Custom Kokoro Voice Blends](help:custom-kokoro-blends).

Pitch and speed effects can change perceived age or delivery, but large changes can introduce artifacts. Start with the unmodified voice and make restrained adjustments in the speaker assignment modal. See [Assign and Test Voices](help:assign-voices).

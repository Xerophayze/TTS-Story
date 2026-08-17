# Reference Voice Prompts

A voice prompt is a clean audio clip used by a cloning engine to guide voice identity and delivery. TTS-Story manages uploaded files and generated voice-design samples under **Voice Prompts** on [Available Voices](app:voices).

![Voice Prompts manager with upload, filtering, preview, and library actions](../../../static/help/screenshots/voice-prompts.png)

*Use the prompt manager to upload clean references, verify metadata, preview clips, and maintain the reusable library.*

> The current interface uploads existing audio; it does not record from a microphone. Record and trim the clip in an audio editor, then upload it.

## Prepare a useful clip

For the most reusable prompt:

- use one speaker only;
- remove music, effects, room echo, and background voices;
- avoid clipping and aggressive noise reduction;
- keep a natural, steady delivery;
- include complete words at the beginning and end; and
- use a standard WAV, MP3, M4A, FLAC, or OGG file.

The manager recommends clips around 5–10 seconds. Chatterbox Turbo enforces a minimum of five seconds when saving through this page. Other engines have their own requirements; a longer clip is not automatically better.

Dot.TTS works best with the exact spoken transcript. A transcript is required before TTS-Story can synchronize a sample to a cloning-capable LocalAI model. VoxCPM and Qwen3 can use cached automatic transcription when a stored transcript is unavailable, but verify pronunciation with a short test rather than assuming transcription succeeded.

## Upload one clip

1. Expand **Voice Prompts**.
2. Enter a friendly Voice Name.
3. choose an audio file.
4. Enter exactly what is spoken in **Exact Transcript** when the sample may be used with LocalAI or another transcript-aware engine.
5. Click **Save Voice**.

For bulk import, drag several supported clips into the drop zone or use **Browse Files**. Filenames become friendly names, so rename source files before import if that saves cleanup.

Files are copied into `data/voice_prompts`; the originals are not linked in place.

## Organize the library

Use search and Source, Gender, and Language filters. Edit metadata to improve filtering or add a missing exact transcript. A **Transcript ready** badge shows which samples can be synchronized to LocalAI; generated voice-design samples are backfilled from their saved preview text when possible. Metadata labels do not alter how the recording sounds.

For a sample without text, select **Generate Transcript** in its row. You can also select several missing samples and use **Generate Transcripts** in the batch toolbar. TTS-Story runs SenseVoice locally, stores the detected text with the sample, and then marks it transcript-ready. The first request may take longer while the ASR model downloads and loads. Review the generated transcript with **Edit**, especially for names, invented words, acronyms, and noisy recordings.

Available row and batch actions include preview, edit metadata, export, archive, and delete. Archived voices remain stored and can be restored with **Unarchive**. Deleting removes the managed prompt and can break assignments that still reference its path.

**Load External Voices** retrieves the current external GitHub catalog. Previewing requires internet access. Download an external voice to make a local prompt available to compatible engines.

## Assign a prompt

1. Open [Generate](app:generate) and select a reference-based engine.
2. Analyze the text and click a speaker chip.
3. Select the prompt in the reference selector.
4. Enter a transcript if the selected engine exposes and requires one.
5. Use **Quick Test**.

The same prompt can behave differently across Chatterbox, VoxCPM, Pocket TTS, Qwen3, OmniVoice, IndexTTS, and Dot.TTS. Test it with the actual engine rather than relying on the raw clip preview.

See [Voice Management Overview](help:available-voices) and the selected article under [Engine Reference and Comparison](help:engine-overview).

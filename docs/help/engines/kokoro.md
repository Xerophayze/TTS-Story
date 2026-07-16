# Kokoro: Local and Replicate

Kokoro is a strong default for lightweight, repeatable narration. TTS-Story provides a local engine and a Replicate-hosted engine. Both use named Kokoro voices; the local engine also supports TTS-Story's saved embedding blends.

## Best for

- Narration on a CPU or modest NVIDIA GPU
- Built-in voices that remain consistent across a long project
- Multiple supported languages when each voice and language code are matched correctly
- Local, private generation after the model has downloaded
- Custom Kokoro blends made in [Available Voices](help:custom-kokoro-blends)

Kokoro does **not** clone a person from reference audio in TTS-Story. A custom blend combines existing compatible Kokoro voice embeddings.

## Local requirements and setup

![Kokoro voice browser showing available voices and preview controls](../../../static/help/screenshots/kokoro-voice-browser.png)

*Use the voice browser to audition a Kokoro voice before assigning it to a long project.*

The normal TTS-Story setup installs the Kokoro package and required speech components. The model and voice files may download on first use. An NVIDIA GPU speeds generation, but the adapter automatically falls back to CPU and Kokoro is one of the more practical local CPU engines.

Open [Settings → Engine Settings](app:settings/kokoro), select **Kokoro**, and leave the 500-character chunk target in place for the first test. On Generate, assign a language and compatible built-in or custom voice to every speaker.

The adapter produces 24 kHz speech before TTS-Story performs any requested final conversion or merge processing.

## Replicate requirements and setup

Create a Replicate API token, enter it under **Engine Settings → API Keys**, and select **Kokoro Replicate** for the job. TTS-Story calls a pinned `jaaari/kokoro-82m` model version and submits text, voice, and speed.

Replicate uses its own supported voice names. A local `custom_...` blend is not uploaded to the hosted model and should be used with local Kokoro instead.

## Controls TTS-Story exposes

- **Chunk Size:** sentence-aware character target; 300–600 is a useful working range and 500 is the default.
- **Voice and language:** selected per speaker on Generate.
- **Speed and FX:** normal TTS-Story speaker/output controls; extreme changes can sound synthetic.
- **Parallel chunks:** useful for Replicate, within provider capacity. It does not make the local model generate several chunks at once.

## Effective-use tips

1. Preview the voice using text in the same language and style as the manuscript.
2. Keep a voice's intended language code; a mismatch is a common cause of pronunciation problems.
3. If phrasing changes abruptly at joins, reduce chunk size modestly or increase crossfade rather than changing several controls together.
4. Build blends from voices that use the same language pipeline. Use conservative weights and test several sentences.
5. Use local Kokoro when reproducibility, privacy, or avoiding per-request cost matters; use Replicate when local inference is inconvenient.

## Time, privacy, cost, and limits

The first local run can be slower while files download and the pipeline loads. Later CPU generation is usually practical, while CUDA is faster. Local text stays on the computer after installation.

Replicate needs internet access and sends each text chunk to Replicate. Predictions are billed to the token owner's account and can wait for a hosted worker to start. Availability and pricing belong to the pinned hosted model and can change independently of TTS-Story.

Kokoro's voice catalog is finite, and very expressive acting or identity cloning is better handled by Chatterbox, Qwen3 Clone, OmniVoice, IndexTTS, Dot.TTS, or a hosted provider.

## Authoritative references

- [Kokoro official repository](https://github.com/hexgrad/kokoro)
- [Kokoro model used by the Replicate adapter](https://replicate.com/jaaari/kokoro-82m)
- [Replicate API-token guidance](https://replicate.com/docs/topics/security/api-tokens/)
- [Replicate prediction lifecycle](https://replicate.com/docs/topics/predictions/create-a-prediction)

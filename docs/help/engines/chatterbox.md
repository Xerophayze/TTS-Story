# Chatterbox Turbo: Local and Replicate

Chatterbox Turbo is TTS-Story's expressive English option for reference cloning and supported non-verbal cues such as `[laugh]`, `[chuckle]`, and `[cough]`. The same family is available as a local CUDA-focused engine or through Replicate.

## Best for

- Expressive English dialogue and narration
- Reusing a clean reference voice prompt
- Text that deliberately uses Chatterbox's supported paralinguistic tags
- A reusable default reference prompt when one voice should be the fallback

This integration is English-only. Do not assume capabilities from other Chatterbox models are present in the Turbo adapter.

## Local requirements and setup

![Voice Prompts library with upload controls and saved reference voices](../../../static/help/screenshots/voice-prompts.png)

*Add and review Chatterbox reference recordings in Voice Prompts before assigning them to speakers.*

Install Chatterbox Local from **Settings → Engine Settings**. TTS-Story creates a dedicated `engines/chatterbox/.venv` environment because Chatterbox requires an older Transformers version than Qwen3. The Turbo model downloads on first use. TTS-Story presents the local engine as requiring roughly 8 GB of VRAM; a supported NVIDIA CUDA GPU is strongly recommended. Although the Device field accepts `cpu`, long CPU jobs can be very slow. Uninstalling Chatterbox removes this isolated environment and its model cache without changing Qwen3 or other local engines.

Under [Settings → Engine Settings](app:settings/chatterbox-local), leave **Device** at `auto`, assign a Default Prompt only if one voice should be the fallback, and test the defaults before tuning. A clean reference of roughly ten seconds is a sensible starting point: one speaker, steady volume, no music, and little room echo.

## Replicate requirements and setup

Enter a Replicate token directly under **Engine Settings → Chatterbox Cloud**. The token and Replicate parallel-request limit are shared with Kokoro Cloud. The cloud adapter uses a pinned `resemble-ai/chatterbox-turbo` version. It can use the configured provider voice, or upload the assigned reference prompt for cloning.

Do not replace the pinned model string casually. A different Replicate version may expose a different input schema and fail even when it belongs to the same model page.

## Controls TTS-Story exposes

Local controls include **Device**, **Chunk Size** (450 by default), **Default Prompt**, temperature, top-p, top-k, repetition penalty, CFG weight, exaggeration, and prompt/output loudness normalization.

Replicate controls include the pinned **Model Version**, provider **Default Voice**, temperature, top-p, top-k, repetition penalty, and an optional seed. The Replicate API token is shared with Kokoro Replicate.

Keep the supplied sampling defaults for a baseline. Change only one of temperature, top-p, or top-k at a time. A fixed seed can make cloud comparisons easier, but it does not guarantee identical output if the hosted implementation changes.

## Effective-use tips

- Match the reference's speaking style to the desired result. A whispered or highly emotional prompt can carry that behavior into neutral prose.
- Normalize a prompt when it is much quieter or louder than other prompts; disable prompt normalization if it damages an already clean recording.
- Use only documented tags and spell them exactly. Unsupported bracketed text may be spoken aloud.
- Keep tags sparse. Several cues in a short chunk can make delivery unstable.
- If words repeat or loop, first restore defaults, then try a small increase in repetition penalty or a shorter chunk.
- Review one representative scene before committing a full book. See [Speaker and Expression Tags](help:speaker-tags) and [Reference Voice Prompts](help:voice-prompts).

## Time, privacy, cost, and limits

Local first use includes a large download and model load. CUDA out-of-memory errors usually require closing other GPU applications, shortening chunks, or choosing a lighter engine. Local synthesis keeps manuscript and prompts on this computer after downloads.

Replicate avoids local VRAM use but sends text and any assigned prompt audio to Replicate. It requires internet access, incurs prediction charges, and can have cold-start or queue delays. TTS-Story cannot guarantee provider capacity.

Chatterbox-generated audio is watermarked by the upstream model. The adapter does not expose every control described for other Chatterbox family models.

## Authoritative references

- [Resemble AI Chatterbox repository](https://github.com/resemble-ai/chatterbox)
- [Chatterbox Turbo on Replicate](https://replicate.com/resemble-ai/chatterbox-turbo)
- [Replicate API-token guidance](https://replicate.com/docs/topics/security/api-tokens/)

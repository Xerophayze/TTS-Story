# Dot.TTS

Dot.TTS is a local 2B-parameter, 48 kHz zero-shot cloning engine. It is most effective with a clean reference recording of about ten seconds and an exact transcript.

## Best for

- High-quality local reference cloning
- Projects that benefit from native 48 kHz model output
- Users with a capable NVIDIA GPU and time to prepare accurate prompts

Dot.TTS is not the fastest first engine to try. Use it when cloning quality justifies a large model and longer startup.

## Requirements and setup

The normal setup clones the official repository into `engines/dots-tts/repo`, creates `engines/dots-tts/.venv`, and installs a compatible PyTorch stack. The model prefetch is opt-in, so the first Dot.TTS job normally downloads the selected weights unless setup was run with `PREFETCH_DOTS_TTS_MODEL=1`.

CUDA is strongly recommended. Setup can install a CPU runtime when no NVIDIA GPU is found, but a 2B model can be impractically slow there. The ASR Device field controls transcript auto-detection; Dot.TTS itself handles model placement within its worker.

## Prepare the reference

![Voice Prompts library for adding and managing reference recordings](../../../static/help/screenshots/voice-prompts.png)

*Add the Dot.TTS reference recording here and verify its transcript before a long cloning job.*

Use approximately ten seconds of one speaker with no background music, overlapping voices, clipping, heavy denoising, or room echo. Enter exactly what is spoken, including filler words. TTS-Story can use SenseVoice to fill a missing transcript and cache it, but inspect that result when names, accents, or unusual words matter.

Leave **Allow reference-audio-only fallback** off for the best-conditioned path. Enable it only when a transcript truly cannot be obtained; the x-vector-only result can lose identity or style detail.

## Controls TTS-Story exposes

Open [Settings → Engine Settings](app:settings/dots-tts) and select **Dot.TTS**.

- **Model:** SOAR for best cloning, MeanFlow (`mf`) for speed, or base
- **Chunk Size:** 250 characters by default; reduce it if long chunks stall
- **Precision:** auto uses bfloat16 on CUDA and float32 on CPU; explicit types are available for troubleshooting
- **Default Prompt** and **Default Prompt Transcript**
- **Sampling Steps:** SOAR/base commonly use 10–32; MeanFlow is designed for 4
- **Guidance Scale:** 1.2 initially; approximately 0.8–1.0 can sound calmer
- **Speaker Scale:** 1.5 initially; lower it if reference style is exaggerated
- **Seed:** 42 by default for easier comparison; blank permits unfixed variation
- **Language:** `none`, `auto_detect`, or a supported language token
- **Normalize Text**, **Optimize with torch.compile**, and **Allow reference-audio-only fallback**

## Effective-use tips

1. Start with SOAR, 10 steps, the exact transcript, and a short representative passage.
2. Compare MeanFlow at 4 steps when speed matters; do not judge it at SOAR's step count.
3. Change guidance or speaker scale in small increments and replay the same sentence.
4. Use a fixed seed while tuning. Change it only when seeking a different delivery.
5. `torch.compile` incurs a slow warm-up and helps only after successful compilation and repeated inference.
6. Reduce chunk size for stalls, long silences, or identity drift before raising output limits elsewhere.

## Time, privacy, and limitations

First use is long: it can include a multi-gigabyte download, ASR model initialization, Dot.TTS model loading, and optional compilation. Later chunks are the meaningful performance measure. SOAR with more steps is slower than MeanFlow at four steps.

After assets are cached, transcription and synthesis remain local with no provider fee. The isolated runtime and multiple model variants consume significant disk space.

The adapter cannot guarantee equal quality across every language accepted by upstream. Transcript accuracy and reference cleanliness materially affect results. A CPU-compatible installation should not be interpreted as a promise of practical audiobook throughput.

## Authoritative reference

- [RedNote Dot.TTS official repository](https://github.com/rednote-hilab/dots.tts)

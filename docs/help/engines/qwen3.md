# Qwen3-TTS: Custom Voice, Clone, and Design

TTS-Story uses three distinct Qwen3-TTS model modes. **CustomVoice** and **Voice Clone** are selectable job engines. **VoiceDesign** is used only from Voice Creation to make a preview that can be saved as a reusable reference prompt.

## Choose the correct mode

![Voice Creation workspace for designing and previewing Qwen3 voices](../../../static/help/screenshots/voice-creation.png)

*Voice Creation lets you describe, preview, and save a Qwen3 VoiceDesign result for later use.*

- **Qwen3 CustomVoice:** choose one of the speakers reported by the installed CustomVoice model and optionally describe delivery with an instruction such as “calm, warm narration.” This directs an existing identity; it does not invent a new speaker.
- **Qwen3 Voice Clone:** condition the Base model with reference audio. TTS-Story can transcribe the prompt automatically with SenseVoice when no transcript is supplied.
- **Qwen3 VoiceDesign:** describe a voice in [Voice Creation](help:voice-creation), generate a short local preview, and save that audio as a prompt. It is not a normal full-manuscript engine in the Generate list.

## Requirements and setup

The normal setup installs the Qwen TTS runtime. Each mode uses a separate 1.7B model by default, so selecting a new mode can trigger another multi-gigabyte download. A supported NVIDIA GPU is strongly recommended. CPU is selectable but can be impractically slow for long work.

The defaults are:

- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base` for cloning
- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` for Voice Creation

Open [Settings → Engine Settings](app:settings/qwen3), leave Device at `auto`, and use the default model IDs unless deliberately testing a compatible replacement.

## Controls TTS-Story exposes

Both job modes share a 500-character chunk target and expose **Device**, **DType**, **Attention**, and **Default Language**.

- **bfloat16** is the initial dtype and is appropriate on supported recent GPUs. Try float16 if the GPU lacks good bfloat16 support. Float32 uses substantially more memory.
- **Flash Attention 2** can reduce memory use and improve speed, but it requires a compatible GPU, half-precision dtype, and a successful optional installation. If initialization reports a FlashAttention error, select **eager** rather than repeatedly retrying the job.
- **Auto language** lets the model infer or use its normal behavior. An explicit language can improve consistency when the manuscript is known to be monolingual.
- **Default Instruction** applies to CustomVoice. Per-speaker instructions can describe pace, emotion, or delivery.
- **Default Prompt** and **Prompt Transcript** apply to Clone. A per-speaker prompt takes priority.

The speaker and language choices shown by TTS-Story are built-in compatibility lists. The metadata endpoint deliberately does not load the large model, so a custom replacement model may support a different set of choices than the interface displays.

## Effective-use tips

1. Use CustomVoice when a built-in identity is acceptable and consistent instruction control matters.
2. Keep instructions short and non-conflicting. “Calm, intimate narration” is more reliable than a paragraph of competing traits.
3. For Clone, use clean single-speaker audio and verify the automatic transcript. A transcript mismatch can reduce similarity or intelligibility.
4. Start with 500-character chunks. Reduce the target if long sentences cause drift or memory pressure.
5. Test dtype and attention changes with the same passage. A successful first sample is more important than the theoretically fastest setting.
6. VoiceDesign results vary by prompt and seed behavior. Save only previews that have been auditioned with text similar to the intended project.

## Time, privacy, and limitations

The first use of each model can spend significant time downloading, loading, and warming GPU kernels. Switching among CustomVoice, Clone, and VoiceDesign can unload and load different weights. Later chunks are a better speed measurement than the first preview.

After model downloads, synthesis, design, and automatic prompt transcription run locally. There is no provider usage fee and manuscript text is not submitted to a cloud TTS service.

TTS-Story does not expose Qwen VoiceDesign as a regular job engine. CustomVoice instructions affect delivery but do not train a permanent identity. Supported languages and speakers depend on the installed model, and output quality can vary across languages.

## Authoritative reference

- [Qwen3-TTS official repository](https://github.com/QwenLM/Qwen3-TTS)

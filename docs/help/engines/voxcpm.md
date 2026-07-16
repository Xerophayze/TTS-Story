# VoxCPM 1.5

TTS-Story integrates **VoxCPM 1.5**, a local English-and-Chinese continuation model for expressive zero-shot voice cloning. Guidance for newer VoxCPM releases does not describe this adapter.

## Best for

- Local English or Chinese voice cloning
- Expressive speech conditioned by a reference recording
- Users with an NVIDIA GPU who want prompts and manuscript text to remain local

VoxCPM 1.5 is not a preset-voice engine. Assign a reference prompt for each cloned speaker, or configure a Default Prompt.

## Requirements and setup

![Voice Prompts library used to manage cloning reference recordings](../../../static/help/screenshots/voice-prompts.png)

*Upload a clean VoxCPM reference here, then select it in the speaker assignment or engine defaults.*

The main setup installs the VoxCPM runtime and SenseVoice transcription dependency. The model and transcription files can download on first use. Upstream reports roughly 6 GB of VRAM for VoxCPM 1.5; available memory also depends on the rest of the application and the selected chunk.

Open [Settings → Engine Settings](app:settings/voxcpm), choose **VoxCPM 1.5**, leave **Device** at `auto`, and keep the model ID at `openbmb/VoxCPM1.5`. Select a clean prompt in the speaker assignment or enter a fallback Default Prompt.

The prompt transcript is optional because TTS-Story can transcribe it with SenseVoice and cache the result. A manually verified, exact transcript is still preferable when names, accents, or noisy audio could confuse ASR.

## Controls TTS-Story exposes

- **Model ID** and **Device** (`auto`, `cuda`, or `cpu`)
- **Chunk Size:** 550 characters by default; approximately 400–600 is a practical range
- **Default Prompt** and **Prompt Transcript**
- **CFG Guidance:** controls conditioning strength; the application starts conservatively
- **Timesteps:** more sampling work generally costs more time; lower values should be evaluated by listening
- **Normalize numbers/abbreviations:** helpful for dates, quantities, and abbreviations, but review text whose literal form matters
- **Denoise output:** optional; leave off unless a test demonstrates an improvement

## Effective-use tips

1. Use one speaker, clean speech, and a stable recording level in the reference.
2. Keep the transcript verbatim, including the words actually spoken rather than the intended script.
3. Match prompt and target language when possible.
4. Begin with the saved defaults. If generation is too slow, reduce timesteps gradually and compare the same sentence.
5. If a long passage stalls or drifts, reduce chunk size before increasing guidance.
6. Listen for identity and pacing across several chunks; a good single sentence does not guarantee long-form consistency.

See [Reference Voice Prompts](help:voice-prompts) for recording guidance and [Performance Tuning](help:performance-tuning) before changing several GPU settings.

## Time, privacy, and limitations

First use can take several minutes because both synthesis and transcription assets may download and initialize. Later speed depends heavily on GPU, timesteps, text length, and whether another model is already using VRAM. CPU is selectable but is not a practical baseline for a long audiobook.

After required files are downloaded, synthesis and automatic transcription run locally. No per-character provider fee applies.

This adapter supports English and Chinese and targets VoxCPM 1.5. It does not expose capabilities advertised for VoxCPM2, such as its broader language set or newer model modes.

## Authoritative reference

- [OpenBMB VoxCPM repository](https://github.com/OpenBMB/VoxCPM)

# Engine Reference and Comparison

TTS-Story offers sixteen engines in the Generate list. A seventeenth registered workflow, OmniVoice Design, is used by Voice Creation and Library tools rather than as a normal full-job engine.

Choose an engine for the voice source you need first, then consider hardware, language, privacy, and cost. A fast engine with the wrong voice is rarely the best choice for a long project.

## Quick comparison

| Engine | Runs | Voice source | Best fit | Main tradeoff |
|---|---|---|---|---|
| Kokoro | Local, CPU or GPU | Built-in voices and local blends | Lightweight narration and repeatable character voices | No reference-audio cloning |
| Kokoro Replicate | Cloud | Provider's Kokoro voices | Kokoro without local inference | Replicate token, network, and usage cost |
| Chatterbox Turbo | Local CUDA recommended | Default voice or reference clone | Expressive English and supported non-verbal tags | Large model and substantial VRAM |
| Chatterbox Turbo Replicate | Cloud | Provider voice or reference clone | Expressive English without local GPU work | Reference text/audio leave the computer; billed by Replicate |
| VoxCPM 1.5 | Local CUDA recommended | Reference clone | Expressive English or Chinese cloning | Large first download and slower local inference |
| Pocket TTS Clone | Local CPU | Reference audio or saved voice state | CPU-only English cloning | Older, English-only adapter; cloning is slower than presets |
| Pocket TTS Preset | Local CPU | Installed preset voices | Simple private CPU generation | Limited voice catalog |
| Qwen3 CustomVoice | Local CUDA recommended | Model-provided speakers plus instruction | Directed tone and delivery with stable built-ins | Large model; instructions do not create a new identity |
| Qwen3 Voice Clone | Local CUDA recommended | Reference clone | High-quality multilingual cloning | Large model and reference preparation |
| OmniVoice Clone | Local; CUDA recommended | Reference clone | Broad multilingual coverage | Isolated runtime and several-GB model |
| KittenTTS | Local CPU | Eight built-in voices | Small, simple, low-resource English TTS | No cloning and fewer controls |
| IndexTTS | Local CUDA recommended | Reference clone | English/Chinese zero-shot cloning | Isolated runtime; many performance controls |
| Dot.TTS | Local CUDA strongly recommended | Reference clone plus transcript | High-quality 48 kHz cloning | 2B model, long first run, clean transcript-dependent prompt |
| Azure AI Speech | Cloud | Voices discovered from an Azure region | Supported production cloud speech, styles, roles, and SSML prosody | Azure account, key, region, and usage billing |
| Edge TTS | Cloud, experimental | Microsoft's current consumer voice catalog | No-key testing and personal use | Unofficial service with no availability guarantee |
| ElevenLabs | Cloud | Voices and models available to the account | Polished hosted voices and continuity | API key, quota, subscription limits, and character cost |

OmniVoice Design is available through [Voice Creation](help:voice-creation). It turns an instruction into speech locally, but it is not in the normal Generate engine list.

## Hardware and privacy

- **Easiest CPU choices:** KittenTTS, Pocket TTS Preset, and Kokoro. Kokoro can also use CUDA.
- **GPU-focused local choices:** Chatterbox Turbo, VoxCPM 1.5, Qwen3-TTS, OmniVoice, IndexTTS, and Dot.TTS. Some adapters permit CPU selection, but large-model generation can be impractically slow.
- **Cloud choices:** the two Replicate engines, Azure Speech, Edge TTS, and ElevenLabs need a working internet connection. No local inference GPU is required.
- **Private after download:** local engines keep manuscript and generated speech on this computer. Initial installation, model downloads, and optional model updates still contact package or model hosts.
- **Cloud boundary:** cloud engines send the text being spoken to the provider. Cloning engines also send their reference audio when the provider performs synthesis.

## Voice cloning versus voice design

Cloning reproduces characteristics from a clean reference recording. Use a short, single-speaker clip with little noise, music, reverberation, or compression. Engines that request a transcript work best when it matches the clip exactly. See [Reference Voice Prompts](help:voice-prompts).

Voice design starts from a written description. TTS-Story exposes Qwen3 VoiceDesign and OmniVoice Design through Voice Creation. A designed result can be saved as a reusable prompt, but it is not the same as a permanently trained custom model.

Kokoro custom voices are weighted blends of compatible Kokoro embeddings. They are neither reference clones nor newly trained voices.

## Settings shared by engines

![Engine Settings navigation with the engine tabs and configuration panels](../../../static/help/screenshots/engine-settings-navigation.png)

*Expand Engine Settings, then select an engine tab to reveal its dedicated controls.*

Engine-specific controls live under [Settings → Engine Settings](app:settings). Output format, bitrate, chunk transitions, silence, global speed, chapter behavior, and VRAM cleanup are separate generation settings. Per-speaker assignments can override the default voice, language, prompt, speed, and supported engine extras.

Start with defaults and a short representative passage. Change one setting at a time, use Quick Test, and keep the result that sounds better rather than assuming a larger number means higher quality. See [Performance Tuning](help:performance-tuning) and [Generation Time and ETA](help:generation-times).

## First-run expectations

Cloud engines normally start after credentials and catalog discovery, although provider queues and rate limits can delay individual requests. Local engines may download anything from tens of megabytes to several gigabytes. Large engines also spend time loading weights, warming kernels, or compiling on their first request. A first test is therefore not a reliable speed benchmark; compare later chunks from the same passage.

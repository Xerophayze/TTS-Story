# Pocket TTS: Presets and Voice Clone

Pocket TTS is a local, CPU-only English engine. TTS-Story exposes two job choices backed by the same pinned runtime: **Pocket TTS Preset** for installed built-in voices and **Pocket TTS Clone** for a reference recording or saved voice state.

## Best for

- Private generation without an NVIDIA GPU
- A compact CPU workflow with reusable preset voices
- English voice cloning when GPU-focused models are unavailable

Preset voices initialize and generate more simply than a new audio prompt. Use Clone only when a specific reference identity is important.

## Requirements and setup

![Voice Prompts library for uploading and organizing reference voices](../../../static/help/screenshots/voice-prompts.png)

*Pocket TTS Clone uses a saved reference from Voice Prompts; Preset mode does not require one.*

The normal setup installs `pocket-tts==1.0.3`. TTS-Story intentionally uses the older `b6369a24` model variant and declares this integration English-only, even if a newer upstream release advertises additional capabilities.

No GPU is used. Initial model or voice assets may download the first time they are selected. The built-in preset voices use the non-cloning model and do not require Hugging Face authentication. Custom voice cloning uses Kyutai's gated weights.

To enable cloning:

1. Sign in and open the [Kyutai Pocket TTS model page](https://huggingface.co/kyutai/pocket-tts). Review and accept its access conditions, including the requirement for explicit and lawful consent when cloning a voice.
2. Open [Hugging Face Access Tokens](https://huggingface.co/settings/tokens) and create either a **Read** token or a fine-grained token with read access to `kyutai/pocket-tts`.
3. In TTS-Story, open **Settings → Engine Settings → Pocket TTS**, paste the token, and select **Verify Voice-Cloning Access**.
4. Save Settings. The token is stored only in the local configuration, passed to the isolated Pocket TTS process, and removed by TTS-Story's repository-sync scrubber.

Choose the preset or clone engine on Generate. For cloning, assign a supported audio prompt such as WAV or MP3, or a compatible `.safetensors` voice state. See [Reference Voice Prompts](help:voice-prompts).

## Controls TTS-Story exposes

- **Model Variant:** keep `b6369a24` unless validating an adapter-compatible alternative
- **Hugging Face Access Token:** required only for custom voice cloning; use **Verify Voice-Cloning Access** before starting a long job
- **Chunk Size:** 450 characters by default; 300–500 is a useful starting range
- **Temperature:** starts at 0.7; higher values can add variation and instability
- **Decode Steps:** starts at 1; extra work is not automatically better for every passage
- **Noise Clamp:** optional and blank by default
- **EOS Threshold:** controls end-of-speech behavior; restore `-4.0` if clips end too early or continue unexpectedly after experimentation
- **Default Voice Prompt** and **Truncate Prompt Audio**
- **CPU Threads** and **Interop Threads:** blank means let the runtime choose

The preset voice list is read from the installed Pocket TTS package, so the exact names shown by TTS-Story belong to that pinned installation.

## Effective-use tips

- Start with automatic thread counts. Assigning every logical core can make the desktop less responsive and does not always improve throughput.
- Use a short, clean, single-speaker prompt. Very long prompt audio adds preparation work and may include unwanted style changes.
- Test Preset first to separate general runtime problems from prompt-conditioning problems.
- Shorten chunks if endings are clipped or delivery becomes unstable.
- Change temperature or decode settings only after saving a baseline sample.

## Time, privacy, and limitations

Pocket TTS is designed for CPU use, but total generation time still varies with processor, thread settings, prompt conditioning, and chunk count. The first cloned use may be slower while the prompt is converted to a voice state; a saved compatible state can load faster later.

After model download, text, prompts, and generated audio stay local and there is no provider usage charge.

TTS-Story's pinned adapter is English-only. It does not expose every capability of current Pocket TTS releases, and it does not use CUDA. Cloning quality depends strongly on the prompt and may be less consistent than a larger GPU model.

## Authoritative reference

- [Kyutai Pocket TTS repository](https://github.com/kyutai-labs/pocket-tts)
- [Kyutai Pocket TTS gated model and access conditions](https://huggingface.co/kyutai/pocket-tts)
- [Hugging Face token documentation](https://huggingface.co/docs/hub/security-tokens)

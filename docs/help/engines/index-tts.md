# IndexTTS

TTS-Story uses IndexTTS as a local English-and-Chinese zero-shot cloning engine. It runs in its own environment and uses reference audio from the shared Voice Prompts library.

## Best for

- Local English or Chinese reference cloning
- Users with an NVIDIA GPU who want detailed speed/quality controls
- Rebuilding individual chunks with the same reference identity

TTS-Story defaults to IndexTTS-2, but the adapter currently uses its cloning and sampling path only. It does **not** expose IndexTTS-2's upstream emotion-reference, emotion-vector, emotion-text, or duration-control interfaces.

## Requirements and setup

![Voice Prompts library used to store cloning references](../../../static/help/screenshots/voice-prompts.png)

*Store the clean reference in Voice Prompts, then select it for IndexTTS in the speaker or engine settings.*

The normal setup clones the official IndexTTS repository into `engines/index-tts`, creates its isolated environment with `uv`, and installs dependencies. Model weights are downloaded automatically on first use and are approximately 2–4 GB.

A CUDA GPU is strongly recommended. CPU is selectable but large jobs can be very slow. The Windows setup deliberately skips the optional DeepSpeed extra because it usually cannot build without specialized CUDA tooling.

Assign a clean prompt for each speaker or configure a Default Prompt under [Settings → Engine Settings](app:settings/index-tts). See [Reference Voice Prompts](help:voice-prompts).

## Controls TTS-Story exposes

- **Model Version:** IndexTTS-2, 1.5, or legacy 1.0
- **Device**, **Default Prompt**, and **Chunk Size** (400 by default)
- **Beam Search Width:** 1 is fastest; wider search adds GPT-stage work
- **Diffusion Steps:** TTS-Story starts at 12; more steps increase the slow synthesis stage
- **Temperature, top-p, and top-k:** sampling variation
- **Repetition Penalty:** raise cautiously if audio stutters or loops
- **Max Mel Tokens:** output-length safety cap; 1500 is roughly 68 seconds according to the UI estimate
- **Max Text Tokens per Segment:** internal splitting limit
- **FP16, DeepSpeed, torch.compile, and Accel:** optional performance paths whose support depends on the isolated runtime and GPU

Saved application defaults leave DeepSpeed, torch.compile, and Accel off. FP16 can reduce memory on supported GPUs, but make a baseline before enabling several acceleration paths together.

## Effective-use tips

1. Start with one beam, 12 diffusion steps, and the supplied sampling defaults.
2. Use a clean single-speaker prompt. A noisy reference is usually a larger quality problem than a sampling value.
3. If output loops, shorten the chunk first, then test a small repetition-penalty increase.
4. If a chunk is cut short, check Max Mel Tokens and sentence length before raising every limit.
5. `torch.compile` can make the first compiled request much slower while later requests improve. Benchmark after warm-up.
6. Treat DeepSpeed as unsupported unless you intentionally installed and verified it inside the IndexTTS environment.

## Time, privacy, and limitations

The first job includes model download, weight loading, and possibly compilation. Later speed varies with diffusion steps, beams, precision, text length, and GPU. Changing model versions can trigger another download.

After assets are cached, synthesis is local and has no provider usage charge. The isolated environment occupies additional disk space by design.

This adapter supports English and Chinese. Despite upstream IndexTTS-2 capabilities and older interface text that may mention emotion control, TTS-Story does not currently send emotion or duration parameters to the worker.

## Authoritative reference

- [IndexTTS official repository](https://github.com/index-tts/index-tts)

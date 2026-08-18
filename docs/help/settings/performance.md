# Performance Tuning for GPU and CPU Systems

Tune from evidence, not from the largest available number. Generate one representative chapter with defaults, record its timing and quality, and change one control at a time.

![Settings foundations with default engine and configuration groups visible](../../../static/help/screenshots/settings-foundations.png)

*Begin with the saved defaults, then change one engine or performance control at a time so comparisons remain meaningful.*

## Establish a baseline

1. Restart TTS-Story so the test begins from a known state.
2. Select the intended engine and one representative voice.
3. Use a chapter containing narration, dialogue, names, and long sentences.
4. Keep sampling, chunking, and FX at defaults.
5. Generate the chapter and inspect **Metrics** in [Library](app:library).
6. Record the processing time and chunk count shown there, the output duration from the player or file, and any audible artifacts. TTS-Story does not record per-job peak VRAM; use an operating-system or GPU-vendor monitor if that measurement matters.

The first local run may download or load model data. Run the same sample again before deciding the normal speed.

## GPU engines

Use `auto` device selection first. If a tab supports `cuda:n`, choose a numbered device only when the system has more than one compatible GPU.

For CUDA out-of-memory errors:

- close other GPU-heavy applications;
- reduce the engine-specific character chunk size;
- disable optional accelerators or high-memory settings;
- reduce beam, diffusion, or output-token settings where the engine exposes them;
- enable **Unload GPU model after job** when switching engines; and
- restart the application to release stale model state.

FP16 or another lower-precision option can reduce memory on supporting GPUs, but use only values offered by that engine tab. Do not apply IndexTTS, Dot.TTS, Qwen3, or OmniVoice advice to a different model.

Every optional local engine uses an engine-owned isolated runtime. Its dependencies and models are not represented solely by the main Python environment. Install, repair, or remove it through [Engine Settings](help:engine-management) instead of modifying the core `venv`.

## CPU engines

Pocket TTS and KittenTTS are practical CPU-focused choices. Pocket exposes a CPU thread control; raising it beyond the computer's useful physical resources can make the system less responsive without improving throughput.

Use smaller character chunks when memory is limited, but avoid extremely small chunks that create excessive overhead and audible joins. Keep other workloads light during the baseline.

## Cloud engines

Cloud speed depends on network latency, service load, quota, request size, and provider concurrency. Increasing parallelism can produce 429 throttling instead of a speedup.

- **Simultaneous Cloud Projects** limits how many cloud-backed projects can run together.
- Each supporting cloud provider has its own request-concurrency control in that provider's settings.
- Azure also exposes requests per minute.
- LocalAI is self-hosted and uses the local worker queue; begin with one job unless the LocalAI host and selected model are known to support more.

Start with defaults, watch the first complete chapter, and reduce concurrency after rate-limit or intermittent timeout errors.

## Quality controls cost time

More diffusion steps, beams, output tokens, or complex sampling can increase generation time. Post-generation pitch/speed effects and ACX normalization add processing. M4B export adds another encoding pass.

Use Quick Test for voice decisions, but benchmark a complete chapter because short previews do not include the same merge and packaging work.

## Avoid false speed comparisons

Compare the same text, voice, engine model, device, output format, and effects. Do not compare a preset CPU voice with a large zero-shot cloning model as if only hardware changed.

For timing interpretation, see [Generation Time and ETA](help:generation-times). For failures, see [GPU, CPU, Model, and Dependency Errors](help:gpu-cpu-errors).

# Generation Time and ETA

There is no reliable hardware-independent promise for audiobook generation time. Local model size, CPU/GPU capability, reference processing, text chunk complexity, cloud throttling, audio format, and post-processing can all dominate the result.

## How the queue estimates time

After chunks complete, TTS-Story estimates the remaining time from elapsed time per processed chunk. Early estimates can move sharply because there is little history. Later estimates can still change when chunks differ in length or complexity.

![Job Queue progress and estimated time information for an active generation](../../../static/help/screenshots/job-queue.png)

*Queue progress becomes a more useful planning estimate after several representative chunks have completed.*

The duration shown in Generate statistics is different: it assumes about 150 spoken words per minute and estimates listening length, not synthesis time.

## Measure your own baseline

The most useful estimate comes from one representative chapter.

1. Use the same engine, voice type, output format, and effects planned for the book.
2. Generate a chapter containing narration and dialogue.
3. Open **Metrics** on its [Library](app:library) card when metrics are available.
4. Record total processing time, chunk count, and audio duration.
5. Scale cautiously for the remaining word or chunk count.

The first local run may include model download and model loading. Do not use that run alone as the steady-state benchmark.

### Worked scaling example

If a representative 1,500-word chapter takes 12 minutes after the model is already loaded, a 60,000-word manuscript contains about 40 equivalent chapters. The simplest baseline is `40 × 12 = 480 minutes`, or about 8 hours. Treat that as a planning baseline, not a promise: add time for the initial model load, unusually long or difficult chunks, retries, chapter compilation, and final exports. Recalculate after several chapters if the measured average changes.

## Factors that increase time

- Larger or more compute-intensive local models
- CPU fallback when a model was expected to use CUDA
- Large diffusion-step, beam, or sampling settings
- Voice cloning that analyzes a prompt for many chunks
- Per-chunk speed/pitch processing
- ACX normalization and compressed output encoding
- Chapter, Full Story, ZIP, or M4B compilation
- Provider rate limits, network latency, or account concurrency limits
- Very small text chunks, which increase request and merge overhead

TTS-Story processes separate jobs serially. Submitting several jobs is useful for unattended work, but it does not make them run simultaneously.

## Improve throughput safely

- Start from the engine defaults.
- Test with effects disabled, then add only the needed effects.
- Keep chunks near the engine's recommended range.
- Use Replicate parallelism only within account and provider limits.
- For CPU engines, adjust documented thread controls rather than running multiple application instances.
- Enable **Unload GPU model after job** only when reclaiming VRAM matters more than the next job's model-load delay.
- Keep TTS-Story, the output directory, and model cache on responsive storage.

Do not increase every concurrency or chunk setting at once. A faster short test can become less reliable on a full book if it triggers memory or provider limits.

See [Audio and Generation Settings](help:settings-audio) and [Performance Tuning for GPU and CPU Systems](help:performance-tuning).

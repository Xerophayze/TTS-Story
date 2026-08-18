# GPU, CPU, Model, and Dependency Errors

Local engines depend on the main Python environment, engine-specific isolated environments, downloaded models, and audio command-line tools. A working web page does not guarantee that every optional engine is installed and ready.

![Engine settings navigation showing the tabs used to inspect local engine configuration](../../../static/help/screenshots/engine-settings-navigation.png)

*Open the affected engine's own Settings tab before changing device, model, chunk-size, or isolated-runtime options.*

## Confirm installation and health

After moving the project to another computer or updating dependencies, run the platform setup script and allow it to finish. Core setup installs TTS-Story and shared tools; it does not force every optional engine onto the computer. Open **Settings → Engine Settings** to install or repair the affected local engine.

Restart TTS-Story after setup. Then inspect `http://localhost:5000/api/health` and run:

```text
python scripts/engine_versions.py
```

The health response includes availability and unavailable-reason fields for engines such as Chatterbox Turbo, IndexTTS, Dot.TTS, and Edge TTS.

## First-run model downloads

Qwen3, OmniVoice, IndexTTS, Dot.TTS, and other local engines can download gigabytes of model data. The first request may appear slow while the model is fetched or loaded. TTS-Story displays a one-time first-job notice for each engine; keep the backend running and watch Job Queue and the terminal.

Check free disk space, internet access, the terminal, and engine-specific cache directories. Do not terminate the job merely because a large download pauses briefly. If a download is incomplete or the worker no longer starts, use that engine panel's uninstall/reinstall workflow rather than copying a partial cache into source control.

## Isolated engines

Supported local engines use engine-owned environments under their `engines` directories to avoid dependency conflicts. Installing a package only into the main `venv` may not fix their worker.

- OmniVoice uses `engines/omnivoice/.venv`.
- Dot.TTS uses `engines/dots-tts/.venv` and an upstream repository/model cache.
- IndexTTS uses its own environment/checkpoint layout under `engines/index-tts`.

Use the live engine-management log and health unavailable reason before modifying those environments manually. The normal repair path is [Install, Remove, and Reinstall TTS Engines](help:engine-management).

## CUDA out of memory

1. Stop other GPU applications.
2. Restart TTS-Story to unload cached models.
3. Reduce the selected engine's character chunk size.
4. Disable optional high-memory acceleration or quality settings.
5. Use a supported lower-precision mode if the engine tab offers it.
6. Try a smaller engine or CPU-focused engine.

**Unload GPU model after job** releases cached engine state after completion but makes the next job reload the model.

If CUDA is unavailable rather than out of memory, confirm the installed PyTorch build and GPU driver are compatible. Do not force `cuda` in Settings until `/api/health` reports CUDA available.

## CPU problems

CPU inference can be much slower without being stuck. Test one sentence and watch process activity. Pocket TTS exposes CPU threads; excessive threads can reduce responsiveness. KittenTTS is a lightweight alternative when large local engines are impractical.

An instruction-set or native-library import error is different from slow inference. Preserve the complete import error and rerun the current setup script.

## Missing audio tools

- **FFmpeg** is required for reliable merging/encoding and M4B export.
- **SoX** supports high-quality speed and pitch processing paths.
- **Rubber Band** may support higher-quality time stretching when available.

Windows setup places supported executables under `tools/ffmpeg`, `tools/sox`, and `tools/rubberband` when installation succeeds. Linux/macOS require the corresponding native tools described by the setup/README instructions. A raw chunk may play even when final merge or M4B fails because those stages use different tools.

For systematic collection of evidence, see [Prepare a Useful Issue Report](help:report-an-issue).

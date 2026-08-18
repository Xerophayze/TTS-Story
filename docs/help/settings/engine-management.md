# Install, Remove, and Reinstall TTS Engines

TTS-Story installs its core application first and lets you add only the speech engines you intend to use. Open [Settings → Engine Settings](app:settings) to manage them.

The engine chips are divided into two groups:

- **Local / Installable Engines** run model inference through files installed on this computer.
- **Cloud / Remote Providers** connect TTS-Story to an online service or a separately hosted server such as LocalAI.

Red chips require installation or configuration. Green chips are ready and appear in the engine selectors under **Quick Settings** and **Generate → Generation Options**. A yellow/busy state means an installation or removal is still running.

![Engine Settings navigation showing engine status chips and configuration panels](../../../static/help/screenshots/engine-settings-navigation.png)

*Choose the engine group and chip first; the panel below contains its setup, connection, or removal controls.*

## Install a local engine

1. Expand **Settings → Engine Settings**.
2. Select a red chip under **Local / Installable Engines**.
3. Review the hardware and disk-space guidance in that engine's help article.
4. Select **Install Engine**.
5. Leave TTS-Story running while the live log downloads and installs the runtime.
6. You may leave Settings or refresh the page. Return to the same engine panel to reconnect to the active log.
7. Wait for the operation to finish. If TTS-Story displays **Restart TTS-Story**, use that button and wait for the page to reconnect.

Install and uninstall controls are disabled while an engine-management operation is active, preventing the same installer from being started twice.

## What “isolated environment” means

Each supported local engine receives its own Python virtual environment, normally under `engines/<engine>/.venv`, plus engine-owned model and cache directories. This keeps packages such as PyTorch, Transformers, NumPy, and engine-specific libraries from replacing incompatible versions used by TTS-Story or another engine.

The main TTS-Story `venv` runs the web application. Engine workers run in their own environments and communicate with the application. Installing a missing package manually into the main `venv` therefore may not repair an isolated engine; use the engine panel's installer or reinstall action instead.

Isolation also makes removal predictable. It does not duplicate your projects, generated audio, saved voice prompts, or global Settings.

## The first generation can take longer

Completing **Install Engine** does not always download every model weight. Some upstream packages fetch model files only when the engine is used for synthesis.

When the first job for an engine is submitted, TTS-Story displays a one-time message explaining that the engine may be downloading weights, loading them into RAM or VRAM, warming kernels, or preparing a transcription model. Keep the backend running and monitor **Job Queue** and the terminal. Later jobs normally start faster.

Reinstalling a local engine resets this notice because its removed model files may need to be downloaded again.

## Remove a local engine

1. Finish, pause, or cancel work that is using the engine.
2. Open the green local-engine chip.
3. Select **Uninstall Engine**.
4. Read the confirmation carefully and approve the removal.
5. Follow the live log until it completes, then restart when prompted.

Removal deletes that engine's isolated runtime and engine-owned downloaded model files. It preserves:

- the TTS-Story core environment and shared audio tools;
- other installed TTS engines;
- `config.json` and provider settings;
- saved projects and Job Queue records;
- Library audio and chunk metadata; and
- uploaded, downloaded, and designed voice prompts.

An assignment can still refer to a voice or engine that is no longer available. After removing an engine, review saved projects before submitting them with a replacement.

## Reinstall or repair an engine

After removal, the chip turns red and offers **Install Engine** again. Reinstallation creates a clean compatible environment and redownloads required packages and models. Use this when an engine's worker will not start, dependencies are corrupted, or an interrupted download cannot recover.

A normal TTS-Story update does not reinstall every optional engine. Update the core first, then reinstall only the affected engine when its help article or an error specifically calls for it.

## Cloud and remote providers

Cloud and remote chips do not install model runtimes into TTS-Story. They become green after their required endpoint, credentials, region, model, or voice catalog is configured and **Save Settings** is selected. Their models run on the provider or remote host, so there is no local engine environment to uninstall.

To stop using one, select another default engine and remove or replace its saved credentials privately. Deleting a TTS-Story setting does not delete an account, subscription, model, container, or saved voice profile on the external service.

For a self-hosted speech server, see [LocalAI TTS](help:engine-localai-tts). For account-based services, see [Configure Online Services Safely](help:online-services).

## If management appears stuck

- Return to the engine panel; the operation and log survive page navigation and refreshes while the backend remains running.
- Do not start a second installation from another browser window.
- Check the TTS-Story terminal for package, disk-space, network, CUDA, or permission errors.
- If the backend was closed during installation, start it again and reinstall that engine.
- If automatic restart is unavailable, close TTS-Story and launch it with `run.bat` or `run.sh`; those launchers support the in-app restart action.

See [GPU, CPU, Model, and Dependency Errors](help:gpu-cpu-errors) for model-loading and hardware failures.

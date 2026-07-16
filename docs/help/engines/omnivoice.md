# OmniVoice: Clone and Voice Design

OmniVoice provides local multilingual voice cloning and instruction-based voice design through one model. TTS-Story runs it in an isolated environment so its PyTorch and Transformers requirements do not conflict with other engines.

## Best for

- Reference cloning across a very broad language catalog
- Local synthesis when multilingual coverage matters
- Creating an English- or Chinese-directed voice preview from attributes such as age, pitch, accent, dialect, or whisper style

**OmniVoice Clone** appears in the Generate engine list. **OmniVoice Design** is registered for [Voice Creation](help:voice-creation) and Library workflows rather than normal full-job selection.

## Requirements and setup

![Voice Creation workspace with the Qwen3 and OmniVoice design modes](../../../static/help/screenshots/voice-creation.png)

*Switch Voice Creation to OmniVoice to combine concrete voice attributes and audition a designed prompt.*

The normal TTS-Story setup creates `engines/omnivoice/.venv`, installs the compatible runtime, and normally attempts to prefetch `k2-fsa/OmniVoice`. If prefetch was skipped or failed, first generation retries the several-gigabyte download.

CUDA is the practical choice for long jobs. The adapter also accepts `cpu`, `mps`, and explicit CUDA device names, subject to the installed runtime and hardware. Leave Device at `auto` initially. The default float16 dtype minimizes memory on compatible accelerators; use float32 only when necessary because it increases memory and time.

Do not install OmniVoice into TTS-Story's main `venv`; the isolated environment is intentional. If its setup failed, rerun `setup.bat` rather than mixing dependencies manually.

## Controls TTS-Story exposes

Open [Settings → Engine Settings](app:settings/omnivoice) and select **OmniVoice**.

- **Model ID:** `k2-fsa/OmniVoice` by default
- **Chunk Size:** 500 characters for both clone and design workflows
- **Device** and **DType**
- **Diffusion Steps:** 32 favors quality; 16 is a useful faster comparison
- **Post-Process Output:** trims/manages sentence-boundary silence; disable it if short sentences run together unnaturally
- **Clone Default Prompt** and **Prompt Transcript:** Whisper transcribes locally when the transcript is blank
- **Design Default Instruction:** a comma-separated description such as “female, middle aged, low pitch, British accent”

Per-speaker reference prompts and transcripts override clone defaults.

## Effective-use tips

- For cloning, provide clean, short, single-speaker audio and an exact transcript when possible. Automatic Whisper transcription is convenient, not infallible.
- Match reference and target language for the most stable identity. Test multilingual switching before a full job.
- Reduce diffusion steps only after making a 32-step baseline with the same passage.
- Keep design descriptions concrete. Specify a few compatible traits rather than a narrative biography.
- Upstream reports voice design is trained primarily for Chinese and English and can be less stable in low-resource languages. Use cloning when preserving an identity matters more than inventing one.
- If pauses disappear, test with Post-Process disabled before altering global silence and crossfade settings.

## Time, privacy, and limitations

Initial setup or first use can take a long time because the isolated runtime and model are large. Each worker startup loads the model, and first inference may warm GPU kernels. CPU generation can be extremely slow.

Once the model is cached, reference transcription and synthesis remain local with no per-character fee. Model and package downloads still require internet access.

“600+ languages” is an upstream model claim, not a promise of equal quality in every language, dialect, script, or code-switched passage. Voice Design is not exposed as a normal Generate job, and a designed preview must be saved and auditioned before reuse.

## Authoritative reference

- [k2-fsa OmniVoice official repository](https://github.com/k2-fsa/OmniVoice)

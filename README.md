## ❤️ Thank You for Supporting Our Work

We are deeply grateful to everyone who uses, shares, contributes to, and supports TTS-Story. Your encouragement helps us continue improving the project, adding new features, fixing problems, and keeping it freely available to the community.

If you appreciate what we do and would like to support ongoing development, you can make a donation here:

👉 **[Support TTS-Story and our other projects](https://xerophayze.com/store.html?category=patron+support)**

Every contribution is appreciated. Thank you for helping make this work possible! 🙏

---

# Current Updates and Notes - updated 07-15-2026
- **Complete illustrated in-app user guide** - use the new searchable Help Center for screenshot-guided workflows, engine comparisons, setup instructions, troubleshooting, and contextual help from throughout the interface.
- **Edge TTS and ElevenLabs support** - generate speech with dynamically discovered Edge voices or the voices and models available through your ElevenLabs account.
- **Microsoft Azure AI Speech support** - use an Azure Speech key and region to access Microsoft's multilingual cloud voices and expression controls.
- **Atlas Cloud and OpenRouter support** - process and prepare text using models available through either LLM provider.
- **Dot.TTS support** - generate high-quality local voice-cloned audio using the RedNote HiLab Dot.TTS engine.
- **Improved installation and platform support** - expanded setup, diagnostics, and audio-tool compatibility across Windows, Linux, macOS, and Pinokio.

### Previous Updates
- Improved section handling, chapter organization, review, regeneration, and audiobook export.
- Added Pinokio installation and update support.
- Added M4B audiobook export with chapter markers and cover art.
- Added dynamic section-heading detection and chapter controls.
- Improved voice-prompt audio quality and management.

# TTS-Story

A web-based Text-to-Speech application supporting multiple TTS engines including **Kokoro-82M**, **Chatterbox**, **VoxCPM 1.5**, **Qwen3 TTS** (Custom Voice, Clone, Voice Creation), **Pocket TTS** (CPU-friendly), **OmniVoice**, **KittenTTS** (ultra-lightweight CPU-only), **IndexTTS** (zero-shot voice cloning by Bilibili), **Dot.TTS** (48 kHz zero-shot voice cloning by rednote-hilab), **Microsoft Azure AI Speech**, **Microsoft Edge TTS**, and **ElevenLabs**, with local and cloud options for generating multi-voice audiobooks and stories.

<div align="center">
  <table>
    <tr>
      <td>
        <a href="https://github.com/user-attachments/assets/fdec637a-e543-4000-88d9-050ca68a413f" target="_blank">
          <img src="https://github.com/user-attachments/assets/fdec637a-e543-4000-88d9-050ca68a413f" alt="chrome_uQg512nBym" width="280" />
        </a>
      </td>
      <td>
        <a href="https://github.com/user-attachments/assets/00dd7984-d685-4482-8401-2ad03dac44e4" target="_blank">
          <img src="https://github.com/user-attachments/assets/00dd7984-d685-4482-8401-2ad03dac44e4" alt="chrome_Y4WyrXGpRI" width="280" />
        </a>
      </td>
      <td>
        <a href="https://github.com/user-attachments/assets/dcf91a06-5f26-45a1-858f-157aff6d60ca" target="_blank">
          <img src="https://github.com/user-attachments/assets/dcf91a06-5f26-45a1-858f-157aff6d60ca" alt="chrome_YKrqBtk5GU" width="280" />
        </a>
      </td>
    </tr>
    <tr>
      <td>
        <a href="https://github.com/user-attachments/assets/508fd274-a8a4-4b6e-8b8f-2ceb7ae36571" target="_blank">
          <img src="https://github.com/user-attachments/assets/508fd274-a8a4-4b6e-8b8f-2ceb7ae36571" alt="chrome_52iXxPMM4R" width="280" />
        </a>
      </td>
      <td>
        <a href="https://github.com/user-attachments/assets/b961444c-6b1f-46a2-b09c-a618e8557ea2" target="_blank">
          <img src="https://github.com/user-attachments/assets/b961444c-6b1f-46a2-b09c-a618e8557ea2" alt="chrome_CP9EEaBnE5" width="280" />
        </a>
      </td>
      <td>
        <a href="https://github.com/user-attachments/assets/9af12bf6-47f7-45f5-8c0a-e6220b694497" target="_blank">
          <img src="https://github.com/user-attachments/assets/9af12bf6-47f7-45f5-8c0a-e6220b694497" alt="chrome_d8ZrL1laNn" width="280" />
        </a>
      </td>
    </tr>
    <tr>
      <td>
        <a href="https://github.com/user-attachments/assets/2a57d2cc-eddb-4648-89c8-27b353479549" target="_blank">
          <img src="https://github.com/user-attachments/assets/2a57d2cc-eddb-4648-89c8-27b353479549" alt="chrome_rJUicZZFGM" width="280" />
        </a>
      </td>
      <td>
        <a href="https://github.com/user-attachments/assets/3307938d-b628-4852-90fa-655a4eca2164" target="_blank">
          <img src="https://github.com/user-attachments/assets/3307938d-b628-4852-90fa-655a4eca2164" alt="chrome_3heAn2FRjF" width="280" />
        </a>
      </td>
      <td></td>
    </tr>
  </table>
</div>

## Features

### TTS Engines
- **Multi-Engine Support**: Choose from sixteen TTS engine options:
  - **Kokoro · Local GPU** - Run Kokoro-82M locally on your NVIDIA GPU
  - **Kokoro · Replicate** - Use Kokoro via Replicate cloud API
  - **Chatterbox · Local GPU** - Run Chatterbox locally with voice cloning (~8GB VRAM required)
  - **Chatterbox · Replicate** - Use Chatterbox via Replicate cloud API (`resemble-ai/chatterbox-turbo`)
  - **VoxCPM 1.5 · Local GPU** - Run VoxCPM 1.5 locally with voice cloning and automatic transcription
  - **Pocket TTS · Preset Voices** - CPU-only preset voices with fast local generation
  - **Pocket TTS · Voice Clone** - CPU-only voice cloning using reference prompts
  - **Qwen3 TTS · Custom Voice** - Generate with Qwen3 TTS custom voice prompts
  - **Qwen3 TTS · Clone** - Clone a voice from reference audio using Qwen3 TTS
  - **OmniVoice · Voice Clone** - Voice cloning in an isolated environment
  - **KittenTTS** - Ultra-lightweight CPU-only engine, no GPU required
  - **IndexTTS** - Zero-shot voice cloning by Bilibili, runs in an isolated venv
  - **Dot.TTS** - 48 kHz zero-shot voice cloning by rednote-hilab, runs in an isolated venv
  - **Microsoft Azure Speech · Cloud** - Use your Azure Speech resource with dynamically discovered multilingual neural voices, styles, roles, and SSML prosody controls
  - **Microsoft Edge TTS · Experimental Cloud** - Use Microsoft's dynamically discovered consumer Edge voices without an API key
  - **ElevenLabs · Cloud** - Use the voices and text-to-speech models available through your ElevenLabs account
- **Unified Replicate API**: Single API token works for both Kokoro and Chatterbox Replicate engines
- **Voice Cloning**: Upload your own voice recordings (10-15 seconds recommended) for supported cloning engines including Chatterbox, VoxCPM, Qwen3 Clone, OmniVoice, Pocket TTS Clone, IndexTTS, and Dot.TTS
- **Voice Prompt Management**: Add, rename, delete, and preview custom voice prompts with drag-and-drop bulk upload
- **External Voice Library**: Browse and download 500+ voice samples from the [TTS Samples](https://github.com/yaph/tts-samples) repository directly in the app
- **Qwen3 TTS Modes**: Dedicated flows for **Custom Voice**, **Clone**, and **Voice Creation** to generate, clone, or design new voices

### Voice & Audio
- **Multi-Voice Support**: Use built-in, blended, or cloned voices for any number of characters in your story
- **Custom Voice Blending**: Mix any combination of Kokoro voices with weighted ratios to create reusable "custom_*" voice codes
- **Speaker Tags & Auto Detection**: Automatically parse `[speaker1]...[/speaker1]` or `[alice]...[/alice]` tags
- **Smart Text Chunking**: Automatically splits long texts into manageable chunks with per-engine configurable character limits
- **Alternate Word Registry**: Define word substitutions for terms TTS engines mispronounce — replacements are applied automatically before synthesis
- **Seamless Audio Merging**: Merges chunks into a single file with configurable crossfade
- **Intro & Inter-Segment Silence Controls**: Dial in precise empty space before the first line and between chunks

### AI & Processing
- **Multi-Provider LLM Pre-Processing**: Clean up and tag text with Gemini, Atlas Cloud, OpenRouter, LM Studio, or Ollama
- **Atlas Cloud Model Discovery**: Save an Atlas Cloud API key and retrieve current LLM choices from the authenticated API and public LLM catalog
- **OpenRouter Model Discovery**: Retrieve the text models allowed by your key's provider preferences, privacy settings, and guardrails
- **Speaker Memory Between Chunks**: LLM requests carry forward discovered speaker tags for consistency
- **Local GPU Processing**: Run entirely on your machine for privacy and speed
- **Cloud API Options**: Use Replicate, Microsoft Azure Speech, experimental Edge TTS, or ElevenLabs when you don't have local GPU resources

### Job Management & Library
- **Job Queue**: Submit multiple jobs, track real-time progress with ETA, cancel, and download results
- **Job Queue Tab**: Dedicated UI to monitor all jobs with progress bars and chunk counts
- **Audio Library**: Browsable list of all completed outputs with inline players, **engine indicator** showing which TTS engine was used, and delete/clear controls
- **Chapter Collections + Full Audiobook**: Toggle per-chapter outputs and optionally create a single combined audiobook

### UI & Configuration
- **Bundled User Guide**: Search 53 post-install guides, follow complete workflows, and open context-specific articles from the **?** buttons throughout the interface
- **Available Voices & Previews**: Browse all Kokoro voices grouped by language, generate preview samples
- **Configurable Settings**: Control TTS engine, speed, chunk size, output format, bitrate, crossfade
- **Dynamic LLM Controls**: Save cloud API keys and fetch available Gemini, Atlas Cloud, OpenRouter, or local models on demand
- **Web Interface**: Modern single-page UI built with Flask and vanilla JS

## Available Voices

### Kokoro Voices

TTS-Story exposes the full Kokoro-82M voice set, grouped by language.

### American English 🇺🇸 (lang_code `a`)
- Female: `af_alloy`, `af_aoede`, `af_bella`, `af_heart`, `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky`
- Male: `am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `am_santa`

### British English 🇬🇧 (lang_code `b`)
- Female: `bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily`
- Male: `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis`

### Spanish 🇪🇸 (lang_code `e`)
- `ef_dora`, `em_alex`, `em_santa`

### French 🇫🇷 (lang_code `f`)
- `ff_siwis`

### Hindi 🇮🇳 (lang_code `h`)
- `hf_alpha`, `hf_beta`, `hm_omega`

### Japanese 🇯🇵 (lang_code `j`)
- `jf_alpha`, `jf_gongitsune`, `jf_nezumi`, `jf_tebukuro`, `jm_kumo`

### Mandarin Chinese 🇨🇳 (lang_code `z`)
- `zf_xiaobei`, `zf_xiaoni`, `zf_xiaoxiao`, `zf_xiaoyi`

### Brazilian Portuguese 🇧🇷 (lang_code `p`)
- `pf_dora`, `pm_alex`, `pm_santa`

All of these voices are browsable in the **Available Voices** tab, where you can generate and play preview samples.

### Voice Prompts (Shared Cloning Engines)

Chatterbox, VoxCPM, Qwen3 Clone, OmniVoice, Pocket TTS Clone, IndexTTS, and Dot.TTS support voice cloning from audio recordings. Voice prompts are shared between these engines where compatible.

#### Adding Voice Prompts

1. Go to the **Available Voices** tab → **Voice Prompts** section
2. Upload a voice recording (WAV, MP3, M4A, FLAC, or OGG format)
   - **Recommended duration**: 10-15 seconds of clear speech
   - Avoid background noise for best results
3. Give the voice a descriptive name and click **Save Voice**
4. Your custom voices appear in the supported voice-cloning engine dropdowns

You can also drag-and-drop multiple audio files for bulk upload.

#### Voice Prompt Management

The Voice Prompts section provides a sortable list view with:
- **Name, Gender, Language, Duration, Source** columns
- **Sortable headers** - Click any column to sort
- **Filtering** - Filter by gender, language, or source (local vs external)
- **Search** - Find voices by name
- **Edit** - Click Edit to modify name, gender, and language metadata
- **Preview** - Play any voice sample directly
- **Delete** - Remove unwanted voice prompts

#### External Voice Library

TTS-Story integrates with the [TTS Samples](https://github.com/yaph/tts-samples) repository, providing access to 500+ pre-recorded voice samples in multiple languages:

1. In the Voice Prompts section, external voices appear with an "External" source badge
2. Click **Download** to save any external voice locally
3. Downloaded voices become local and can be edited/deleted
4. Filter by source to show only local or external voices

#### Voice Dropdown Enhancements

All voice selection dropdowns (main screen and library) now show:
- **Gender indicator**: `[M]` for Male, `[F]` for Female
- **Language**: Human-readable language name (e.g., "English (UK)")
- **Duration**: Sample length in seconds
- **Filter controls**: Filter by gender and language before selecting

## Installation

### Prerequisites
- Python 3.9 or higher
- NVIDIA GPU with CUDA support (optional, for local GPU inference)
- Internet connection (for downloading dependencies)

### Automatic Installation (Recommended)

1. **Run the installer/updater**

To download the installer, *Right-Click* the link and click "Save As":
[Install-Update.bat](https://github.com/Xerophayze/TTS-Story/raw/52b8d3a8edd6ac1ad8acfb1b83421bb4508d8d01/install-update.bat)

The installer will clone or update the repository, then run the setup script which will automatically:
- ✅ Detect your Python version
- ✅ Create a Python virtual environment
- ✅ Detect your NVIDIA GPU and CUDA version
- ✅ Install PyTorch with appropriate CUDA support (or CPU-only if no GPU)
- ✅ Install a CUDA 12.8 PyTorch build for Blackwell / RTX 50-series GPUs
- ✅ Download and install espeak-ng automatically
- ✅ Install all other required dependencies
- ✅ Download the Rubber Band CLI and wire it up for high-quality pitch/tempo FX
- ✅ Verify the installation

**Supported CUDA Versions:**
- CUDA 12.9, 12.8, 12.6, 12.4, 12.1
- CUDA 11.8
- CPU-only (automatic fallback if no GPU detected)

**Blackwell / RTX 50-series note:** these GPUs require a PyTorch build compiled
with `sm_120` support. The installer detects compute capability `12.x` or RTX
50-series GPU names and installs the stable PyTorch CUDA 12.8 wheel. If stable
PyTorch wheels ever lag a new GPU architecture, rerun setup with
`USE_TORCH_NIGHTLY=1` to opt into PyTorch nightly CUDA 12.8 wheels.

3. **Start the application**
```bash
run.bat
```

4. **Open your browser**
```
http://localhost:5000
```

### Manual Installation

If you prefer to install manually or the automatic setup fails:

1. **Install espeak-ng**
   - Download from [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases)
   - Install the `espeak-ng-X64.msi` file for Windows

2. **Install Rubber Band CLI (for pitch/tempo FX quality)**
   - Download the Windows zip from [breakfastquay.com/rubberband](https://breakfastquay.com/rubberband/)
   - Extract it and add the folder containing `rubberband.exe` to your `PATH`

2. **Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

3. **Install PyTorch with CUDA support**
```bash
# For Blackwell / RTX 50-series GPUs
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# For CUDA 12.4 GPUs
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

After installing PyTorch, verify that the wheel supports your GPU:
```bash
python scripts/torch_cuda_probe.py --test-cuda

# Blackwell / RTX 50-series should include sm_120
python scripts/torch_cuda_probe.py --require-arch sm_120 --test-cuda
```

4. **Install other dependencies**
```bash
pip install -r requirements.txt
```

5. **Run the application**
```bash
python app.py
```

## Usage

### Built-in User Guide

Open the **Help** tab for the complete bundled guide. It includes a first-run checklist, a screenshot-guided first-audio walkthrough, all TTS and LLM provider setup, voice assignment, Alternate Word Registry behavior, job and Library workflows, performance guidance, and troubleshooting. Search works across every article, and the **?** buttons beside interface controls open the relevant guide directly.

The same canonical Markdown articles are available in [`docs/help`](docs/help) for reading on GitHub or offline. Start with [Welcome to TTS-Story](docs/help/start-here/welcome.md), [First-Run Checklist](docs/help/start-here/first-run.md), or [Generate Your First Audio](docs/help/start-here/quick-start.md).

### Selecting a TTS Engine

TTS-Story supports sixteen TTS engine options. In the **Settings** tab, choose your preferred default engine:

| Engine | Description | Requirements |
|--------|-------------|--------------|
| **Kokoro · Local GPU** | Run Kokoro-82M locally | NVIDIA GPU with CUDA |
| **Kokoro · Replicate** | Kokoro via cloud API | Replicate API token |
| **Chatterbox · Local GPU** | Chatterbox with voice cloning | NVIDIA GPU (~8GB VRAM) |
| **Chatterbox · Replicate** | Chatterbox via cloud API | Replicate API token |
| **VoxCPM 1.5 · Local GPU** | VoxCPM with voice cloning & auto-transcription | NVIDIA GPU (~6GB VRAM) |
| **Qwen3 TTS · Custom Voice** | Qwen3 TTS custom voice prompts | NVIDIA GPU (local) |
| **Qwen3 TTS · Clone** | Qwen3 TTS voice cloning from reference audio | NVIDIA GPU (local) |
| **OmniVoice · Voice Clone** | Voice cloning from reference prompts | NVIDIA GPU recommended; isolated venv |
| **Pocket TTS · Preset Voices** | CPU-only preset voices, no GPU needed | CPU only |
| **Pocket TTS · Voice Clone** | CPU-only voice cloning from reference prompts | CPU only |
| **KittenTTS** | Ultra-lightweight CPU-only, 8 built-in voices | CPU only |
| **IndexTTS** | Zero-shot voice cloning, English + Chinese | NVIDIA GPU recommended; isolated venv |
| **Dot.TTS** | High-similarity zero-shot voice cloning with reference transcript | NVIDIA GPU recommended; isolated venv |
| **Microsoft Azure Speech · Cloud** | Regional multilingual neural voices with styles, roles, and SSML controls | Azure Speech resource key and region |
| **Microsoft Edge TTS · Experimental Cloud** | Dynamically discovered Microsoft consumer voices | Internet connection; no API key |
| **ElevenLabs · Cloud** | Account voices and current text-to-speech models | ElevenLabs API key and available character quota |

You can also override the engine per-job in the **Generate** tab.

### Microsoft Azure AI Speech Engine

Azure Speech is a bring-your-own-account cloud engine. TTS-Story uses the regional REST endpoints directly, so it does not add the native Azure Speech SDK or require a GPU.

#### Azure Speech Setup

1. Create or select an Azure AI Speech resource and copy one resource key plus its region from **Keys and Endpoint**. The key and region must belong to the same resource.
2. Open **Settings → Engine Settings → Azure Speech**.
3. Enter the resource key and region, then click **Test Connection & Load Voices**.
4. Select a default voice, output quality, and optional default expression settings, then save.
5. Select **Microsoft Azure Speech · Cloud** on the Generate tab and assign a voice to each speaker.

The voice list is retrieved from the selected Azure region. Each speaker can use a different locale and voice. When the selected voice advertises them, TTS-Story also exposes speaking style, role, style intensity, speed, pitch, and volume controls. Unsupported style or role combinations are not offered in the main assignment UI.

#### Azure Speech Settings

| Setting | Default | Description |
|---------|---------|-------------|
| Default Voice | `en-US-AvaMultilingualNeural` | Fallback voice when a speaker does not override it |
| Azure WAV Quality | 24 kHz, 16-bit mono PCM | Intermediate synthesis quality; 48 kHz is also available |
| Chunk Size | `1000` chars | Sentence-aware target size; larger chunks reduce request count |
| Request Limit | `20` per minute | Local rolling limiter; set this to the quota appropriate for your Azure tier, or `0` to disable local throttling |
| Timeout | `60` seconds | Per-request timeout |
| Default Style / Role | Neutral / default | Applied when a speaker does not choose another supported expression |
| Style Intensity | `1.0` | Azure style degree, clamped to `0.01`–`2.0` |

Azure usage and data handling are governed by your Azure resource and subscription. Review the current [Azure Speech pricing](https://azure.microsoft.com/pricing/details/speech/) and [Azure text-to-speech documentation](https://learn.microsoft.com/azure/ai-services/speech-service/text-to-speech) before large jobs. TTS-Story retries transient failures and `429` responses, but Azure quota and billing still apply.

### Microsoft Edge TTS Engine (Experimental)

Edge TTS provides a large multilingual voice catalog without an API key or local GPU. Run the installer/update script, open **Settings → Engine Settings → Edge TTS**, load the current voices, choose a default, and save. The Generate tab supports per-speaker voices plus the existing speed and pitch controls.

This provider uses Microsoft's consumer Edge speech service through the unofficial `edge-tts` client. It is therefore marked experimental: availability, throttling, and protocol behavior can change without notice. It is best suited to personal use and testing rather than a guaranteed production service.

### ElevenLabs Engine

1. Create an API key in your ElevenLabs account.
2. Open **Settings → Engine Settings → ElevenLabs** and enter the key.
3. Click **Test Connection & Load Catalog** to retrieve the voices and TTS models available to the account and view character usage.
4. Select a default model and voice, save, then choose **ElevenLabs · Cloud** on the Generate tab.

TTS-Story supports per-speaker ElevenLabs voices along with configurable stability, similarity, style, speaker boost, generation speed, previews, full jobs, and Library regeneration. Account concurrency and character limits still apply. Review [ElevenLabs pricing](https://elevenlabs.io/pricing) before large jobs.

### Dot.TTS Engine

Dot.TTS (by rednote-hilab) is a 48 kHz zero-shot voice cloning engine:

- **Continuation Voice Cloning**: Uses reference audio plus the exact reference transcript for best similarity and stability
- **Transcript Fallback**: If a voice prompt has no stored transcript, the app attempts SenseVoice transcription and caches the result in `data/voice_prompts/transcripts.json`
- **Shared Voice Prompts Library**: Uses the same prompt library and library-regeneration controls as Chatterbox, VoxCPM, Qwen3 Clone, OmniVoice Clone, and Pocket TTS Clone
- **Isolated Environment**: Runs in `engines/dots-tts/.venv` with the upstream repo cloned to `engines/dots-tts/repo`
- **Optional X-Vector Fallback**: Can run with reference audio only if enabled, but reference audio plus matching text remains the recommended path
- **Windows Compatibility**: Setup skips WeTextProcessing because its `pynini` dependency does not provide a clean Windows wheel; the worker falls back to no-op text normalization when WeTextProcessing is unavailable

#### Dot.TTS Setup

1. **Run `setup.bat`** — it clones `rednote-hilab/dots.tts`, creates the isolated venv, and installs the compatible Dot.TTS runtime dependencies
2. **Optional model prefetch**: set `PREFETCH_DOTS_TTS_MODEL=1` before running setup to download `rednote-hilab/dots.tts-soar`; otherwise the model downloads on first use
3. **Select Dot.TTS** from the engine dropdown in Settings or the Generate tab
4. **Assign voice prompts** per speaker and make sure each reference has transcript text

#### Dot.TTS Settings

| Setting | Default | Description |
|---------|---------|-------------|
| Model | `rednote-hilab/dots.tts-soar` | Best voice cloning checkpoint |
| Chunk Size | `250` chars | Sentence-aware target size; lower values are safer for long passages |
| Precision | `auto` | Uses `bfloat16` on CUDA and `float32` on CPU to avoid upstream CPU bf16 dtype mismatches |
| Default Voice Prompt | _(empty)_ | Fallback reference audio path |
| Default Prompt Transcript | _(empty)_ | Fallback transcript for the reference audio |
| Sampling Steps | `10` | Flow-matching sampling steps |
| Guidance Scale | `1.2` | CFG scale; lower values can reduce intensity |
| Speaker Scale | `1.5` | Reference speaker conditioning strength; lower values can reduce exaggerated style |
| Normalize Text | `false` | Apply Dot.TTS text normalization when supported |
| Optimize | `false` | Enable Dot.TTS `torch.compile` optimization |

### VoxCPM 1.5 Engine

VoxCPM 1.5 is a powerful voice cloning engine with unique features:

- **Automatic Transcription**: If no transcript is provided, VoxCPM uses SenseVoice ASR to automatically transcribe the reference audio
- **Shared Voice Prompts**: Uses the shared voice prompt library used by the local voice-cloning engines
- **Lower VRAM Requirements**: Runs on GPUs with ~6GB VRAM
- **High Quality Output**: Produces natural-sounding speech with good prosody

### IndexTTS Engine

IndexTTS (by Bilibili) is a state-of-the-art zero-shot voice cloning engine with emotion control:

- **Zero-Shot Voice Cloning**: Clone any voice from a short reference audio (up to 15 seconds)
- **Reuses Voice Prompts Library**: Uses the shared voice prompt files used by the local voice-cloning engines — no new voice management needed
- **Emotion Control** (IndexTTS-2): Condition speech on a separate emotion reference audio, an 8-value emotion vector, or a text description
- **English + Chinese**: Supports both languages natively
- **GPU Recommended**: Runs on CUDA, MPS (Apple Silicon), or CPU (slow)
- **FP16 Support**: Halves VRAM usage with minimal quality loss
- **Isolated Environment**: Runs in its own `uv` venv under `engines/index-tts/` to avoid dependency conflicts
- **Multiple Model Versions**: IndexTTS-2 (recommended), IndexTTS-1.5, IndexTTS-1.0

#### IndexTTS Setup

1. **Run `setup.bat`** — it will automatically clone the repo and run `uv sync` if `git` and `uv` are available
2. **Download model weights** (one-time, several GB):
   ```bash
   cd engines/index-tts
   uv tool run huggingface-cli download IndexTeam/IndexTTS-2 --local-dir=checkpoints
   ```
3. **Select IndexTTS** from the engine dropdown in Settings or the Generate tab
4. **Assign voice prompts** per speaker — uses the existing Voice Prompts library

#### IndexTTS Settings

| Setting | Default | Description |
|---------|---------|-------------|
| Model Version | `IndexTTS-2` | Which model version to use |
| Chunk Size | `400` chars | Character limit per synthesis chunk |
| Device | `auto` | cuda, cuda:0, cpu, or auto |
| Default Voice Prompt | _(empty)_ | Fallback reference audio path |
| Use FP16 | `true` | Half-precision inference (faster, less VRAM) |
| Use DeepSpeed | `false` | Optional inference speedup |

### KittenTTS Engine

KittenTTS is an ultra-lightweight, CPU-only TTS engine — ideal for machines without a GPU:

- **No GPU Required**: Runs entirely on CPU, no CUDA needed
- **Small CPU Models**: The recommended mini model is about 80 MB; the experimental nano-int8 option is about 25 MB
- **8 Built-in Voices**: Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo
- **Multiple Model Sizes**: Choose from `kitten-tts-mini-0.8` (recommended), `kitten-tts-micro-0.8`, or `kitten-tts-nano-0.8` (fp32/int8) for speed vs. quality trade-offs
- **English Only**: Optimised for English narration
- **Installation**: Install separately via `pip install https://github.com/KittenML/KittenTTS/releases/download/0.8/kittentts-0.8.0-py3-none-any.whl`

#### KittenTTS Settings

| Setting | Default | Description |
|---------|---------|-------------|
| Model ID | `KittenML/kitten-tts-mini-0.8` | Which KittenTTS model to load |
| Default Voice | `Jasper` | Fallback voice when no per-speaker assignment is set |
| Chunk Size | `300` chars | Character limit per synthesis chunk (lower = faster on CPU) |

### Basic Workflow

1. Open your browser to `http://localhost:5000`
2. In **Settings**, select your preferred TTS engine
3. If using a cloud engine, configure its Replicate token or Azure Speech resource key and region
4. Paste your text with or without speaker tags in the **Generate** tab
5. Select a **Default Voice** (used for plain text / unassigned speakers)
6. If you use speaker tags, TTS-Story automatically analyzes the text and lets you assign voices per speaker
7. Click **Generate Audio**
8. The job is added to the **Job Queue**, processed in the background, and the result appears in:
   - **Job Queue** tab (with real-time progress, ETA, and player)
   - **Library** tab (all past generations with engine indicator)

### Using Voice Cloning

When using voice-cloning engines:

1. First, add voice recordings in **Available Voices → Voice Prompts**
2. In the **Generate** tab, select your cloned voice from the **Reference Prompt** dropdown
3. Use the gender and language filters to quickly find the right voice
4. Each speaker can use a different cloned voice for multi-character stories

### Quick Test Previews

- A shared "Quick Test Text" field lives above the Assigned Voices section so you can type once and preview any speaker with matching FX.
- Each speaker row includes an inline Quick Test button beside the tone controls.

**Note:** Local modes run entirely on your machine. Replicate and Azure Speech send synthesis text to the selected cloud provider and may incur provider charges.

### Silence & Timing Controls

- **Intro Silence (ms):** Adds empty space before the very first spoken line
- **Silence Between Segments (ms):** Inserts a gap after each chunk/line before the next one begins
- Both settings are configurable in the **Generation Settings** panel (0–2000 ms, 100 ms steps)

### Replicate API Setup

Both Kokoro · Replicate and Chatterbox · Replicate use the same API token:

1. Get your API key from [Replicate](https://replicate.com) (starts with `r8_...`)
2. In **Settings**, enter your token in the **Replicate API** section
3. Click **Save Settings**
4. Select either Replicate engine from the dropdown

### Speaker Tag Format

You can use either numbered speakers or named speakers:

**Numbered Format:**
```
[speaker1]Hello, my name is Alice.[/speaker1]
[speaker2]Nice to meet you, Alice! I'm Bob.[/speaker2]
[speaker1]It's great to meet you too![/speaker1]
```

**Named Format:**
```
[narrator]Once upon a time, in a land far away...[/narrator]
[alice]Hello, my name is Alice.[/alice]
[bob]Nice to meet you, Alice! I'm Bob.[/bob]
[narrator]And so their adventure began.[/narrator]
```

You can use any alphanumeric name (letters, numbers, underscores). The system will automatically detect all unique speakers and let you assign voices to each one.

### LLM Pre-Processing Workflow

Need to tidy a manuscript or add consistent speaker tags before running TTS? Use the **Prep Text** button:

1. In **Settings**, select Gemini, Atlas Cloud, OpenRouter, LM Studio, or Ollama as the LLM provider.
   - For Atlas Cloud, enter your API key and click **Fetch Atlas Models**, then select a model from the returned list.
   - For OpenRouter, enter your API key and click **Fetch OpenRouter Models** to load the text models allowed by that key.
   - For Gemini, enter your API key and use **Fetch Models**.
   - For LM Studio or Ollama, enter the local server URL and use **Fetch Local Models**.
2. Paste your story in the **Generate** tab and decide whether "Generate separate audio files for each chapter" should be enabled.
3. Select a **Prompt Preset** (see below) or write your own custom prompt.
4. Click **Prep Text**:
   - If chapter splitting is enabled, TTS-Story reuses the detected chapter list and sends each one to the selected LLM with your pre-prompt and the running speaker list.
   - If chapter splitting is disabled, the whole manuscript (plus pre-prompt) is sent in one request to respect the context window.
   - A real-time progress bar shows which chapter or full-text step is running.
5. When the LLM finishes, the cleaned/expanded narrative replaces the input field. Chapter headings stay inside the narrator tags so audio splitting still works.
6. Re-run **Analyze Text** if needed. Your voice assignments and FX settings remain untouched unless you explicitly reset them.

Because the speaker list is tracked across sections, characters that appear later continue to use the same tag, which keeps the voice assignment UI tidy and prevents duplicate dropdowns.

#### Pre-loaded Prompt Presets

TTS-Story includes pre-configured LLM prompt presets optimized for different use cases:

| Preset | Best For | Description |
|--------|----------|-------------|
| **Chatterbox Natural Dialogue Conversation** | Chatterbox engines | Transforms text into natural-sounding dialogue with paralinguistic tags (laughter, sighs, pauses) and human speech quirks. Ideal for conversational content where you want expressive, lifelike output. |
| **Chatterbox Audio Book Conversion** | Chatterbox engines | Maintains strict adherence to the original text while converting symbols and abbreviations that TTS engines struggle with into speakable words (e.g., "/" → "slash", "-" → "dash", "Dr." → "Doctor"). |
| **Strict Book Narration V1** | Kokoro & other engines | Preserves the exact text of the book while adding speaker tags and preparing the content for TTS conversion. Improved instruction adherence ensures the original prose is never paraphrased or summarised. |

Select a preset from the LLM section, or create your own custom prompts and save them for reuse.

### Plain Text Mode

If no speaker tags are found, the entire text will be processed with a single voice.

### Job Queue & Library

- **Job Queue** tab shows all jobs with:
  - Real-time progress bars and chunk counts
  - ETA estimates during processing
  - Status indicators (`queued`, `processing`, `completed`, `failed`, `cancelled`)
  - Per-job controls (cancel, download)
- **Library** tab lists all completed outputs (sorted newest first) with:
  - **Engine indicator** showing which TTS engine was used (Kokoro, Kokoro Replicate, Chatterbox, Chatterbox Replicate)
  - Inline audio players
  - Download links
  - Delete and "Clear All" controls

### Available Voices & Previews

- **Available Voices** tab lists all Kokoro-82M voices grouped by language.
- You can:
  - Generate preview samples for all voices
  - Regenerate (overwrite) samples if you change text or update voices
  - Click any voice to play its preview sample

### Custom Voice Blends

- Open the **Custom Voice Blends** panel inside the **Available Voices** tab to create bespoke voices.
- Click **New Custom Voice** (or Edit on any card) to open the modal where you can:
  - Name the blend and choose its language group (lang_code)
  - Add one or more component voices and set their mix weights (e.g., 0.5 narrator + 0.5 af_heart)
  - Optionally add notes for future reference
- Saved blends appear in the grid with metadata (code, language, updated time) and can be edited or deleted at any time.
- All custom voices automatically show up in:
  - Default voice dropdowns
  - Per-speaker assignment selects (grouped by language under “Custom Blends” optgroups)
  - `/api/voices` responses (`custom_voices` arrays per language) so automation scripts can use them.
- When the generator encounters a `custom_*` voice, the backend blends the component embeddings on the fly and caches the tensor for fast reuse.

> Tip: The API exposes the full CRUD workflow under `/api/custom-voices`, so you can script voice creation or keep predefined blends in source control.

## Configuration

Settings are stored in `config.json`:

```json
{
  "replicate_api_key": "",
  "chunk_size": 500,
  "sample_rate": 24000,
  "speed": 1.0,
  "output_format": "mp3",
  "output_bitrate_kbps": 128,
  "crossfade_duration": 0.1,
  "intro_silence_ms": 0,
  "inter_chunk_silence_ms": 0,
  "tts_engine": "kokoro",
  "chatterbox_turbo_local_device": "auto",
  "chatterbox_turbo_local_temperature": 0.8,
  "chatterbox_turbo_replicate_model": "resemble-ai/chatterbox-turbo",
  "voxcpm_local_device": "auto",
  "llm_provider": "gemini",
  "gemini_api_key": "",
  "gemini_model": "gemini-2.0-flash",
  "atlas_cloud_api_key": "",
  "atlas_cloud_base_url": "https://api.atlascloud.ai/v1",
  "atlas_cloud_model": "deepseek-v3",
  "atlas_cloud_timeout": 120,
  "openrouter_api_key": "",
  "openrouter_base_url": "https://openrouter.ai/api/v1",
  "openrouter_model": "openrouter/auto",
  "openrouter_timeout": 120,
  "azure_speech_key": "",
  "azure_speech_region": "",
  "azure_speech_default_voice": "en-US-AvaMultilingualNeural",
  "azure_speech_output_format": "riff-24khz-16bit-mono-pcm",
  "azure_speech_requests_per_minute": 20,
  "azure_speech_chunk_size": 1000,
  "edge_tts_default_voice": "en-US-AriaNeural",
  "edge_tts_chunk_size": 1000,
  "edge_tts_max_parallel": 2,
  "elevenlabs_api_key": "",
  "elevenlabs_base_url": "https://api.elevenlabs.io",
  "elevenlabs_model": "eleven_multilingual_v2",
  "elevenlabs_default_voice": "JBFqnCBsd6RMkjVDRZzb",
  "elevenlabs_chunk_size": 4000,
  "elevenlabs_max_parallel": 2
}
```

Cloud API keys are stored locally in `config.json`. The repository sync script scrubs these fields before staging or committing that file.

### TTS Engine Options

| Value | Description |
|-------|-------------|
| `kokoro` | Kokoro-82M local GPU inference |
| `kokoro_replicate` | Kokoro via Replicate cloud API |
| `chatterbox_turbo_local` | Chatterbox local GPU with voice cloning |
| `chatterbox_turbo_replicate` | Chatterbox via Replicate cloud API |
| `voxcpm_local` | VoxCPM 1.5 local GPU with voice cloning |
| `qwen3_custom_voice` | Qwen3 TTS custom voice mode |
| `qwen3_clone` | Qwen3 TTS voice cloning mode |
| `qwen3_voice_creation` | Qwen3 TTS voice creation mode |
| `omnivoice_clone` | OmniVoice voice cloning mode |
| `pocket_tts` | Pocket TTS voice clone mode (CPU-only) |
| `pocket_tts_preset` | Pocket TTS preset voices (CPU-only) |
| `kitten_tts` | KittenTTS CPU-only engine |
| `index_tts` | IndexTTS zero-shot voice cloning |
| `dots_tts` | Dot.TTS 48 kHz zero-shot voice cloning |
| `azure_speech` | Microsoft Azure AI Speech regional REST API |
| `edge_tts` | Experimental Microsoft Edge online speech service |
| `elevenlabs` | ElevenLabs REST API with account voice/model discovery |

Any settings you override in the Generate tab (format, bitrate, engine) are sent along with the job payload while keeping the saved defaults intact.

## Project Structure

```
TTS-Story/
├── app.py                 # Flask web server
├── docs/help/             # Canonical Markdown source for the bundled user guide
├── requirements.txt       # Python dependencies
├── setup.bat             # Windows setup script
├── run.bat               # Windows run script
├── config.json           # Configuration file
├── src/
│   ├── tts_engine.py     # TTS engine registry and factory
│   ├── replicate_api.py  # Replicate API integration (Kokoro)
│   ├── text_processor.py # Text chunking and parsing
│   ├── help_center.py    # Validated Markdown catalog and in-app Help API
│   ├── audio_merger.py   # Audio file merging
│   ├── voice_manager.py  # Voice configuration and preview sample metadata
│   ├── voice_sample_generator.py # Batch generation of voice preview samples
│   └── engines/
│       ├── kokoro_engine.py              # Kokoro-82M local engine
│       ├── chatterbox_turbo_local_engine.py    # Chatterbox local GPU engine
│       ├── chatterbox_turbo_replicate_engine.py # Chatterbox Replicate engine
│       ├── voxcpm_local_engine.py        # VoxCPM 1.5 local GPU engine
│       ├── pocket_tts_engine.py          # Pocket TTS CPU-only engine
│       ├── qwen3_engine.py               # Qwen3 TTS engine (custom/clone/design)
│       ├── omnivoice_clone_engine.py     # OmniVoice subprocess adapter
│       ├── kitten_tts_engine.py          # KittenTTS CPU-only engine
│       ├── index_tts_engine.py           # IndexTTS subprocess adapter
│       ├── dots_tts_engine.py            # Dot.TTS subprocess adapter
│       ├── azure_speech_engine.py         # Microsoft Azure Speech REST adapter
│       ├── edge_tts_engine.py             # Experimental Microsoft Edge adapter
│       ├── elevenlabs_engine.py           # ElevenLabs REST adapter
│       └── cloud_audio.py                 # Shared cloud-audio WAV conversion
├── engines/
│   ├── index-tts/                    # Cloned IndexTTS repo (isolated venv)
│   │   ├── .venv/                        # uv-managed isolated environment
│   │   ├── checkpoints/                  # Downloaded model weights
│   │   └── tts_worker.py                 # Worker script called by the adapter
│   ├── omnivoice/                    # OmniVoice worker and isolated venv
│   └── dots-tts/                     # Dot.TTS worker, cloned repo, model cache, and isolated venv
├── static/
│   ├── css/
│   │   ├── style.css
│   │   └── help.css       # Responsive Help Center reader styles
│   ├── js/
│   │   ├── help.js        # Search, deep links, and contextual Help navigation
│   │   ├── main.js
│   │   ├── queue.js
│   │   ├── library.js
│   │   ├── voice-manager.js
│   │   └── settings.js
│   ├── help/screenshots/   # Privacy-safe screenshots embedded in the user guide
│   ├── audio/            # Generated audio files (per-job subdirectories)
│   └── samples/          # Voice preview samples and manifest.json
├── data/
│   └── voice_prompts/    # Chatterbox voice recordings for cloning
└── templates/
    └── index.html        # Web interface
```

## API Endpoints

- `GET /` - Main web interface
- `GET /api/help/catalog` - Get the searchable bundled guide catalog
- `GET /api/help/articles/<article_id>` - Render one bundled guide article, including aliases used by contextual help
- `GET /api/health` - Health check (TTS engine availability, CUDA status, and loaded engines)
- `GET /api/voices` - Get available voices and preview sample status
- `POST /api/voices/samples` - Generate or regenerate voice preview samples
- `GET /api/settings` - Get current settings
- `POST /api/settings` - Update settings
- `POST /api/analyze` - Analyze text and return statistics/speakers
- `POST /api/gemini/sections` - Preview the sections (chapters/chunks) the configured LLM will process
- `POST /api/gemini/process-section` - Send one section to the configured LLM (called in sequence by the frontend for live progress updates)
- `POST /api/gemini/process` - Process the entire text through the configured LLM in one backend call (legacy route name retained for compatibility)
- `POST /api/gemini/models` - Fetch available Gemini models after providing an API key
- `POST /api/atlas-cloud/models` - Fetch Atlas Cloud LLM models using an entered or saved API key
- `POST /api/openrouter/models` - Fetch text-output models allowed by an entered or saved OpenRouter API key
- `GET /api/azure-speech/voices` - Fetch the saved Azure resource's regional voice catalog
- `POST /api/azure-speech/voices` - Validate an entered Azure key/region and fetch its regional voice catalog
- `GET|POST /api/edge-tts/voices` - Fetch the current experimental Edge voice catalog
- `GET|POST /api/elevenlabs/catalog` - Validate ElevenLabs access and fetch account voices, TTS models, and permitted usage data
- `POST /api/local-llm/models` - Fetch models from LM Studio or Ollama
- `POST /api/generate` - Queue a new audio generation job
- `GET /api/status/<job_id>` - Check status of a specific job
- `POST /api/cancel/<job_id>` - Cancel a queued or running job
- `GET /api/queue` - Get all jobs, their status, and current queue size
- `GET /api/download/<job_id>` - Download generated audio file
- `GET /api/library` - List all completed audio files
- `DELETE /api/library/<job_id>` - Delete a specific library item
- `POST /api/library/clear` - Delete all library items
- `GET /api/custom-voices` - List custom voice blends (includes normalized metadata and component weights)
- `POST /api/custom-voices` - Create a new custom voice blend
- `GET /api/custom-voices/<voice_id>` - Retrieve a specific custom voice (ID or `custom_` code)
- `PUT /api/custom-voices/<voice_id>` - Update an existing custom voice blend
- `DELETE /api/custom-voices/<voice_id>` - Delete a custom voice and invalidate cached tensors
- `GET /api/chatterbox-voices` - List saved voice prompts for shared voice-cloning engines
- `POST /api/chatterbox-voices` - Upload a new voice prompt
- `PUT /api/chatterbox-voices/<voice_id>/update` - Update voice metadata (name, gender, language)
- `DELETE /api/chatterbox-voices/<voice_id>` - Delete a voice prompt
- `GET /api/chatterbox-voices/<voice_id>/preview` - Preview a voice prompt
- `GET /api/voice-prompts` - List voice prompts with metadata (gender, language, duration)
- `GET /api/external-voices` - List available external voice samples from GitHub
- `POST /api/external-voices/<voice_id>/download` - Download an external voice sample locally

## Performance

### Kokoro · Local GPU (NVIDIA RTX 3090)
- ~2 seconds per chunk (500 words)
- No API costs
- Full privacy

### Kokoro · Replicate
- ~2-3 seconds per chunk (varies by input)
- Cost varies by usage
- No GPU required
- Model: [jaaari/kokoro-82m](https://replicate.com/jaaari/kokoro-82m)

### Chatterbox · Local GPU
- Uses **Chatterbox Turbo**, not a V2/V3 selector. Setup pins `chatterbox-tts==0.1.6` and loads the public `ResembleAI/chatterbox-turbo` model.
- Current adapter language support: English.
- Requires ~8GB VRAM
- Voice cloning from 10-15 second audio samples
- No API costs
- Full privacy

### Chatterbox · Replicate
- Voice cloning via cloud API
- Model: [resemble-ai/chatterbox-turbo](https://replicate.com/resemble-ai/chatterbox-turbo)
- Cost varies by usage
- No GPU required

### VoxCPM 1.5 · Local GPU
- Requires ~6GB VRAM
- Voice cloning from audio samples
- Automatic transcription via SenseVoice ASR
- No API costs
- Full privacy

### Pocket TTS · CPU
- Setup pins `pocket-tts==1.0.3`. The current TTS-Story adapter supports English.
- No GPU required — runs on any CPU
- Preset voices and voice cloning from reference **audio files**. A reference prompt means a WAV/MP3/M4A/FLAC/OGG voice recording, not a text prompt.
- No API costs
- Full privacy

### Checking installed engine versions

Package versions can differ on installations created before the current pins. To include exact environment information in a bug report, run:

```bash
python scripts/engine_versions.py
```

Use `python scripts/engine_versions.py --json` for machine-readable output. OmniVoice, IndexTTS, and Dot.TTS use isolated environments, so the report lists their configured model identifiers separately from packages installed in the main environment.

### KittenTTS · CPU
- No GPU required — the default mini model is about 80 MB, with smaller variants down to roughly 25 MB
- 8 built-in English voices
- Default chunk size: 300 chars (tunable in Settings → Engine Settings)
- No API costs
- Full privacy
- Install: `pip install https://github.com/KittenML/KittenTTS/releases/download/0.8/kittentts-0.8.0-py3-none-any.whl`

### IndexTTS · Local GPU (NVIDIA recommended)
- Zero-shot voice cloning from reference audio
- GPU strongly recommended; CPU mode works but is slow
- FP16 inference halves VRAM usage
- Default chunk size: 400 chars
- Runs in isolated `uv` venv — no dependency conflicts with main project
- No API costs
- Full privacy
- Model download: `uv tool run huggingface-cli download IndexTeam/IndexTTS-2 --local-dir=checkpoints`

### OmniVoice · Local GPU (NVIDIA recommended)
- Voice cloning from shared reference voice prompts
- Runs in an isolated venv to avoid dependency conflicts
- Reference prompt transcripts are reused when available
- No API costs
- Full privacy

### Dot.TTS · Local GPU (NVIDIA recommended)
- 48 kHz zero-shot voice cloning from reference audio plus matching reference text
- `dots.tts-soar` is the default high-similarity cloning model; `dots.tts-mf` is available for faster MeanFlow generation
- Sentence-aware chunking defaults to 250 characters and keeps section/chapter folder outputs aligned with detected headings
- Runs in an isolated venv with project-local model caching for Windows-friendly downloads
- No API costs
- Full privacy

### Microsoft Azure Speech · Cloud
- No local model download or GPU required
- Regional voice catalog with multilingual neural voices
- Voice-dependent speaking styles and roles plus SSML rate, pitch, and volume
- 24 kHz or 48 kHz PCM synthesis before final audiobook encoding
- Usage limits, privacy, and cost depend on your Azure subscription

### Microsoft Edge TTS · Experimental Cloud
- No API key, local model download, or GPU required
- Current voice list is loaded dynamically
- Consumer service availability and throttling are not guaranteed

### ElevenLabs · Cloud
- Loads the voices and TTS models available to your API key
- Displays current character usage in Settings when the API key permits subscription access
- Account plan controls cost, quota, and concurrency

## Troubleshooting

### espeak-ng not found
Make sure espeak-ng is installed and in your PATH.

### CUDA out of memory
Reduce chunk_size in settings or use a Replicate engine instead of local GPU.

### Audio quality issues
Adjust the speed parameter (0.5 - 2.0) in settings.

### Azure Speech rejects the key, region, voice, or request

- Confirm the resource key and region came from the same Azure Speech resource; enter the region name such as `eastus`, not the full endpoint URL.
- Click **Test Connection & Load Voices** after changing credentials. Only voices returned for that resource and region are shown.
- A `429` response means Azure throttled the resource. Reduce **Parallel Chunks**, lower the configured request limit to match the subscription quota, or retry later.
- If Azure rejects a style or role, return the speaker to neutral/default or reload the voice catalog; expression support varies by voice.

### Edge TTS is unavailable or stops responding

- Run the current installer/update script so `edge-tts` is installed, then restart TTS-Story.
- Reload the voice catalog and reduce parallel requests if the service temporarily blocks or throttles requests.
- Because the Edge consumer protocol is unofficial, an upstream Microsoft change may require a TTS-Story or `edge-tts` update.

### ElevenLabs rejects the key, voice, model, or request

- Test the connection in Settings and reload the account catalog after changing the API key.
- A `402` normally indicates exhausted quota or billing allowance; a `429` indicates concurrency or rate limiting.
- Reduce parallel requests, select a model/voice returned by the catalog, and verify the account's current character allowance.

### Chatterbox Turbo is unavailable

Run `setup.bat` on Windows or `./setup.sh` on Linux/macOS. Setup now validates the Chatterbox import, and `/api/health` reports the underlying dependency error in `chatterbox_turbo_unavailable_reason`. Include that value and the output from `python scripts/engine_versions.py` in the issue report.

### Virtual environment Python cannot start

Rerun the platform setup script. Setup detects virtual environments that reference a removed or relocated Python installation and recreates them automatically.

### SoX is unavailable or permission is denied

Windows uses the bundled `tools/sox/sox.exe`. Linux and macOS require a native SoX installation on `PATH`; the Windows executable is never used on those platforms.

```bash
# Ubuntu / Debian / Linux Mint
sudo apt-get install sox libsox-dev

# macOS
brew install sox
```

## License

Apache 2.0 - Same as Kokoro-82M

## Credits

- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad
- [Chatterbox](https://github.com/resemble-ai/chatterbox) by Resemble AI
- [VoxCPM](https://github.com/openvpi/VoxCPM) by OpenVPI
- [KittenTTS](https://github.com/KittenML/KittenTTS) by KittenML
- [IndexTTS](https://github.com/index-tts/index-tts) by Bilibili IndexTTS Team
- [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) by k2-fsa
- [Dot.TTS](https://github.com/rednote-hilab/dots.tts) by RedNote HiLab
- [TTS Samples](https://github.com/yaph/tts-samples) by yaph - External voice sample library
- [StyleTTS2](https://github.com/yl4579/StyleTTS2) by yl4579
- [Replicate](https://replicate.com) for cloud API

## Support

For issues and questions, please open an issue on GitHub.

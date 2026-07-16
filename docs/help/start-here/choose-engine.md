# Choose the Right Engine

The best engine is the one that meets the voice, language, hardware, privacy, and budget requirements of the current story. There is no universally highest-quality option for every speaker.

For exact controls and limitations, use the [Engine Reference and Comparison](help:engine-overview). This article helps narrow the first choice.

## Start with the deciding constraint

### No compatible GPU

- **Pocket TTS** offers local CPU preset voices and local CPU voice cloning.
- **KittenTTS** is a very small local CPU engine with built-in voices.
- **Azure Speech**, **Edge TTS**, **ElevenLabs**, and Replicate-backed engines move synthesis online.

CPU-only does not mean instant. Test a representative paragraph and measure it before starting a long book.

### Strict local privacy

Choose a local engine and a local LLM—or skip **Prep Text** entirely. Local synthesis keeps story text away from speech providers, but downloading models and dependencies may still require an internet connection during setup or first use.

Voice cloning with a local engine also keeps the reference recording local. Review [Reference Voice Prompts](help:voice-prompts) for recording quality and storage guidance.

### Fast setup and a large online voice catalog

- **Edge TTS** needs no API key and offers many online voices, but uses an unofficial consumer-service client and is marked experimental.
- **Azure Speech** uses your regional Azure Speech resource and offers voice-dependent styles, roles, and SSML prosody.
- **ElevenLabs** exposes voices and models allowed by your account and can produce highly natural results, subject to quota and billing.

Read [Configure Online Services Safely](help:online-services) before sending a manuscript to any provider.

### Built-in voices without reference recordings

- **Kokoro** has a broad built-in catalog and supports custom blends.
- **KittenTTS** and **Pocket TTS · Preset Voices** provide lightweight CPU choices.
- **Qwen3-TTS · Custom Voice** provides instructed built-in speakers with language and instruction controls.
- Cloud catalogs also provide ready-made voices.

### Voice cloning from a recording

Chatterbox, VoxCPM, Pocket TTS Clone, Qwen3 Clone, OmniVoice Clone, IndexTTS, and Dot.TTS use reference audio in different ways. Their quality depends heavily on a clean, representative recording. Some also benefit from or require an accurate reference transcript.

Do not assume a prompt that works well in one cloning engine will sound identical in another. Use **Quick Test** after changing engines.

### Expressive dialogue or special controls

- Chatterbox supports its own expressive behavior and non-verbal tags.
- Qwen3 can use custom voice instructions and detected emotion information.
- OmniVoice exposes its own supported paralinguistic tags.
- Azure styles and roles depend on the selected voice.
- ElevenLabs offers stability, similarity, style, and speaker-boost settings.

Only use expression tags shown for the selected engine. Unsupported bracketed text may be spoken literally or ignored.

## Local versus cloud tradeoffs

| Question | Local engine | Cloud engine |
|---|---|---|
| Hardware | Uses your CPU/GPU and memory | Provider supplies inference hardware |
| Startup | May load/download a model | Usually little local startup work |
| Cost | No per-request provider fee | May consume credits or incur charges |
| Privacy | Text and reference audio stay local during synthesis | Content is sent to the provider |
| Reliability | Depends on local installation and resources | Depends on internet, provider, quota, and account |
| Throughput | Limited by the computer | Limited by provider rate/concurrency rules |

## A practical evaluation method

1. Create a 30–60 second test containing narration, dialogue, a difficult name, punctuation, and the target language.
2. Select the engine under **Generation Options**.
3. Assign a compatible voice to every speaker.
4. Leave advanced controls at defaults.
5. Use **Quick Test**, then generate the entire short sample.
6. Record perceived quality, first-run time, normal generation time, and any cost.
7. Change one control or one engine and repeat the same passage.

The preview validates a voice and short request. A generated sample also tests chunking, transitions, final encoding, and chapter behavior.

## Default engine and job engine

**Settings → Quick Settings → Default Engine** chooses the initial engine for new work. **Generate → Generation Options → Engine** selects the engine for the current job. Changing the job engine can repopulate or clear voice selectors, so choose it before doing detailed assignments.

![Engine Settings navigation showing the available TTS engine tabs](../../../static/help/screenshots/engine-settings-navigation.png)

*Open Engine Settings and select the tab matching the engine you want to configure.*

When you have selected an engine, continue with [Assign and Test Voices](help:assign-voices) and [Generation and Output Options](help:generation-options).

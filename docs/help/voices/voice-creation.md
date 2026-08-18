# Create Voices with Qwen3 or OmniVoice

Voice Creation synthesizes a short voice sample from a written description. An accepted preview can be saved to Voice Prompts and reused as reference audio by compatible cloning engines.

Open [Available Voices](app:voices), expand **Voice Creation**, and choose **Qwen3-TTS** or **OmniVoice**.

![Voice Creation panel with engine selection, sample text, instructions, and preview controls](../../../static/help/screenshots/voice-creation.png)

*Describe the intended voice, generate a short preview, and save only an approved result to Voice Prompts.*

## Qwen3-TTS voice design

Provide:

- a Voice Name;
- optional Gender metadata;
- Language;
- a short descriptive label;
- Sample Text; and
- a Voice Style Instruction.

Use a concrete instruction such as “calm middle-aged narrator, low pitch, measured pace, warm but restrained” rather than subjective terms alone. The Sample Text should resemble the target manuscript and be long enough to reveal rhythm and pronunciation.

Click **Generate Preview**, listen to the entire result, and click **Save to Voice Prompts** only when satisfied. Saving is disabled until a preview exists.

If Qwen3-TTS is not installed, **Generate Preview** and the speaker **Generate Voice/Generate Voices** actions are disabled. Use the displayed **Open Qwen3 Settings** action, select **Install Engine**, and wait for the isolated runtime installation to finish.

## OmniVoice design

Provide a name, optional gender metadata, a short description, and sample text. Build the Voice Instruction with the selectable tags for gender, age, pitch, whisper, and accent. Select at least one instruction tag before generating.

Click **Generate Preview**, review it, then **Save to Voice Prompts**.

OmniVoice runs in an isolated local environment and may download a model of several gigabytes on first setup/use. Qwen3 also requires its local model and dependencies. First preview generation can therefore take much longer than later previews.

## What saving does

Saving stores the preview audio in the reference prompt library with its name and available metadata. It does not create a new built-in Qwen or OmniVoice model, and it does not automatically change existing assignments unless the Generate workflow explicitly assigns the saved prompt.

Use [Reference Voice Prompts](help:voice-prompts) to preview, archive, export, or delete the result.

## Create voices from speaker profiles

Prep Text can generate speaker profiles. Back on [Generate](app:generate), **Generate Voices** uses Qwen3 VoiceDesign to create samples sequentially for all detected speakers, optionally adding a name prefix. See [Generate and Auto-Assign Voices](help:auto-assign-voices).

## Quality and consent

Generate several short candidates rather than committing a full book to the first preview. Confirm language, names, emotional range, and long-sentence stability with Quick Tests.

Use only voices and reference material you have the right and consent to use. A generated design can still resemble recognizable speech characteristics, and cloud services may impose additional usage policies.

For engine-specific settings, see [Qwen3-TTS: Custom Voice, Clone, and Design](help:engine-qwen3) or [OmniVoice: Clone and Voice Design](help:engine-omnivoice).

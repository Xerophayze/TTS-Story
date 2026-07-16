# LLM Preparation Overview

TTS-Story can send a manuscript through a large language model before speech generation. This step is optional. It is useful for cleaning punctuation, applying speaker tags, normalizing headings, or following a repeatable editorial prompt, but it can also rewrite names, facts, dialogue, or formatting.

> **Keep a source copy.** Prep Text replaces the text in the Generate editor when processing completes. Save the manuscript outside TTS-Story or save a Project before you begin.

## Choose whether Prep Text is appropriate

Prep Text is a good fit when you need to:

- clean predictable OCR or punctuation problems;
- add speaker tags to consistently formatted dialogue;
- standardize chapter headings;
- apply the same focused instruction to every section; or
- create speaker profiles for the Generate Voices workflow.

Avoid broad prompts such as “improve this story” when fidelity matters. For a finished manuscript, a narrow cleanup prompt and a careful comparison with the source are safer.

Cloud providers receive the text sent for processing. If the manuscript must remain on the computer, use LM Studio or Ollama and confirm that the local server is running. See [LM Studio and Ollama](help:llm-local). For provider privacy, billing, and credential considerations, see [Configure Online Services Safely](help:online-services).

## Prepare the provider

1. Open [Settings](app:settings).
2. Expand **LLM Pre-Processing**.
3. Select Gemini, Atlas Cloud, OpenRouter, or Local.
4. Enter the provider settings and use its **Fetch Models** button.
5. Select a returned model.
6. Click **Save Settings**.

Fetching models verifies the URL and credentials well enough to retrieve a catalog; it does not guarantee that every listed model accepts the same parameters or has sufficient quota for a long manuscript.

## Run Prep Text

![Prep Text progress and resume controls on the Generate page](../../../static/help/screenshots/prep-text-resume.png)

*Prep Text shows section progress and provides pause, resume, restart, and abort controls for longer runs.*

1. Open [Generate](app:generate) and load the source text.
2. Save a Project or preserve the original file.
3. Select a prompt preset beside **Prep Text**, or rely on the configured Prompt Prefix.
4. Click **Prep Text**.
5. Watch the section progress. You may pause, resume, restart, or abort the operation.
6. When it finishes, review the entire replacement text before generating audio.
7. Wait for automatic text analysis, then correct any speaker-tag or heading warnings.

TTS-Story divides the text into sections and processes them one at a time. A section failure marked retryable by the provider, including an HTTP 503 response, is retried up to five times with increasing delays. Authentication, validation, and other non-retryable failures stop immediately. Progress is retained locally so a paused or interrupted preparation can be resumed when its source text still matches.

After successful preparation, TTS-Story also requests speaker profiles for detected speakers. Those profiles can guide Qwen3 or OmniVoice voice design; they are not a substitute for listening to previews. Learn more in [Generate and Auto-Assign Voices](help:auto-assign-voices).

## Review before synthesis

Compare at least these items with the source:

- proper names, invented words, dates, and numbers;
- every opening and closing speaker tag;
- dialogue ownership and narrator passages;
- chapter and book headings;
- text near section boundaries; and
- any content the model may have summarized or expanded.

Speaker tags have strict behavior: if valid speaker-tagged blocks are present, untagged passages are not synthesized. Review [Speaker and Expression Tags](help:speaker-tags) before submitting a long job.

Next: [Prompt Presets, Chunking, and Review](help:llm-prompts).

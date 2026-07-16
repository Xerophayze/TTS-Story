# Welcome to TTS-Story

TTS-Story turns written material into multi-voice audio. You prepare the text on **Generate**, assign a compatible voice to every detected speaker, submit a background job, monitor it in **Job Queue**, and review the result in **Library**.

The program can work entirely on your computer, use online speech services, or combine local and online tools. Nothing requires you to configure every engine or every provider. A good first setup is the smallest one that fits your hardware and the kind of voices you need.

## The main pages

| Page | What you do there |
|---|---|
| **Generate** | Enter or import text, optionally prepare it with an LLM, review speakers and sections, assign voices, and submit an audio job. |
| **Job Queue** | Monitor queued and active work, inspect progress, and respond to a paused, failed, or cancelled job. |
| **Library** | Play completed stories and chapters, review or regenerate chunks, repair saved work, and download final audio. |
| **Available Voices** | Browse built-in voices, make Kokoro blends, create voices, and manage reference recordings for cloning engines. |
| **Settings** | Choose saved defaults, configure individual engines, enter cloud credentials, and tune audio or LLM behavior. |
| **Help** | Read this guide or open a contextual article from a **?** button elsewhere in the interface. |

![TTS-Story Generate page with application status, navigation tabs, input text, and preparation controls](../../../static/help/screenshots/generate-overview.png)

*The Generate page is the starting point; the top tabs lead to the queue, Library, voices, Settings, and this guide.*

Open any page directly: [Generate](app:generate), [Job Queue](app:queue), [Library](app:library), [Available Voices](app:voices), or [Settings](app:settings).

## A dependable production workflow

1. Keep an untouched copy of the manuscript outside TTS-Story.
2. Choose and configure one TTS engine.
3. Enter or import a short representative passage.
4. Wait for **Text Statistics** to appear; analysis happens automatically after you edit the input.
5. Correct speaker tags and review the detected speakers and sections.
6. Assign and **Quick Test** every important voice.
7. Generate a short job before committing to a full book.
8. Monitor the job in **Job Queue**.
9. Review the completed audio in **Library**, correcting only the chunks that need attention.
10. Generate or rebuild the final chapter and audiobook files.

The defaults are intended to be a safe starting point. Change one performance or voice control at a time and test the same passage again; this makes improvements and regressions much easier to identify.

## Local and online processing

Local engines run synthesis on your CPU or GPU and do not charge per request. They may download models on first use and can take longer to start. Online engines avoid local model requirements, but they send synthesis text to a provider and can be subject to credentials, quotas, billing, network delays, or service changes.

LLM text preparation is separate from speech generation. For example, you can prepare text with OpenRouter and synthesize it locally, or skip LLM preparation entirely and send your original text directly to a cloud speech engine.

See [Choose the Right Engine](help:choose-engine) and [Configure Online Services Safely](help:online-services) before deciding.

## Three ideas that prevent most mistakes

> **Protect the source.** **Prep Text** replaces the text currently in the input field. Document imports append to existing input. Projects are browser-local snapshots, not portable backup files.

> **Test small.** A few paragraphs containing narration, dialogue, difficult names, and chapter headings reveal more than a generic one-line sample.

> **Review before export.** A completed job is not necessarily the final audiobook. Use Library chunk review to fix pronunciation, voice, or pacing problems without regenerating everything.

## Where to begin

- New installation: [First-Run Checklist](help:first-run)
- First end-to-end result: [Generate Your First Audio](help:quick-start)
- Understand how data moves through the app: [How the Pages Work Together](help:workflow-map)
- Add a manuscript: [Enter or Import Your Text](help:input-text)

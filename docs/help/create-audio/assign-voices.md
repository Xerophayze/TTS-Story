# Assign and Test Voices

Every analyzed job needs a voice assignment. Tagged manuscripts receive one card per detected speaker; untagged text receives a single card named **Speaker 1**.

## Select the engine before assigning

Under **Generation Options**, choose the job **Engine** first. Voice selectors and advanced controls depend on that engine. Changing engines afterward can replace the available list or leave an earlier selection incompatible.

If you have not chosen an engine, read [Choose the Right Engine](help:choose-engine).

## Understand the assignment control

Depending on the engine, a speaker card can show:

- **Select Voice...** for a built-in or cloud-catalog voice
- **Voice Sample** for a saved reference recording used by a cloning engine
- **Language** and **Custom Instruction** for Qwen3
- Azure **Speaking Style**, **Role**, **Style Intensity**, and **Volume Change (%)** when supported by the selected voice
- **Pitch**, **Speed**, and **Quick Test** controls

Cloud voices are populated from the catalog loaded in Settings. Cloning voices come from **Available Voices → Voice Prompts**. If the appropriate list is empty, configure the provider or add a reference before continuing.

![Voice assignment cards for analyzed speakers with voice and test controls](../../../static/help/screenshots/voice-assignments.png)

*Each detected speaker receives an assignment card; verify its compatible voice and test controls before submission.*

## Assign a voice to every speaker

1. Review the detected speaker chips under **Text Statistics**.
2. In **Assign Voices**, choose a voice or sample for the first speaker.
3. Repeat for every remaining speaker.
4. Confirm that two similarly named speakers were not detected by mistake.
5. For an untagged manuscript, assign **Speaker 1**.

TTS-Story prevents submission when no valid assignments have been collected. If at least one assignment exists but a detected speaker has no selection of its own, that speaker inherits the first/default assignment. This fallback avoids a failed job, but it can give two characters the same voice, so review and assign every speaker explicitly.

## Use Quick Test effectively

Select **Quick Test** on a speaker card to preview the assignment and its current FX. Built-in and provider-catalog engines synthesize the preview text. For reference/prompt engines, Quick Test replays the saved reference recording with the selected pitch and speed; it does not call the chosen TTS engine or prove that it can synthesize the manuscript. For useful results:

- For a synthesized Quick Test, use a sentence representative of the actual character when a preview-text field is available.
- Include difficult names, punctuation, the target language, and emotional delivery.
- Test the same sentence after changing a voice or engine.
- Wait for the first local model load to finish before judging normal preview speed on engines that synthesize the test.
- Remember that a short preview does not test chapter merging or long-form consistency.

After a good preview, generate a short full job. That tests text chunking, transitions, final encoding, and service limits as well as the voice.

## Tune Pitch and Speed conservatively

- **Pitch** is measured in semitone-style steps around `0.0 st`.
- **Speed** defaults to `1.00x`.

Start at the defaults. Large pitch changes can create artifacts or an unnatural identity, and large speed changes can reduce intelligibility. Make small changes, Quick Test, and listen on headphones before applying the same settings across a book.

Global **Speech Speed** in Settings and per-speaker speed can both affect pacing. Change one level at a time so you know which control produced the result.

## Voice-cloning assignments

A reference recording influences pronunciation, tone, accent, and noise. Choose a clip that is:

- Clean and free of music or room echo
- Long enough for the selected engine
- Mostly one speaker
- Representative of the intended delivery and language
- Accompanied by an accurate transcript where the engine uses one

Use the **Play** button beside **Voice Sample** to verify the source itself. If the sample sounds noisy or clipped, replace it rather than trying to repair every generated chunk. See [Reference Voice Prompts](help:voice-prompts).

## Rename and organize speakers

Each card includes a speaker-name field and **Apply**. Applying a new name normalizes it for tags, rewrites exact matching opening and closing tags in Input Text, and triggers analysis again.

Review the manuscript after renaming. Spaces become hyphens, uppercase becomes lowercase, and unsupported punctuation is removed.

Selecting a speaker chip opens **Edit Speaker**, where profile information and the **Mark as ready** checkbox can help track preparation. **Mark as ready** is an organizational indicator; it does not assign a voice, validate a sample, or bypass generation checks.

## Generate Voices and Auto Assign

For tagged manuscripts, **Generate Voices** can design samples from speaker-profile information, while **Auto Assign** matches detected speaker names to available samples using a threshold.

These are accelerators, not editorial decisions:

1. Run **Prep Text** only if you want LLM-generated profiles.
2. Inspect every generated or matched voice.
3. Quick Test each assignment.
4. Manually correct weak or ambiguous matches.

See [Generate and Auto-Assign Voices](help:auto-assign-voices).

## Common assignment problems

- **The selector is empty:** load the cloud catalog or add compatible Voice Prompts.
- **A selection vanished after changing engines:** select a voice compatible with the new engine.
- **The wrong card appears:** repair the speaker tags and wait for automatic analysis.
- **A clone sounds unlike the source:** improve the reference recording and retest the engine defaults.
- **Quick Test succeeds but a full job drifts:** review individual chunks and reduce extreme controls.

When all assignments are verified, continue with [Generation and Output Options](help:generation-options).

# Assign and Test Voices

Every analyzed job needs a voice assignment. Tagged manuscripts receive one row per detected speaker in the **Cast** list; untagged text receives a single row named **Speaker 1**.

## Select the engine before assigning

In the **Output** section, choose the job **Engine** first. Voice selectors and advanced controls depend on that engine. Changing engines afterward can replace the available list or leave an earlier selection incompatible.

If you have not chosen an engine, read [Choose the Right Engine](help:choose-engine).

The Cast summary shows how many speakers are ready and how many still need a voice. **Review next** opens a speaker missing a voice first, then the next speaker not marked ready. The **Choose the engine** link takes you directly to the Output selector.

## Understand the assignment control

Click a speaker in the Cast list (or its **Review** link) to open the **Edit speaker** drawer. Depending on the engine, its **Voice** section can show:

- **Select a voice…** for a built-in or cloud-catalog voice
- **Voice Sample** for a saved reference recording used by a cloning engine
- **Language** and **Custom Instruction** for Qwen3
- Azure **Speaking Style**, **Role**, **Style Intensity**, and **Volume Change (%)** when supported by the selected voice
- **Pitch**, **Speed**, and **Quick test** (an editable test line with a **Test** button)

Cloud voices are populated from the catalog loaded in Settings. Cloning voices come from **Voices → Voice prompts**. If the appropriate list is empty, configure the provider or add a reference before continuing.

![Edit speaker drawer with Name and Rename, Character, Voice, Pitch, Speed, Quick test, Design a new voice, and Mark as ready](../../../static/help/screenshots/voice-assignments.png)

*Each detected speaker has its own Edit speaker drawer; verify its compatible voice and test it before submission.*

## Assign a voice to every speaker

1. Review the speakers in the **Cast** list.
2. Click the first speaker and choose a voice or sample in the drawer; click **Done**.
3. Repeat for every remaining speaker.
4. Confirm that two similarly named speakers were not detected by mistake.
5. For an untagged manuscript, assign **Speaker 1**.

TTS-Story checks the entire cast before submission. If a speaker lacks a valid assignment or configured default, generation stops and opens that speaker's properties. A configured default can still be shared, but assign characters explicitly when they should sound different. **Mark as ready** protects a prepared speaker from default batch actions; it is not required merely to generate audio once the voice is assigned.

## Use Quick test effectively

In the drawer, edit the **Quick test** line if needed and click **Test** to preview the assignment and its current FX. Built-in and provider-catalog engines synthesize the preview text. For reference/prompt engines, Quick test replays the saved reference recording with the selected pitch and speed; it does not call the chosen TTS engine or prove that it can synthesize the manuscript. For useful results:

- For a synthesized Quick test, use a sentence representative of the actual character.
- Include difficult names, punctuation, the target language, and emotional delivery.
- Test the same sentence after changing a voice or engine.
- Wait for the first local model load to finish before judging normal preview speed on engines that synthesize the test.
- Remember that a short preview does not test chapter merging or long-form consistency.

After a good preview, generate a short full job. That tests text chunking, transitions, final encoding, and service limits as well as the voice.

## Tune Pitch and Speed conservatively

- **Pitch** is measured in semitone-style steps around `0.0 st`.
- **Speed** defaults to `1.00×`.

Start at the defaults. Large pitch changes can create artifacts or an unnatural identity, and large speed changes can reduce intelligibility. Make small changes, run **Quick test**, and listen on headphones before applying the same settings across a book.

Global **Speech speed** (Settings → Audio & generation) and per-speaker speed can both affect pacing. Change one level at a time so you know which control produced the result.

## Voice-cloning assignments

A reference recording influences pronunciation, tone, accent, and noise. Choose a clip that is:

- Clean and free of music or room echo
- Long enough for the selected engine
- Mostly one speaker
- Representative of the intended delivery and language
- Accompanied by an accurate transcript where the engine uses one

Use the **Play** button beside **Voice Sample** to verify the source itself. If the sample sounds noisy or clipped, replace it rather than trying to repair every generated chunk. See [Reference Voice Prompts](help:voice-prompts).

## Rename and organize speakers

The drawer's **Name** field and **Rename** button rename the speaker; this normalizes the name, rewrites exact matching opening and closing tags in the manuscript, and triggers analysis again.

Review the manuscript after renaming. Spaces become hyphens, uppercase becomes lowercase, and unsupported punctuation is removed.

Clicking a speaker opens its **Edit speaker** drawer, where the **Character** profile and the **Mark as ready** switch (drawer footer) help track preparation. Speakers not marked ready are listed above **Generate audio**; click that notice to open the first one.

For text that is already speaker-tagged, use **Build profile** to analyze only the selected speaker:

1. Confirm the correct tagged speaker is shown at the top of the drawer.
2. Click **Build profile** in the **Character** section header.
3. Review and, if necessary, edit the generated character profile and **Voice type**.
4. Click **Generate voice** (in **Design a new voice**) only when you want to design and save a reference voice from those fields.
5. Select or verify the resulting voice assignment and run **Quick test**.

Build profile uses only bounded excerpts tagged for the selected speaker. It does not run Prep text, rewrite the manuscript, clear other speakers' profiles, or generate audio. The **Build profiles** button below the Cast list remains available when every speaker needs a profile.

**Mark as ready** is an organizational indicator; it does not assign a voice, validate a sample, or bypass generation checks.

## Generate voices and Auto assign

For tagged manuscripts, **Generate voices** beside **Build profiles** designs samples from speaker-profile information, while **Auto assign** matches detected speaker names to available samples using a threshold. Batch actions skip speakers marked **Ready** by default. Enable **Include ready speakers in batch actions** only when you deliberately want to prepare approved speakers again.

These are accelerators, not editorial decisions:

1. Run **Prep text** when the manuscript needs preparation, use **Build profiles** for all detected speakers, or use **Build profile** for one already tagged speaker.
2. Inspect every generated or matched voice.
3. Run **Quick test** for each assignment.
4. Manually correct weak or ambiguous matches.

See [Generate and Auto-Assign Voices](help:auto-assign-voices).

## Common assignment problems

- **The selector is empty:** load the cloud catalog or add compatible voice prompts.
- **A selection vanished after changing engines:** select a voice compatible with the new engine.
- **The wrong speaker appears in the Cast list:** repair the speaker tags and wait for automatic analysis.
- **A clone sounds unlike the source:** improve the reference recording and retest the engine defaults.
- **Quick test succeeds but a full job drifts:** review individual chunks and reduce extreme controls.

When all assignments are verified, continue with [Generation and Output Options](help:generation-options).

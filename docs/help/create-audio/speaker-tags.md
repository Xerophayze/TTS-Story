# Speaker and Expression Tags

Speaker tags divide a manuscript into passages that can use different voices. Expression or paralinguistic tags ask certain engines for a non-verbal sound or delivery cue. They use similar brackets but serve different purposes.

## Speaker tag format

For experiments, Prep Text also offers **Strict Book Conversion V3 - Opening Tags (Experimental)**. It asks the language model for opening speaker tags only, then converts its explicitly marked response to normal paired tags before displaying it. It does not replace the standard V3 prompt. Review speaker assignments, directions, and every source passage; the model can still omit or alter text. If the response cannot be safely converted, preparation reports an error rather than silently guessing.

Wrap every spoken passage in a matching opening and closing tag:

```text
[narrator]The room fell silent.[/narrator]
[alice]I thought everyone had gone home.[/alice]
[bob]Not everyone.[/bob]
```

Speaker names may contain letters, numbers, underscores, and hyphens. Do not put spaces or punctuation in the brackets. These are valid examples:

```text
[speaker1]...[/speaker1]
[mary-jane]...[/mary-jane]
[detective_2]...[/detective_2]
```

Use the same spelling for both halves of a pair. Tags are matched without regard to letter case during validation, but consistent lowercase names are easier to maintain and match elsewhere.

## Plain text is valid

You do not have to add tags to a single-voice story. When no paired speaker tags are detected, TTS-Story analyzes the manuscript as one speaker named **Speaker 1**. You must still assign that speaker a voice before generation.

## Keep passages simple and non-overlapping

Prefer sequential blocks:

```text
[narrator]She opened the letter.[/narrator]
[alice]This cannot be true.[/alice]
```

Avoid nesting one speaker inside another or leaving narration outside an otherwise tagged manuscript. Simple, balanced blocks make analysis, chunking, voice assignment, and later review predictable.

## The unmatched-tag warning

TTS-Story checks recognized speaker-tag pairs shortly after you type or paste. When it finds an orphaned or mismatched recognized tag, the toolbar shows **N tag issues** and an amber strip appears above the manuscript (*N speaker/direction tag issues found*) with:

- up and down arrow buttons (*Previous issue* / *Next issue*) that switch the editor to **Raw tags** and highlight each affected tag
- **Auto-fix** to insert a likely missing opening or closing tag

![Tag-issue strip with previous/next arrows and Auto-fix above the Manuscript editor in Raw tags view](../../../static/help/screenshots/speaker-tag-warning.png)

*The warning identifies unbalanced recognized speaker tags; review each location before using Auto-fix.*

Generation is blocked while those unmatched speaker tags remain.

The validator distinguishes speaker tags from one-part expression cues such as `[laugh]` by treating a name as a speaker only when a closing form for that name appears somewhere in the manuscript. Consequently, a one-off opening tag such as `[alice]Hello there.` with no `[/alice]` anywhere is treated as an expression cue and may not trigger the warning. Before generating, search for every intended speaker's opening and closing forms manually, especially when that speaker appears only once.

> **Review every Auto-fix.** It uses nearby tags and text boundaries to choose an insertion point. It can make the structure balanced without necessarily expressing the conversation you intended.

Typical corrections are:

```text
# Earlier valid pair establishes alice as a speaker
[alice]Earlier line.[/alice]

# Later passage is missing its close
[alice]Hello there.

# Corrected later passage
[alice]Hello there.[/alice]
```

```text
# Mismatch
[alice]Hello.[/bob]

# Corrected
[alice]Hello.[/alice]
```

## Rename a detected speaker

After analysis, click the speaker in the **Cast** list to open its **Edit speaker** drawer, enter a new name in **Name**, and click **Rename**. TTS-Story normalizes it to a tag-safe name, rewrites exact matching opening and closing tags in the manuscript, and analyzes again.

Spaces become hyphens, uppercase becomes lowercase, and unsupported punctuation is removed. Review the rewritten manuscript, especially if similar speaker names exist.

## Expression and paralinguistic tags

The **Insert tag** menu in the manuscript toolbar lists the paralinguistic tags of the selected engine (it is hidden for engines without them). Choosing a tag inserts it at the cursor in **Raw tags**, or replaces the selected text; from the **Script** view it goes into the paragraph at the top of the screen and the editor switches to Raw tags.

Examples can include:

```text
[alice]I suppose that is one way to solve it. [sigh][/alice]
```

Standalone tags such as `[laugh]` have no closing form and are not treated as speakers by the tag-balance checker. Support is engine-specific:

- Use only cues displayed for the current engine.
- Preview the exact sentence; unsupported cues may be ignored or spoken aloud.
- Do not assume a cue supported by Chatterbox, Kokoro, Qwen3, or OmniVoice has the same effect in another engine.
- Use cues sparingly. Repeated non-verbal generations can vary and may interrupt narration rhythm.

See the selected engine's guide, beginning with [Engine Reference and Comparison](help:engine-overview).

## Passage-level delivery directions

Breeze TTS 2 can retain an assigned reference voice while changing how a passage is performed. Put one direction immediately before the speaker block:

```text
[direction]Begin as a whisper, then become firm and urgent on the final sentence.[/direction]
[alice]I know what I saw. We have to leave now.[/alice]
```

TTS-Story also accepts the older `[emotion]...[/emotion]` form. A direction is attached to the following speaker passage, carried into its chunks, saved in review metadata, and reused during regeneration. Other engines may ignore the direction; it is not spoken as manuscript text.

## Speaker tags created by an LLM

**Prep text** can add or normalize speaker tags, but an LLM can also change names, omit a close, wrap too much narration, or rewrite the prose. When preparation finishes:

1. Compare the output with the source.
2. Resolve the unmatched-tag warning.
3. Review the speakers and any possible-duplicates notice in the **Cast** list.
4. Confirm that each passage belongs to the intended speaker.
5. Assign voices only after the tags are stable.

Read [Prepare Text with an LLM](help:prep-text) and [Analyze Text and Review Statistics](help:text-statistics) next.

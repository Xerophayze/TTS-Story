# Save and Restore Projects

**Manage Projects** saves a snapshot of Generate-page preparation so you can resume or reuse it. Projects are convenient working copies, but they are not portable project files and not a complete backup of TTS-Story.

## Save a project

1. Open [Generate](app:generate).
2. Select **Manage Projects**.
3. Enter a clear **Project name**.
4. Select **Save**.

![Manage Projects dialog with several clearly named preparation checkpoints](../../../static/help/screenshots/project-manager.png)

*Use stage-specific names so imported, LLM-prepared, and voice-approved snapshots remain easy to distinguish.*

Saving the same name again prompts before overwriting that project's snapshot. Saving does not continue automatically as you edit; select **Save** again whenever you want a new checkpoint.

Useful names include a title and stage, for example:

- `Dracula - imported source`
- `Dracula - LLM prepared`
- `Dracula - voices approved`
- `Dracula - chapter test settings`

## What a project records

The current snapshot includes much of the Generate-page state, including:

- Input text
- Job engine
- Default or reference voice choices where applicable
- Per-speaker assignments
- Speaker FX and ready indicators
- Speaker profiles
- Qwen language and instruction choices
- Chapter splitting, full-story, and heading configuration
- Output format, bitrate, and ACX choice
- Selected LLM prompt/preset information
- Current Alternate Word Registry entries

Loading the project analyzes the restored text and then reapplies saved assignments.

## What it does not contain

A saved project does not embed or back up:

- Generated audio or Library items
- Job Queue history
- API keys
- Cloud accounts, model access, quota, or voice catalogs
- Uploaded reference-audio files themselves
- Downloaded local models or isolated engine environments
- A standalone file that can be copied to another computer

If a saved assignment refers to a voice, prompt, or cloud catalog entry that no longer exists, choose a replacement after loading.

## Projects use the shared TTS-Story project library

> **Important:** Projects are stored by the TTS-Story backend in `data/projects.json`.

This means:

- Browsers and devices connected to the same running TTS-Story installation see the same project list.
- `localhost:5000`, `127.0.0.1:5000`, and a LAN address for that installation use the same projects.
- Clearing browser site data does not delete the backend project library.
- Moving TTS-Story to another computer does not transfer them.
- Deleting or losing `data/projects.json` removes the saved project snapshots.

When an upgraded browser still contains projects from the older browser-local system, TTS-Story imports them into the shared library and removes the legacy browser copy only after the server confirms the import.

Keep the original manuscript and critical prepared versions in normal backed-up files. See [Local Data, API Keys, and Backups](help:data-storage).

## Load carefully

Select **Load** beside a project to replace the current Generate-page state. The current interface does not first prompt you to save unsaved text.

Before loading:

1. Copy important current text externally or save it under another project name.
2. Note any unsaved Alternate Word entries or voice changes.
3. Confirm that the target project's timestamp and name are correct.

After loading, wait for automatic analysis and verify the engine, voices, section settings, output choices, and cloud catalogs. A successful load means the snapshot was applied; it does not prove that every referenced engine resource is still available.

## Delete carefully

The project list offers **Delete**. Deletion is immediate in the current interface and has no recovery or confirmation step. It removes only the saved project snapshot; it does not delete a separately submitted Library job or the external manuscript.

## Recommended checkpoint strategy

For a long audiobook:

1. Keep the source document outside TTS-Story.
2. Save an `imported source` snapshot after document cleanup.
3. Save a separate `LLM prepared` snapshot after reviewing model output.
4. Save `voices approved` after Quick Tests.
5. Submit a representative chapter.
6. Preserve final manuscripts and exported audio in your normal backup system.

Do not repeatedly overwrite the only known-good snapshot while experimenting with prompts or voices.

## Projects versus Library items

- A **project** helps you resume preparation on Generate.
- A **Library item** represents submitted/generated work with audio, chunk metadata, and regeneration tools.

Loading a project does not reopen an existing Library item, and deleting a project does not remove Library audio. Read [How the Pages Work Together](help:workflow-map) and [Use the Audio Library](help:audio-library).

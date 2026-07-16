# Use the Audio Library

Completed generation jobs are collected in [Audio Library](app:library). Library is the working area for listening, downloading, correcting chunks, and rebuilding final audio. It is also where review takes place; review opens in a modal on the same page.

![Audio Library with a completed item expanded to show playback and chapter actions](../../../static/help/screenshots/audio-library.png)

*Expand a Library item to play its output and reach chapter, review, rebuild, metadata, and download actions.*

## Read a Library card

Each card identifies the item by title, date, engine, size, and output format when that information is available. Expand the card to show its player and chapter controls.

The main actions can include:

- **Alt Words:** Save pronunciation replacements for future regeneration of this item.
- **Metrics:** Inspect recorded render timing when the job contains timing metrics.
- **Time Codes:** Generate chapter start times for a video description.
- **Download:** Download the normal full output.
- **Download M4B:** Build an audiobook file with chapter markers for chapter-mode items.
- **Edit Metadata:** Set title, author, genre, year, and description.
- **Rebuild:** Recompile outputs from the saved chunk audio.
- **Delete:** Permanently remove the item and its generated files.

Click the displayed title to edit the Library collection title. This changes the saved Library metadata; it does not alter the source manuscript.

## Play the result

Use the card's audio player for the selected output. Only one Library player is kept active at a time.

For chapter-mode items, select a chapter pill to load that chapter. Its action menu includes:

- **Review Chunks**
- **Rename Chapter**
- **Download Chapter**

The **Full Story** pill offers Review Chunks and, when a combined file exists, Download Full Story.

## Review and correct audio

Choose **Review Chunks** from a chapter or Full Story menu. TTS-Story loads the saved chunks into an in-page modal where you can edit text, change voices or engines, apply FX, and regenerate selected audio.

After regeneration, rebuild the affected chapter or all outputs. A previously compiled chapter does not update merely because one of its source chunks changed. Follow [Review and Regenerate Chunks](help:job-review).

## Use Library-specific alternate words

The Generate-page Alternate Word Registry is temporary unless saved in a Project, but a Library item's **Alt Words** list is stored with that item. Edit it before regenerating a recurring mispronunciation.

Replacements are case-insensitive literal substrings and run in list order. They are not whole-word rules. Test a narrow replacement before applying it across a speaker or book. See [Alternate Word Registry](help:alt-word-registry).

## Refresh and delete safely

Use **Refresh Library** after an export or rebuild if the card has not updated yet.

Deleting a Library item removes its audio directory, chunk data, metadata, and job log. This is different from removing a Queue record, which preserves completed audio. **Clear All** permanently deletes every Library item; download or back up anything you need first.

For storage locations and backup boundaries, see [Local Data, API Keys, and Backups](help:data-storage).

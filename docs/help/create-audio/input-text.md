# Enter or Import Your Text

The **Input Text** field on [Generate](app:generate) is the working manuscript for analysis and the next submitted job. You can type, paste, drag documents onto the field, or select **Load Document**.

Keep an untouched source copy outside TTS-Story before making changes.

![Generate page with the Input Text editor, document import, project, and Prep Text controls](../../../static/help/screenshots/generate-overview.png)

*Enter or import the manuscript in Input Text; the surrounding controls manage projects, documents, pronunciation, and optional LLM preparation.*

## Type or paste text

Plain text produces one **Speaker 1** assignment. Add paired speaker tags when different passages need different voices:

```text
[narrator]The station clock struck midnight.[/narrator]
[elena]We are out of time.[/elena]
```

After an edit, TTS-Story waits briefly and analyzes the field automatically. **Text Statistics**, detected speakers, voice assignments, and section information update without a separate Analyze button. Large manuscripts or slower computers can take longer than a short passage.

Read [Speaker and Expression Tags](help:speaker-tags) before tagging a long multi-character work.

## Load documents

Select **Load Document**, choose one or more files, and wait for the extraction status. You can also drag files onto the Input Text area.

Supported extensions are:

- TXT
- PDF
- Word `.doc` and `.docx`
- RTF
- EPUB
- ODT
- Markdown
- HTML

Document extraction retrieves text; it is not a layout-preserving conversion. Review headings, page furniture, footnotes, tables, image captions, hyphenation, smart punctuation, and reading order—especially after PDF extraction.

## Imports always append

> **Important:** Loading a document does not replace the current field. Every successfully extracted document is appended after the existing text, separated by blank lines.

This behavior is useful for assembling multiple chapters, but it can also duplicate a manuscript:

- Clear the field before importing a replacement copy.
- If loading several files, confirm their selected order and inspect the combined result.
- A supported file that fails extraction is skipped while other valid files can still append.
- Unsupported files are ignored; the status reports when no supported documents were found.

The small **C** button in the input field clears all text. If the field is not empty, TTS-Story asks for confirmation.

## Preserve a clean source

Several actions deliberately modify the working field:

- **Load Document** appends extracted text.
- **Prep Text** replaces the entire field with LLM output.
- Renaming a detected speaker with **Apply** rewrites matching speaker tags.
- **Auto-Fix** inserts tags to repair detected imbalances.
- Editing a heading in **Review detected sections** changes that heading in the input.
- Loading a browser project replaces the current Generate-page state.

Before any large change, save the original file externally. You can also create a named snapshot with **Manage Projects**, but projects are stored only in the current browser profile; see [Save and Restore Projects](help:projects).

## Clean the extracted text

Work from the top down:

1. Remove repeated headers, page numbers, navigation text, and legal boilerplate that should not be spoken.
2. Restore broken paragraphs and words split by PDF line endings.
3. Make chapter headings consistent if you want separate outputs.
4. Add balanced speaker tags, or use [Prepare Text with an LLM](help:prep-text) and review its changes.
5. Add pronunciation substitutions only after the wording is stable; see [Alternate Word Registry](help:alt-word-registry).
6. Wait for **Text Statistics** and confirm that the word, speaker, chunk, and section counts are plausible.

## Large manuscripts

For a book-length source:

- Import once rather than repeatedly clicking **Load Document**.
- Save a browser project after cleanup and again after voice assignment.
- Test one representative chapter before generating everything.
- Use clear, consistent headings covered by [Chapters, Books, and Sections](help:chapters-and-sections).
- Expect automatic analysis to take longer as the field grows.

When the text is clean, continue with [Analyze Text and Review Statistics](help:text-statistics).

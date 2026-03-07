# Resilient Gemini Prep Text Plan

## Goal

Improve the Prep Text flow so transient Gemini failures, especially HTTP `503` / model overload responses, do not wipe out completed work. The recommended approach is to move section-processing state from ephemeral frontend memory into a lightweight backend prep job, add bounded retry with backoff for retryable upstream failures, and let the UI resume from persisted completed sections instead of restarting from scratch.

## Steps

1. Define the new prep-text execution model around a persisted backend job instead of a purely client-side loop. Create a dedicated Gemini prep job record with fields for source text, derived sections, completed section outputs, known speakers, failure metadata, retry counters, and status/progress. This step blocks all later steps because the API contract depends on it.
2. Reuse the existing persistence patterns in `app.py` from the audio jobs database helpers (`_init_jobs_db`, `_serialize_job_entry`, `_persist_job_state`) to add a parallel persistence path for Gemini prep jobs. Prefer a separate table or separate JSON file namespace for prep jobs rather than overloading audio-job rows, because prep-text lifecycle and payload shape are different.
3. Add backend endpoints for prep-job lifecycle in `app.py`: start job, poll job status, process next pending section or continue processing, and optionally resume an interrupted job for the same input. Persist each completed section immediately after success so section `N` failure never discards sections `0..N-1`.
4. Add retry-on-`503` behavior in the LLM call path. Update `src/gemini_processor.py` and the Gemini branch inside `_run_llm_prompt()` call flow so retryable upstream failures are detected explicitly. Retry only bounded transient cases such as `503`, `UNAVAILABLE`, or rate-limit-style overload responses, using small exponential backoff with jitter and clear logging. Do not retry validation errors, empty prompts, auth/config issues, or malformed responses.
5. Preserve failure semantics instead of collapsing everything into HTTP `400`. In `app.py`, map retry-exhausted transient upstream Gemini failures to a `503`-style API response, or return a structured `retryable: true` failure payload so the frontend can distinguish overload from bad input. Keep actual client mistakes as `400`. This directly addresses the current misleading `400 (BAD REQUEST)` wrapping a Gemini `503 UNAVAILABLE`.
6. Update the frontend flow in `static/js/main.js` so `processWithGemini()` no longer stores all progress only in the local `outputs` array. Instead, start or resume a prep job, poll persisted progress, render section counts from backend state, and on completion rebuild the final text from stored section outputs. If the browser reloads or a section fails, the UI should be able to reattach to the job and continue.
7. Improve the UX for partial failure and resume in `static/js/main.js`, `templates/index.html`, and `static/css/style.css`. Show when a retry is in progress, when the system is waiting to retry, when a job is resumable, and whether some sections are already complete. Add an animated spinner/state indicator with distinct visual treatment for states such as generating, retrying, waiting-to-retry, completed, and failed so users can understand the live prep state at a glance. On retry exhaustion, keep the progress bar and expose a resume/retry action instead of dropping back to an all-or-nothing alert.
8. Ensure speaker continuity remains deterministic when resuming. Persist the running `known_speakers` set after each completed section and restore it before processing the next one so resumed runs keep the same tag guidance that the current sequential loop provides.
9. Add backend and frontend verification coverage. Backend tests should cover retry classification, bounded retries, persistence after each section, resume after failure, and correct HTTP/status payload mapping. Frontend checks should cover progress restoration, final-text reconstruction from partial outputs, and non-destructive handling of exhausted retries.

## Relevant Files

- `app.py` — modify Gemini prep endpoints around `get_gemini_sections()`, `process_gemini_section()`, and helper `_run_llm_prompt()` call sites; reuse database/persistence patterns near `_init_jobs_db()` and `_persist_job_state()`.
- `static/js/main.js` — replace the current in-memory sequential `processWithGemini()` orchestration with persisted job start/resume/poll behavior and clearer retry/resume UX.
- `src/gemini_processor.py` — add retryable error classification, bounded retry/backoff, and structured exception details for Gemini-specific transient failures.
- `src/llm_processor.py` — review whether unified multi-provider behavior needs matching retry/error-shaping so Gemini paths behave consistently if routed through this abstraction elsewhere.
- `templates/index.html` — add any minimal UI affordances needed for resumable prep-text status, spinner/state indicator, and retry feedback.
- `static/css/style.css` — add animated spinner and per-state styling for Prep Text progress and retry status.
- `README.md` — update Prep Text documentation and API endpoint docs once implementation is complete.

## Verification

1. Trigger prep text on multi-section input with Gemini intentionally returning a transient `503` once, verify the backend retries automatically, and confirm the UI remains on the same job instead of alerting immediately.
2. Trigger a repeated `503` beyond retry limit, verify completed sections remain persisted, the API surfaces a retryable/resumable failure, and resuming continues from the first incomplete section rather than starting over.
3. Refresh the page mid-run and verify the UI can reattach to the active prep job and continue showing correct processed count, current section, and the appropriate animated state indicator.
4. Verify non-retryable failures such as missing API key, empty section content, and invalid prompt/config still fail fast without retry and remain classified as `400`-level validation/config errors.
5. Confirm final processed text exactly matches the concatenation of persisted section outputs in original order and that extracted speaker tags remain stable across a resumed run.
6. Verify the Prep Text spinner/state indicator changes styling appropriately for generating, waiting-to-retry, retrying, completed, and failed states without becoming visually stuck.
7. Run regression checks for the existing audio job queue to ensure the new Gemini prep persistence path does not interfere with audio job schema, queue behavior, or library operations.

## Decisions

- Included: resilient prep-text processing for section-based Gemini runs, persisted incremental progress, retry on transient `503`-like overloads, better status/error mapping, frontend resume behavior, and explicit animated prep-state feedback in the UI.
- Included: preserving and restoring `known_speakers` state between sections so resume does not change tagging behavior.
- Excluded: redesigning the full single-call `/api/gemini/process` endpoint unless it shares retry helpers cleanly.
- Excluded: broad provider-agnostic retries for OpenAI or Anthropic unless discovery during implementation shows shared code should adopt the same structured transient-error model.
- Recommendation: prefer a dedicated Gemini prep-job persistence model rather than trying to repurpose the existing audio `jobs` rows, because the payload, lifecycle, and UI semantics are materially different.

## Further Considerations

1. Persistence medium choice: Option A (recommended) use SQLite alongside the existing jobs DB for crash-safe resume and easy status queries; Option B use per-job JSON files for lower schema effort but weaker query ergonomics.
2. Resume matching policy: Option A (recommended) explicit prep job ID returned from start endpoint; Option B implicit lookup by exact input text hash and prompt settings; Option A is less surprising and easier to reason about.
3. Retry policy defaults: start with `2-3` retries and short exponential backoff with jitter so temporary Gemini overloads are masked without making users wait excessively on hard failures.

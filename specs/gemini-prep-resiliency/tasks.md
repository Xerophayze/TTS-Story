# Resilient Gemini Prep Text Tasks

## Implementation Rules

- Do **not** start production code changes until the task owner confirms the task sequencing and API shape below.
- Prefer **small, testable commits** by milestone.
- Keep the existing Prep Text UX functional during refactor; avoid a long-lived broken intermediate state.
- Do not reuse the existing audio job row schema directly for Gemini prep jobs.
- Treat upstream transient Gemini overloads as retryable; treat validation/config/auth issues as non-retryable.

## Progress Snapshot — 2026-03-07

- Backend prep jobs are implemented with a dedicated SQLite `prep_jobs` table in `app.py`.
- Gemini retry classification/backoff is implemented in `src/gemini_processor.py` and mapped to retry-aware API responses in `app.py`.
- Frontend orchestration has been moved to persisted prep jobs in `static/js/main.js` with refresh restore + resume support.
- Prep status UI updates landed in `templates/index.html` and `static/css/style.css`.
- Focused backend regression coverage was added in `tests/test_gemini_prep_resiliency.py` and is passing.
- Manual browser validation and broader audio-queue regression checks are still pending.

## Recommended Delivery Sequence

1. Finalize prep-job persistence schema and API contract
2. Add Gemini retry classification and bounded retry helper
3. Implement backend prep-job lifecycle endpoints
4. Refactor frontend orchestration to use prep jobs
5. Add resume/retry UX polish
6. Add verification coverage and documentation

## Milestone 1 — Prep Job Data Model and Persistence

### Objective

Create a dedicated persisted job model for Gemini Prep Text so section-level progress survives retries, browser refreshes, and server restarts.

### Target files

- `app.py`
- optionally `data/jobs/` if a file-based fallback is needed

### Tasks

- [x] Choose persistence medium and record decision in code comments and README notes:
  - [x] **Preferred:** add a dedicated SQLite table alongside the existing jobs DB
  - [ ] Fallback: per-job JSON files under `data/jobs/`
- [x] Define prep job fields:
  - [x] `job_id`
  - [x] `status` (`queued`, `processing`, `retrying`, `failed`, `completed`, `cancelled`)
  - [x] `created_at`
  - [x] `updated_at`
  - [x] source text or source text hash
  - [x] prompt override / preset metadata
  - [x] section list snapshot
  - [x] completed section outputs keyed by section index
  - [x] `known_speakers`
  - [x] `current_section_index`
  - [x] `processed_sections`
  - [x] `total_sections`
  - [x] `progress`
  - [x] `last_error`
  - [x] `retryable`
  - [x] `retry_count`
  - [x] `max_retries`
  - [x] `resume_token` or equivalent job identifier used by the UI
- [x] Add schema initialization or storage bootstrap logic
- [x] Add serialization and deserialization helpers
- [x] Add helpers to:
  - [x] create a prep job
  - [x] load a prep job
  - [x] update a prep job after each section
  - [x] reconstruct final output from persisted section results
  - [x] resume a failed or interrupted job from the first incomplete section
- [x] Persist progress immediately after each successful section
- [x] Ensure in-progress prep jobs can be loaded after a process restart

### Deliverables

- A stable prep-job persistence model in `app.py`
- Helper functions for create/load/update/resume/reconstruct
- Clear separation from audio generation jobs

### Done criteria

- A prep job can be created without processing any sections yet
- After each completed section, the persisted record reflects updated progress and stored output
- Restarting the server does not erase the prep job state
- The final output can be rebuilt from persisted section outputs in original order

### Dependencies

- None

---

## Milestone 2 — Retry Classification and Error Semantics

### Objective

Detect retryable Gemini overload conditions precisely and stop mislabeling transient upstream failures as HTTP `400` errors.

### Target files

- `src/gemini_processor.py`
- `src/llm_processor.py` (review and update only if needed for consistency)
- `app.py`

### Tasks

- [x] Introduce structured Gemini error metadata in `src/gemini_processor.py`
- [x] Classify retryable transient conditions, including:
  - [x] HTTP `503`
  - [x] Gemini `UNAVAILABLE`
  - [x] temporary capacity / overload wording
  - [x] optionally rate-limit responses if they are semantically transient in this code path
- [x] Keep non-retryable failures explicit:
  - [x] empty prompt
  - [x] missing API key
  - [x] invalid config
  - [x] malformed or empty model response
  - [x] auth / permission errors
- [x] Add bounded retry with exponential backoff and jitter
- [x] Choose and document retry defaults:
  - [x] retry count
  - [x] base delay
  - [x] maximum delay
- [x] Log each retry attempt with section/job context
- [x] Return structured exception details so `app.py` can map failures by type
- [x] Update API response mapping in `app.py` so retry-exhausted transient failures return either:
  - [x] HTTP `503`, or
  - [x] a structured `success: false` payload with `retryable: true`

### Deliverables

- Retryable/non-retryable Gemini error classification
- Bounded retry helper behavior
- Correct API-level status semantics for transient failures

### Done criteria

- A single transient Gemini overload does not fail the whole prep flow immediately
- Retry exhaustion produces a retryable/resumable failure state, not a misleading bad-request response
- Validation and configuration failures still fail fast without retry

### Dependencies

- Milestone 1 can be parallelized partially, but final status mapping must align with the prep-job model

---

## Milestone 3 — Backend Prep Job API

### Objective

Expose a backend-owned prep-job lifecycle so the frontend no longer owns critical progress state.

### Target files

- `app.py`
- `README.md` (after implementation)

### Required API capabilities

- [x] Start a prep job from input text and current prompt options
- [x] Get prep job status and progress
- [x] Continue or resume a prep job
- [x] Optionally cancel a prep job
- [x] Return completed final text when the job finishes

### Proposed API shape to confirm before coding

- [x] `POST /api/gemini/prep-jobs`
  - input: text, prompt override, custom heading, sectioning options
  - output: `job_id`, `status`, `processed_sections`, `total_sections`
- [x] `GET /api/gemini/prep-jobs/<job_id>`
  - output: job status, progress, retry state, partial completion summary
- [x] `POST /api/gemini/prep-jobs/<job_id>/resume`
  - output: updated job state or completion payload
- [x] optional `POST /api/gemini/prep-jobs/<job_id>/cancel`

### Tasks

- [x] Define the final response contract for each endpoint
- [x] Add backend validation for malformed requests
- [x] Ensure each completed section updates:
  - [x] stored output text
  - [x] processed count
  - [x] progress percent
  - [x] `known_speakers`
  - [x] retry metadata
  - [x] resumable state
- [x] Ensure a failed job reports enough information for the UI to show:
  - [x] retryable vs non-retryable
  - [x] completed sections so far
  - [x] current/failed section index
  - [x] human-readable error message
- [x] Ensure completion reconstructs final processed text in original section order

### Deliverables

- Prep-job start/status/resume API endpoints
- Stable JSON response contracts documented in code comments

### Done criteria

- The frontend can retrieve all state needed to render progress without storing section outputs in memory
- A failed job can be resumed from persisted state
- A completed job returns or exposes the final processed text

### Dependencies

- Milestone 1
- Milestone 2

---

## Milestone 4 — Frontend Orchestration Refactor

### Objective

Move Prep Text orchestration in `static/js/main.js` from an in-memory section loop to a backend-job-driven flow.

### Target files

- `static/js/main.js`
- `templates/index.html`
- `static/css/style.css`

### Tasks

- [x] Refactor `processWithGemini()` to:
  - [x] create a prep job instead of iterating sections directly in the browser
  - [x] poll prep job status
  - [x] update the existing progress UI from backend state
  - [x] update a prep-state indicator from backend state
  - [x] replace input text only after final completion
- [x] Remove dependency on the in-memory-only `outputs` array as the source of truth
- [x] Preserve existing prompt override and section-building behavior where possible
- [x] Handle backend states explicitly:
  - [x] `queued`
  - [x] `processing`
  - [x] `retrying`
  - [x] `waiting_to_retry`
  - [x] `failed`
  - [x] `completed`
- [x] Define a frontend prep-state mapping table that controls:
  - [x] label text
  - [x] spinner visibility
  - [x] spinner style variant
  - [x] progress bar behavior
  - [x] action button availability
- [x] Allow the page to reattach to an existing active prep job after refresh
- [x] Keep notifications informative but non-destructive

### Deliverables

- A backend-driven `processWithGemini()` flow
- Progress bar updates sourced from persisted job state
- A frontend prep-state model that can drive spinner/status rendering consistently

### Done criteria

- Refreshing the page no longer guarantees lost prep progress
- Completed text is still written back into the input field on success
- The UI can distinguish retrying from permanently failed states
- The UI shows the correct animated state indicator while work is generating, waiting to retry, retrying, completed, or failed

### Dependencies

- Milestone 3

---

## Milestone 5 — Resume and Retry UX

### Objective

Make failures survivable and understandable in the UI instead of reducing everything to a blocking alert.

### Target files

- `static/js/main.js`
- `templates/index.html`
- `static/css/style.css`

### Tasks

- [x] Add minimal UI affordances for:
  - [x] resume action
  - [x] retrying indicator
  - [x] waiting-to-retry indicator
  - [x] partial completion summary
  - [x] failure message with retryability state
- [x] Add an animated spinner or equivalent live activity indicator for Prep Text status
- [x] Define distinct visual variants for at least:
  - [x] generating / processing
  - [x] waiting to retry
  - [x] actively retrying
  - [x] completed
  - [x] failed
- [x] Preserve the current progress bar where possible instead of redesigning the UI
- [x] Ensure users can tell whether:
  - [x] work is still in progress
  - [x] the backend is retrying automatically
  - [x] the system is paused briefly before the next retry attempt
  - [x] a job is resumable
  - [x] a job is permanently failed
- [x] Avoid clearing completed progress when a retryable failure occurs
- [x] Ensure spinner/state styling is accessible and does not rely on color alone
- [x] Ensure animations stop or transition cleanly when the state changes or the job finishes

### Deliverables

- Minimal resumable-failure UX for Prep Text
- Animated spinner/state indicator with clear per-state styling

### Done criteria

- Users can resume a partially completed prep job without reprocessing completed sections
- Retrying state is visible while automatic retries are happening
- Failure messaging no longer implies user input was invalid when the real problem was transient model overload
- Waiting-to-retry and active-retry states are visually distinguishable
- Spinner/state UI does not get stuck in an old state after completion or failure

### Dependencies

- Milestone 4

---

## Milestone 6 — Speaker Continuity

### Objective

Preserve speaker-tag consistency across retries, resumes, and page refreshes.

### Target files

- `app.py`
- `static/js/main.js`

### Tasks

- [x] Persist `known_speakers` after every successful section
- [x] Restore `known_speakers` before processing the next section on resume
- [x] Verify normalization stays consistent with the current lowercased speaker tracking approach
- [x] Ensure resumed jobs do not produce duplicate or drifted speaker tags because context was lost

### Deliverables

- Stable speaker continuity during resumed runs

### Done criteria

- A resumed job continues using the same accumulated speaker guidance as before failure
- Speaker tag extraction does not regress versus the current sequential implementation

### Dependencies

- Milestone 1
- Milestone 3
- Milestone 4

---

## Milestone 7 — Verification and Documentation

### Objective

Prove the refactor works and document the new operational behavior.

### Target files

- tests or ad hoc verification harnesses added during implementation
- `README.md`

### Backend verification checklist

- [x] Retryable error classification covers the observed Gemini `503 UNAVAILABLE` case
- [ ] Automatic retry succeeds when the transient failure resolves within the retry budget
- [x] Retry exhaustion leaves a resumable persisted prep job
- [x] Completed section outputs remain available after failure
- [x] Resume starts from the first incomplete section, not from section `0`
- [x] API status mapping distinguishes validation failures from transient upstream failures

### Frontend verification checklist

- [ ] Progress state survives page refresh
- [ ] Final text is reconstructed from persisted outputs in the correct order
- [ ] Retrying and failed states render correctly
- [ ] Partial progress is not lost on retry exhaustion
- [ ] Spinner/state indicator styling changes correctly across generating, waiting-to-retry, retrying, completed, and failed states
- [ ] Spinner animation starts and stops correctly on state transitions

### Regression checklist

- [ ] Existing audio job queue behavior remains unchanged
- [ ] Existing `/api/gemini/process` behavior is unaffected unless intentionally updated
- [ ] Existing speaker analysis flow still runs after prep completes

### Documentation tasks

- [x] Update `README.md` Prep Text workflow description
- [x] Document retry/resume behavior in user-facing language
- [x] Document any new API endpoints in the API section

### Deliverables

- Verification notes or tests
- Updated README documentation

### Done criteria

- The implementation is verified against transient overload, retry exhaustion, resume flow, and regression checks
- Documentation accurately reflects the new backend-owned Prep Text flow

### Dependencies

- Milestones 1 through 6

---

## Decisions Resolved During Implementation

- [x] Persistence medium confirmed: dedicated SQLite table in the existing jobs DB (`prep_jobs`)
- [x] Final endpoint names and payload shapes confirmed around `/api/gemini/prep-jobs`
- [x] Retryable failures use HTTP `503` plus explicit `retryable` metadata
- [x] Default retry settings confirmed: `max_retries=3`, `base_delay=1.0s`, `max_delay=8.0s`
- [x] Prep jobs include explicit cancellation in v1

## Suggested Work Breakdown by Commit

1. Prep-job schema and persistence helpers
2. Gemini retry classification and retry helper
3. Prep-job start/status/resume backend endpoints
4. Frontend orchestration refactor in `processWithGemini()`
5. Resume/retry UI polish
6. Verification coverage and README updates

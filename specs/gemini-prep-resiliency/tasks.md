# Resilient Gemini Prep Text Tasks

## Phase 1: Backend data model and persistence

- [ ] Define a Gemini prep job schema with:
  - [ ] `job_id`
  - [ ] source text or text hash
  - [ ] prompt override / preset metadata
  - [ ] derived sections
  - [ ] completed section outputs keyed by section index
  - [ ] `known_speakers`
  - [ ] current section index
  - [ ] total section count
  - [ ] status (`queued`, `processing`, `retrying`, `failed`, `completed`)
  - [ ] retry metadata and last error
- [ ] Decide persistence medium:
  - [ ] Preferred: add a dedicated SQLite table alongside the existing jobs DB
  - [ ] Alternative: per-job JSON files under `data/jobs/`
- [ ] Implement serialization / deserialization helpers in `app.py`
- [ ] Persist progress after every successfully completed section
- [ ] Support loading an in-progress prep job after process restart

## Phase 2: Retry and error semantics

- [ ] Add structured Gemini transient error classification in `src/gemini_processor.py`
- [ ] Detect retryable overload conditions, including:
  - [ ] HTTP `503`
  - [ ] Gemini `UNAVAILABLE`
  - [ ] similar temporary capacity / overload responses
- [ ] Add bounded retry with exponential backoff and jitter
- [ ] Keep non-retryable errors as immediate failures:
  - [ ] empty prompt
  - [ ] invalid config
  - [ ] missing API key
  - [ ] malformed response
  - [ ] auth / permission failures
- [ ] Return structured error details so `app.py` can map them correctly
- [ ] Update API responses so transient upstream failures are not mislabeled as `400`

## Phase 3: Backend API flow

- [ ] Add endpoint to create a Gemini prep job from input text
- [ ] Add endpoint to get prep job status / progress
- [ ] Add endpoint to continue or resume an existing prep job
- [ ] Optionally add endpoint to cancel or clear a prep job
- [ ] Ensure each section completion updates:
  - [ ] stored output text
  - [ ] processed count
  - [ ] progress percent
  - [ ] `known_speakers`
  - [ ] resumable state
- [ ] Ensure completion reconstructs final processed text in original section order

## Phase 4: Frontend orchestration and UX

- [ ] Refactor `processWithGemini()` in `static/js/main.js` to use backend prep jobs
- [ ] Replace in-memory-only `outputs` accumulation with server-backed progress
- [ ] Poll job status and update progress UI from persisted backend state
- [ ] Handle retrying state in the progress UI
- [ ] Handle resumable failure state without clearing completed progress
- [ ] Allow page refresh / reattach to an in-progress job
- [ ] Keep final text replacement behavior once the backend job completes
- [ ] Add minimal UI affordances in `templates/index.html` for:
  - [ ] resume action
  - [ ] retrying status
  - [ ] partial completion feedback

## Phase 5: Speaker continuity

- [ ] Persist `known_speakers` after each successful section
- [ ] Restore `known_speakers` when resuming a prep job
- [ ] Verify speaker tag continuity remains stable across retries and resumes

## Phase 6: Validation and regression coverage

- [ ] Add backend tests for:
  - [ ] retryable error classification
  - [ ] bounded retry behavior
  - [ ] persistence after each section
  - [ ] resume from first incomplete section
  - [ ] correct status / error mapping
- [ ] Add frontend verification for:
  - [ ] progress restoration after refresh
  - [ ] final text reconstruction from persisted outputs
  - [ ] non-destructive partial failure handling
- [ ] Run regression checks to ensure audio job queue behavior remains unchanged
- [ ] Update `README.md` documentation for the new Prep Text flow and retry/resume behavior

## Recommended implementation order

1. Persistence model
2. Gemini retry classification and bounded retry
3. Backend start/status/resume endpoints
4. Frontend orchestration refactor
5. UX polish for resumable failures
6. Tests and README updates

# Estudio integration and workflow improvements

## Scope and status

The user approved merging Estudio PR #43 before further usability work. The PR was merged on 2026-10-29 as `fd7db71560d4484d54b4e8cf83f0f886d618dab4`, and the main checkout was fast-forwarded. Work on AMD/ROCm and multi-GPU expansion is deferred. IndexTTS issue #39 is an optional later enhancement, not part of the immediate release gate.

## Implementation stages

1. Implemented locally: preserve ready speakers during Build profiles, Generate voices, and Auto assign; merge profile responses without clearing casting metadata; keep pending auditions; make intentional ready-speaker inclusion explicit; preserve that policy on resume; visibly label Generate voices. Browser checks confirm all-ready batches send no generation requests, and assignments/Ready flags survive a backend/browser reload. Mixed-ready and pending-audition behavior also have automated coverage.
2. Implemented locally: Kokoro honors the selected device, accepts an unspecified sample rate as 24 kHz, and no longer advertises sample cloning. Kokoro/Pocket register finished chunks before pause/cancel callbacks; Pocket grouping preserves original segment and narrative indices. Nonpersistent isolated workers now stop their owned process tree on interruption rather than orphaning a Windows Python child. Real preset, CPU/CUDA, blend, interruption, and authenticated Pocket cloning results are recorded below.
3. Reference transcription is now a separately installed utility under Engine Settings → Utilities. Its CPU-only runtime and pinned SenseVoiceSmall model are isolated from the web server and other engines. Installation downloads the model; each transcription runs in a short-lived offline worker. Missing-tool responses link users to installation; removing it preserves samples/transcripts and needs no backend restart. Real installation and transcription succeeded.
4. Implemented engine-first Cast guidance, assignment counts, Review next navigation, and partial-cast submission checks. Buttons wrap cleanly in the sidebar. Undated legacy projects no longer break sorting. Library listing is deferred while Generate is active; regeneration updates its text summary. Both Breeze options were restored to the remaining chunk/alternate-word menus without changing synthesis or direction processing.
5. Help and README updated, including contributor attribution. The oversized Gemini preview block was removed from above Thank You; Current updates now has one short Gemini entry linking to its detailed help page.

## Verification gates

- A marked-ready speaker's profile, voice selection, candidate settings, and audition metadata remain unchanged in default batches.
- All-ready batches send no provider generation requests. Missing/partial profile responses preserve existing entries.
- Resume never repeats completed speakers and rechecks ready flags before the next request.
- Pending candidates are distinguishable from newly generated, skipped, and failed speakers.
- Profile responses cannot overwrite another project or manuscript switched during a request.
- CPU/GPU diagnostics reflect the actual worker device. Pocket preset access is tested separately from gated cloning access.
- Preview, whole-job generation, Library regeneration, and pause/resume are verified for Pocket and Kokoro before calling them fully functional.
- Existing user projects and approved production audio are preserved. The user authorized starting/stopping a tracked test backend; use one listener with no reloader and stop owned test processes afterward. Do not make paid cloud requests solely for automated tests.

## Attribution

Credit LaMarCab76 for the Estudio redesign in README Credits and link PR #43. That attribution has been added; it makes no claim that all acceptance gates above have passed.

## Baseline results

Before merging, PR #43 passed 45 focused Python tests for queue APIs, library durations, and speaker chunk statistics. Both the original main revision and the PR returned 55 passing and five failing JavaScript tests. Those failures involved missing shared LocalAI setup in test fixtures; the routing fixture also needed to distinguish the opened drawer from the intentionally lazy hidden holding list. These fixtures have been repaired.

Current automated checks (2026-10-30): all 81 JavaScript tests pass. The complete Python suite passes 618 tests plus 789 subtests with the bundled FFmpeg on PATH. Its only warning is the existing audioop deprecation. Cache-version tests now enforce a minimum fresh version instead of rejecting every later cache bust. The earlier focused selection passed 172 tests plus 720 subtests.

Actual local verification (not provider billing):

- Installed isolated transcription, Pocket, and Kokoro runtimes. SenseVoice transcribed a known reference in about 12 seconds. Its model revision is pinned to `3847d57b6bdf2dd8875cb1508d2af43d80a16bf7`, and no remote model Python is trusted.
- Pocket presets generated two ordered non-silent 24 kHz chunks in a CLI test (~17 seconds), and a real two-speaker UI job completed in ~10 seconds. A separate interruption test retained the finished chunk and stopped the worker tree (~8 seconds).
- Plain text uses the visible **Speaker 1**, not the analyzer's internal **default** pseudo-speaker, in assignment preflight. A real Marius job paused after its final chunk (~14 seconds), survived restarting only the tracked backend, then resumed and completed chapter/full-story exports. The saved WAV SHA-256 did not change. This exposed and fixed a pre-existing Resume edge case: zero recorded post-processing did not mean the export was complete, and saving an already-completed job while holding the non-reentrant queue lock could deadlock all queue requests. Four regression cases cover pending and finished exports.
- Kokoro completed a real two-speaker UI job on CUDA (~34 seconds including first-use downloads; the second chunk was ~0.8 seconds). Explicit CPU synthesis produced two chunks in ~13 seconds; a CUDA interruption test stopped after one registered chunk (~11 seconds). Real CUDA blending of am_echo/am_adam produced a valid 24 kHz clip without modifying the custom-voice store.
- A five-chunk Kokoro job paused after two completed chunks and resumed successfully. Its job log shows only three remaining segments submitted on resume, the prior audio hashes stayed unchanged, and timing retained five real measurements with a three-chunk resumed session (no instant-completion samples).
- Both real `/api/preview` paths returned valid 24 kHz WAVs: Kokoro ~16 seconds and Pocket preset ~8 seconds, including runtime/model initialization. Pocket Library regeneration changed the text and selected voice to Marius, produced different audio bytes, and SenseVoice independently transcribed the new wording.
- Library regeneration changed the disposable QA chunk's text and audio; transcription independently confirmed the new wording. Current engine/voice and Play controls were present in the expanded speaker section.
- Loaded the existing Sherlock project read-only: 1,028,608 manuscript characters and 66 ready speakers, five idle audio elements, zero audio sources loaded at project load or after opening one speaker. Saved-project store SHA-256 was unchanged. Library pagination previously showed 20 idle players, zero sources, across the large existing catalog.

- Browser acceptance with a separate, disposable mixed-ready project: one ready speaker was skipped and one pending two-candidate audition was retained without starting a generation job. Closing/reopening the drawer and saving/reloading the project retained both candidates despite an already-selected voice. Choosing Candidate A updated the sample selector and removed the audition cards; the unrelated ready profile was unchanged. This used copied local test audio, not paid generation or existing user voice candidates.

- Authenticated Pocket cloning passed on 2026-10-30 after the user configured and verified Hugging Face access. A real `/api/preview` request using an existing Qwen-designed reference produced a valid 6.16-second 24 kHz clip in ~19 seconds, with the reference hash unchanged. SenseVoice transcribed the requested two-sentence test correctly. The isolated cloning batch then produced two ordered non-silent 24 kHz chunks on CPU in ~11 seconds; its first chunk's transcript also matched. The CLI smoke helper now reads the saved Pocket token (or environment fallback) without printing it or placing it on the command line, with a regression test proving credentials are not passed to unrelated engines.

## Acceptance outcome

The implementation and required acceptance gates are complete as of 2026-10-30. The Pocket access blocker is resolved. Mixed-ready profile responses and paid-provider audition behavior have automated coverage; no paid requests were made for acceptance testing. Disposable QA productions, projects, and sample entries were removed and original stores checked for changes. Cloning used only temporary cached output files and did not modify its reference sample. No test engine/transcription workers remain; the user's manually started backend was left running, with no additional backend started for the final tests.

This follow-up release includes the implementation, regression tests, help, and condensed README described above. GitHub's API independently confirmed PR #43 merged at `2026-10-29T23:35:39Z` to the expected merge SHA (the public HTML page was stale). Deferred AMD/multi-GPU work and IndexTTS emotion-vector enhancements remain outside this approved implementation scope.

### Gemini bulk assignment follow-up — 2026-10-30

User testing found a provider-specific gap: all Gemini bulk auditions stayed pending, including single voices, while Auto assign only updated local reference paths. A successful single Gemini candidate now updates the speaker's hosted-voice dropdown and selected-voice metadata automatically when Gemini narration is selected. Multiple candidates and partial failures remain pending; ready speakers are protected. Existing unselected singles can be recovered without creating new voices.

Auto assign now offers custom Gemini hosted IDs, uses the speaker's own saved audition identity ahead of fuzzy name matching, and excludes presets/local audio. A blank or unmatched Gemini proposal preserves the existing assignment. Completion messaging distinguishes automatic assignments and auditions requiring review. Regression tests cover prefixed names, distinct per-speaker dropdown values replacing Iapetus, recovery without rebilling, assignment failure/retry, partial/multiple auditions, and compatible narration. Provider creation responses were mocked for these regression tests; no paid Google generation or cloud deletion requests were made for the fix.

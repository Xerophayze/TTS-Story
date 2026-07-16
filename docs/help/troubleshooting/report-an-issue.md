# Prepare a Useful Issue Report

A good report lets another person reproduce the failure without receiving your API keys, private manuscript, or voice recordings. Reduce the problem to the smallest safe example before opening an issue.

## Search and reproduce

Check existing reports at [TTS-Story Issues](https://github.com/Xerophayze/TTS-Story/issues). If the problem is not already covered:

1. Restart TTS-Story.
2. Reproduce with one short, non-private paragraph.
3. Use the same engine and the fewest settings needed to trigger it.
4. Record the exact expected result and actual result.
5. Note whether the problem is consistent or intermittent.

Do not remove the failed Queue or Library item until evidence is collected.

![Job Details dialog with the non-sensitive fields useful in an issue report](../../../static/help/screenshots/job-details.png)

*Record the Job ID, selected engine, status, progress, and exact error, then redact private text and paths before sharing.*

## Include environment details

- TTS-Story revision or release
- Operating system and version
- Python version
- CPU and RAM
- GPU model, VRAM, and driver when relevant
- Whether CUDA is reported available
- Selected engine and local/cloud mode
- Model/version, device, chunk size, format, and relevant non-secret controls
- Whether current setup/update completed without warnings

Run:

```text
python scripts/engine_versions.py
```

For structured output:

```text
python scripts/engine_versions.py --json
```

While the app is running, `http://localhost:5000/api/health` provides availability, CUDA/VRAM, loaded-engine, and selected configuration status without returning credential values.

## Include reproduction steps

Write numbered steps beginning from startup. For example:

1. Start TTS-Story and select Edge TTS.
2. Load the voice catalog and choose a named voice.
3. Paste the attached two-sentence sample.
4. Assign the voice and click Quick Test.
5. Observe the exact error.

Attach a tiny synthetic/public-domain input when possible. State whether Quick Test, full generation, Library regeneration, rebuild, or export fails; those use different paths.

## Include safe diagnostic output

Useful evidence includes:

- the exact error text and HTTP status;
- terminal traceback beginning with the first relevant error;
- Job ID, status, and last completed chunk;
- the per-job `static/audio/<job-id>/job.log` after reviewing it; and
- a screenshot with private fields obscured.

Job Details can include full manuscript text. Job logs and paths can reveal usernames, filenames, speaker names, or project titles. Read every attachment and redact private material before posting.

## Never include

- `config.json` unless every credential and personal value has been removed;
- API keys, bearer tokens, cookies, or authorization headers;
- private manuscripts or LLM prompts containing the manuscript;
- a real person's reference recording without permission;
- `.env`, key, certificate, database, or browser-profile files; or
- a complete user-data/archive directory.

If a secret was exposed, revoke it immediately. Editing the issue afterward is not sufficient because notifications and caches may retain the original.

## Suggested issue template

```text
Summary:
Expected:
Actual:

Steps to reproduce:
1.
2.
3.

Engine and relevant settings:
Environment:
Health/version results:
Job ID/status (if applicable):
Exact error:

Minimal sample attached: yes/no
Regression (worked before): yes/no/unknown
```

For preliminary diagnosis, use [Troubleshooting Checklist](help:troubleshooting-overview).

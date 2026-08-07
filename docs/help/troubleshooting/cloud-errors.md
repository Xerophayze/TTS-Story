# Cloud Credentials, Quota, and Network Errors

Cloud failures can come from TTS-Story configuration, the local network, account policy, or the provider itself. Preserve the exact HTTP status and provider message; “API error” alone is not enough to distinguish them.

## Interpret common status codes

| Status | Typical meaning | First checks |
| --- | --- | --- |
| 400 | Invalid request, model, voice, or unsupported parameter | Re-fetch catalogs; return tuning to defaults |
| 401 | Missing, invalid, expired, or malformed credential | Re-enter the key/token and save |
| 402 | Payment or credit required on providers that use this code | Check billing and account balance |
| 403 | Permission, region, resource, or policy restriction | Verify account/resource access and Azure region |
| 404 | Wrong Base URL, unavailable model/voice, or removed endpoint | Restore the documented URL and fetch the catalog again |
| 408 or timeout | Request exceeded client/provider time | Test less text; check service/network; then adjust timeout |
| 429 | Rate, concurrency, or quota limit | Pause, reduce parallelism, and inspect account usage |
| 5xx | Provider or upstream service failure | Retry a tiny test later and check provider status |

Provider messages take precedence over this table.

## Verify configuration in order

1. Open [Settings](app:settings).
2. Select the provider or engine tab.
3. Re-enter the credential without leading/trailing spaces.
4. Restore the documented Base URL unless the provider explicitly gave another.
5. Use **Fetch Models**, **Test & Load Voices**, or the provider's catalog button.
6. Select an item returned for the current account.
7. Click **Save Settings**.
8. Test one sentence before retrying the manuscript.

![Cloud LLM settings with provider, credential, model catalog, and connection controls](../../../static/help/screenshots/llm-cloud-settings.png)

*Verify the provider URL and credential, refresh the live catalog, select an available model, and save before a minimal test.*

Model/voice discovery proves only that the catalog request works. Generation may require separate model access, sufficient credits, or a compatible parameter set.

## Provider-specific checks

**Gemini, Atlas Cloud, and OpenRouter:** Confirm the chosen model still appears for the key. Atlas normally uses `https://api.atlascloud.ai/v1`; OpenRouter normally uses `https://openrouter.ai/api/v1`.

**Replicate:** Confirm the token, model/version setting, billing, file-upload permission for reference prompts, and effective parallel request count.

**Azure Speech:** The Speech key and region must belong to the same Azure Speech resource. Re-fetch voices after changing either. A valid key with the wrong region can fail authorization.

**Edge TTS:** There is no API key. It uses Microsoft's unofficial consumer speech protocol, so throttling or an upstream protocol change can break it even when local configuration is unchanged. Reduce maximum parallel requests and test again; run the current setup/update if the `edge-tts` package is unavailable.

**ElevenLabs:** Re-fetch the account catalog and usage data. Confirm the selected voice/model is permitted by the account and that character and concurrency limits remain.

## Timeouts and retries

Increase a timeout only when the provider is accepting requests but legitimately needs longer. A longer timeout does not fix 401, 403, 404, or 429 responses.

Prep Text treats temporary capacity and hard quota errors differently. High demand, overload, timeout, connection, and temporary service failures retry the same profile up to five times before advancing. Explicit quota exhaustion and a profile's local daily request cap advance immediately. After a backup succeeds, later sections continue with it; Pause and Resume retain that active profile. Authentication, invalid model, validation, and other non-retryable failures stop immediately. Correct Settings, then resume or restart as appropriate.

Daily profile counters reset at midnight Pacific Time and count attempts made by this TTS-Story installation, including failed attempts. They do not include use of the same key by another application or another TTS-Story installation. Set a profile limit to `0` only when unlimited local use is intentional.

Build Profiles and the individual Build Profile action are independent requests. Each begins with the primary LLM again. If a specific backup fails, verify its provider-level settings and any model or API-key overrides; a blank override inherits the chosen provider's normal value.

## Credential safety

Saved keys are plain text in local `config.json`. Never post that file or include request headers in a public report. Redact tokens from screenshots, terminal output, and copied commands. If a key appears anywhere public, revoke it rather than merely editing the post.

See [Configure Online Services Safely](help:online-services) and [Local Data, API Keys, and Backups](help:data-storage).

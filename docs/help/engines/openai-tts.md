# OpenAI-compatible TTS

This cloud engine can use OpenAI's speech API or another service that implements the same `/audio/speech` request format.

## Connect the service

![Engine settings navigation where the OpenAI TTS tab is available](../../../static/help/screenshots/engine-settings-navigation.png)

1. Open [Settings → Engine Settings](app:settings/openai-tts) and select **OpenAI TTS**.
2. For OpenAI, enter your API key and leave the base URL at `https://api.openai.com/v1`.
3. For a compatible provider, enter its base URL or full `/audio/speech` URL. A local service may not require a key.
4. Enter the model and default voice, save, and generate a short preview before starting a long job.

## Voices and models

The voice selector includes OpenAI's built-in voice names. You can add comma-separated provider voice IDs in **Additional Voice IDs**. An OpenAI custom voice ID beginning with `voice_` is sent in the custom-voice object format; other names are sent as ordinary voice strings for broad endpoint compatibility.

OpenAI custom voice creation is not a general voice-cloning feature. It requires an eligible account, an audio sample, and a separate recorded consent workflow. TTS-Story does not create consent records or custom voices; after an eligible user creates a voice through OpenAI, its `voice_...` ID can be added here and used for generation.

The model field is editable because compatible providers may expose different model names. Voice instructions work only with models that support them; legacy `tts-1` models ignore instructions.

## Speed and voice effects

Speaker speed is sent directly as the API's native `speed` value. TTS-Story applies pitch and tone to the returned WAV without applying speed a second time, so previews, full generation, and Library regeneration use the same pacing rule.

## Troubleshooting

- **401:** Check the key and make sure it belongs to the configured provider.
- **403:** Verify model and voice access.
- **429:** Reduce parallel requests or check quota and billing.
- **404:** Confirm the URL ends at `/v1` or the full `/audio/speech` route.
- **Voice rejected:** Use a built-in name or an exact custom/provider voice ID.

Treat API keys like passwords. They are stored in local configuration and scrubbed by the repository sync safety check.

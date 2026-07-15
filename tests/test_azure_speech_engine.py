import tempfile
import unittest
from pathlib import Path

from src.engines.azure_speech_engine import AzureSpeechEngine, AzureSpeechError
from src.engines.base import VoiceAssignment


WAV_BYTES = b"RIFF" + (b"\x00" * 20)


class FakeResponse:
    def __init__(self, *, status_code=200, payload=None, content=b"", text="", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.content = content
        self.text = text
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("not JSON")
        return self._payload


class RecordingRequester:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if not self.responses:
            raise AssertionError("Unexpected Azure request")
        return self.responses.pop(0)


class AzureSpeechEngineTests(unittest.TestCase):
    def make_engine(self, requester, **overrides):
        return AzureSpeechEngine(
            subscription_key="azure-test-key",
            region="westus2",
            requests_per_minute=0,
            request_func=requester,
            sleep_func=lambda _seconds: None,
            **overrides,
        )

    def test_requires_key_and_valid_region(self):
        with self.assertRaisesRegex(AzureSpeechError, "resource key"):
            AzureSpeechEngine(subscription_key="", region="westus2")
        with self.assertRaisesRegex(AzureSpeechError, "region"):
            AzureSpeechEngine(subscription_key="key", region="https://westus2")

    def test_voice_catalog_is_normalized_and_sorted(self):
        requester = RecordingRequester([
            FakeResponse(payload=[
                {
                    "ShortName": "en-US-GuyNeural",
                    "DisplayName": "Guy",
                    "LocalName": "Guy",
                    "Gender": "Male",
                    "Locale": "en-US",
                    "LocaleName": "English (United States)",
                    "StyleList": ["newscast", "cheerful"],
                    "RolePlayList": ["YoungAdultMale"],
                    "SecondaryLocaleList": ["de-DE"],
                    "SampleRateHertz": "48000",
                    "VoiceType": "Neural",
                    "Status": "GA",
                    "WordsPerMinute": "150",
                }
            ])
        ])
        engine = self.make_engine(requester)

        voices = engine.list_voices()

        self.assertEqual(voices[0]["short_name"], "en-US-GuyNeural")
        self.assertEqual(voices[0]["styles"], ["newscast", "cheerful"])
        self.assertEqual(voices[0]["roles"], ["YoungAdultMale"])
        self.assertEqual(voices[0]["sample_rate_hertz"], 48000)
        method, url, kwargs = requester.calls[0]
        self.assertEqual(method, "GET")
        self.assertEqual(
            url,
            "https://westus2.tts.speech.microsoft.com/cognitiveservices/voices/list",
        )
        self.assertEqual(kwargs["headers"]["Ocp-Apim-Subscription-Key"], "azure-test-key")

    def test_ssml_escapes_text_and_includes_expression_and_prosody(self):
        engine = self.make_engine(RecordingRequester([]))
        assignment = VoiceAssignment(
            voice="en-US-GuyNeural",
            lang_code="en-US",
            fx_payload={"pitch": 2.5},
            speed_override=1.2,
            extra={
                "style": "newscast-casual",
                "style_degree": 1.4,
                "role": "YoungAdultMale",
                "volume": 8,
            },
        )

        ssml = engine.build_ssml('News & <updates> "today"', assignment)

        self.assertIn('xml:lang="en-US"', ssml)
        self.assertIn('name="en-US-GuyNeural"', ssml)
        self.assertIn("News &amp; &lt;updates&gt; \"today\"", ssml)
        self.assertIn('style="newscast-casual"', ssml)
        self.assertIn('styledegree="1.40"', ssml)
        self.assertIn('role="YoungAdultMale"', ssml)
        self.assertIn('rate="1.200"', ssml)
        self.assertIn('pitch="+2.50st"', ssml)
        self.assertIn('volume="+8.00%"', ssml)

    def test_generate_audio_posts_ssml_and_returns_wav(self):
        requester = RecordingRequester([FakeResponse(content=WAV_BYTES)])
        engine = self.make_engine(
            requester,
            output_format="riff-48khz-16bit-mono-pcm",
        )

        audio = engine.generate_audio(
            "Hello Azure",
            voice="en-US-AvaMultilingualNeural",
            lang_code="en-US",
        )

        self.assertEqual(audio, WAV_BYTES)
        self.assertEqual(engine.sample_rate, 48000)
        method, url, kwargs = requester.calls[0]
        self.assertEqual(method, "POST")
        self.assertEqual(
            url,
            "https://westus2.tts.speech.microsoft.com/cognitiveservices/v1",
        )
        self.assertEqual(
            kwargs["headers"]["X-Microsoft-OutputFormat"],
            "riff-48khz-16bit-mono-pcm",
        )
        self.assertIn(b"Hello Azure", kwargs["data"])

    def test_retries_throttled_requests_and_honors_retry_after(self):
        sleeps = []
        requester = RecordingRequester([
            FakeResponse(status_code=429, text="throttled", headers={"Retry-After": "0.25"}),
            FakeResponse(content=WAV_BYTES),
        ])
        engine = AzureSpeechEngine(
            subscription_key="azure-test-key",
            region="westus2",
            requests_per_minute=0,
            request_func=requester,
            sleep_func=sleeps.append,
        )

        audio = engine.generate_audio("Retry me", voice="en-US-GuyNeural")

        self.assertEqual(audio, WAV_BYTES)
        self.assertEqual(len(requester.calls), 2)
        self.assertEqual(sleeps, [0.25])

    def test_batch_keeps_chronological_order_and_reports_chunks(self):
        requester = RecordingRequester([
            FakeResponse(content=WAV_BYTES),
            FakeResponse(content=WAV_BYTES),
        ])
        engine = self.make_engine(requester)
        progress = []
        chunks = []

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = engine.generate_batch(
                segments=[{"speaker": "Narrator", "chunks": ["First", "Second"]}],
                voice_config={
                    "Narrator": {
                        "voice": "en-US-GuyNeural",
                        "lang_code": "en-US",
                        "extra": {"style": "narration-professional"},
                    }
                },
                output_dir=Path(temp_dir),
                progress_cb=lambda: progress.append(True),
                chunk_cb=lambda index, metadata, path: chunks.append((index, metadata, path)),
            )

            self.assertEqual([Path(path).name for path in paths], [
                "azure_chunk_000000.wav",
                "azure_chunk_000001.wav",
            ])
            self.assertTrue(all(Path(path).read_bytes() == WAV_BYTES for path in paths))

        self.assertEqual(len(progress), 2)
        self.assertEqual([entry[0] for entry in chunks], [0, 1])
        self.assertEqual([entry[1]["text"] for entry in chunks], ["First", "Second"])
        self.assertIn(b'narration-professional', requester.calls[0][2]["data"])

    def test_surfaces_credential_error_without_response_body(self):
        requester = RecordingRequester([
            FakeResponse(status_code=401, text="sensitive provider response")
        ])
        engine = self.make_engine(requester)

        with self.assertRaisesRegex(AzureSpeechError, "resource key or region") as caught:
            engine.generate_audio("Hello", voice="en-US-GuyNeural")
        self.assertNotIn("sensitive provider response", str(caught.exception))


if __name__ == "__main__":
    unittest.main()

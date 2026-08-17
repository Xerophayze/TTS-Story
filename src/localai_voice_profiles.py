"""Synchronize TTS-Story reference samples with LocalAI voice profiles."""
from __future__ import annotations

import hashlib
import json
import mimetypes
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import quote, unquote

import requests

from .localai_tts_client import normalize_localai_urls


TTS_STORY_VOICE_PREFIX = "tts-story://voice-prompts/"


class LocalAIVoiceProfileError(RuntimeError):
    """Raised when a TTS-Story sample cannot become a LocalAI profile."""


def build_tts_story_voice_reference(file_name: str) -> str:
    safe_name = Path(str(file_name or "")).name
    if not safe_name:
        raise LocalAIVoiceProfileError("The TTS-Story voice sample filename is missing.")
    return f"{TTS_STORY_VOICE_PREFIX}{quote(safe_name, safe='')}"


class LocalAIVoiceProfileManager:
    """Create each LocalAI profile once and reuse it across chunks and jobs."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        *,
        consent_confirmed: bool = False,
        timeout: int = 120,
        voice_prompt_dir: Path = Path("data/voice_prompts"),
        registry_path: Path = Path("data/chatterbox_voices.json"),
        mappings_path: Path = Path("data/localai_voice_profiles.json"),
        request_func: Callable[..., Any] = requests.request,
    ) -> None:
        self.server_root, _ = normalize_localai_urls(base_url)
        self.api_key = str(api_key or "").strip()
        self.consent_confirmed = bool(consent_confirmed)
        self.timeout = max(10, min(int(timeout or 120), 600))
        self.voice_prompt_dir = Path(voice_prompt_dir)
        self.registry_path = Path(registry_path)
        self.transcripts_path = self.voice_prompt_dir / "transcripts.json"
        self.mappings_path = Path(mappings_path)
        self._request = request_func
        self._lock = threading.RLock()
        self._resolved: Dict[str, str] = {}
        self._remote_voices: Optional[set[str]] = None

    @property
    def headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def resolve(self, voice: str) -> str:
        reference = str(voice or "").strip()
        if not reference.startswith(TTS_STORY_VOICE_PREFIX):
            return reference
        with self._lock:
            if reference in self._resolved:
                return self._resolved[reference]
            if not self.consent_confirmed:
                raise LocalAIVoiceProfileError(
                    "LocalAI voice-profile synchronization requires the rights/consent confirmation in Settings."
                )
            file_name = unquote(reference[len(TTS_STORY_VOICE_PREFIX):])
            if not file_name or Path(file_name).name != file_name:
                raise LocalAIVoiceProfileError("The selected TTS-Story voice reference is invalid.")
            audio_path = (self.voice_prompt_dir / file_name).resolve()
            prompt_root = self.voice_prompt_dir.resolve()
            if audio_path.parent != prompt_root or not audio_path.is_file():
                raise LocalAIVoiceProfileError(
                    f"The TTS-Story voice sample is missing: {file_name}"
                )
            transcript = self._transcript_for(audio_path)
            if not transcript:
                raise LocalAIVoiceProfileError(
                    f"The voice sample '{file_name}' needs an exact transcript before LocalAI can clone it. "
                    "Add the transcript in Available Voices → Voice Prompts → Edit."
                )
            metadata = self._metadata_for(file_name)
            language = str(metadata.get("language") or "en-US").strip() or "en-US"
            mapping_key = self._mapping_key(audio_path, transcript, language)
            mappings = self._load_mappings()
            mapped = mappings.get(mapping_key) if isinstance(mappings, dict) else None
            mapped_voice = str((mapped or {}).get("voice") or "").strip()
            if mapped_voice and mapped_voice in self._remote_voice_uris():
                self._resolved[reference] = mapped_voice
                return mapped_voice

            profile = self._create_profile(audio_path, transcript, metadata, language)
            voice_uri = str(profile.get("voice") or "").strip()
            profile_id = str(profile.get("id") or "").strip()
            if not voice_uri and profile_id:
                voice_uri = f"localai://voice-profiles/{profile_id}"
            if not voice_uri:
                raise LocalAIVoiceProfileError("LocalAI created the profile but returned no voice URI.")
            mappings[mapping_key] = {
                "voice": voice_uri,
                "profile_id": profile_id,
                "file_name": file_name,
                "name": metadata.get("name") or audio_path.stem,
                "server_root": self.server_root,
            }
            self._save_mappings(mappings)
            if self._remote_voices is not None:
                self._remote_voices.add(voice_uri)
            self._resolved[reference] = voice_uri
            return voice_uri

    def _transcript_for(self, audio_path: Path) -> str:
        try:
            payload = json.loads(self.transcripts_path.read_text(encoding="utf-8"))
            transcripts = payload.get("transcripts", {}) if isinstance(payload, dict) else {}
        except (OSError, UnicodeError, json.JSONDecodeError):
            return ""
        stat = audio_path.stat()
        key_data = f"{audio_path.name}:{stat.st_size}:{stat.st_mtime}"
        key = hashlib.md5(key_data.encode()).hexdigest()[:16]
        return str(transcripts.get(key) or "").strip()

    def _metadata_for(self, file_name: str) -> Dict[str, Any]:
        try:
            entries = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            entries = []
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict) and entry.get("file_name") == file_name:
                    return entry
        return {"name": Path(file_name).stem}

    def _mapping_key(self, audio_path: Path, transcript: str, language: str) -> str:
        digest = hashlib.sha256()
        digest.update(self.server_root.lower().encode())
        digest.update(b"\0")
        digest.update(audio_path.read_bytes())
        digest.update(b"\0")
        digest.update(transcript.encode())
        digest.update(b"\0")
        digest.update(language.encode())
        return digest.hexdigest()

    def _remote_voice_uris(self) -> set[str]:
        if self._remote_voices is not None:
            return self._remote_voices
        try:
            response = self._request(
                "GET", f"{self.server_root}/api/voice-profiles",
                headers=self.headers, timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise LocalAIVoiceProfileError(
                f"Unable to list LocalAI voice profiles: {exc}"
            ) from exc
        if not 200 <= int(getattr(response, "status_code", 0) or 0) < 300:
            raise LocalAIVoiceProfileError(
                f"LocalAI could not list voice profiles ({response.status_code})."
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise LocalAIVoiceProfileError("LocalAI returned an invalid voice-profile catalog.") from exc
        profiles = payload.get("data", payload.get("profiles", [])) if isinstance(payload, dict) else payload
        self._remote_voices = set()
        for profile in profiles or []:
            if not isinstance(profile, dict):
                continue
            voice_uri = str(profile.get("voice") or profile.get("uri") or "").strip()
            profile_id = str(profile.get("id") or "").strip()
            if not voice_uri and profile_id:
                voice_uri = f"localai://voice-profiles/{profile_id}"
            if voice_uri:
                self._remote_voices.add(voice_uri)
        return self._remote_voices

    def _create_profile(
        self, audio_path: Path, transcript: str, metadata: Dict[str, Any], language: str
    ) -> Dict[str, Any]:
        mime_type = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
        data = {
            "name": str(metadata.get("name") or audio_path.stem),
            "description": str(metadata.get("description") or "Synchronized from TTS-Story"),
            "language": language,
            "transcript": transcript,
            "consent_confirmed": "true",
        }
        try:
            with audio_path.open("rb") as handle:
                response = self._request(
                    "POST", f"{self.server_root}/api/voice-profiles",
                    headers=self.headers,
                    data=data,
                    files={"audio": (audio_path.name, handle, mime_type)},
                    timeout=self.timeout,
                )
        except (OSError, requests.RequestException) as exc:
            raise LocalAIVoiceProfileError(f"Unable to upload the voice sample to LocalAI: {exc}") from exc
        if int(getattr(response, "status_code", 0) or 0) != 201:
            detail = str(getattr(response, "text", "") or "").strip()[:300]
            raise LocalAIVoiceProfileError(
                f"LocalAI rejected the voice profile ({response.status_code}){': ' + detail if detail else '.'}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise LocalAIVoiceProfileError("LocalAI returned an invalid profile response.") from exc
        if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
            return payload["data"]
        return payload if isinstance(payload, dict) else {}

    def _load_mappings(self) -> Dict[str, Dict[str, Any]]:
        try:
            payload = json.loads(self.mappings_path.read_text(encoding="utf-8"))
            mappings = payload.get("mappings", {}) if isinstance(payload, dict) else {}
            return mappings if isinstance(mappings, dict) else {}
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}

    def _save_mappings(self, mappings: Dict[str, Dict[str, Any]]) -> None:
        self.mappings_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.mappings_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"mappings": mappings}, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(self.mappings_path)


__all__ = [
    "LocalAIVoiceProfileError", "LocalAIVoiceProfileManager",
    "TTS_STORY_VOICE_PREFIX", "build_tts_story_voice_reference",
]

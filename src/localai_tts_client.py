"""Discovery helpers for LocalAI TTS models and saved voice profiles."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple
from urllib.parse import urlsplit, urlunsplit

import requests


class LocalAITTSDiscoveryError(RuntimeError):
    """Raised when a LocalAI catalog cannot be discovered."""


def normalize_localai_urls(base_url: str) -> Tuple[str, str]:
    raw = str(base_url or "http://localhost:8080/v1").strip().rstrip("/")
    if not raw.startswith(("http://", "https://")):
        raise LocalAITTSDiscoveryError("The LocalAI URL must begin with http:// or https://.")
    parsed = urlsplit(raw)
    path = parsed.path.rstrip("/")
    if path.endswith("/audio/speech"):
        path = path[:-13].rstrip("/")
    if path.endswith("/v1"):
        root_path = path[:-3].rstrip("/")
        v1_path = path
    else:
        root_path = path
        v1_path = f"{path}/v1"
    server_root = urlunsplit((parsed.scheme, parsed.netloc, root_path, "", "")).rstrip("/")
    v1_root = urlunsplit((parsed.scheme, parsed.netloc, v1_path, "", "")).rstrip("/")
    return server_root, v1_root


def discover_localai_tts_catalog(
    base_url: str,
    api_key: str = "",
    *,
    timeout: int = 20,
    request_func: Callable[..., Any] = requests.request,
) -> Dict[str, Any]:
    server_root, v1_root = normalize_localai_urls(base_url)
    headers = {"Accept": "application/json"}
    if str(api_key or "").strip():
        headers["Authorization"] = f"Bearer {str(api_key).strip()}"

    def get_json(url: str, *, optional: bool = False) -> Any:
        try:
            response = request_func("GET", url, headers=headers, timeout=max(3, min(int(timeout), 120)))
        except requests.RequestException as exc:
            if optional:
                return None
            raise LocalAITTSDiscoveryError(f"Unable to reach LocalAI at {server_root}: {exc}") from exc
        if not 200 <= int(getattr(response, "status_code", 0) or 0) < 300:
            if optional:
                return None
            detail = str(getattr(response, "text", "") or "").strip()[:240]
            raise LocalAITTSDiscoveryError(
                f"LocalAI catalog request failed ({response.status_code}){': ' + detail if detail else '.'}"
            )
        try:
            return response.json()
        except ValueError as exc:
            if optional:
                return None
            raise LocalAITTSDiscoveryError("LocalAI returned an invalid catalog response.") from exc

    metadata = get_json(f"{server_root}/.well-known/localai.json", optional=True) or {}
    server_capabilities = metadata.get("capabilities") if isinstance(metadata, dict) else {}
    model_payload = get_json(f"{v1_root}/models/capabilities", optional=True)
    used_capabilities = model_payload is not None
    if model_payload is None:
        model_payload = get_json(f"{v1_root}/models")
    raw_models = model_payload.get("data", model_payload if isinstance(model_payload, list) else [])
    models: List[Dict[str, Any]] = []
    for entry in raw_models or []:
        if isinstance(entry, str):
            entry = {"id": entry}
        if not isinstance(entry, dict):
            continue
        model_capabilities = [str(value).lower() for value in entry.get("capabilities") or []]
        output_modalities = [str(value).lower() for value in entry.get("output_modalities") or []]
        if used_capabilities and "tts" not in model_capabilities and "audio" not in output_modalities:
            continue
        model_id = str(entry.get("id") or entry.get("model") or entry.get("name") or "").strip()
        if model_id:
            model_config = get_json(
                f"{server_root}/api/models/config-json/{model_id}", optional=True
            ) or {}
            tts_config = model_config.get("tts") if isinstance(model_config, dict) else {}
            voice_cloning = (
                bool(tts_config.get("voice_cloning"))
                if isinstance(tts_config, dict) and "voice_cloning" in tts_config else None
            )
            models.append({
                "model_id": model_id,
                "name": str(entry.get("name") or model_id),
                "capabilities": model_capabilities,
                "voice_cloning": voice_cloning,
            })

    profiles_payload = get_json(f"{server_root}/api/voice-profiles", optional=True)
    raw_profiles = (
        profiles_payload.get("profiles", profiles_payload.get("data", []))
        if isinstance(profiles_payload, dict) else (profiles_payload or [])
    )
    voices: List[Dict[str, Any]] = []
    for profile in raw_profiles or []:
        if not isinstance(profile, dict):
            continue
        profile_id = str(profile.get("id") or "").strip()
        voice_id = str(profile.get("voice") or profile.get("uri") or "").strip()
        if not voice_id and profile_id:
            voice_id = f"localai://voice-profiles/{profile_id}"
        if not voice_id:
            continue
        voices.append({
            "short_name": voice_id,
            "voice_id": voice_id,
            "display_name": str(profile.get("name") or profile_id or voice_id),
            "locale": str(profile.get("language") or profile.get("locale") or ""),
            "gender": str(profile.get("gender") or ""),
            "category": "LocalAI profile",
            "profile_id": profile_id,
            "duration_ms": profile.get("duration_ms"),
        })

    return {
        "server_root": server_root,
        "v1_root": v1_root,
        "version": metadata.get("version") or metadata.get("name") or "",
        "voice_profiles_supported": bool(
            isinstance(server_capabilities, dict) and server_capabilities.get("voice_profiles")
        ),
        "models": models,
        "voices": voices,
    }


__all__ = ["LocalAITTSDiscoveryError", "discover_localai_tts_catalog", "normalize_localai_urls"]

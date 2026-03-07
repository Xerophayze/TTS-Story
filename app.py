"""
TTS-Story - Web-based TTS application
"""
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import base64
import asyncio
import webbrowser
import copy
import inspect
import hashlib
import io
import json
import concurrent.futures
import logging
import math
import mimetypes
import os
import queue
import re
import shutil
import sqlite3
import stat
import subprocess
import tempfile
import threading
import time
import zipfile
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple
from werkzeug.utils import secure_filename
import soundfile as sf

from src.audio_effects import VoiceFXSettings
from src.audio_merger import AudioMerger
from src.custom_voice_store import (
    CUSTOM_CODE_PREFIX,
    delete_custom_voice,
    get_custom_voice,
    get_custom_voice_by_code,
    list_custom_voice_entries,
    replace_custom_voice,
    save_custom_voice,
)
from src.document_extractor import extract_text_from_file, get_supported_formats
from src.gemini_processor import GeminiProcessor, GeminiProcessorError
from src.local_llm_processor import (
    DEFAULT_LOCAL_LLM_BASE_URLS,
    LocalLLMProcessor,
    LocalLLMProcessorError,
    LLM_PROVIDER_LMSTUDIO,
    LLM_PROVIDER_OLLAMA,
)
from src.replicate_api import ReplicateAPI
from src.text_processor import TextProcessor
from src.engines import TtsEngineBase
from src.engines.chatterbox_turbo_local_engine import (
    CHATTERBOX_TURBO_AVAILABLE,
)
from src.engines.voxcpm_local_engine import VOXCPM_AVAILABLE
from src.engines.qwen3_custom_voice_engine import QWEN3_AVAILABLE
from src.engines.qwen3_voice_clone_engine import QWEN3_AVAILABLE as QWEN3_CLONE_AVAILABLE
from src.engines.pocket_tts_engine import POCKET_TTS_AVAILABLE
from src.engines.kitten_tts_engine import (
    KITTEN_TTS_AVAILABLE,
    KITTEN_TTS_BUILTIN_VOICES,
    KITTEN_TTS_DEFAULT_MODEL,
)
from src.engines.index_tts_engine import (
    INDEX_TTS_AVAILABLE,
    INDEX_TTS_UNAVAILABLE_REASON,
    INDEX_TTS_SAMPLE_RATE,
)
from src.engines.chatterbox_turbo_replicate_engine import (
    DEFAULT_CHATTERBOX_TURBO_REPLICATE_MODEL,
    DEFAULT_CHATTERBOX_TURBO_REPLICATE_VOICE,
)
from src.tts_engine import (
    TTSEngine,
    KOKORO_AVAILABLE,
    DEFAULT_SAMPLE_RATE,
    get_engine,
    AVAILABLE_ENGINES,
)
from src.voice_manager import VoiceManager
from src.voice_sample_generator import generate_voice_samples

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
CONFIG_FILE = "config.json"
# Use Flask's static folder so paths stay correct even when the server is launched
# from a different working directory.
OUTPUT_DIR = Path(app.static_folder) / "audio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DATA_DIR = Path("data/jobs")
JOBS_ARCHIVE_DIR = JOBS_DATA_DIR / "archive"
JOBS_DB_PATH = JOBS_DATA_DIR / "jobs.db"
JOBS_DATA_DIR.mkdir(parents=True, exist_ok=True)
JOBS_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
VOICE_PROMPT_DIR = Path("data/voice_prompts")
VOICE_PROMPT_DIR.mkdir(parents=True, exist_ok=True)
VOICE_PROMPT_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
CHATTERBOX_VOICE_REGISTRY = Path("data/chatterbox_voices.json")
EXTERNAL_VOICES_ARCHIVE_FILE = Path("data/external_voice_archives.json")
JOB_METADATA_FILENAME = "metadata.json"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
DEFAULT_LLM_PROVIDER = "gemini"
LIBRARY_CACHE_TTL = 5  # seconds
MIN_CHATTERBOX_PROMPT_SECONDS = 5.0
DEFAULT_CONFIG = {
    "replicate_api_key": "",
    "chunk_size": 500,
    "kokoro_chunk_size": 500,
    "chatterbox_turbo_local_chunk_size": 450,
    "voxcpm_chunk_size": 550,
    "qwen3_chunk_size": 500,
    "pocket_tts_chunk_size": 450,
    "sample_rate": 24000,
    "speed": 1.0,
    "output_format": "mp3",
    "output_bitrate_kbps": 128,
    "crossfade_duration": 0.1,
    "intro_silence_ms": 0,
    "inter_chunk_silence_ms": 0,
    "gemini_api_key": "",
    "gemini_model": DEFAULT_GEMINI_MODEL,
    "gemini_prompt": "",
    "gemini_prompt_presets": [],
    "gemini_speaker_profile_prompt": "",
    "llm_provider": DEFAULT_LLM_PROVIDER,
    "llm_local_provider": LLM_PROVIDER_LMSTUDIO,
    "llm_local_base_url": DEFAULT_LOCAL_LLM_BASE_URLS[LLM_PROVIDER_LMSTUDIO],
    "llm_local_model": "",
    "llm_local_api_key": "",
    "llm_local_timeout": 120,
    "llm_local_temperature": 0.2,
    "llm_local_top_p": 1.0,
    "llm_local_top_k": 0,
    "llm_local_repeat_penalty": 1.0,
    "llm_local_max_tokens": 0,
    "llm_local_disable_reasoning": False,
    "llm_gemini_chunk_size": 500,
    "llm_local_chunk_size": 500,
    "llm_gemini_chunk_chapters": True,
    "llm_local_chunk_chapters": True,
    "tts_engine": "kokoro",
    "chatterbox_turbo_local_default_prompt": "",
    "chatterbox_turbo_local_temperature": 0.8,
    "chatterbox_turbo_local_top_p": 0.95,
    "chatterbox_turbo_local_top_k": 1000,
    "chatterbox_turbo_local_repetition_penalty": 1.2,
    "chatterbox_turbo_local_cfg_weight": 0.0,
    "chatterbox_turbo_local_exaggeration": 0.0,
    "chatterbox_turbo_local_norm_loudness": True,
    "chatterbox_turbo_local_prompt_norm_loudness": True,
    "chatterbox_turbo_local_device": "auto",
    "chatterbox_turbo_replicate_api_token": "",
    "chatterbox_turbo_replicate_model": DEFAULT_CHATTERBOX_TURBO_REPLICATE_MODEL,
    "chatterbox_turbo_replicate_voice": DEFAULT_CHATTERBOX_TURBO_REPLICATE_VOICE,
    "chatterbox_turbo_replicate_temperature": 0.8,
    "chatterbox_turbo_replicate_top_p": 0.95,
    "chatterbox_turbo_replicate_top_k": 1000,
    "chatterbox_turbo_replicate_repetition_penalty": 1.2,
    "chatterbox_turbo_replicate_seed": None,
    "voxcpm_local_model_id": "openbmb/VoxCPM1.5",
    "voxcpm_local_device": "auto",
    "voxcpm_local_default_prompt": "",
    "voxcpm_local_default_prompt_text": "",
    "voxcpm_local_cfg_value": 2.0,
    "voxcpm_local_inference_timesteps": 32,
    "voxcpm_local_normalize": True,  # Enable text normalization for numbers/abbreviations
    "voxcpm_local_denoise": False,
    "qwen3_custom_model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "qwen3_custom_device": "auto",
    "qwen3_custom_dtype": "bfloat16",
    "qwen3_custom_attn_implementation": "flash_attention_2",
    "qwen3_custom_default_language": "Auto",
    "qwen3_custom_default_instruct": "",
    "qwen3_clone_model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "qwen3_clone_device": "auto",
    "qwen3_clone_dtype": "bfloat16",
    "qwen3_clone_attn_implementation": "flash_attention_2",
    "qwen3_clone_default_language": "Auto",
    "qwen3_clone_default_prompt": "",
    "qwen3_clone_default_prompt_text": "",
    "qwen3_voice_design_model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "pocket_tts_model_variant": "b6369a24",
    "pocket_tts_temp": 0.7,
    "pocket_tts_lsd_decode_steps": 1,
    "pocket_tts_noise_clamp": None,
    "pocket_tts_eos_threshold": -4.0,
    "pocket_tts_default_prompt": "",
    "pocket_tts_prompt_truncate": False,
    "pocket_tts_num_threads": None,
    "pocket_tts_interop_threads": None,
    "kitten_tts_model_id": "KittenML/kitten-tts-mini-0.8",
    "kitten_tts_default_voice": "Jasper",
    "kitten_tts_chunk_size": 300,
    "index_tts_model_version": "IndexTTS-2",
    "index_tts_use_fp16": True,
    "index_tts_use_deepspeed": False,
    "index_tts_use_torch_compile": True,
    "index_tts_use_accel": True,
    "index_tts_num_beams": 1,
    "index_tts_diffusion_steps": 12,
    "index_tts_temperature": 0.8,
    "index_tts_top_p": 0.8,
    "index_tts_top_k": 30,
    "index_tts_repetition_penalty": 10.0,
    "index_tts_max_mel_tokens": 1500,
    "index_tts_max_text_tokens_per_segment": 120,
    "index_tts_device": "auto",
    "index_tts_default_prompt": "",
    "index_tts_chunk_size": 400,
    "parallel_chunks": 3,
    "group_chunks_by_speaker": False,
    "cleanup_vram_after_job": False,
}

POCKET_TTS_PRESET_VOICES = [
    "alba",
    "marius",
    "javert",
    "jean",
    "fantine",
    "cosette",
    "eponine",
    "azelma",
]

CHATTERBOX_TURBO_LOCAL_SETTING_KEYS = {
    "chatterbox_turbo_local_default_prompt",
    "chatterbox_turbo_local_temperature",
    "chatterbox_turbo_local_top_p",
    "chatterbox_turbo_local_top_k",
    "chatterbox_turbo_local_repetition_penalty",
    "chatterbox_turbo_local_cfg_weight",
    "chatterbox_turbo_local_exaggeration",
    "chatterbox_turbo_local_norm_loudness",
    "chatterbox_turbo_local_prompt_norm_loudness",
    "chatterbox_turbo_local_device",
}
VOXCPM_LOCAL_SETTING_KEYS = {
    "voxcpm_local_model_id",
    "voxcpm_local_device",
    "voxcpm_local_default_prompt",
    "voxcpm_local_default_prompt_text",
    "voxcpm_local_cfg_value",
    "voxcpm_local_inference_timesteps",
    "voxcpm_local_normalize",
    "voxcpm_local_denoise",
}
QWEN3_CUSTOM_SETTING_KEYS = {
    "qwen3_custom_model_id",
    "qwen3_custom_device",
    "qwen3_custom_dtype",
    "qwen3_custom_attn_implementation",
    "qwen3_custom_default_language",
    "qwen3_custom_default_instruct",
}
QWEN3_CLONE_SETTING_KEYS = {
    "qwen3_clone_model_id",
    "qwen3_clone_device",
    "qwen3_clone_dtype",
    "qwen3_clone_attn_implementation",
    "qwen3_clone_default_language",
    "qwen3_clone_default_prompt",
    "qwen3_clone_default_prompt_text",
}
POCKET_TTS_SETTING_KEYS = {
    "pocket_tts_model_variant",
    "pocket_tts_temp",
    "pocket_tts_lsd_decode_steps",
    "pocket_tts_noise_clamp",
    "pocket_tts_eos_threshold",
    "pocket_tts_default_prompt",
    "pocket_tts_prompt_truncate",
    "pocket_tts_num_threads",
    "pocket_tts_interop_threads",
}
CHATTERBOX_TURBO_LOCAL_OPTION_ALIASES = {
    "default_prompt": "chatterbox_turbo_local_default_prompt",
    "prompt": "chatterbox_turbo_local_default_prompt",
    "temperature": "chatterbox_turbo_local_temperature",
    "top_p": "chatterbox_turbo_local_top_p",
    "top_k": "chatterbox_turbo_local_top_k",
    "repetition_penalty": "chatterbox_turbo_local_repetition_penalty",
    "cfg_weight": "chatterbox_turbo_local_cfg_weight",
    "exaggeration": "chatterbox_turbo_local_exaggeration",
    "norm_loudness": "chatterbox_turbo_local_norm_loudness",
    "prompt_norm_loudness": "chatterbox_turbo_local_prompt_norm_loudness",
    "device": "chatterbox_turbo_local_device",
}
VOXCPM_LOCAL_OPTION_ALIASES = {
    "model": "voxcpm_local_model_id",
    "model_id": "voxcpm_local_model_id",
    "device": "voxcpm_local_device",
    "default_prompt": "voxcpm_local_default_prompt",
    "prompt_text": "voxcpm_local_default_prompt_text",
    "cfg_value": "voxcpm_local_cfg_value",
    "inference_timesteps": "voxcpm_local_inference_timesteps",
    "normalize": "voxcpm_local_normalize",
    "denoise": "voxcpm_local_denoise",
}
QWEN3_CUSTOM_OPTION_ALIASES = {
    "model": "qwen3_custom_model_id",
    "model_id": "qwen3_custom_model_id",
    "device": "qwen3_custom_device",
    "dtype": "qwen3_custom_dtype",
    "attn_implementation": "qwen3_custom_attn_implementation",
    "default_language": "qwen3_custom_default_language",
    "default_instruct": "qwen3_custom_default_instruct",
}
QWEN3_CLONE_OPTION_ALIASES = {
    "model": "qwen3_clone_model_id",
    "model_id": "qwen3_clone_model_id",
    "device": "qwen3_clone_device",
    "dtype": "qwen3_clone_dtype",
    "attn_implementation": "qwen3_clone_attn_implementation",
    "default_language": "qwen3_clone_default_language",
    "default_prompt": "qwen3_clone_default_prompt",
    "prompt_text": "qwen3_clone_default_prompt_text",
}
POCKET_TTS_OPTION_ALIASES = {
    "model": "pocket_tts_model_variant",
    "model_variant": "pocket_tts_model_variant",
    "temp": "pocket_tts_temp",
    "temperature": "pocket_tts_temp",
    "lsd_decode_steps": "pocket_tts_lsd_decode_steps",
    "noise_clamp": "pocket_tts_noise_clamp",
    "eos_threshold": "pocket_tts_eos_threshold",
    "default_prompt": "pocket_tts_default_prompt",
    "prompt": "pocket_tts_default_prompt",
    "prompt_truncate": "pocket_tts_prompt_truncate",
    "num_threads": "pocket_tts_num_threads",
    "interop_threads": "pocket_tts_interop_threads",
}
CHATTERBOX_TURBO_LOCAL_BOOLEAN_SETTINGS = {
    "chatterbox_turbo_local_norm_loudness",
    "chatterbox_turbo_local_prompt_norm_loudness",
}
CHATTERBOX_TURBO_LOCAL_FLOAT_SETTINGS = {
    "chatterbox_turbo_local_temperature": (0.05, 2.0, 0.8),
    "chatterbox_turbo_local_top_p": (0.1, 1.0, 0.95),
    "chatterbox_turbo_local_repetition_penalty": (1.0, 2.0, 1.2),
    "chatterbox_turbo_local_cfg_weight": (0.0, 2.0, 0.0),
    "chatterbox_turbo_local_exaggeration": (0.0, 2.0, 0.0),
}
CHATTERBOX_TURBO_LOCAL_INT_SETTINGS = {
    "chatterbox_turbo_local_top_k": (1, 4000, 1000),
    "chatterbox_turbo_local_chunk_size": (100, 1000, 450),
}
VOXCPM_LOCAL_BOOLEAN_SETTINGS = {
    "voxcpm_local_normalize",
    "voxcpm_local_denoise",
}
VOXCPM_LOCAL_FLOAT_SETTINGS = {
    "voxcpm_local_cfg_value": (0.1, 5.0, 2.0),
}
VOXCPM_LOCAL_INT_SETTINGS = {
    "voxcpm_local_inference_timesteps": (10, 100, 32),
}

CHATTERBOX_TURBO_REPLICATE_SETTING_KEYS = {
    "chatterbox_turbo_replicate_api_token",
    "chatterbox_turbo_replicate_model",
    "chatterbox_turbo_replicate_voice",
    "chatterbox_turbo_replicate_temperature",
    "chatterbox_turbo_replicate_top_p",
    "chatterbox_turbo_replicate_top_k",
    "chatterbox_turbo_replicate_repetition_penalty",
    "chatterbox_turbo_replicate_seed",
}
CHATTERBOX_TURBO_REPLICATE_OPTION_ALIASES = {
    "api_token": "chatterbox_turbo_replicate_api_token",
    "token": "chatterbox_turbo_replicate_api_token",
    "model": "chatterbox_turbo_replicate_model",
    "voice": "chatterbox_turbo_replicate_voice",
    "temperature": "chatterbox_turbo_replicate_temperature",
    "top_p": "chatterbox_turbo_replicate_top_p",
    "top_k": "chatterbox_turbo_replicate_top_k",
    "repetition_penalty": "chatterbox_turbo_replicate_repetition_penalty",
    "seed": "chatterbox_turbo_replicate_seed",
}
CHATTERBOX_TURBO_REPLICATE_FLOAT_SETTINGS = {
    "chatterbox_turbo_replicate_temperature": (0.05, 2.0, 0.8),
    "chatterbox_turbo_replicate_top_p": (0.1, 1.0, 0.95),
    "chatterbox_turbo_replicate_repetition_penalty": (1.0, 2.0, 1.2),
}
CHATTERBOX_TURBO_REPLICATE_INT_SETTINGS = {
    "chatterbox_turbo_replicate_top_k": (1, 4000, 1000),
}


def _measure_audio_duration(audio_path: Path) -> Optional[float]:
    """
    Return the duration of an audio file in seconds, or None if it cannot be determined.
    """
    try:
        info = sf.info(str(audio_path))
        if not info.frames or not info.samplerate:
            return None

        duration = info.frames / float(info.samplerate)
        return duration if duration > 0 else None
    except Exception as exc:  # pragma: no cover - logging only
        logger.warning("Unable to measure duration for %s: %s", audio_path, exc)
        return None


def _is_external_voice_id(voice_id: str) -> bool:
    return bool(voice_id) and voice_id.startswith("external:")


def _strip_external_voice_id(voice_id: str) -> str:
    return voice_id.replace("external:", "", 1)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return bool(value)


def _coerce_int(
    value: Any,
    *,
    minimum: int = 1,
    maximum: Optional[int] = None,
    fallback: int = 1,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    if parsed < minimum:
        parsed = minimum
    if maximum is not None and parsed > maximum:
        parsed = maximum
    return parsed


def _coerce_float(
    value: Any,
    *,
    minimum: float,
    maximum: float,
    fallback: float,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = fallback
    if parsed < minimum:
        parsed = minimum


def _normalize_engine_options(engine_name: str, options: Dict[str, Any]) -> Dict[str, Any]:
    if engine_name == "chatterbox_turbo_local":
        return _normalize_chatterbox_turbo_local_options(options)
    if engine_name == "chatterbox_turbo_replicate":
        return _normalize_chatterbox_turbo_replicate_options(options)
    if engine_name == "voxcpm_local":
        return _normalize_voxcpm_local_options(options)
    if engine_name == "qwen3_custom":
        return _normalize_qwen3_custom_options(options)
    if engine_name == "qwen3_clone":
        return _normalize_qwen3_clone_options(options)
    if engine_name in {"pocket_tts", "pocket_tts_preset"}:
        return _normalize_pocket_tts_options(options)
    if engine_name == "kitten_tts":
        return _normalize_kitten_tts_options(options)
    if engine_name == "index_tts":
        return _normalize_index_tts_options(options)
    return {}


def _normalize_chatterbox_turbo_local_options(options: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for raw_key, value in options.items():
        if raw_key is None:
            continue
        key = str(raw_key).strip().lower()
        canonical = CHATTERBOX_TURBO_LOCAL_OPTION_ALIASES.get(key)
        if not canonical and key in CHATTERBOX_TURBO_LOCAL_SETTING_KEYS:
            canonical = key
        if canonical and canonical in CHATTERBOX_TURBO_LOCAL_SETTING_KEYS:
            normalized[canonical] = value

    result: Dict[str, Any] = {}
    for key, value in normalized.items():
        if key in CHATTERBOX_TURBO_LOCAL_BOOLEAN_SETTINGS:
            result[key] = _coerce_bool(value)
            continue
        if key in CHATTERBOX_TURBO_LOCAL_FLOAT_SETTINGS:
            minimum, maximum, fallback = CHATTERBOX_TURBO_LOCAL_FLOAT_SETTINGS[key]
            result[key] = _coerce_float(value, minimum=minimum, maximum=maximum, fallback=fallback)
            continue
        if key in CHATTERBOX_TURBO_LOCAL_INT_SETTINGS:
            minimum, maximum, fallback = CHATTERBOX_TURBO_LOCAL_INT_SETTINGS[key]
            result[key] = _coerce_int(value, minimum=minimum, maximum=maximum, fallback=fallback)
            continue
        result[key] = (value or "").strip() if isinstance(value, str) else (value or "")
    return result


def _normalize_pocket_tts_options(options: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for raw_key, value in options.items():
        if raw_key is None:
            continue
        key = str(raw_key).strip().lower()
        canonical = POCKET_TTS_OPTION_ALIASES.get(key)
        if not canonical and key in POCKET_TTS_SETTING_KEYS:
            canonical = key
        if canonical and canonical in POCKET_TTS_SETTING_KEYS:
            normalized[canonical] = value

    result: Dict[str, Any] = {}
    for key, value in normalized.items():
        if key in {"pocket_tts_temp", "pocket_tts_noise_clamp", "pocket_tts_eos_threshold"}:
            result[key] = None if value in (None, "") else _coerce_float(
                value,
                minimum=-10.0,
                maximum=10.0,
                fallback=DEFAULT_CONFIG.get(key),
            )
            continue
        if key == "pocket_tts_lsd_decode_steps":
            result[key] = _coerce_int(value, minimum=1, maximum=8, fallback=DEFAULT_CONFIG.get(key, 1))
            continue
        if key == "pocket_tts_prompt_truncate":
            result[key] = _coerce_bool(value)
            continue
        if key in {"pocket_tts_num_threads", "pocket_tts_interop_threads"}:
            result[key] = None if value in (None, "") else _coerce_int(
                value,
                minimum=1,
                maximum=64,
                fallback=DEFAULT_CONFIG.get(key),
            )
            continue
        result[key] = (value or "").strip() if isinstance(value, str) else value
    return result


def _normalize_kitten_tts_options(options: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for raw_key, value in options.items():
        if raw_key is None:
            continue
        key = str(raw_key).strip().lower()
        if key == "kitten_tts_model_id":
            result[key] = (value or KITTEN_TTS_DEFAULT_MODEL).strip()
        elif key == "kitten_tts_default_voice":
            v = (value or "Jasper").strip()
            result[key] = v if v in KITTEN_TTS_BUILTIN_VOICES else "Jasper"
        elif key == "kitten_tts_chunk_size":
            result[key] = _coerce_int(value, minimum=50, maximum=600, fallback=300)
    return result


def _normalize_index_tts_options(options: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for raw_key, value in options.items():
        if raw_key is None:
            continue
        key = str(raw_key).strip().lower()
        if key == "index_tts_model_version":
            v = (value or "IndexTTS-2").strip()
            result[key] = v if v in {"IndexTTS-2", "IndexTTS-1.5", "IndexTTS"} else "IndexTTS-2"
        elif key == "index_tts_use_fp16":
            result[key] = _coerce_bool(value)
        elif key == "index_tts_use_deepspeed":
            result[key] = _coerce_bool(value)
        elif key == "index_tts_use_torch_compile":
            result[key] = _coerce_bool(value)
        elif key == "index_tts_use_accel":
            result[key] = _coerce_bool(value)
        elif key == "index_tts_num_beams":
            result[key] = max(1, int(value or 1))
        elif key == "index_tts_diffusion_steps":
            result[key] = max(1, int(value or 25))
        elif key == "index_tts_temperature":
            result[key] = max(0.01, float(value or 0.8))
        elif key == "index_tts_top_p":
            result[key] = max(0.01, min(1.0, float(value or 0.8)))
        elif key == "index_tts_top_k":
            result[key] = max(1, int(value or 30))
        elif key == "index_tts_repetition_penalty":
            result[key] = max(1.0, float(value or 10.0))
        elif key == "index_tts_max_mel_tokens":
            result[key] = max(100, int(value or 1500))
        elif key == "index_tts_max_text_tokens_per_segment":
            result[key] = max(20, int(value or 120))
        elif key == "index_tts_device":
            v = (value or "auto").strip().lower()
            result[key] = v if v else "auto"
        elif key == "index_tts_default_prompt":
            result[key] = (value or "").strip()
        elif key == "index_tts_chunk_size":
            result[key] = _coerce_int(value, minimum=100, maximum=1000, fallback=400)
    return result


def _normalize_qwen3_clone_options(options: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for raw_key, value in options.items():
        if raw_key is None:
            continue
        key = str(raw_key).strip().lower()
        canonical = QWEN3_CLONE_OPTION_ALIASES.get(key)
        if not canonical and key in QWEN3_CLONE_SETTING_KEYS:
            canonical = key
        if canonical and canonical in QWEN3_CLONE_SETTING_KEYS:
            normalized[canonical] = value

    result: Dict[str, Any] = {}
    for key in QWEN3_CLONE_SETTING_KEYS:
        if key in normalized:
            result[key] = normalized[key]
    return result


def _normalize_qwen3_custom_options(options: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for raw_key, value in options.items():
        if raw_key is None:
            continue
        key = str(raw_key).strip().lower()
        canonical = QWEN3_CUSTOM_OPTION_ALIASES.get(key)
        if not canonical and key in QWEN3_CUSTOM_SETTING_KEYS:
            canonical = key
        if canonical and canonical in QWEN3_CUSTOM_SETTING_KEYS:
            normalized[canonical] = value

    result: Dict[str, Any] = {}
    for key, value in normalized.items():
        result[key] = (value or "").strip() if isinstance(value, str) else (value or "")
    return result


def _normalize_voxcpm_local_options(options: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for raw_key, value in options.items():
        if raw_key is None:
            continue
        key = str(raw_key).strip().lower()
        canonical = VOXCPM_LOCAL_OPTION_ALIASES.get(key)
        if not canonical and key in VOXCPM_LOCAL_SETTING_KEYS:
            canonical = key
        if canonical and canonical in VOXCPM_LOCAL_SETTING_KEYS:
            normalized[canonical] = value

    result: Dict[str, Any] = {}
    for key, value in normalized.items():
        if key in VOXCPM_LOCAL_BOOLEAN_SETTINGS:
            result[key] = _coerce_bool(value)
            continue
        if key in VOXCPM_LOCAL_FLOAT_SETTINGS:
            minimum, maximum, fallback = VOXCPM_LOCAL_FLOAT_SETTINGS[key]
            result[key] = _coerce_float(value, minimum=minimum, maximum=maximum, fallback=fallback)
            continue
        if key in VOXCPM_LOCAL_INT_SETTINGS:
            minimum, maximum, fallback = VOXCPM_LOCAL_INT_SETTINGS[key]
            result[key] = _coerce_int(value, minimum=minimum, maximum=maximum, fallback=fallback)
            continue
        result[key] = (value or "").strip() if isinstance(value, str) else (value or "")
    return result


def _normalize_chatterbox_turbo_replicate_options(options: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for raw_key, value in options.items():
        if raw_key is None:
            continue
        key = str(raw_key).strip().lower()
        canonical = CHATTERBOX_TURBO_REPLICATE_OPTION_ALIASES.get(key)
        if not canonical and key in CHATTERBOX_TURBO_REPLICATE_SETTING_KEYS:
            canonical = key
        if canonical and canonical in CHATTERBOX_TURBO_REPLICATE_SETTING_KEYS:
            normalized[canonical] = value

    result: Dict[str, Any] = {}
    for key, value in normalized.items():
        if key in CHATTERBOX_TURBO_REPLICATE_FLOAT_SETTINGS:
            minimum, maximum, fallback = CHATTERBOX_TURBO_REPLICATE_FLOAT_SETTINGS[key]
            result[key] = _coerce_float(value, minimum=minimum, maximum=maximum, fallback=fallback)
            continue
        if key in CHATTERBOX_TURBO_REPLICATE_INT_SETTINGS:
            minimum, maximum, fallback = CHATTERBOX_TURBO_REPLICATE_INT_SETTINGS[key]
            result[key] = _coerce_int(value, minimum=minimum, maximum=maximum, fallback=fallback)
            continue
        if key == "chatterbox_turbo_replicate_seed":
            const_value = value
            if isinstance(const_value, str):
                const_value = const_value.strip()
            if const_value not in (None, ""):
                try:
                    result[key] = int(const_value)
                except (TypeError, ValueError):
                    pass
            continue
        result[key] = (value or "").strip() if isinstance(value, str) else (value or "")
    return result


def _apply_engine_option_overrides(config: Dict[str, Any], engine_name: str, options: Optional[Dict[str, Any]]):
    overrides = _normalize_engine_options(engine_name, options or {})
    config.update(overrides)
# Allow headings like [narrator]\nChapter 1 or Chapter 1 without tags.
BOOK_HEADING_PATTERN = re.compile(
    r'^\s*(?:\[[^\]]+\]\s*)*(book\b[^\n\r]*)$',
    re.IGNORECASE | re.MULTILINE
)

SECTION_HEADING_KEYWORDS = [
    "chapter",
    "section",
    "letter",
    "part",
    "prologue",
    "epilogue",
]

def _normalize_custom_headings(value: Optional[Any]) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        raw_values = value
    else:
        raw_values = str(value).split(",")
    normalized = []
    for entry in raw_values:
        if not entry:
            continue
        candidate = str(entry).strip()
        if candidate:
            normalized.append(candidate)
    return normalized


def _keyword_to_regex(keyword: str) -> str:
    escaped = re.escape(keyword.strip())
    if not escaped:
        return ""
    return escaped.replace(r"\ ", r"\s+")


def _clean_heading_text(value: Optional[str]) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"\[[^\]]+\]", "", str(value))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _build_section_heading_pattern(custom_heading: Optional[Any] = None) -> re.Pattern:
    keywords = list(SECTION_HEADING_KEYWORDS)
    for custom in _normalize_custom_headings(custom_heading):
        lowered = custom.lower()
        if lowered not in keywords:
            keywords.append(lowered)
    keyword_regex = "|".join(filter(None, (_keyword_to_regex(word) for word in keywords)))
    if not keyword_regex:
        keyword_regex = "chapter"
    return re.compile(
        rf'^\s*(?:\[[^\]]+\]\s*)*(({keyword_regex})\b[^\n\r]*)$',
        re.IGNORECASE | re.MULTILINE
    )

# Exceptions
class JobCancelled(Exception):
    """Raised when a job is cancelled mid-processing."""


class JobPaused(Exception):
    """Raised when a job is paused after completing a chunk."""


# Global state
jobs = {}  # Track all jobs (queued, processing, completed)
job_queue = queue.Queue()  # Thread-safe job queue
current_job_id = None  # Currently processing job
cancel_flags = {}  # Cancellation flags for jobs
cancel_events = {}  # Cancellation events for immediate job aborts
pause_flags = {}  # Pause flags for jobs
queue_lock = threading.Lock()  # Lock for thread-safe operations
worker_thread = None  # Background worker thread
tts_engine_instances: Dict[str, TtsEngineBase] = {}
engine_config_signatures: Dict[str, str] = {}
tts_engine_lock = threading.Lock()
# Lock to prevent concurrent GPU inference across all TTS operations
# This prevents GPU contention and "badcase" retry loops with VoxCPM
gpu_inference_lock = threading.Lock()
# Use max_workers=1 to prevent parallel GPU inference which causes contention
# and "badcase" retry loops with VoxCPM and other GPU-based engines
chunk_regen_executor = ThreadPoolExecutor(max_workers=1)
qwen3_voice_design_model = None
qwen3_voice_design_signature = None
library_cache = {
    "items": None,
    "timestamp": 0.0,
}


def _job_dir_from_entry(job_id: str, job_entry: Dict[str, Any]) -> Path:
    directory = job_entry.get("job_dir")
    if directory:
        candidate = Path(directory)
        try:
            resolved = candidate.resolve()
            expected_root = (OUTPUT_DIR / job_id).resolve()
            if resolved == expected_root:
                return resolved
            if expected_root in resolved.parents:
                return resolved
        except Exception:
            pass
    return OUTPUT_DIR / job_id


def _chunk_file_url(job_id: str, relative_path: Optional[str]) -> Optional[str]:
    if not relative_path:
        return None
    rel = Path(relative_path).as_posix()
    return f"/static/audio/{job_id}/{rel}"


def _find_chunk_record(job_entry: Dict[str, Any], chunk_id: str):
    for idx, chunk in enumerate(job_entry.get("chunks") or []):
        if chunk.get("id") == chunk_id:
            return idx, chunk
    return None, None


def _clone_voice_assignment(assignment: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(assignment, dict):
        return None
    return copy.deepcopy(assignment)


def _voice_label_from_assignment(assignment: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(assignment, dict):
        return None
    if assignment.get("voice"):
        return assignment.get("voice")
    # Check for audio_prompt_path (Chatterbox uses this)
    prompt = assignment.get("audio_prompt_path")
    if isinstance(prompt, str) and prompt:
        # Extract filename and remove .wav extension for cleaner display
        filename = Path(prompt).name if "/" in prompt or "\\" in prompt else prompt
        return filename.replace('.wav', '') if filename.endswith('.wav') else filename
    return None


def _normalize_voice_payload(raw_payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(raw_payload, dict):
        return None
    cleaned = {}
    for key, value in raw_payload.items():
        if value is None:
            continue
        if isinstance(value, str):
            trimmed = value.strip()
            if not trimmed:
                continue
            cleaned[key] = trimmed
            continue
        cleaned[key] = value
    return cleaned or None


def _normalize_speaker_key(speaker: Optional[str]) -> str:
    return (speaker or "").strip().lower()


def _normalize_voice_assignments_map(voice_assignments: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(voice_assignments, dict):
        return {}
    normalized: Dict[str, Dict[str, Any]] = {}
    for raw_key, assignment in voice_assignments.items():
        if not isinstance(assignment, dict):
            continue
        if raw_key == "default":
            normalized["default"] = assignment
            continue
        normalized[_normalize_speaker_key(raw_key)] = assignment
    return normalized


def _prepare_voice_assignments(text: str, voice_assignments: Any) -> Dict[str, Dict[str, Any]]:
    normalized = _normalize_voice_assignments_map(voice_assignments)
    if not normalized:
        return {}
    if "default" not in normalized and len(normalized) == 1:
        normalized["default"] = next(iter(normalized.values()))
    speakers = _extract_speakers_for_text(text)
    if "default" in speakers and "default" not in normalized:
        normalized["default"] = next(iter(normalized.values()))
    return normalized


def _extract_speakers_for_text(text: str) -> List[str]:
    processor = TextProcessor()
    speakers = processor.extract_speakers(text)
    return speakers or ["default"]


def _check_speaker_tag_balance(text: str) -> List[str]:
    """Return a list of human-readable error strings for any unbalanced speaker tags.

    Detects:
    - Opening tags with no matching close: [narrator] ... (no [/narrator])
    - Closing tags with no matching open: [/narrator] at the start
    - Mismatched nesting: [narrator] ... [/sola]
    """
    open_re = re.compile(r'\[([a-zA-Z0-9_\-]+)\]')
    close_re = re.compile(r'\[/([a-zA-Z0-9_\-]+)\]')
    reserved = {"default"}

    events: List[tuple] = []
    for m in open_re.finditer(text):
        tag = m.group(1).lower()
        if tag not in reserved:
            events.append((m.start(), "open", tag))
    for m in close_re.finditer(text):
        tag = m.group(1).lower()
        if tag not in reserved:
            events.append((m.start(), "close", tag))
    events.sort(key=lambda e: e[0])

    errors: List[str] = []
    stack: List[str] = []
    for _pos, kind, tag in events:
        if kind == "open":
            stack.append(tag)
        else:
            if not stack:
                errors.append(f"Closing tag [/{tag}] has no matching opening tag")
            elif stack[-1] != tag:
                errors.append(
                    f"Mismatched tags: expected [/{stack[-1]}] but found [/{tag}]"
                )
                stack.pop()
            else:
                stack.pop()
    for unclosed in stack:
        errors.append(f"Opening tag [{unclosed}] has no matching closing tag")
    return errors


def _validate_voice_assignments_for_engine(
    engine_name: str,
    text: str,
    voice_assignments: Any,
    config: Dict[str, Any],
) -> None:
    engine_name = _normalize_engine_name(engine_name)

    # Check for unbalanced/orphaned speaker tags before anything else.
    # Only run when the text actually contains speaker tags to avoid
    # false positives on plain text submissions.
    processor = TextProcessor()
    if processor.has_speaker_tags(text):
        tag_errors = _check_speaker_tag_balance(text)
        if tag_errors:
            raise ValueError(
                "Speaker tags are unbalanced — please fix the text before submitting.\n"
                + "\n".join(f"  • {e}" for e in tag_errors)
            )

    speakers = _extract_speakers_for_text(text)
    normalized_assignments = _normalize_voice_assignments_map(voice_assignments)
    default_assignment = normalized_assignments.get("default") or {}
    if not default_assignment and len(normalized_assignments) == 1:
        default_assignment = next(iter(normalized_assignments.values())) or {}
    missing_voices: List[str] = []
    missing_prompts: List[str] = []

    for speaker in speakers:
        assignment = normalized_assignments.get(speaker) or default_assignment or {}
        voice = (assignment.get("voice") or "").strip()
        prompt = (assignment.get("audio_prompt_path") or "").strip()

        if engine_name in {"kokoro", "qwen3_custom"} and not voice:
            missing_voices.append(speaker)

        if engine_name == "chatterbox_turbo_replicate":
            default_voice = (config.get("chatterbox_turbo_replicate_voice") or "").strip()
            if not prompt and not voice and not default_voice:
                missing_voices.append(speaker)

        if engine_name == "chatterbox_turbo_local":
            default_prompt = (config.get("chatterbox_turbo_local_default_prompt") or "").strip()
            if not prompt and not default_prompt:
                missing_prompts.append(speaker)

        if engine_name == "qwen3_clone":
            default_prompt = (config.get("qwen3_clone_default_prompt") or "").strip()
            if not prompt and not default_prompt:
                missing_prompts.append(speaker)

        if engine_name in {"pocket_tts", "pocket_tts_preset"}:
            default_prompt = (config.get("pocket_tts_default_prompt") or "").strip()
            if not prompt and not voice and not default_prompt:
                missing_prompts.append(speaker)

        if engine_name == "kitten_tts":
            default_voice = (config.get("kitten_tts_default_voice") or "").strip()
            if not voice and not default_voice:
                missing_voices.append(speaker)

        if engine_name == "index_tts":
            default_prompt = (config.get("index_tts_default_prompt") or "").strip()
            if not prompt and not default_prompt:
                missing_prompts.append(speaker)

    if missing_voices or missing_prompts:
        pieces = []
        if missing_voices:
            pieces.append(
                "Select a voice for: " + ", ".join(missing_voices)
            )
        if missing_prompts:
            pieces.append(
                "Select a reference audio prompt for: " + ", ".join(missing_prompts)
            )
        raise ValueError(". ".join(pieces))


def _has_active_regen_tasks(job_entry: Dict[str, Any]) -> bool:
    tasks = job_entry.get("regen_tasks") or {}
    for task in tasks.values():
        if (task or {}).get("status") in {"queued", "running"}:
            return True
    return False


def _ensure_review_ready(job_entry: Dict[str, Any]):
    if not job_entry.get("review_mode"):
        raise ValueError("Job was not created with review mode enabled.")
    if not job_entry.get("job_dir"):
        raise ValueError("Job output directory is unavailable.")


def _load_review_manifest(job_id: str, job_entry: Dict[str, Any]) -> Dict[str, Any]:
    manifest_name = job_entry.get("review_manifest") or "review_manifest.json"
    job_dir = _job_dir_from_entry(job_id, job_entry)
    manifest_path = job_dir / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError("Review manifest not found for this job.")
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _update_review_post_progress(job_id: str, step_index: int, ratio: float) -> None:
    with queue_lock:
        entry = jobs.get(job_id)
        if not entry:
            return
        total_steps = int(entry.get("post_process_total") or 0)
        if total_steps <= 0:
            return
        clamped = max(0.0, min(float(ratio or 0), 1.0))
        overall = ((max(step_index - 1, 0)) + clamped) / total_steps
        entry["post_process_percent"] = int(min(overall * 100, 100))
        entry["post_process_active"] = True


def _update_regen_status(job_id: str, chunk_id: str, **fields):
    with queue_lock:
        job_entry = jobs.get(job_id)
        if not job_entry:
            return
        regen_tasks = job_entry.setdefault("regen_tasks", {})
        task_state = regen_tasks.setdefault(chunk_id, {})
        task_state.update(fields)


def _perform_chunk_regeneration(
    job_id: str,
    chunk_id: str,
    text_to_render: str,
    voice_override: Optional[Dict[str, Any]] = None,
    engine_override: Optional[str] = None,
):
    with queue_lock:
        job_entry = jobs.get(job_id)
        if not job_entry:
            raise ValueError("Job not found.")
        idx, chunk = _find_chunk_record(job_entry, chunk_id)
        if chunk is None:
            raise ValueError("Chunk not found.")
        config_snapshot = copy.deepcopy(job_entry.get("config_snapshot") or load_config())
        job_voice_assignments = copy.deepcopy(job_entry.get("voice_assignments") or {})
        job_dir = _job_dir_from_entry(job_id, job_entry)
        speaker = chunk.get("speaker") or "default"
        relative_file = chunk.get("relative_file")
        speed = config_snapshot.get("speed", 1.0)
        sample_rate = config_snapshot.get("sample_rate")
        # Allow engine override for cross-engine regeneration
        if engine_override:
            config_snapshot["tts_engine"] = engine_override

    normalized_override = _normalize_voice_payload(voice_override)
    chunk_voice_assignment = _clone_voice_assignment(chunk.get("voice_assignment"))
    default_assignment = _clone_voice_assignment(
        job_voice_assignments.get(speaker) or job_voice_assignments.get("default")
    )
    effective_assignment = chunk_voice_assignment or default_assignment
    if normalized_override:
        effective_assignment = {**(effective_assignment or {}), **normalized_override}

    voice_config = copy.deepcopy(job_voice_assignments) if job_voice_assignments else {}
    if effective_assignment:
        voice_config = voice_config or {}
        voice_config[speaker] = effective_assignment

    word_replacements = job_entry.get("word_replacements") or []
    chunk_text = (text_to_render or "").strip()
    if not chunk_text:
        raise ValueError("Chunk text cannot be empty.")
    if not relative_file:
        raise ValueError("Chunk does not have an associated file path.")

    # Use system temp directory instead of job_dir to avoid Windows permission issues
    tmp_dir = Path(tempfile.mkdtemp(prefix="tts_chunk_regen_"))
    generated_files: List[str] = []
    try:
        regen_text = _apply_word_replacements(chunk_text, word_replacements) if word_replacements else chunk_text
        segments = [{
            "speaker": speaker,
            "text": regen_text,
            "chunks": [regen_text],
        }]
        engine_name = _normalize_engine_name(config_snapshot.get("tts_engine"))
        
        # Acquire GPU lock to prevent concurrent inference which causes
        # GPU contention and "badcase" retry loops with VoxCPM
        with gpu_inference_lock:
            engine = get_tts_engine(engine_name, config=config_snapshot)
            generated_files = engine.generate_batch(
                segments=segments,
                voice_config=voice_config,
                output_dir=str(tmp_dir),
                speed=speed,
                sample_rate=sample_rate,
            )

        if not generated_files:
            raise RuntimeError("TTS engine did not return any audio for the chunk.")
        temp_file = Path(generated_files[0])
        target_path = job_dir / relative_file
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(temp_file), str(target_path))
    finally:
        # Clean up temp directory - retry with longer delays on Windows
        for attempt in range(5):
            try:
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
                break
            except Exception as e:
                if attempt < 4:
                    time.sleep(0.2 * (attempt + 1))  # Increasing delay: 0.2, 0.4, 0.6, 0.8s
                else:
                    logger.warning("Failed to clean up temp dir %s: %s", tmp_dir, e)

    with queue_lock:
        job_entry = jobs.get(job_id)
        if not job_entry:
            return
        _, chunk = _find_chunk_record(job_entry, chunk_id)
        if chunk:
            chunk["text"] = chunk_text
            chunk["regenerated_at"] = datetime.now().isoformat()
            chunk["engine"] = engine_name
            if effective_assignment:
                chunk["voice_assignment"] = copy.deepcopy(effective_assignment)
                voice_label = _voice_label_from_assignment(effective_assignment)
                if voice_label:
                    chunk["voice_label"] = voice_label
                else:
                    chunk.pop("voice_label", None)

    _persist_chunks_metadata(job_id, job_dir)


def _persist_chunks_metadata(job_id: str, job_dir: Path):
    """Update the chunks_metadata.json file with current chunk state."""
    with queue_lock:
        job_entry = jobs.get(job_id)
        if not job_entry:
            return
        job_chunks = job_entry.get("chunks") or []
        config_snapshot = job_entry.get("config_snapshot") or {}
        engine_name = _normalize_engine_name(config_snapshot.get("tts_engine"))

    chunks_meta_path = job_dir / "chunks_metadata.json"
    existing_meta = {}
    if chunks_meta_path.exists():
        try:
            with chunks_meta_path.open("r", encoding="utf-8") as handle:
                existing_meta = json.load(handle)
        except Exception:
            pass

    chunks_meta = {
        "engine": engine_name,
        "created_at": existing_meta.get("created_at", datetime.now().isoformat()),
        "updated_at": datetime.now().isoformat(),
        "chunks": job_chunks,
    }
    with chunks_meta_path.open("w", encoding="utf-8") as handle:
        json.dump(chunks_meta, handle, indent=2)


def _load_chunks_metadata(job_dir: Path) -> List[Dict[str, Any]]:
    chunks_meta_path = job_dir / "chunks_metadata.json"
    if not chunks_meta_path.exists():
        return []
    try:
        with chunks_meta_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        chunks = payload.get("chunks") if isinstance(payload, dict) else None
        return [dict(item) for item in (chunks or []) if isinstance(item, dict)]
    except Exception:
        return []


def _schedule_chunk_regeneration(
    job_id: str,
    chunk_id: str,
    text_to_render: str,
    voice_payload: Optional[Dict[str, Any]] = None,
    engine_override: Optional[str] = None,
):
    requested_at = datetime.now().isoformat()
    normalized_voice = _normalize_voice_payload(voice_payload)
    normalized_engine = _normalize_engine_name(engine_override) if engine_override else None
    with queue_lock:
        job_entry = jobs.get(job_id)
        if not job_entry:
            raise ValueError("Job not found.")
        regen_tasks = job_entry.setdefault("regen_tasks", {})
        regen_tasks[chunk_id] = {
            "status": "queued",
            "requested_at": requested_at,
            "error": None,
            "voice": normalized_voice,
            "engine": normalized_engine,
        }

    def task():
        try:
            _update_regen_status(job_id, chunk_id, status="running", started_at=datetime.now().isoformat(), error=None)
            _perform_chunk_regeneration(job_id, chunk_id, text_to_render, voice_override=normalized_voice, engine_override=normalized_engine)
            _update_regen_status(job_id, chunk_id, status="completed", completed_at=datetime.now().isoformat())
        except Exception as exc:  # noqa: BLE001
            logger.error("Chunk regeneration failed for job %s chunk %s: %s", job_id, chunk_id, exc, exc_info=True)
            _update_regen_status(
                job_id,
                chunk_id,
                status="failed",
                error=str(exc),
                completed_at=datetime.now().isoformat(),
            )

    chunk_regen_executor.submit(task)


@contextmanager
def log_request_timing(label: str, warn_ms: float = 750.0):
    """Log slow endpoint or processing sections."""
    start = perf_counter()
    try:
        yield
    finally:
        duration_ms = (perf_counter() - start) * 1000.0
        if duration_ms >= warn_ms:
            logger.warning(f"{label} took {duration_ms:.1f} ms")
        else:
            logger.debug(f"{label} took {duration_ms:.1f} ms")


def invalidate_library_cache():
    """Clear cached library listing so next request reloads from disk."""
    library_cache["items"] = None
    library_cache["timestamp"] = 0.0


def _get_jobs_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(JOBS_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_jobs_db() -> None:
    with _get_jobs_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT,
                created_at TEXT,
                updated_at TEXT,
                text_preview TEXT,
                text_path TEXT,
                text_length INTEGER,
                engine TEXT,
                split_by_chapter INTEGER,
                generate_full_story INTEGER,
                review_mode INTEGER,
                custom_heading TEXT,
                merge_options TEXT,
                voice_assignments TEXT,
                config_snapshot TEXT,
                job_dir TEXT,
                total_chunks INTEGER,
                processed_chunks INTEGER,
                progress INTEGER,
                eta_seconds INTEGER,
                post_process_total INTEGER,
                post_process_done INTEGER,
                post_process_percent INTEGER,
                post_process_active INTEGER,
                chapter_mode INTEGER,
                chapter_count INTEGER,
                book_mode INTEGER,
                book_count INTEGER,
                full_story_requested INTEGER,
                paused_at TEXT,
                interrupted_at TEXT,
                last_completed_chunk_index INTEGER,
                resume_from_chunk_index INTEGER,
                archived INTEGER DEFAULT 0,
                error TEXT,
                job_payload TEXT
            )
            """
        )
        conn.commit()


def _job_text_dir(job_id: str) -> Path:
    job_dir = JOBS_DATA_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def _write_job_text(job_id: str, text: str) -> str:
    job_dir = _job_text_dir(job_id)
    text_path = job_dir / "input.txt"
    text_path.write_text(text or "", encoding="utf-8")
    return str(text_path)


def _load_job_text(text_path: Optional[str]) -> str:
    if not text_path:
        return ""
    try:
        return Path(text_path).read_text(encoding="utf-8")
    except Exception:
        return ""


def _serialize_job_payload(job_entry: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(job_entry.get("job_payload") or {})
    if not isinstance(payload, dict):
        payload = {}
    for key in ("timing_metrics", "started_at", "completed_at"):
        value = job_entry.get(key)
        if value is not None:
            payload[key] = value
    return payload


def _serialize_job_entry(job_id: str, job_entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "status": job_entry.get("status"),
        "created_at": job_entry.get("created_at"),
        "updated_at": datetime.now().isoformat(),
        "text_preview": job_entry.get("text_preview"),
        "text_path": job_entry.get("text_path"),
        "text_length": job_entry.get("text_length"),
        "engine": job_entry.get("engine"),
        "split_by_chapter": int(bool(job_entry.get("chapter_mode"))),
        "generate_full_story": int(bool(job_entry.get("full_story_requested"))),
        "review_mode": int(bool(job_entry.get("review_mode"))),
        "custom_heading": job_entry.get("custom_heading"),
        "merge_options": json.dumps(job_entry.get("merge_options") or {}),
        "voice_assignments": json.dumps(job_entry.get("voice_assignments") or {}),
        "config_snapshot": json.dumps(job_entry.get("config_snapshot") or {}),
        "job_dir": job_entry.get("job_dir"),
        "total_chunks": job_entry.get("total_chunks"),
        "processed_chunks": job_entry.get("processed_chunks"),
        "progress": job_entry.get("progress"),
        "eta_seconds": job_entry.get("eta_seconds"),
        "post_process_total": job_entry.get("post_process_total"),
        "post_process_done": job_entry.get("post_process_done"),
        "post_process_percent": job_entry.get("post_process_percent"),
        "post_process_active": int(bool(job_entry.get("post_process_active"))),
        "chapter_mode": int(bool(job_entry.get("chapter_mode"))),
        "chapter_count": job_entry.get("chapter_count"),
        "book_mode": int(bool(job_entry.get("book_mode"))),
        "book_count": job_entry.get("book_count"),
        "full_story_requested": int(bool(job_entry.get("full_story_requested"))),
        "paused_at": job_entry.get("paused_at"),
        "interrupted_at": job_entry.get("interrupted_at"),
        "last_completed_chunk_index": job_entry.get("last_completed_chunk_index"),
        "resume_from_chunk_index": job_entry.get("resume_from_chunk_index"),
        "archived": int(bool(job_entry.get("archived"))),
        "error": job_entry.get("error"),
        "job_payload": json.dumps(_serialize_job_payload(job_entry)),
    }


def _persist_job_state(job_id: str, job_entry: Optional[Dict[str, Any]] = None, force: bool = False) -> None:
    with queue_lock:
        entry = job_entry or jobs.get(job_id)
        if not entry:
            return
        now = time.monotonic()
        last_persisted = entry.get("_last_persisted")
        if not force and last_persisted and (now - last_persisted) < 2:
            return
        entry["_last_persisted"] = now
        payload = _serialize_job_entry(job_id, entry)

    with _get_jobs_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, status, created_at, updated_at, text_preview, text_path, text_length, engine,
                split_by_chapter, generate_full_story, review_mode, custom_heading, merge_options,
                voice_assignments, config_snapshot, job_dir, total_chunks, processed_chunks, progress,
                eta_seconds, post_process_total, post_process_done, post_process_percent, post_process_active,
                chapter_mode, chapter_count, book_mode, book_count, full_story_requested, paused_at,
                interrupted_at, last_completed_chunk_index, resume_from_chunk_index, archived, error, job_payload
            ) VALUES (
                :job_id, :status, :created_at, :updated_at, :text_preview, :text_path, :text_length, :engine,
                :split_by_chapter, :generate_full_story, :review_mode, :custom_heading, :merge_options,
                :voice_assignments, :config_snapshot, :job_dir, :total_chunks, :processed_chunks, :progress,
                :eta_seconds, :post_process_total, :post_process_done, :post_process_percent, :post_process_active,
                :chapter_mode, :chapter_count, :book_mode, :book_count, :full_story_requested, :paused_at,
                :interrupted_at, :last_completed_chunk_index, :resume_from_chunk_index, :archived, :error, :job_payload
            )
            ON CONFLICT(job_id) DO UPDATE SET
                status=excluded.status,
                updated_at=excluded.updated_at,
                text_preview=excluded.text_preview,
                text_path=excluded.text_path,
                text_length=excluded.text_length,
                engine=excluded.engine,
                split_by_chapter=excluded.split_by_chapter,
                generate_full_story=excluded.generate_full_story,
                review_mode=excluded.review_mode,
                custom_heading=excluded.custom_heading,
                merge_options=excluded.merge_options,
                voice_assignments=excluded.voice_assignments,
                config_snapshot=excluded.config_snapshot,
                job_dir=excluded.job_dir,
                total_chunks=excluded.total_chunks,
                processed_chunks=excluded.processed_chunks,
                progress=excluded.progress,
                eta_seconds=excluded.eta_seconds,
                post_process_total=excluded.post_process_total,
                post_process_done=excluded.post_process_done,
                post_process_percent=excluded.post_process_percent,
                post_process_active=excluded.post_process_active,
                chapter_mode=excluded.chapter_mode,
                chapter_count=excluded.chapter_count,
                book_mode=excluded.book_mode,
                book_count=excluded.book_count,
                full_story_requested=excluded.full_story_requested,
                paused_at=excluded.paused_at,
                interrupted_at=excluded.interrupted_at,
                last_completed_chunk_index=excluded.last_completed_chunk_index,
                resume_from_chunk_index=excluded.resume_from_chunk_index,
                archived=excluded.archived,
                error=excluded.error,
                job_payload=excluded.job_payload
            """,
            payload,
        )
        conn.commit()


def _load_jobs_from_db() -> Dict[str, Dict[str, Any]]:
    loaded: Dict[str, Dict[str, Any]] = {}
    with _get_jobs_db_connection() as conn:
        rows = conn.execute("SELECT * FROM jobs WHERE status != 'deleted'").fetchall()
    for row in rows:
        job_id = row["job_id"]
        loaded[job_id] = {
            "status": row["status"],
            "created_at": row["created_at"],
            "text_preview": row["text_preview"],
            "text_path": row["text_path"],
            "text_length": row["text_length"],
            "engine": row["engine"],
            "chapter_mode": bool(row["chapter_mode"]),
            "full_story_requested": bool(row["full_story_requested"]),
            "review_mode": bool(row["review_mode"]),
            "custom_heading": row["custom_heading"],
            "merge_options": json.loads(row["merge_options"] or "{}"),
            "voice_assignments": json.loads(row["voice_assignments"] or "{}"),
            "config_snapshot": json.loads(row["config_snapshot"] or "{}"),
            "job_dir": row["job_dir"],
            "total_chunks": row["total_chunks"],
            "processed_chunks": row["processed_chunks"],
            "progress": row["progress"],
            "eta_seconds": row["eta_seconds"],
            "post_process_total": row["post_process_total"],
            "post_process_done": row["post_process_done"],
            "post_process_percent": row["post_process_percent"],
            "post_process_active": bool(row["post_process_active"]),
            "chapter_count": row["chapter_count"],
            "book_mode": bool(row["book_mode"]),
            "book_count": row["book_count"],
            "paused_at": row["paused_at"],
            "interrupted_at": row["interrupted_at"],
            "last_completed_chunk_index": row["last_completed_chunk_index"],
            "resume_from_chunk_index": row["resume_from_chunk_index"],
            "archived": bool(row["archived"]),
            "error": row["error"],
            "job_payload": json.loads(row["job_payload"] or "{}"),
            "chunks": [],
            "regen_tasks": {},
        }
        _extra = loaded[job_id]["job_payload"]
        for key in ("timing_metrics", "started_at", "completed_at"):
            if key in _extra:
                loaded[job_id][key] = _extra[key]
    return loaded


def _purge_stale_jobs(days: int = 7) -> None:
    """Hard-delete terminal-state jobs (cancelled/failed/interrupted/deleted) older than `days` days."""
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    terminal_statuses = "('cancelled', 'failed', 'interrupted', 'deleted')"
    with _get_jobs_db_connection() as conn:
        stale = conn.execute(
            f"SELECT job_id, status FROM jobs WHERE status IN {terminal_statuses} AND updated_at < ?",
            (cutoff,),
        ).fetchall()
        # Also purge any deleted-status rows regardless of age
        deleted_rows = conn.execute(
            "SELECT job_id, status FROM jobs WHERE status = 'deleted'"
        ).fetchall()
    all_to_purge = {row["job_id"]: row["status"] for row in stale}
    all_to_purge.update({row["job_id"]: row["status"] for row in deleted_rows})
    for job_id, status in all_to_purge.items():
        try:
            with _get_jobs_db_connection() as conn:
                conn.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
                conn.commit()
            logger.info("Purged stale job %s (status=%s)", job_id, status)
        except Exception as exc:
            logger.warning("Failed to purge stale job %s: %s", job_id, exc)


def _archive_old_jobs(max_jobs: int = 500) -> None:
    with _get_jobs_db_connection() as conn:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM jobs WHERE archived=0").fetchone()
        count = int(row["cnt"] or 0) if row else 0
        if count <= max_jobs:
            return
        to_archive = conn.execute(
            "SELECT job_id FROM jobs WHERE archived=0 ORDER BY created_at ASC LIMIT ?",
            (count - max_jobs,),
        ).fetchall()
    for row in to_archive:
        job_id = row["job_id"]
        job_audio_dir = OUTPUT_DIR / job_id
        job_data_dir = JOBS_DATA_DIR / job_id
        archive_root = JOBS_ARCHIVE_DIR / job_id
        try:
            archive_root.mkdir(parents=True, exist_ok=True)
            if job_audio_dir.exists():
                shutil.move(str(job_audio_dir), str(archive_root / "audio"))
            if job_data_dir.exists():
                shutil.move(str(job_data_dir), str(archive_root / "data"))
            with _get_jobs_db_connection() as conn:
                conn.execute(
                    "UPDATE jobs SET archived=1, status='archived' WHERE job_id=?",
                    (job_id,),
                )
                conn.commit()
        except Exception as exc:
            logger.warning("Failed to archive job %s: %s", job_id, exc)


def _build_job_payload(job_id: str, text: str, job_entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "text_path": job_entry.get("text_path"),
        "voice_assignments": job_entry.get("voice_assignments") or {},
        "config_snapshot": job_entry.get("config_snapshot") or {},
        "split_by_chapter": bool(job_entry.get("chapter_mode")),
        "generate_full_story": bool(job_entry.get("full_story_requested")),
        "review_mode": bool(job_entry.get("review_mode")),
        "merge_options": job_entry.get("merge_options") or {},
        "job_dir": job_entry.get("job_dir"),
        "custom_heading": job_entry.get("custom_heading"),
        "total_chunks": job_entry.get("total_chunks"),
        "engine": job_entry.get("engine"),
        "text_preview": (text or "")[:200],
    }


def _build_job_data_from_entry(job_id: str, job_entry: Dict[str, Any]) -> Dict[str, Any]:
    payload = job_entry.get("job_payload") or {}
    text_path = payload.get("text_path") or job_entry.get("text_path")
    text = _load_job_text(text_path)
    config = job_entry.get("config_snapshot") or load_config()
    return {
        "job_id": job_id,
        "text": text,
        "voice_assignments": job_entry.get("voice_assignments") or {},
        "config": config,
        "split_by_chapter": bool(job_entry.get("chapter_mode")),
        "generate_full_story": bool(job_entry.get("full_story_requested")),
        "total_chunks": job_entry.get("total_chunks") or 0,
        "review_mode": bool(job_entry.get("review_mode")),
        "merge_options": job_entry.get("merge_options") or {},
        "job_dir": job_entry.get("job_dir"),
        "custom_heading": job_entry.get("custom_heading"),
        "resume_from_chunk_index": job_entry.get("resume_from_chunk_index") or 0,
    }


def _restore_jobs_from_db() -> None:
    loaded = _load_jobs_from_db()
    with queue_lock:
        jobs.clear()
        jobs.update(loaded)
        for job_id, job_entry in jobs.items():
            status = job_entry.get("status")
            if status in {"processing", "queued", "pausing"}:
                if status == "processing":
                    # Check if the job actually finished — output files on disk mean it completed
                    job_dir_str = job_entry.get("job_dir")
                    _actually_completed = False
                    if job_dir_str:
                        _jdir = Path(job_dir_str)
                        # Completed jobs have output.mp3/wav or chapter subdirs with audio
                        _audio_exts = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
                        if any(f.suffix in _audio_exts for f in _jdir.glob("output*")):
                            _actually_completed = True
                        elif any(f.suffix in _audio_exts for f in _jdir.rglob("chapter_*/*")):
                            _actually_completed = True
                        elif any(f.suffix in _audio_exts for f in _jdir.rglob("*/Title*")):
                            _actually_completed = True
                    if _actually_completed:
                        job_entry["status"] = "completed"
                        job_entry["progress"] = 100
                        job_entry["interrupted_at"] = None
                        job_entry["eta_seconds"] = 0
                    else:
                        job_entry["status"] = "interrupted"
                        job_entry["interrupted_at"] = datetime.now().isoformat()
                        last_completed = job_entry.get("last_completed_chunk_index")
                        if last_completed is not None:
                            job_entry["resume_from_chunk_index"] = max(0, int(last_completed) + 1)
                else:
                    job_entry["status"] = "paused"
                job_entry["eta_seconds"] = None
    for job_id in loaded.keys():
        _persist_job_state(job_id, force=True)


def _normalize_engine_name(name: Optional[str]) -> str:
    value = (name or DEFAULT_CONFIG["tts_engine"]).strip().lower()
    return value or DEFAULT_CONFIG["tts_engine"]


def _resolve_qwen_device(device: str) -> str:
    import torch
    candidate = (device or "auto").strip().lower()
    if candidate == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if candidate.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but no GPU is available.")
    return candidate


def _resolve_qwen_dtype(dtype_value: str):
    import torch
    normalized = (dtype_value or "").strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def _resolve_qwen_attn(attn_value: Optional[str]) -> Optional[str]:
    normalized = (attn_value or "").strip().lower().replace("-", "_")
    if normalized in {"", "auto"}:
        return None
    if normalized in {"flash_attention_2", "flash_attention2", "flash"}:
        try:
            import flash_attn  # type: ignore  # noqa: F401
        except Exception:
            logger.warning("flash-attn not installed; falling back to eager attention for Qwen3")
            return "eager"
        return "flash_attention_2"
    return normalized


def _ensure_qwen3_model(model_id: str) -> Path:
    from huggingface_hub import snapshot_download
    local_model_dir = Path(__file__).parent / "models" / "qwen3"
    local_model_dir.mkdir(parents=True, exist_ok=True)
    model_path = local_model_dir / model_id.replace("/", "_")
    if not model_path.exists() or not any(model_path.iterdir()):
        logger.info("Downloading Qwen3 model to %s (this may take a few minutes)...", model_path)
        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=str(model_path),
                local_dir_use_symlinks=False,
            )
        except Exception as exc:
            logger.error("Failed to download Qwen3 model %s: %s", model_id, exc, exc_info=True)
            raise RuntimeError(
                "Qwen3 VoiceDesign model files are missing. Download the model while online "
                "(repo: %s) into %s, then retry." % (model_id, model_path)
            ) from exc
    return model_path


def _qwen3_voice_design_signature(config: Dict[str, Any]) -> str:
    model_id = (config.get("qwen3_voice_design_model_id") or "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign").strip()
    device = (config.get("qwen3_custom_device") or "auto").strip()
    dtype = (config.get("qwen3_custom_dtype") or "bfloat16").strip()
    attn = (config.get("qwen3_custom_attn_implementation") or "flash_attention_2").strip()
    return f"{model_id}|{device}|{dtype}|{attn}"


def _get_qwen3_voice_design_model(config: Dict[str, Any]):
    if not QWEN3_AVAILABLE:
        raise ImportError("qwen-tts is not installed. Run setup to enable Qwen3-TTS local mode.")
    global qwen3_voice_design_model, qwen3_voice_design_signature
    config = config or {}
    signature = _qwen3_voice_design_signature(config)
    with tts_engine_lock:
        if qwen3_voice_design_model is not None and qwen3_voice_design_signature == signature:
            return qwen3_voice_design_model
        from qwen_tts import Qwen3TTSModel  # type: ignore
        model_id = (config.get("qwen3_voice_design_model_id") or "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign").strip()
        device = _resolve_qwen_device(config.get("qwen3_custom_device") or "auto")
        dtype = _resolve_qwen_dtype(config.get("qwen3_custom_dtype") or "bfloat16")
        attn = _resolve_qwen_attn(config.get("qwen3_custom_attn_implementation") or "flash_attention_2")
        model_path = _ensure_qwen3_model(model_id)
        logger.info(
            "Loading Qwen3 VoiceDesign model=%s device=%s dtype=%s attn=%s",
            model_id,
            device,
            dtype,
            attn or "auto",
        )
        qwen3_voice_design_model = Qwen3TTSModel.from_pretrained(
            str(model_path),
            device_map=device,
            dtype=dtype,
            attn_implementation=attn or None,
        )
        qwen3_voice_design_signature = signature
        return qwen3_voice_design_model


def _engine_signature(engine_name: str, config: Dict) -> str:
    """Generate a signature capturing settings that require a fresh engine."""
    config = config or {}
    if engine_name == "chatterbox_turbo_local":
        parts = (
            (config.get("chatterbox_turbo_local_default_prompt") or "").strip(),
            str(config.get("chatterbox_turbo_local_temperature")),
            str(config.get("chatterbox_turbo_local_top_p")),
            str(config.get("chatterbox_turbo_local_top_k")),
            str(config.get("chatterbox_turbo_local_repetition_penalty")),
            str(config.get("chatterbox_turbo_local_cfg_weight")),
            str(config.get("chatterbox_turbo_local_exaggeration")),
            str(bool(config.get("chatterbox_turbo_local_norm_loudness", True))),
            str(bool(config.get("chatterbox_turbo_local_prompt_norm_loudness", True))),
        )
        return f"{engine_name}::{'|'.join(parts)}"
    if engine_name == "chatterbox_turbo_replicate":
        parts = (
            (config.get("chatterbox_turbo_replicate_api_token") or "").strip(),
            (config.get("chatterbox_turbo_replicate_model") or "").strip(),
            (config.get("chatterbox_turbo_replicate_voice") or "").strip(),
            str(config.get("chatterbox_turbo_replicate_temperature")),
            str(config.get("chatterbox_turbo_replicate_top_p")),
            str(config.get("chatterbox_turbo_replicate_top_k")),
            str(config.get("chatterbox_turbo_replicate_repetition_penalty")),
            str(config.get("chatterbox_turbo_replicate_seed")),
        )
        return f"{engine_name}::{'|'.join(parts)}"
    if engine_name == "voxcpm_local":
        parts = (
            (config.get("voxcpm_local_model_id") or "").strip(),
            (config.get("voxcpm_local_default_prompt") or "").strip(),
            (config.get("voxcpm_local_default_prompt_text") or "").strip(),
            str(config.get("voxcpm_local_cfg_value")),
            str(config.get("voxcpm_local_inference_timesteps")),
            str(bool(config.get("voxcpm_local_normalize", False))),
            str(bool(config.get("voxcpm_local_denoise", False))),
        )
        return f"{engine_name}::{'|'.join(parts)}"
    if engine_name == "qwen3_custom":
        parts = (
            (config.get("qwen3_custom_model_id") or "").strip(),
            (config.get("qwen3_custom_device") or "").strip(),
            (config.get("qwen3_custom_dtype") or "").strip(),
            (config.get("qwen3_custom_attn_implementation") or "").strip(),
            (config.get("qwen3_custom_default_language") or "").strip(),
            (config.get("qwen3_custom_default_instruct") or "").strip(),
        )
        return f"{engine_name}::{'|'.join(parts)}"
    if engine_name == "qwen3_clone":
        parts = (
            (config.get("qwen3_clone_model_id") or "").strip(),
            (config.get("qwen3_clone_device") or "").strip(),
            (config.get("qwen3_clone_dtype") or "").strip(),
            (config.get("qwen3_clone_attn_implementation") or "").strip(),
            (config.get("qwen3_clone_default_language") or "").strip(),
            (config.get("qwen3_clone_default_prompt") or "").strip(),
            (config.get("qwen3_clone_default_prompt_text") or "").strip(),
        )
        return f"{engine_name}::{'|'.join(parts)}"
    if engine_name in {"pocket_tts", "pocket_tts_preset"}:
        parts = (
            (config.get("pocket_tts_model_variant") or "").strip(),
            str(config.get("pocket_tts_temp")),
            str(config.get("pocket_tts_lsd_decode_steps")),
            str(config.get("pocket_tts_noise_clamp")),
            str(config.get("pocket_tts_eos_threshold")),
            (config.get("pocket_tts_default_prompt") or "").strip(),
            str(bool(config.get("pocket_tts_prompt_truncate", False))),
            str(config.get("pocket_tts_num_threads")),
            str(config.get("pocket_tts_interop_threads")),
        )
        return f"{engine_name}::{'|'.join(parts)}"
    if engine_name == "kokoro_replicate":
        parts = (
            (config.get("replicate_api_key") or "").strip(),
        )
        return f"{engine_name}::{'|'.join(parts)}"
    return engine_name


def _create_engine(engine_name: str, config: Dict) -> TtsEngineBase:
    """Instantiate a specific engine with configuration-derived options."""
    config = config or {}
    if engine_name == "kokoro":
        if not KOKORO_AVAILABLE:
            raise ImportError("Kokoro is not installed. Run setup to enable local mode.")
        device = config.get("device", "auto")
        logger.info("Creating Kokoro engine with device='%s'", device)
        engine = TTSEngine(device=device)
        logger.info("Kokoro engine created on device='%s'", engine.device)
        return engine

    if engine_name == "chatterbox_turbo_local":
        if not CHATTERBOX_TURBO_AVAILABLE:
            raise ImportError(
                "chatterbox-tts is not installed. Run setup to enable the local Chatterbox Turbo engine."
            )
        device = (config.get("chatterbox_turbo_local_device") or config.get("device") or "auto").strip()
        return get_engine(
            "chatterbox_turbo_local",
            device=device or "auto",
            default_prompt=(config.get("chatterbox_turbo_local_default_prompt") or "").strip() or None,
            temperature=float(config.get("chatterbox_turbo_local_temperature") or 0.8),
            top_p=float(config.get("chatterbox_turbo_local_top_p") or 0.95),
            top_k=int(config.get("chatterbox_turbo_local_top_k") or 1000),
            repetition_penalty=float(config.get("chatterbox_turbo_local_repetition_penalty") or 1.2),
            cfg_weight=float(config.get("chatterbox_turbo_local_cfg_weight") or 0.0),
            exaggeration=float(config.get("chatterbox_turbo_local_exaggeration") or 0.0),
            norm_loudness=bool(config.get("chatterbox_turbo_local_norm_loudness", True)),
            prompt_norm_loudness=bool(config.get("chatterbox_turbo_local_prompt_norm_loudness", True)),
        )

    if engine_name == "chatterbox_turbo_replicate":
        # Use shared replicate_api_key, fall back to engine-specific token for backward compatibility
        api_token = (config.get("replicate_api_key") or config.get("chatterbox_turbo_replicate_api_token") or "").strip()
        if not api_token:
            raise ValueError("Replicate API token is required for Chatterbox (Replicate). Configure it in the Kokoro · Replicate settings section.")
        return get_engine(
            "chatterbox_turbo_replicate",
            api_token=api_token,
            model_version=(config.get("chatterbox_turbo_replicate_model") or DEFAULT_CHATTERBOX_TURBO_REPLICATE_MODEL).strip()
            or DEFAULT_CHATTERBOX_TURBO_REPLICATE_MODEL,
            default_voice=(config.get("chatterbox_turbo_replicate_voice") or DEFAULT_CHATTERBOX_TURBO_REPLICATE_VOICE).strip()
            or DEFAULT_CHATTERBOX_TURBO_REPLICATE_VOICE,
            temperature=float(config.get("chatterbox_turbo_replicate_temperature") or 0.8),
            top_p=float(config.get("chatterbox_turbo_replicate_top_p") or 0.95),
            top_k=int(config.get("chatterbox_turbo_replicate_top_k") or 1000),
            repetition_penalty=float(config.get("chatterbox_turbo_replicate_repetition_penalty") or 1.2),
            seed=(
                int(config["chatterbox_turbo_replicate_seed"])
                if config.get("chatterbox_turbo_replicate_seed") not in (None, "")
                else None
            ),
        )

    if engine_name == "voxcpm_local":
        if not VOXCPM_AVAILABLE:
            raise ImportError("voxcpm is not installed. Run setup to enable VoxCPM local mode.")
        device = (config.get("voxcpm_local_device") or "auto").strip()
        return get_engine(
            "voxcpm_local",
            device=device or "auto",
            model_id=(config.get("voxcpm_local_model_id") or "openbmb/VoxCPM1.5").strip(),
            default_prompt=(config.get("voxcpm_local_default_prompt") or "").strip() or None,
            default_prompt_text=(config.get("voxcpm_local_default_prompt_text") or "").strip() or None,
            cfg_value=float(config.get("voxcpm_local_cfg_value") or 2.0),
            inference_timesteps=int(config.get("voxcpm_local_inference_timesteps") or 32),
            normalize=bool(config.get("voxcpm_local_normalize", False)),
            denoise=bool(config.get("voxcpm_local_denoise", False)),
        )

    if engine_name == "qwen3_custom":
        if not QWEN3_AVAILABLE:
            raise ImportError("qwen-tts is not installed. Run setup to enable Qwen3-TTS local mode.")
        device = (config.get("qwen3_custom_device") or "auto").strip()
        return get_engine(
            "qwen3_custom",
            device=device or "auto",
            model_id=(config.get("qwen3_custom_model_id") or "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice").strip(),
            dtype=(config.get("qwen3_custom_dtype") or "bfloat16").strip(),
            attn_implementation=(config.get("qwen3_custom_attn_implementation") or "flash_attention_2").strip(),
            default_language=(config.get("qwen3_custom_default_language") or "Auto").strip() or "Auto",
            default_instruct=(config.get("qwen3_custom_default_instruct") or "").strip() or None,
        )

    if engine_name == "qwen3_clone":
        if not QWEN3_CLONE_AVAILABLE:
            raise ImportError("qwen-tts is not installed. Run setup to enable Qwen3-TTS local mode.")
        device = (config.get("qwen3_clone_device") or "auto").strip()
        return get_engine(
            "qwen3_clone",
            device=device or "auto",
            model_id=(config.get("qwen3_clone_model_id") or "Qwen/Qwen3-TTS-12Hz-1.7B-Base").strip(),
            dtype=(config.get("qwen3_clone_dtype") or "bfloat16").strip(),
            attn_implementation=(config.get("qwen3_clone_attn_implementation") or "flash_attention_2").strip(),
            default_language=(config.get("qwen3_clone_default_language") or "Auto").strip() or "Auto",
            default_prompt=(config.get("qwen3_clone_default_prompt") or "").strip() or None,
            default_prompt_text=(config.get("qwen3_clone_default_prompt_text") or "").strip() or None,
        )

    if engine_name in {"pocket_tts", "pocket_tts_preset"}:
        if not POCKET_TTS_AVAILABLE:
            raise ImportError("pocket-tts is not installed. Run setup to enable Pocket TTS.")
        return get_engine(
            "pocket_tts",
            device="cpu",
            model_variant=(config.get("pocket_tts_model_variant") or "b6369a24").strip(),
            temp=float(config.get("pocket_tts_temp") or 0.7),
            lsd_decode_steps=int(config.get("pocket_tts_lsd_decode_steps") or 1),
            noise_clamp=config.get("pocket_tts_noise_clamp"),
            eos_threshold=float(config.get("pocket_tts_eos_threshold") or -4.0),
            default_prompt=(config.get("pocket_tts_default_prompt") or "").strip() or None,
            prompt_truncate=bool(config.get("pocket_tts_prompt_truncate", False)),
            num_threads=config.get("pocket_tts_num_threads"),
            interop_threads=config.get("pocket_tts_interop_threads"),
        )

    if engine_name == "kitten_tts":
        if not KITTEN_TTS_AVAILABLE:
            raise ImportError(
                "kittentts is not installed. Run: "
                "pip install https://github.com/KittenML/KittenTTS/releases/download/0.8/kittentts-0.8.0-py3-none-any.whl"
            )
        return get_engine(
            "kitten_tts",
            model_id=(config.get("kitten_tts_model_id") or KITTEN_TTS_DEFAULT_MODEL).strip(),
            default_voice=(config.get("kitten_tts_default_voice") or "Jasper").strip(),
        )

    if engine_name == "index_tts":
        if not INDEX_TTS_AVAILABLE:
            raise ImportError(
                f"IndexTTS is not available: {INDEX_TTS_UNAVAILABLE_REASON}"
            )
        device_raw = (config.get("index_tts_device") or "auto").strip().lower()
        device = None if device_raw in ("auto", "") else device_raw
        return get_engine(
            "index_tts",
            model_version=(config.get("index_tts_model_version") or "IndexTTS-2").strip(),
            use_fp16=bool(config.get("index_tts_use_fp16", True)),
            use_deepspeed=bool(config.get("index_tts_use_deepspeed", False)),
            use_torch_compile=bool(config.get("index_tts_use_torch_compile", True)),
            use_accel=bool(config.get("index_tts_use_accel", True)),
            num_beams=int(config.get("index_tts_num_beams", 1)),
            diffusion_steps=int(config.get("index_tts_diffusion_steps", 12)),
            temperature=float(config.get("index_tts_temperature", 0.8)),
            top_p=float(config.get("index_tts_top_p", 0.8)),
            top_k=int(config.get("index_tts_top_k", 30)),
            repetition_penalty=float(config.get("index_tts_repetition_penalty", 10.0)),
            max_mel_tokens=int(config.get("index_tts_max_mel_tokens", 1500)),
            max_text_tokens_per_segment=int(config.get("index_tts_max_text_tokens_per_segment", 120)),
            device=device,
            default_prompt=(config.get("index_tts_default_prompt") or "").strip() or None,
        )

    if engine_name == "kokoro_replicate":
        api_key = (config.get("replicate_api_key") or "").strip()
        if not api_key:
            raise ValueError("Replicate API key is required for Kokoro (Replicate).")
        return ReplicateAPI(api_key)

    raise ValueError(f"Unsupported local TTS engine '{engine_name}'.")


def get_tts_engine(engine_name: Optional[str] = None, config: Optional[Dict] = None):
    """Return a shared engine instance keyed by engine name and config signature.
    
    When switching to a different engine, unloads other engines to free GPU memory.
    """
    import gc
    
    selected = _normalize_engine_name(engine_name)
    config = config or load_config()
    signature = _engine_signature(selected, config)

    with tts_engine_lock:
        cached = tts_engine_instances.get(selected)
        if cached and engine_config_signatures.get(selected) == signature:
            return cached

        # Unload OTHER engines to free GPU memory before loading new one
        # This prevents multiple large models from being loaded simultaneously
        other_engines = [name for name in tts_engine_instances.keys() if name != selected]
        for other_name in other_engines:
            other_engine = tts_engine_instances.get(other_name)
            if other_engine:
                try:
                    logger.info("Unloading engine '%s' to free GPU memory for '%s'", other_name, selected)
                    other_engine.cleanup()
                except Exception:
                    logger.warning("Failed to cleanup engine '%s'", other_name, exc_info=True)
                tts_engine_instances.pop(other_name, None)
                engine_config_signatures.pop(other_name, None)
        
        # Force garbage collection and clear CUDA cache after unloading
        if other_engines:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    allocated = torch.cuda.memory_allocated(0) / 1024**3
                    logger.info("GPU memory after unloading other engines: %.2f GB", allocated)
            except Exception:
                pass

        if cached:
            try:
                cached.cleanup()
            except Exception:
                logger.warning("Failed to cleanup engine '%s' before reload.", selected, exc_info=True)

        engine = _create_engine(selected, config)
        tts_engine_instances[selected] = engine
        engine_config_signatures[selected] = signature
        return engine


def clear_cached_custom_voice(voice_code: str | None = None) -> int:
    """Ensure cached blended tensors stay in sync after CRUD operations."""
    engine = tts_engine_instances.get("kokoro")
    if engine is None:
        return 0
    return engine.clear_custom_voice_cache(voice_code)


def _cleanup_engine_vram(engine_name: Optional[str] = None) -> None:
    """Clean up VRAM for a specific engine or all engines."""
    import gc
    
    with tts_engine_lock:
        if engine_name:
            engine = tts_engine_instances.get(engine_name)
            if engine:
                try:
                    engine.cleanup()
                    logger.info("VRAM cleanup completed for engine: %s", engine_name)
                except Exception as e:
                    logger.warning("VRAM cleanup failed for engine %s: %s", engine_name, e)
                # Remove from cache to force reload on next use
                tts_engine_instances.pop(engine_name, None)
                engine_config_signatures.pop(engine_name, None)
        else:
            # Clean all engines
            for name, engine in list(tts_engine_instances.items()):
                try:
                    engine.cleanup()
                except Exception as e:
                    logger.warning("VRAM cleanup failed for engine %s: %s", name, e)
            tts_engine_instances.clear()
            engine_config_signatures.clear()
    
    gc.collect()
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _voice_manager_for_custom_voices() -> VoiceManager:
    """Create a fresh VoiceManager to validate custom voice payloads."""
    return VoiceManager()


def _normalize_component(component) -> Dict[str, float]:
    """Normalize a component entry into {'voice': str, 'weight': float}."""
    voice = None
    weight = 1.0

    if isinstance(component, str):
        voice = component.strip()
    elif isinstance(component, dict):
        voice = str(component.get("voice") or component.get("name") or "").strip()
        weight_candidate = component.get("weight") or component.get("ratio") or component.get("mix")
        if weight_candidate is not None:
            try:
                weight = float(weight_candidate)
            except (TypeError, ValueError):
                raise ValueError("Component weight must be numeric.")
    else:
        raise ValueError("Invalid component format.")

    if not voice:
        raise ValueError("Component voice is required.")
    if weight <= 0:
        raise ValueError("Component weight must be greater than zero.")

    return {"voice": voice, "weight": weight}


def _prepare_custom_voice_payload(data: dict, existing: Optional[dict] = None) -> dict:
    """Validate and normalize incoming custom voice payloads."""
    if not isinstance(data, dict):
        raise ValueError("Invalid payload format.")

    existing = existing or {}
    name = (data.get("name") or existing.get("name") or "Custom Voice").strip()
    if not name:
        raise ValueError("Custom voice name cannot be empty.")
    if len(name) > 80:
        raise ValueError("Custom voice name must be 80 characters or fewer.")

    lang_code = (data.get("lang_code") or existing.get("lang_code") or "a").lower()
    manager = _voice_manager_for_custom_voices()
    if not manager.supports_lang_code(lang_code):
        raise ValueError(f"Unsupported language code '{lang_code}'.")

    components_input = data.get("components")
    if components_input is None:
        components_input = existing.get("components")
    if not components_input:
        raise ValueError("Custom voice requires at least one component voice.")

    normalized_components = [_normalize_component(component) for component in components_input]
    for component in normalized_components:
        if not manager.validate_voice(component["voice"], lang_code):
            raise ValueError(f"Voice '{component['voice']}' is not available for language '{lang_code}'.")

    total_weight = sum(component["weight"] for component in normalized_components)
    if total_weight <= 0:
        raise ValueError("Total component weight must be greater than zero.")

    notes_value = (data.get("notes") or existing.get("notes") or "").strip()

    payload = existing.copy()
    payload.update({
        "name": name,
        "lang_code": lang_code,
        "components": normalized_components,
        "notes": notes_value or None,
    })
    return payload


def _to_public_custom_voice(entry: dict) -> dict:
    """Ensure API responses include normalized metadata."""
    if not entry:
        return {}
    public = entry.copy()
    if "code" not in public and public.get("id"):
        public["code"] = f"{CUSTOM_CODE_PREFIX}{public['id']}"
    public["components"] = public.get("components", [])
    return public


def _get_raw_custom_voice(identifier: str) -> Optional[dict]:
    """Fetch raw custom voice definition by id or code."""
    if not identifier:
        return None
    voice_id = identifier
    if identifier.startswith(CUSTOM_CODE_PREFIX):
        entry = get_custom_voice_by_code(identifier)
        voice_id = entry.get("id") if entry else None
    if not voice_id:
        return None
    return get_custom_voice(voice_id)


def slugify_filename(value: str, default: str = "chapter", max_length: int = 120) -> str:
    """Create a filesystem-friendly slug."""
    if not value:
        return default
    value = re.sub(r'[^A-Za-z0-9]+', '-', value)
    value = re.sub(r'-{2,}', '-', value).strip('-')
    if max_length and len(value) > max_length:
        value = value[:max_length].rstrip('-')
    return value or default


def _build_sections_from_matches(
    text: str,
    matches: List[re.Match],
    default_label: str,
    base_offset: int = 0
) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    if not matches:
        clean_text = text.strip()
        if clean_text:
            sections.append({"title": "Full Story", "content": clean_text})
        return sections

    first_start = matches[0].start()
    if first_start > 0:
        pre_content = text[:first_start].strip()
        if pre_content:
            # Create a "Title" section for content before the first heading.
            sections.append({"title": "Title", "content": pre_content})

    # Matches a closing speaker tag at the very start of a string (possibly after whitespace)
    _leading_close_tag_re = re.compile(r'^(\s*\[/([a-zA-Z0-9_\-]+)\])', re.DOTALL)
    # Matches a lone opening speaker tag on its own line immediately before a heading
    _lone_open_tag_re = re.compile(r'\[([a-zA-Z0-9_\-]+)\]\s*$')

    for idx, match in enumerate(matches):
        start = match.start()
        raw_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)

        # The chapter-heading regex allows optional tag prefixes like [/narrator]\n\n[narrator]
        # before the heading keyword. When the next match begins with a closing speaker tag
        # (e.g. "[/narrator]\n\n[narrator]\nCHAPTER IX"), that closing tag actually belongs
        # to the CURRENT chapter's content — extend end to include it.
        next_chunk = text[raw_end:]
        close_prefix = _leading_close_tag_re.match(next_chunk)
        if close_prefix:
            end = raw_end + len(close_prefix.group(1))
        else:
            end = raw_end

        # If there is a lone speaker-tag opening on the line(s) immediately before
        # the heading, include it so the content slice has a balanced open+close pair.
        # This handles patterns like:
        #   [narrator]
        #   CHAPTER VIII          <- match.start() is here
        #   ...prose...
        #   [/narrator]
        preceding = text[:start]
        preceding_stripped = preceding.rstrip('\r\n ')
        last_newline = preceding_stripped.rfind('\n')
        preceding_line = preceding_stripped[last_newline + 1:] if last_newline >= 0 else preceding_stripped
        if _lone_open_tag_re.match(preceding_line.strip()):
            # Rewind start to the beginning of that opening-tag line
            start = len(preceding_stripped) - len(preceding_line.lstrip())

        content = text[start:end].strip()
        if not content:
            continue
        heading_raw = (match.group(0) or "").strip()
        title = _clean_heading_text(match.group(1) or '')
        sections.append({
            "title": title or f"{default_label} {idx + 1}",
            "content": content,
            "heading": heading_raw,
            "heading_start": match.start() + base_offset,
            "heading_end": match.end() + base_offset,
        })

    return sections


def split_text_into_sections(text: str, custom_heading: Optional[Any] = None) -> List[Dict[str, str]]:
    """
    Split text into logical sections (book/chapter/section/letter/part).
    If book headings exist, split by book and include all chapters beneath each book.
    Otherwise, split by chapter/section/letter/part headings.
    """
    book_matches = list(BOOK_HEADING_PATTERN.finditer(text))
    if book_matches:
        return _build_sections_from_matches(text, book_matches, "Book")

    section_pattern = _build_section_heading_pattern(custom_heading)
    section_matches = list(section_pattern.finditer(text))
    if section_matches:
        return _build_sections_from_matches(text, section_matches, "Section")

    clean_text = text.strip()
    return [{"title": "Full Story", "content": clean_text}] if clean_text else []


def split_text_into_book_sections(text: str, custom_heading: Optional[Any] = None) -> Dict[str, Any]:
    """Return structured book->chapter hierarchy with fallbacks."""
    book_matches = list(BOOK_HEADING_PATTERN.finditer(text))
    section_pattern = _build_section_heading_pattern(custom_heading)

    if book_matches:
        books = _build_sections_from_matches(text, book_matches, "Book")
        for idx, book in enumerate(books, start=1):
            book_content = book.get("content") or ""
            section_matches = list(section_pattern.finditer(book_content))
            if section_matches:
                chapters = _build_sections_from_matches(
                    book_content,
                    section_matches,
                    "Chapter",
                    base_offset=book.get("heading_start") or 0
                )
            else:
                clean_content = book_content.strip()
                chapters = []
                if clean_content:
                    chapters.append({"title": "Full Book", "content": clean_content})
            book["chapters"] = chapters
            book["index"] = idx - 1
        return {"kind": "book", "books": books, "sections": []}

    section_matches = list(section_pattern.finditer(text))
    if section_matches:
        sections = _build_sections_from_matches(text, section_matches, "Section")
        return {"kind": "section", "books": [], "sections": sections}

    clean_text = text.strip()
    sections = [{"title": "Full Story", "content": clean_text}] if clean_text else []
    return {"kind": "none", "books": [], "sections": sections}


def _resolve_llm_chunk_size(config: Dict[str, Any]) -> int:
    provider = (config.get("llm_provider") or DEFAULT_LLM_PROVIDER).lower().strip()
    key = "llm_gemini_chunk_size" if provider == "gemini" else "llm_local_chunk_size"
    raw_value = config.get(key, 500)
    try:
        resolved = int(raw_value)
    except (TypeError, ValueError):
        resolved = 500
    return max(50, resolved)


def _resolve_llm_chunk_chapters(config: Dict[str, Any]) -> bool:
    provider = (config.get("llm_provider") or DEFAULT_LLM_PROVIDER).lower().strip()
    key = "llm_gemini_chunk_chapters" if provider == "gemini" else "llm_local_chunk_chapters"
    return bool(config.get(key, True))


def _chunk_text_by_paragraph_words(text: str, max_words: int) -> List[str]:
    content = (text or "").strip()
    if not content:
        return []
    try:
        max_words = int(max_words)
    except (TypeError, ValueError):
        max_words = 500
    max_words = max(50, max_words)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", content) if part.strip()]
    if not paragraphs:
        return [content]

    chunks: List[str] = []
    current_parts: List[str] = []
    current_words = 0

    for paragraph in paragraphs:
        paragraph_words = len(paragraph.split())
        if not current_parts:
            current_parts = [paragraph]
            current_words = paragraph_words
            continue

        if current_words >= max_words:
            chunks.append("\n\n".join(current_parts).strip())
            current_parts = [paragraph]
            current_words = paragraph_words
            continue

        if current_words + paragraph_words > max_words:
            current_parts.append(paragraph)
            chunks.append("\n\n".join(current_parts).strip())
            current_parts = []
            current_words = 0
            continue

        current_parts.append(paragraph)
        current_words += paragraph_words

    if current_parts:
        chunks.append("\n\n".join(current_parts).strip())
    return chunks


def _append_llm_chunks(
    sections: List[Dict[str, Any]],
    content: str,
    title: Optional[str],
    source: str,
    max_words: int,
) -> None:
    clean_content = (content or "").strip()
    if not clean_content:
        return
    chunks = _chunk_text_by_paragraph_words(clean_content, max_words)
    if not chunks:
        chunks = [clean_content]
    if len(chunks) == 1:
        sections.append({
            "title": title,
            "content": chunks[0],
            "source": source,
        })
        return
    for idx, chunk in enumerate(chunks):
        sections.append({
            "title": title if idx == 0 else None,
            "content": chunk,
            "source": source,
        })


def build_gemini_sections(text: str, prefer_chapters: bool, config: dict, custom_heading: Optional[Any] = None):
    """Create sections for Gemini processing based on detected sections or chunks."""
    sections = []
    if not text:
        return sections

    llm_chunk_size = _resolve_llm_chunk_size(config)
    llm_chunk_chapters = _resolve_llm_chunk_chapters(config)

    book_matches = list(BOOK_HEADING_PATTERN.finditer(text))
    section_pattern = _build_section_heading_pattern(custom_heading)
    section_matches = list(section_pattern.finditer(text)) if not book_matches else []

    if prefer_chapters and book_matches:
        hierarchy = split_text_into_book_sections(text, custom_heading)
        for book_idx, book in enumerate(hierarchy.get("books") or [], start=1):
            book_title = (book.get("title") or f"Book {book_idx}").strip()
            for chapter in book.get("chapters") or []:
                chapter_title = (chapter.get("title") or "").strip()
                title = f"{book_title} — {chapter_title}".strip(" —") if chapter_title else book_title
                if llm_chunk_chapters:
                    _append_llm_chunks(
                        sections,
                        chapter.get("content") or "",
                        title,
                        "section",
                        llm_chunk_size,
                    )
                else:
                    sections.append({
                        "title": title,
                        "content": (chapter.get("content") or "").strip(),
                        "source": "section",
                    })
    elif prefer_chapters and section_matches:
        for chapter in split_text_into_sections(text, custom_heading):
            if llm_chunk_chapters:
                _append_llm_chunks(
                    sections,
                    chapter.get("content") or "",
                    chapter.get("title"),
                    "section",
                    llm_chunk_size,
                )
            else:
                sections.append({
                    "title": chapter.get("title"),
                    "content": (chapter.get("content") or "").strip(),
                    "source": "section",
                })
    elif prefer_chapters:
        clean_text = text.strip()
        if clean_text:
            if llm_chunk_chapters:
                chunks = _chunk_text_by_paragraph_words(clean_text, llm_chunk_size)
                if not chunks:
                    chunks = [clean_text]
                if len(chunks) == 1:
                    sections.append({
                        "title": "Full Story",
                        "content": chunks[0],
                        "source": "full"
                    })
                else:
                    for chunk in chunks:
                        sections.append({
                            "title": None,
                            "content": chunk,
                            "source": "chunk"
                        })
            else:
                sections.append({
                    "title": "Full Story",
                    "content": clean_text,
                    "source": "full"
                })
    else:
        chunks = _chunk_text_by_paragraph_words(text, llm_chunk_size)
        if not chunks:
            chunks = [text]
        for chunk in chunks:
            clean_chunk = chunk.strip()
            if not clean_chunk:
                continue
            sections.append({
                "title": None,
                "content": clean_chunk,
                "source": "chunk"
            })

    return sections


def compose_gemini_prompt(section: dict, prompt_prefix: str = "", known_speakers=None) -> str:
    """Build the prompt for a Gemini section, optionally referencing known speakers."""
    parts = []
    if prompt_prefix:
        parts.append(prompt_prefix.strip())

    speakers = [s for s in (known_speakers or []) if s]
    if speakers:
        speaker_line = (
            "Known speaker tags so far (use these exact tags when they apply; "
            "only introduce a new tag if the speaker is truly new): "
            + ", ".join(speakers)
        )
        parts.append(speaker_line)

    content = (section.get("content") or "").strip()
    if content:
        parts.append(content)
    return "\n\n".join(parts).strip()


def _run_llm_prompt(prompt: str, config: Dict[str, Any]) -> str:
    """Run a prompt through the configured LLM provider (Gemini or local)."""
    provider = (config.get("llm_provider") or DEFAULT_LLM_PROVIDER).lower().strip()
    if provider == "gemini":
        api_key = (config.get("gemini_api_key") or "").strip()
        if not api_key:
            raise GeminiProcessorError("Gemini API key not configured")
        model_name = config.get("gemini_model") or DEFAULT_GEMINI_MODEL
        processor = GeminiProcessor(api_key=api_key, model_name=model_name)
        return processor.generate_text(prompt)

    local_provider = (config.get("llm_local_provider") or "").lower().strip()
    base_url = (config.get("llm_local_base_url") or "").strip()
    model_name = (config.get("llm_local_model") or "").strip()
    api_key = (config.get("llm_local_api_key") or "").strip()
    timeout = int(config.get("llm_local_timeout") or 120)
    temperature = config.get("llm_local_temperature")
    top_p = config.get("llm_local_top_p")
    top_k = config.get("llm_local_top_k")
    repeat_penalty = config.get("llm_local_repeat_penalty")
    max_tokens = config.get("llm_local_max_tokens")
    disable_reasoning = bool(config.get("llm_local_disable_reasoning", False))

    processor = LocalLLMProcessor(
        provider=local_provider,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        timeout=timeout,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        max_tokens=max_tokens,
        disable_reasoning=disable_reasoning,
    )
    return processor.generate_text(prompt)


def compose_gemini_speaker_profile_prompt(prompt_prefix: str, speakers: List[str], context: str = "", processed_text: str = "") -> str:
    """Build prompt for Gemini speaker profile table generation."""
    parts = []
    if prompt_prefix:
        parts.append(prompt_prefix.strip())
    if context:
        parts.append(context.strip())
    if processed_text:
        parts.append("The following is the processed story text with speaker tags already applied. Use it to understand each character's personality, role, and speaking style when building their profile:\n\n" + processed_text.strip())
    speaker_line = ", ".join([s for s in speakers if s])
    if speaker_line:
        parts.append(speaker_line)
    return "\n\n".join(parts).strip()


def parse_gemini_speaker_table(text: str) -> Dict[str, Dict[str, str]]:
    """Parse a 3-column table response into a map keyed by normalized speaker name."""
    if not text:
        return {}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return {}
    rows = []
    for line in lines:
        if line.startswith('|') and line.endswith('|'):
            parts = [part.strip() for part in line.strip('|').split('|')]
            if len(parts) >= 3:
                rows.append(parts[:3])
            continue
        if '|' in line:
            parts = [part.strip() for part in line.split('|') if part.strip()]
            if len(parts) >= 3:
                rows.append(parts[:3])
    if not rows:
        return {}
    if rows and all('---' in col for col in rows[0]):
        rows = rows[1:]
    if rows and rows[0][0].lower().startswith('character'):
        rows = rows[1:]
    profiles = {}
    for name, description, voice in rows:
        key = (name or '').strip().lower()
        if not key:
            continue
        profiles[key] = {
            "name": name.strip(),
            "description": description.strip(),
            "voice": voice.strip(),
        }
    return profiles


def _is_chatterbox_engine(engine_name: str) -> bool:
    normalized = _normalize_engine_name(engine_name)
    return normalized.startswith("chatterbox")


def _create_text_processor_for_engine(engine_name: str, chunk_size: int, config: Optional[Dict] = None) -> TextProcessor:
    if _is_chatterbox_engine(engine_name):
        # Use configurable chunk size for Chatterbox, default 450
        chatterbox_chunk_size = 450
        if config:
            chatterbox_chunk_size = config.get("chatterbox_turbo_local_chunk_size", 450)
        # Hard limit is soft limit + 50 to allow sentence completion
        return TextProcessor(
            chunk_strategy="characters",
            char_soft_limit=chatterbox_chunk_size,
            char_hard_limit=chatterbox_chunk_size + 50,
        )
    # Kokoro works best with smaller character-based chunks for faster processing
    if _normalize_engine_name(engine_name) == "kokoro":
        kokoro_chunk_size = 500  # ~80-100 words, generates ~30-40s of audio
        if config:
            kokoro_chunk_size = config.get("kokoro_chunk_size", 500)
        return TextProcessor(
            chunk_strategy="characters",
            char_soft_limit=kokoro_chunk_size,
            char_hard_limit=kokoro_chunk_size + 50,
        )
    if _normalize_engine_name(engine_name) in {"pocket_tts", "pocket_tts_preset"}:
        pocket_chunk_size = 450
        if config:
            pocket_chunk_size = config.get("pocket_tts_chunk_size", pocket_chunk_size)
        return TextProcessor(
            chunk_strategy="characters",
            char_soft_limit=pocket_chunk_size,
            char_hard_limit=pocket_chunk_size + 50,
        )
    if _normalize_engine_name(engine_name) in {"qwen3_custom", "qwen3_clone"}:
        qwen_chunk_size = 500
        if config:
            qwen_chunk_size = config.get("qwen3_chunk_size", qwen_chunk_size)
        return TextProcessor(
            chunk_strategy="characters",
            char_soft_limit=qwen_chunk_size,
            char_hard_limit=qwen_chunk_size + 50,
        )
    if _normalize_engine_name(engine_name) == "voxcpm_local":
        voxcpm_chunk_size = 550
        if config:
            voxcpm_chunk_size = config.get("voxcpm_chunk_size", voxcpm_chunk_size)
        return TextProcessor(
            chunk_strategy="characters",
            char_soft_limit=voxcpm_chunk_size,
            char_hard_limit=voxcpm_chunk_size + 50,
        )
    if _normalize_engine_name(engine_name) == "kitten_tts":
        kitten_chunk_size = 300
        if config:
            kitten_chunk_size = config.get("kitten_tts_chunk_size", kitten_chunk_size)
        return TextProcessor(
            chunk_strategy="characters",
            char_soft_limit=kitten_chunk_size,
            char_hard_limit=kitten_chunk_size + 50,
        )
    if _normalize_engine_name(engine_name) == "index_tts":
        index_chunk_size = 400
        if config:
            index_chunk_size = config.get("index_tts_chunk_size", index_chunk_size)
        return TextProcessor(
            chunk_strategy="characters",
            char_soft_limit=index_chunk_size,
            char_hard_limit=index_chunk_size + 100,
        )
    return TextProcessor(chunk_size=chunk_size)


def estimate_total_chunks(
    text: str,
    split_by_chapter: bool,
    chunk_size: int,
    include_full_story: bool = False,
    engine_name: Optional[str] = None,
    config: Optional[Dict] = None,
    custom_heading: Optional[Any] = None,
) -> int:
    """Estimate total chunk count for a job to power progress indicators."""
    processor = _create_text_processor_for_engine(engine_name or DEFAULT_CONFIG["tts_engine"], chunk_size, config)
    sections = [{"content": text}]
    if split_by_chapter:
        detected = split_text_into_sections(text, custom_heading)
        if detected:
            sections = detected

    total_chunks = 0
    for section in sections:
        section_text = (section.get("content") or "").strip()
        if not section_text:
            continue
        segments = processor.process_text(section_text)
        for segment in segments:
            total_chunks += len(segment.get("chunks", []))

    return max(total_chunks, 1)


def save_job_metadata(job_dir: Path, metadata: dict):
    """Persist metadata for generated outputs."""
    metadata_path = job_dir / JOB_METADATA_FILENAME
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    invalidate_library_cache()


def load_job_metadata(job_dir: Path):
    """Load metadata for a generated job if it exists."""
    metadata_path = job_dir / JOB_METADATA_FILENAME
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as err:
            logger.warning(f"Failed to load metadata from {metadata_path}: {err}")
    return None


def handle_remove_readonly(func, path, exc_info):
    """Handle read-only files on Windows when deleting directories"""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as err:  # pragma: no cover - safeguard
        logger.error(f"Failed to remove read-only attribute for {path}: {err}")

def process_job_worker():
    """Background worker that processes jobs from the queue"""
    global current_job_id
    
    logger.info("Job worker thread started")
    
    while True:
        try:
            # Get next job from queue (blocking)
            job_data = job_queue.get(timeout=1)
            
            if job_data is None:  # Poison pill to stop thread
                logger.info("Job worker thread stopping")
                break
            
            job_id = job_data['job_id']
            
            # Check if job was cancelled while in queue
            if cancel_flags.get(job_id, False):
                logger.info(f"Job {job_id} was cancelled before processing")
                with queue_lock:
                    jobs[job_id]['status'] = 'cancelled'
                _persist_job_state(job_id)
                job_queue.task_done()
                continue

            with queue_lock:
                job_entry = jobs.get(job_id)
                if not job_entry:
                    job_queue.task_done()
                    continue
                if job_entry.get("status") in {"paused", "archived", "deleted", "cancelled"}:
                    job_queue.task_done()
                    continue
            
            # Set as current job
            with queue_lock:
                current_job_id = job_id
                jobs[job_id]['status'] = 'processing'
                jobs[job_id]['started_at'] = datetime.now().isoformat()
            _persist_job_state(job_id)
            
            logger.info(f"Processing job {job_id}")
            
            # Process the job
            try:
                job_type = job_data.get('job_type') or 'audio'
                if job_type == 'audio':
                    process_audio_job(job_data)
                elif job_type == 'qwen3_voice_design_preview':
                    process_qwen3_voice_design_preview_task(job_data)
                elif job_type == 'qwen3_voice_design_save':
                    process_qwen3_voice_design_save_task(job_data)
                else:
                    raise ValueError(f"Unsupported job type: {job_type}")
            except Exception as e:
                logger.error(f"Error processing job {job_id}: {e}", exc_info=True)
                with queue_lock:
                    jobs[job_id]['status'] = 'failed'
                    jobs[job_id]['error'] = str(e)
            
            # Clear current job
            with queue_lock:
                current_job_id = None
            
            job_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Worker thread error: {e}", exc_info=True)
            time.sleep(1)


def _apply_word_replacements(text: str, replacements: list) -> str:
    """Apply a list of {original, replacement} substitutions to text (case-insensitive)."""
    if not replacements or not text:
        return text
    import re as _re
    for entry in replacements:
        original = entry.get('original', '')
        replacement = entry.get('replacement', '')
        if not original or not replacement:
            continue
        try:
            escaped = _re.escape(original)
            text = _re.sub(escaped, replacement, text, flags=_re.IGNORECASE)
        except Exception:
            pass
    return text



def process_audio_job(job_data):
    """Process a single audio generation job"""
    job_id = job_data['job_id']
    text = job_data['text']
    voice_assignments = job_data['voice_assignments']
    config = job_data['config']
    split_by_chapter = job_data.get('split_by_chapter', False)
    generate_full_story = job_data.get('generate_full_story', False)
    custom_heading = job_data.get('custom_heading')
    word_replacements = job_data.get('word_replacements') or []
    job_log = None
    _job_log_handler = None

    try:
        # Check for cancellation
        if cancel_flags.get(job_id, False):
            raise JobCancelled()
        
        review_mode = bool(job_data.get('review_mode', False))
        merge_options_override = job_data.get('merge_options') or {}
        processor = _create_text_processor_for_engine(config.get("tts_engine"), config["chunk_size"], config)
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Per-job file logger — writes job.log inside the job output directory.
        _job_log_path = job_dir / "job.log"
        _job_log_handler = logging.FileHandler(_job_log_path, encoding="utf-8")
        _job_log_handler.setLevel(logging.DEBUG)
        _job_log_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        job_log = logging.getLogger(f"{__name__}.job.{job_id}")
        job_log.setLevel(logging.DEBUG)
        job_log.propagate = False  # write only to file; global logger still handles terminal
        job_log.addHandler(_job_log_handler)

        job_log.info("=" * 72)
        job_log.info("JOB STARTED  id=%s", job_id)
        job_log.info("engine=%s  review_mode=%s  split_by_chapter=%s  full_story=%s",
                     config.get("tts_engine"), review_mode, split_by_chapter, generate_full_story)
        job_log.info("total_chunks=%s  resume_from=%s  chunk_size=%s",
                     job_data.get('total_chunks'), job_data.get('resume_from_chunk_index', 0),
                     config.get('chunk_size'))
        job_log.info("output_format=%s  speed=%s  text_length=%d chars",
                     config.get('output_format'), config.get('speed'), len(text or ''))
        speakers = list(voice_assignments.keys()) if voice_assignments else []
        job_log.info("speakers=%s", speakers)

        with queue_lock:
            job_entry = jobs.get(job_id)
            if job_entry is not None:
                job_entry['job_dir'] = str(job_dir)
        total_chunks = max(1, job_data.get('total_chunks') or jobs.get(job_id, {}).get('total_chunks') or 1)
        resume_from_chunk_index = int(job_data.get("resume_from_chunk_index") or 0)
        processed_chunks = max(0, min(resume_from_chunk_index, total_chunks))
        job_start_time = datetime.now()
        cancel_event = cancel_events.setdefault(job_id, threading.Event())
        remaining_skip = processed_chunks
        _chunk_done_ts: List[float] = []  # monotonic timestamp when each chunk completed

        # Restore historical chunk_times from a prior run so pause/resume preserves the full history.
        # We reconstruct synthetic monotonic timestamps spaced by the saved per-chunk durations.
        _prior_chunk_times: List[float] = []
        with queue_lock:
            _prior_tm = (jobs.get(job_id) or {}).get("timing_metrics") or {}
            _prior_chunk_times = _prior_tm.get("chunk_times") or []
        if _prior_chunk_times and resume_from_chunk_index > 0:
            _synthetic_base = time.monotonic()
            _offset = 0.0
            for _dt in _prior_chunk_times:
                _offset += max(0.0, float(_dt))
                _chunk_done_ts.append(_synthetic_base - (sum(_prior_chunk_times) - _offset))
            # Shift all timestamps so the last one is just before now
            _shift = time.monotonic() - _chunk_done_ts[-1] - 0.001
            _chunk_done_ts = [t + _shift for t in _chunk_done_ts]

        with queue_lock:
            job_entry = jobs.get(job_id)
            if job_entry is not None:
                if not job_entry.get("chunks"):
                    existing_chunks = _load_chunks_metadata(job_dir)
                    if existing_chunks:
                        job_entry["chunks"] = existing_chunks
                job_entry["resume_from_chunk_index"] = resume_from_chunk_index
                if resume_from_chunk_index > 0:
                    job_entry["last_completed_chunk_index"] = resume_from_chunk_index - 1
                job_entry["processed_chunks"] = processed_chunks
                job_entry["total_chunks"] = total_chunks
                job_entry["progress"] = int((processed_chunks / total_chunks) * 100) if total_chunks else 0
        _persist_job_state(job_id)

        def update_progress(increment: int = 1):
            if cancel_flags.get(job_id, False):
                raise JobCancelled()
            nonlocal processed_chunks
            processed_chunks += increment
            processed_chunks = min(processed_chunks, total_chunks)
            if increment > 0:
                _chunk_done_ts.append(time.monotonic())
            elapsed = max((datetime.now() - job_start_time).total_seconds(), 0.001)
            remaining = max(total_chunks - processed_chunks, 0)
            eta_seconds = None
            if processed_chunks and remaining:
                eta_seconds = int((elapsed / processed_chunks) * remaining)
            elif remaining == 0:
                eta_seconds = 0

            percent = int((processed_chunks / total_chunks) * 100)
            percent = max(0, min(100, percent))

            # Build live timing snapshot so job details always show current metrics
            live_tm = None
            if increment > 0 and len(_chunk_done_ts) >= 1:
                _ct: List[float] = []
                if len(_chunk_done_ts) >= 2:
                    _ct = [_chunk_done_ts[i] - _chunk_done_ts[i - 1] for i in range(1, len(_chunk_done_ts))]
                live_tm = {
                    "started_at": job_start_time.isoformat(),
                    "completed_at": None,
                    "total_seconds": round(elapsed, 1),
                    "chunk_count": len(_chunk_done_ts),
                    "avg_chunk_seconds": round(sum(_ct) / len(_ct), 1) if _ct else None,
                    "min_chunk_seconds": round(min(_ct), 1) if _ct else None,
                    "max_chunk_seconds": round(max(_ct), 1) if _ct else None,
                    "chunk_times": [round(t, 2) for t in _ct],
                }

            with queue_lock:
                job_entry = jobs.get(job_id)
                if job_entry:
                    job_entry['processed_chunks'] = processed_chunks
                    job_entry['total_chunks'] = total_chunks
                    job_entry['progress'] = percent if job_entry.get('status') != 'completed' else 100
                    job_entry['eta_seconds'] = eta_seconds
                    job_entry['last_update'] = datetime.now().isoformat()
                    if increment > 0:
                        job_entry['last_completed_chunk_index'] = max(0, processed_chunks - 1)
                        job_entry['resume_from_chunk_index'] = processed_chunks
                        if live_tm is not None:
                            job_entry['timing_metrics'] = live_tm

            _persist_job_state(job_id)

            if pause_flags.get(job_id, False):
                raise JobPaused()

        def pause_cb() -> bool:
            return bool(pause_flags.get(job_id, False))

        def init_post_process(total_steps: int):
            with queue_lock:
                job_entry = jobs.get(job_id)
                if job_entry:
                    job_entry['post_process_total'] = max(int(total_steps or 0), 0)
                    job_entry['post_process_done'] = 0
                    job_entry['post_process_percent'] = 0
                    job_entry['post_process_active'] = True

        def update_post_process(increment: int = 1):
            with queue_lock:
                job_entry = jobs.get(job_id)
                if job_entry:
                    total_steps = int(job_entry.get('post_process_total') or 0)
                    done_steps = int(job_entry.get('post_process_done') or 0) + increment
                    job_entry['post_process_done'] = min(done_steps, total_steps) if total_steps else done_steps
                    if total_steps:
                        job_entry['post_process_percent'] = int(min((job_entry['post_process_done'] / total_steps) * 100, 100))
                    job_entry['post_process_active'] = True

        def update_post_process_progress(start_offset: int, chunk_count: int, ratio: float):
            with queue_lock:
                job_entry = jobs.get(job_id)
                if job_entry:
                    total_steps = int(job_entry.get('post_process_total') or 0)
                    if total_steps <= 0:
                        return
                    clamped = max(0.0, min(float(ratio or 0), 1.0))
                    current_done = int(min(start_offset + (clamped * max(chunk_count, 1)), total_steps))
                    overall = current_done / total_steps
                    job_entry['post_process_done'] = current_done
                    job_entry['post_process_percent'] = int(min(overall * 100, 100))
                    job_entry['post_process_active'] = True
        
        # Determine sections when requested
        chapter_sections = [{"title": "Full Story", "content": text}]
        book_sections = []
        book_mode = False
        if split_by_chapter:
            hierarchy = split_text_into_book_sections(text, custom_heading)
            if hierarchy.get("kind") == "book" and hierarchy.get("books"):
                book_mode = True
                book_sections = hierarchy.get("books") or []
                job_log.info("Text split into book mode: %d books", len(book_sections))
            elif hierarchy.get("sections"):
                chapter_sections = hierarchy.get("sections")
                job_log.info("Text split into %d chapter(s)", len(chapter_sections))
            else:
                logger.info("Section splitting enabled but no headings detected; falling back to single output")
                job_log.info("Section splitting enabled but no headings detected — falling back to single output")
                split_by_chapter = False
        
        chapter_count = len(chapter_sections)
        book_count = len(book_sections) if book_mode else 0
        with queue_lock:
            job_entry = jobs.get(job_id)
            if job_entry:
                job_entry['chapter_count'] = chapter_count
                job_entry['chapter_mode'] = split_by_chapter
                job_entry['full_story_requested'] = generate_full_story
                job_entry['book_mode'] = book_mode
                job_entry['book_count'] = book_count
                job_entry['post_process_total'] = 0
                job_entry['post_process_done'] = 0
                job_entry['post_process_active'] = False
        
        output_format = config['output_format']
        crossfade_seconds = float(config.get('crossfade_duration', 0) or 0)
        merger = None if review_mode else AudioMerger(
            crossfade_ms=int(max(0.0, crossfade_seconds) * 1000),
            intro_silence_ms=int(max(0, config.get('intro_silence_ms', 0) or 0)),
            inter_chunk_silence_ms=int(max(0, config.get('inter_chunk_silence_ms', 0) or 0)),
            bitrate_kbps=int(config.get('output_bitrate_kbps') or 0)
        )
        chapter_outputs = []
        full_story_entry = None
        all_full_story_chunks = [] if (split_by_chapter and generate_full_story) else None
        chunk_dirs_to_cleanup = []
        review_manifest = {
            "chapter_mode": split_by_chapter,
            "book_mode": book_mode,
            "chapters": [],
            "books": [],
            "full_story_requested": generate_full_story,
            "output_format": output_format,
            "chunk_dirs_to_cleanup": [],
            "all_full_story_chunks": [],
        }

        # On resume: restore chapter/book entries that were recorded before the pause.
        # These were saved to review_manifest_partial.json when the job was paused.
        if resume_from_chunk_index > 0 and review_mode:
            _partial_manifest_path = job_dir / "review_manifest_partial.json"
            if _partial_manifest_path.exists():
                try:
                    with _partial_manifest_path.open("r", encoding="utf-8") as _fh:
                        _partial = json.load(_fh)
                    review_manifest["chapters"] = list(_partial.get("chapters") or [])
                    review_manifest["books"] = list(_partial.get("books") or [])
                    review_manifest["chunk_dirs_to_cleanup"] = list(_partial.get("chunk_dirs_to_cleanup") or [])
                    review_manifest["all_full_story_chunks"] = list(_partial.get("all_full_story_chunks") or [])
                    if all_full_story_chunks is not None:
                        all_full_story_chunks = [
                            str(job_dir / rel) for rel in review_manifest["all_full_story_chunks"]
                        ]
                    logger.info(
                        "Job %s: Restored partial review_manifest — %d chapters, %d books from before pause",
                        job_id, len(review_manifest["chapters"]), len(review_manifest["books"]),
                    )
                except Exception as _e:
                    logger.warning("Job %s: Could not load partial review_manifest: %s", job_id, _e)
        
        # Prepare TTS engine
        engine_name = _normalize_engine_name(config.get("tts_engine"))
        logger.info("Job %s: Creating TTS engine '%s'", job_id, engine_name)
        job_log.info("Initializing TTS engine: %s", engine_name)
        engine = get_tts_engine(engine_name, config=config)
        _dev = getattr(engine, 'device', 'unknown')
        logger.info("Job %s: Engine device = %s", job_id, _dev)
        job_log.info("Engine ready — device=%s", _dev)

        # For IndexTTS with split_by_chapter: pre-collect ALL chapters into one
        # subprocess call to avoid per-chapter model-reload overhead (~30-60s each).
        # Results are cached by output_path; generate_chunks returns from cache.
        _prebuilt_audio_cache: Dict[str, str] = {}  # output_path -> file_path (same value)

        job_chunks: List[Dict[str, Any]] = []
        with queue_lock:
            existing_job_chunks = list(jobs.get(job_id, {}).get("chunks") or [])
        if existing_job_chunks:
            job_chunks = existing_job_chunks

        def register_chunk(chapter_idx: int, chunk_idx: int, segment: Dict[str, Any], file_path: str):
            chunk_id = f"{chapter_idx}-{chunk_idx}-{len(job_chunks)}"
            speaker_name = segment.get("speaker")
            speaker_assignment = None
            if voice_assignments:
                candidate = voice_assignments.get(speaker_name) or voice_assignments.get("default")
                speaker_assignment = _clone_voice_assignment(candidate)
            voice_label = _voice_label_from_assignment(speaker_assignment)
            order_index = segment.get("order_index")
            record = {
                "id": chunk_id,
                "order_index": order_index if order_index is not None else len(job_chunks),
                "chapter_index": chapter_idx,
                "chunk_index": chunk_idx,
                "speaker": segment.get("speaker"),
                "engine": engine_name,
                "emotion": segment.get("emotion"),
                "text": segment.get("text"),
                "file_path": file_path,
                "relative_file": os.path.relpath(file_path, job_dir),
                "duration_seconds": segment.get("duration_seconds"),
                "voice_assignment": speaker_assignment,
            }
            if voice_label:
                record["voice_label"] = voice_label
            job_chunks.append(record)
            with queue_lock:
                job_entry = jobs.get(job_id)
                if job_entry is not None:
                    job_entry.setdefault("chunks", []).append(record)

        def make_chunk_callback(chapter_idx: int, output_files: Optional[List[str]] = None):
            def chunk_cb(chunk_idx: int, segment: Dict[str, Any], file_path: str):
                register_chunk(chapter_idx, chunk_idx, segment, file_path)
                if output_files is not None:
                    output_files.append(file_path)
                update_progress(0)  # progress_cb already increments; just register chunk
            return chunk_cb

        def run_with_cancel(operation):
            if cancel_event.is_set() or cancel_flags.get(job_id, False):
                raise JobCancelled()
            result = operation()
            if cancel_event.is_set() or cancel_flags.get(job_id, False):
                raise JobCancelled()
            return result

        def _apply_chunk_skip(segments: List[Dict[str, Any]], skip_count: int) -> tuple[List[Dict[str, Any]], int]:
            remaining = skip_count
            filtered: List[Dict[str, Any]] = []
            for segment in segments:
                chunks = segment.get("chunks") or []
                if remaining >= len(chunks):
                    remaining -= len(chunks)
                    continue
                if remaining > 0:
                    chunks = chunks[remaining:]
                    remaining = 0
                if chunks:
                    updated = dict(segment)
                    updated["chunks"] = chunks
                    filtered.append(updated)
            return filtered, skip_count - remaining

        def generate_chunks(chapter_idx: int, section_text: str, output_dir: Path):
            if cancel_flags.get(job_id, False):
                raise JobCancelled()

            # IndexTTS pre-build: all chunks were already generated in one subprocess.
            # Return the cached files for this chapter's output_dir directly.
            if _prebuilt_audio_cache:
                _out_dir_str = str(output_dir.resolve()).lower()
                cached = sorted(
                    [p for p in _prebuilt_audio_cache
                     if str(Path(p).parent.resolve()).lower() == _out_dir_str],
                    key=lambda p: p,
                )
                if cached:
                    return cached

            segments = processor.process_text(section_text)
            if not segments:
                return []
            nonlocal remaining_skip
            if remaining_skip > 0:
                segments, skipped = _apply_chunk_skip(segments, remaining_skip)
                remaining_skip = max(0, remaining_skip - skipped)
                if not segments:
                    return []
            output_dir.mkdir(parents=True, exist_ok=True)
            # Apply word replacements to each chunk before TTS submission
            if word_replacements:
                for segment in segments:
                    segment["chunks"] = [
                        _apply_word_replacements(chunk, word_replacements)
                        for chunk in (segment.get("chunks") or [])
                    ]
            generated_files: List[str] = []
            chunk_cb = make_chunk_callback(chapter_idx, generated_files)
            supports_chunk_cb = False
            flat_segments: List[Dict[str, Any]] = []
            order_index = 0
            for seg_idx, segment in enumerate(segments):
                speaker = segment.get("speaker")
                chunks = segment.get("chunks") or []
                for chunk_idx, chunk_text in enumerate(chunks):
                    flat_segments.append({
                        "segment_index": seg_idx,
                        "chunk_index": chunk_idx,
                        "order_index": order_index,
                        "speaker": speaker,
                        "text": chunk_text,
                        "emotion": segment.get("emotion"),
                    })
                    order_index += 1
            engine_kwargs = {
                "segments": segments,
                "voice_config": voice_assignments,
                "output_dir": str(output_dir),
                "speed": config['speed'],
                "progress_cb": update_progress,
            }
            sig_params = inspect.signature(engine.generate_batch).parameters
            if "sample_rate" in sig_params:
                engine_kwargs["sample_rate"] = config.get("sample_rate")
            if "chunk_cb" in sig_params:
                engine_kwargs["chunk_cb"] = chunk_cb
                supports_chunk_cb = True
            if "parallel_workers" in sig_params:
                engine_kwargs["parallel_workers"] = max(1, min(8, int(config.get("parallel_chunks", 1) or 1)))
            if "pause_cb" in sig_params:
                engine_kwargs["pause_cb"] = pause_cb
            if "cancel_cb" in sig_params:
                engine_kwargs["cancel_cb"] = lambda: bool(cancel_flags.get(job_id, False))
            if "group_by_speaker" in sig_params:
                engine_kwargs["group_by_speaker"] = bool(config.get("group_chunks_by_speaker", False))
            _total_text_chunks = sum(len(seg.get("chunks") or []) for seg in segments)
            if job_log:
                job_log.info("  chapter_idx=%d  segments=%d  text_chunks=%d  output_dir=%s",
                             chapter_idx, len(segments), _total_text_chunks, output_dir)
            audio_files = run_with_cancel(lambda: engine.generate_batch(**engine_kwargs))
            if not audio_files and generated_files:
                audio_files = list(generated_files)
            if job_log:
                job_log.info("  engine returned %d audio file(s) for chapter_idx=%d",
                             len(audio_files) if audio_files else 0, chapter_idx)

            if not supports_chunk_cb and audio_files:
                for order_idx, file_path in enumerate(audio_files):
                    if order_idx >= len(flat_segments):
                        break
                    descriptor = flat_segments[order_idx]
                    register_chunk(
                        chapter_idx,
                        descriptor["chunk_index"],
                        descriptor,
                        file_path,
                    )
                    update_progress(1)
            return audio_files

        def _prebuild_index_tts_all_chapters():
            """Collect all chapter segments and run a single IndexTTS subprocess.

            Eliminates per-chapter model-reload overhead (~30-60s each) by sending
            all chunks to one tts_worker.py process. Results cached in
            _prebuilt_audio_cache; generate_chunks returns from cache without
            re-launching the engine.
            """
            if not hasattr(engine, 'generate_batch_prebuilt'):
                return
            if not split_by_chapter:
                return
            from src.engines.index_tts_engine import IndexTTSEngine  # noqa: F401
            if not isinstance(engine, IndexTTSEngine):
                return

            logger.info("Job %s: IndexTTS batch mode — pre-collecting all chapters into one subprocess", job_id)

            all_worker_chunks: List[Dict] = []
            all_chunk_meta: List[Dict] = []
            global_order = 0

            # Build parallel lists: section texts and their target chunk dirs
            sections_to_process: List[str] = []
            chapter_output_dirs: List[Path] = []

            if book_mode:
                for book_idx, book in enumerate(book_sections, start=1):
                    _ch_folder = 1
                    for chapter in (book.get("chapters") or []):
                        sections_to_process.append(chapter.get("content", ""))
                        if (chapter.get("title") or "").strip().lower() == "title":
                            _cdir = job_dir / f"book_{book_idx:02d}" / "title"
                        else:
                            _cdir = job_dir / f"book_{book_idx:02d}" / f"chapter_{_ch_folder:02d}"
                            _ch_folder += 1
                        chapter_output_dirs.append(_cdir / "chunks")
            else:
                _ch_folder = 1
                for chapter in chapter_sections:
                    sections_to_process.append(chapter.get("content", ""))
                    if (chapter.get("title") or "").strip().lower() == "title":
                        _cdir = job_dir / "title"
                    else:
                        _cdir = job_dir / f"chapter_{_ch_folder:02d}"
                        _ch_folder += 1
                    chapter_output_dirs.append(_cdir / "chunks")

            _skip_remaining = int(job_data.get("resume_from_chunk_index") or 0)
            for ch_idx, (section_text, chunk_dir) in enumerate(zip(sections_to_process, chapter_output_dirs)):
                segments = processor.process_text(section_text)
                if not segments:
                    continue
                if _skip_remaining > 0:
                    segments, skipped = _apply_chunk_skip(segments, _skip_remaining)
                    _skip_remaining = max(0, _skip_remaining - skipped)
                    if not segments:
                        continue
                if word_replacements:
                    for seg in segments:
                        seg["chunks"] = [
                            _apply_word_replacements(c, word_replacements)
                            for c in (seg.get("chunks") or [])
                        ]
                chunk_dir.mkdir(parents=True, exist_ok=True)
                local_idx = 0
                for seg_idx, segment in enumerate(segments):
                    speaker = segment.get("speaker")
                    assignment = engine._voice_assignment_for(voice_assignments, speaker)
                    spk_prompt = engine._resolve_prompt(assignment)
                    for chunk_text in (segment.get("chunks") or []):
                        out_path = chunk_dir / f"chunk_{local_idx:04d}.wav"
                        all_worker_chunks.append({
                            "text": chunk_text,
                            "spk_audio_prompt": spk_prompt,
                            "output_path": str(out_path),
                            "_order_index": global_order,
                        })
                        all_chunk_meta.append({
                            "speaker": speaker,
                            "text": chunk_text,
                            "segment_index": seg_idx,
                            "chunk_index": local_idx,
                            "chapter_index": ch_idx,
                            "output_path": str(out_path),
                            "assignment": assignment,
                            "_order_index": global_order,
                        })
                        local_idx += 1
                        global_order += 1

            if not all_worker_chunks:
                return

            logger.info(
                "Job %s: IndexTTS single-subprocess batch — %d total chunks across %d chapters",
                job_id, len(all_worker_chunks), len(sections_to_process),
            )

            group_spk = bool(config.get("group_chunks_by_speaker", False))

            def _batch_chunk_cb(chunk_idx: int, segment: Dict[str, Any], file_path: str):
                chapter_idx = segment.get("chapter_index", 0)
                make_chunk_callback(chapter_idx)(chunk_idx, segment, file_path)

            engine.generate_batch_prebuilt(
                worker_chunks=all_worker_chunks,
                chunk_meta=all_chunk_meta,
                progress_cb=update_progress,
                chunk_cb=_batch_chunk_cb,
                pause_cb=pause_cb,
                cancel_cb=lambda: bool(cancel_flags.get(job_id, False)),
                group_by_speaker=group_spk,
            )

            for item in all_worker_chunks:
                p = item["output_path"]
                if Path(p).exists():
                    _prebuilt_audio_cache[p] = p

            logger.info("Job %s: IndexTTS pre-build complete — %d files cached", job_id, len(_prebuilt_audio_cache))

        _prebuild_index_tts_all_chapters()

        if cancel_flags.get(job_id, False):
            raise JobCancelled()

        if pause_flags.get(job_id, False):
            raise JobPaused()

        try:
            if split_by_chapter:
                if book_mode:
                    if not review_mode:
                        post_total = total_chunks * (2 if generate_full_story else 1)
                        init_post_process(post_total)
                        merge_chunk_offset = 0
                    chapter_global_idx = 0
                    for book_idx, book in enumerate(book_sections, start=1):
                        if cancel_flags.get(job_id, False):
                            raise JobCancelled()

                        book_dir = job_dir / f"book_{book_idx:02d}"
                        book_chunk_files = []
                        rel_book_chunks = []
                        book_chapter_indices = []
                        chapter_folder_idx = 1
                        for chapter_idx, chapter in enumerate(book.get("chapters") or [], start=1):
                            # Skip "Title" sections (book number/title before first chapter)
                            if chapter.get("title") == "Title":
                                chapter_dir = book_dir / "title"
                            else:
                                chapter_dir = book_dir / f"chapter_{chapter_folder_idx:02d}"
                                chapter_folder_idx += 1
                            chunk_dir = chapter_dir / "chunks"
                            audio_files = generate_chunks(chapter_global_idx, chapter["content"], chunk_dir)
                            if not audio_files:
                                logger.warning(f"Book {book_idx} chapter {chapter_idx} had no audio chunks; skipping")
                                job_log.warning("Book %d chapter %d produced no audio chunks — skipping", book_idx, chapter_idx)
                                chapter_global_idx += 1
                                continue

                            chapter_global_idx += 1
                            book_chapter_indices.append(chapter_global_idx - 1)
                            book_chunk_files.extend(audio_files)

                            if all_full_story_chunks is not None:
                                all_full_story_chunks.extend(audio_files)
                                chunk_dirs_to_cleanup.append(chunk_dir)

                            rel_chunk_dir = os.path.relpath(chunk_dir, job_dir)
                            rel_chapter_dir = os.path.relpath(chapter_dir, job_dir)
                            rel_chunk_files = [os.path.relpath(path, job_dir) for path in audio_files]
                            rel_book_chunks.extend(rel_chunk_files)
                            if review_mode:
                                review_manifest["chapters"].append({
                                    "index": chapter_global_idx - 1,  # 0-indexed to match chunk.chapter_index
                                    "title": chapter.get("title"),
                                    "chunk_dir": rel_chunk_dir,
                                    "chunk_files": rel_chunk_files,
                                    "chapter_dir": rel_chapter_dir,
                                    "book_index": book_idx - 1,
                                    "book_title": book.get("title"),
                                    "book_order": chapter_idx - 1,
                                })
                                review_manifest["chunk_dirs_to_cleanup"].append(rel_chunk_dir)

                        if not book_chunk_files:
                            continue

                        slug = slugify_filename(book.get('title'), f"book-{book_idx:02d}")
                        output_filename = f"{slug}.{output_format}"
                        output_path = book_dir / output_filename

                        if review_mode:
                            rel_book_dir = os.path.relpath(book_dir, job_dir)
                            review_manifest["books"].append({
                                "index": book_idx - 1,
                                "title": book.get("title"),
                                "chapter_indices": book_chapter_indices,
                                "chunk_files": rel_book_chunks,
                                "book_dir": rel_book_dir,
                                "output_filename": output_filename,
                            })
                        else:
                            if job_log:
                                job_log.info("Merging book %d: %d chunks -> %s", book_idx, len(book_chunk_files), output_path)
                            merger.merge_wav_files(
                                input_files=book_chunk_files,
                                output_path=str(output_path),
                                format=output_format,
                                cleanup_chunks=not generate_full_story,
                                progress_callback=lambda ratio, offset=merge_chunk_offset, count=len(book_chunk_files): (
                                    update_post_process_progress(offset, count, ratio)
                                ),
                            )
                            if job_log:
                                job_log.info("Merge complete: %s (exists=%s)", output_path.name, output_path.exists())
                            update_progress()
                            update_post_process(len(book_chunk_files))
                            merge_chunk_offset += len(book_chunk_files)

                            # Cleanup empty chunk directory
                            if book_dir.exists() and not generate_full_story:
                                for chapter_dir in book_dir.glob("chapter_*"):
                                    try:
                                        (chapter_dir / "chunks").rmdir()
                                    except OSError:
                                        pass

                            relative_path = Path(f"book_{book_idx:02d}") / output_filename
                            chapter_outputs.append({
                                "index": book_idx,
                                "title": book.get("title"),
                                "file_url": f"/static/audio/{job_id}/{relative_path.as_posix()}",
                                "relative_path": relative_path.as_posix()
                            })
                else:
                    if not review_mode:
                        post_total = total_chunks * (2 if generate_full_story else 1)
                        init_post_process(post_total)
                        merge_chunk_offset = 0
                    chapter_folder_idx = 1
                    for idx, chapter in enumerate(chapter_sections, start=1):
                        if cancel_flags.get(job_id, False):
                            raise JobCancelled()

                        section_title = (chapter.get("title") or "").strip()
                        is_title_section = section_title.lower() == "title"
                        if is_title_section:
                            chapter_dir = job_dir / "title"
                        else:
                            chapter_dir = job_dir / f"chapter_{chapter_folder_idx:02d}"
                            chapter_folder_idx += 1
                        chunk_dir = chapter_dir / "chunks"
                        job_log.info("Generating audio for chapter %d: '%s'", idx, chapter.get('title', ''))
                        audio_files = generate_chunks(idx - 1, chapter["content"], chunk_dir)
                        if not audio_files:
                            logger.warning(f"Chapter {idx} had no audio chunks; skipping")
                            job_log.warning("Chapter %d produced no audio chunks — skipping", idx)
                            continue
                        job_log.info("Chapter %d generated %d chunk file(s)", idx, len(audio_files))

                        if all_full_story_chunks is not None:
                            all_full_story_chunks.extend(audio_files)
                            chunk_dirs_to_cleanup.append(chunk_dir)

                        if is_title_section:
                            slug_default = "title"
                        else:
                            slug_default = f"chapter-{chapter_folder_idx - 1:02d}"
                        slug = slugify_filename(chapter.get('title'), slug_default)
                        output_filename = f"{slug}.{output_format}"
                        output_path = chapter_dir / output_filename

                        if review_mode:
                            rel_chunk_dir = os.path.relpath(chunk_dir, job_dir)
                            rel_chapter_dir = os.path.relpath(chapter_dir, job_dir)
                            rel_chunk_files = [os.path.relpath(path, job_dir) for path in audio_files]
                            review_manifest["chapters"].append({
                                "index": idx - 1,  # 0-indexed to match chunk.chapter_index
                                "title": chapter['title'],
                                "chunk_dir": rel_chunk_dir,
                                "chunk_files": rel_chunk_files,
                                "chapter_dir": rel_chapter_dir,
                                "output_filename": output_filename,
                            })
                            review_manifest["chunk_dirs_to_cleanup"].append(rel_chunk_dir)
                        else:
                            if job_log:
                                job_log.info("Merging chapter %d ('%s'): %d chunks -> %s",
                                             idx, chapter.get('title', ''), len(audio_files), output_path)
                            merger.merge_wav_files(
                                input_files=audio_files,
                                output_path=str(output_path),
                                format=output_format,
                                cleanup_chunks=not generate_full_story,
                                progress_callback=lambda ratio, offset=merge_chunk_offset, count=len(audio_files): (
                                    update_post_process_progress(offset, count, ratio)
                                ),
                            )
                            if job_log:
                                job_log.info("Merge complete: %s (exists=%s)", output_path.name, output_path.exists())
                            update_progress()
                            update_post_process(len(audio_files))
                            merge_chunk_offset += len(audio_files)

                            # Cleanup empty chunk directory
                            if chunk_dir.exists() and not generate_full_story:
                                try:
                                    chunk_dir.rmdir()
                                except OSError:
                                    pass

                            relative_path = Path(chapter_dir.name) / output_filename
                            chapter_outputs.append({
                                "index": idx,
                                "title": chapter['title'],
                                "file_url": f"/static/audio/{job_id}/{relative_path.as_posix()}",
                                "relative_path": relative_path.as_posix()
                            })
            else:
                if not review_mode:
                    init_post_process(total_chunks)
                    merge_chunk_offset = 0
                chunk_dir = job_dir / "chunks"
                job_log.info("Generating audio chunks (single section)")
                audio_files = generate_chunks(0, text, chunk_dir)
                if not audio_files:
                    job_log.error("generate_chunks returned no files — raising ValueError")
                    raise ValueError("Unable to generate audio chunks")
                job_log.info("Generated %d chunk file(s)", len(audio_files))
                output_file = job_dir / f"output.{output_format}"
                if review_mode:
                    rel_chunk_dir = os.path.relpath(chunk_dir, job_dir)
                    rel_chapter_dir = os.path.relpath(job_dir, job_dir)
                    rel_chunk_files = [os.path.relpath(path, job_dir) for path in audio_files]
                    review_manifest["chapters"].append({
                        "index": 1,
                        "title": "Full Story",
                        "chunk_dir": rel_chunk_dir,
                        "chunk_files": rel_chunk_files,
                        "chapter_dir": rel_chapter_dir,
                        "output_filename": output_file.name,
                    })
                    review_manifest["chunk_dirs_to_cleanup"].append(rel_chunk_dir)
                else:
                    if job_log:
                        job_log.info("Merging full story: %d chunks -> %s", len(audio_files), output_file)
                    merger.merge_wav_files(
                        input_files=audio_files,
                        output_path=str(output_file),
                        format=output_format,
                        progress_callback=lambda ratio, offset=merge_chunk_offset, count=len(audio_files): (
                            update_post_process_progress(offset, count, ratio)
                        ),
                    )
                    if job_log:
                        job_log.info("Merge complete: %s (exists=%s)", output_file.name, output_file.exists())
                    update_progress()
                    update_post_process(len(audio_files))
                    if chunk_dir.exists():
                        try:
                            chunk_dir.rmdir()
                        except OSError:
                            pass

                    chapter_outputs.append({
                        "index": 1,
                        "title": "Full Story",
                        "file_url": f"/static/audio/{job_id}/output.{output_format}",
                        "relative_path": f"output.{output_format}"
                    })

        except Exception:
            raise


        if all_full_story_chunks and not review_mode:
            full_story_name = f"full_story.{output_format}"
            full_story_path = job_dir / full_story_name
            merger.merge_wav_files(
                input_files=all_full_story_chunks,
                output_path=str(full_story_path),
                format=output_format,
                progress_callback=lambda ratio, offset=merge_chunk_offset, count=len(all_full_story_chunks): (
                    update_post_process_progress(offset, count, ratio)
                ),
            )
            update_progress()
            update_post_process(len(all_full_story_chunks))
            merge_chunk_offset += len(all_full_story_chunks)

            for chunk_dir in chunk_dirs_to_cleanup:
                if chunk_dir.exists():
                    try:
                        chunk_dir.rmdir()
                    except OSError:
                        pass

            full_story_entry = {
                "title": "Full Story",
                "file_url": f"/static/audio/{job_id}/{full_story_name}",
                "relative_path": full_story_name
            }

        if cancel_flags.get(job_id, False):
            raise JobCancelled()

        if review_mode:
            manifest_path = job_dir / "review_manifest.json"
            review_manifest["all_full_story_chunks"] = [
                os.path.relpath(path, job_dir) for path in (all_full_story_chunks or [])
            ]
            with manifest_path.open("w", encoding="utf-8") as handle:
                json.dump(review_manifest, handle, indent=2)
            chunks_meta_path = job_dir / "chunks_metadata.json"
            _tm_for_meta = (jobs.get(job_id) or {}).get("timing_metrics")
            chunks_meta = {
                "engine": engine_name,
                "created_at": datetime.now().isoformat(),
                "chunks": job_chunks,
            }
            if _tm_for_meta:
                chunks_meta["timing_metrics"] = _tm_for_meta
            with chunks_meta_path.open("w", encoding="utf-8") as handle:
                json.dump(chunks_meta, handle, indent=2)
            
            # Auto-finish: merge audio and complete the job (review happens in library)
            with queue_lock:
                job_entry = jobs.get(job_id)
                if job_entry:
                    job_entry['review_manifest'] = manifest_path.name
                    job_entry['chapter_mode'] = split_by_chapter
                    job_entry['full_story_requested'] = generate_full_story
            
            _merge_review_job(job_id, jobs.get(job_id), review_manifest)
            logger.info(f"Job {job_id} auto-finished and moved to library for chunk review")
            job_log.info("Job auto-finished in review mode — %d total chunks written", len(job_chunks))
            job_log.info("review_manifest.json and chunks_metadata.json saved to: %s", job_dir)
            return

        if not chapter_outputs:
            raise ValueError("No audio outputs were generated")

        metadata = {
            "chapter_mode": split_by_chapter,
            "book_mode": book_mode,
            "output_format": output_format,
            "chapters": chapter_outputs,
            "chapter_count": chapter_count,
            "book_count": book_count,
            "books": chapter_outputs if book_mode else [],
            "full_story": full_story_entry
        }
        save_job_metadata(job_dir, metadata)
        
        # Compute timing metrics from chunk completion timestamps
        job_end_time = datetime.now()
        total_job_seconds = (job_end_time - job_start_time).total_seconds()
        chunk_times: List[float] = []
        # Derive per-chunk render times from consecutive completion timestamps.
        # Use diffs between consecutive timestamps (excludes model-load overhead on first chunk).
        if len(_chunk_done_ts) >= 2:
            chunk_times = [
                _chunk_done_ts[i] - _chunk_done_ts[i - 1]
                for i in range(1, len(_chunk_done_ts))
            ]
        elif len(_chunk_done_ts) == 1 and total_chunks == 1:
            # Single chunk: use total job time as best estimate
            chunk_times = [total_job_seconds]
        avg_chunk_seconds = (sum(chunk_times) / len(chunk_times)) if chunk_times else None
        timing_metrics = {
            "started_at": jobs[job_id].get("started_at"),
            "completed_at": job_end_time.isoformat(),
            "total_seconds": round(total_job_seconds, 1),
            "chunk_count": len(_chunk_done_ts),
            "avg_chunk_seconds": round(avg_chunk_seconds, 1) if avg_chunk_seconds is not None else None,
            "min_chunk_seconds": round(min(chunk_times), 1) if chunk_times else None,
            "max_chunk_seconds": round(max(chunk_times), 1) if chunk_times else None,
            "chunk_times": [round(t, 1) for t in chunk_times],
        }

        # Update job as completed
        with queue_lock:
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['progress'] = 100
            jobs[job_id]['processed_chunks'] = total_chunks
            jobs[job_id]['total_chunks'] = total_chunks
            jobs[job_id]['eta_seconds'] = 0
            jobs[job_id]['post_process_percent'] = 100
            jobs[job_id]['post_process_active'] = False
            primary_output = full_story_entry or (chapter_outputs[0] if chapter_outputs else None)
            jobs[job_id]['output_file'] = primary_output['file_url'] if primary_output else ''
            jobs[job_id]['chapter_outputs'] = chapter_outputs
            jobs[job_id]['chapter_mode'] = split_by_chapter
            jobs[job_id]['full_story_requested'] = generate_full_story
            jobs[job_id]['resume_from_chunk_index'] = total_chunks
            if full_story_entry:
                jobs[job_id]['full_story'] = full_story_entry
            jobs[job_id]['completed_at'] = job_end_time.isoformat()
            jobs[job_id]['timing_metrics'] = timing_metrics
        _persist_job_state(job_id, force=True)

        # Also write timing_metrics into chunks_metadata.json so the library can read it from disk.
        try:
            _cmeta_path = job_dir / "chunks_metadata.json"
            if _cmeta_path.exists():
                with _cmeta_path.open("r", encoding="utf-8") as _fh:
                    _cmeta = json.load(_fh)
                _cmeta["timing_metrics"] = timing_metrics
                with _cmeta_path.open("w", encoding="utf-8") as _fh:
                    json.dump(_cmeta, _fh, indent=2)
        except Exception as _e:
            logger.warning("Could not write timing_metrics to chunks_metadata.json: %s", _e)

        logger.info(f"Job {job_id} completed successfully with {len(chapter_outputs)} output file(s)")
        job_log.info("JOB COMPLETED — %d output file(s)  total_time=%.1fs  chunks=%d",
                     len(chapter_outputs),
                     timing_metrics.get('total_seconds', 0),
                     timing_metrics.get('chunk_count', 0))
        for _out in chapter_outputs:
            job_log.info("  output: %s", _out.get('relative_path', ''))
        
        # Optional VRAM cleanup after job completion
        if config.get("cleanup_vram_after_job", False):
            _cleanup_engine_vram(engine_name)
        
    except JobPaused:
        logger.info(f"Job {job_id} paused – stopping after current chunk")
        if job_log:
            job_log.info("JOB PAUSED at chunk %d / %d", processed_chunks, total_chunks)
        # Save the partial review_manifest so chapter/book entries before the pause
        # are preserved and can be restored when the job is resumed.
        if review_mode:
            try:
                _partial_manifest_path = job_dir / "review_manifest_partial.json"
                # Snapshot all_full_story_chunks as relative paths
                _rel_full_story = [
                    os.path.relpath(p, job_dir) for p in (all_full_story_chunks or [])
                ]
                _partial_to_save = dict(review_manifest)
                _partial_to_save["all_full_story_chunks"] = _rel_full_story
                with _partial_manifest_path.open("w", encoding="utf-8") as _fh:
                    json.dump(_partial_to_save, _fh, indent=2)
                logger.info(
                    "Job %s: Saved partial review_manifest — %d chapters, %d books",
                    job_id, len(review_manifest.get("chapters") or []), len(review_manifest.get("books") or []),
                )
            except Exception as _e:
                logger.warning("Job %s: Could not save partial review_manifest: %s", job_id, _e)
        with queue_lock:
            job_entry = jobs.get(job_id)
            if job_entry:
                job_entry['status'] = 'paused'
                job_entry['paused_at'] = datetime.now().isoformat()
                job_entry['eta_seconds'] = None
                job_entry['last_update'] = datetime.now().isoformat()
        _persist_job_state(job_id, force=True)
        return
    except JobCancelled:
        logger.info(f"Job {job_id} cancelled – halting synthesis")
        if job_log:
            job_log.info("JOB CANCELLED at chunk %d / %d", processed_chunks, total_chunks)
        with queue_lock:
            job_entry = jobs.get(job_id)
            if job_entry:
                job_entry['status'] = 'cancelled'
                job_entry['eta_seconds'] = None
                job_entry['last_update'] = datetime.now().isoformat()
        _persist_job_state(job_id, force=True)
        return
    except Exception as e:
        logger.error(f"Error in job {job_id}: {e}", exc_info=True)
        if job_log:
            job_log.error("JOB FAILED: %s", e, exc_info=True)
        with queue_lock:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['error'] = str(e)
            jobs[job_id]['interrupted_at'] = datetime.now().isoformat()
        _persist_job_state(job_id, force=True)
        raise
    finally:
        cancel_flags.pop(job_id, None)
        cancel_events.pop(job_id, None)
        pause_flags.pop(job_id, None)
        if job_log is not None and _job_log_handler is not None:
            try:
                job_log.removeHandler(_job_log_handler)
                _job_log_handler.close()
            except Exception:
                pass


def _enqueue_qwen3_voice_design_task(task_type: str, payload: Dict[str, Any]) -> str:
    start_worker_thread()
    job_id = str(uuid.uuid4())
    job_entry = {
        "status": "queued",
        "progress": 0,
        "created_at": datetime.now().isoformat(),
        "job_type": task_type,
        "title": "Qwen3 VoiceDesign",
    }
    with queue_lock:
        jobs[job_id] = job_entry
    job_payload = {
        "job_id": job_id,
        "job_type": task_type,
        "payload": payload,
    }
    if task_type == "qwen3_voice_design_preview":
        job_payload["config"] = load_config()
    job_queue.put(job_payload)
    return job_id


def process_qwen3_voice_design_preview_task(job_data: Dict[str, Any]) -> None:
    """Process a queued Qwen3 VoiceDesign preview request."""
    job_id = job_data["job_id"]
    payload = job_data.get("payload") or {}
    config = job_data.get("config") or load_config()
    try:
        result = _generate_voice_design_preview(payload, config)
        with queue_lock:
            job_entry = jobs.get(job_id)
            if job_entry:
                job_entry["status"] = "completed"
                job_entry["progress"] = 100
                job_entry["completed_at"] = datetime.now().isoformat()
                job_entry["result"] = result
    except Exception as exc:
        with queue_lock:
            job_entry = jobs.get(job_id)
            if job_entry:
                job_entry["status"] = "failed"
                job_entry["error"] = str(exc)
        raise


def process_qwen3_voice_design_save_task(job_data: Dict[str, Any]) -> None:
    """Process a queued Qwen3 VoiceDesign save request."""
    job_id = job_data["job_id"]
    payload = job_data.get("payload") or {}
    try:
        result = _save_voice_design_payload(payload)
        with queue_lock:
            job_entry = jobs.get(job_id)
            if job_entry:
                job_entry["status"] = "completed"
                job_entry["progress"] = 100
                job_entry["completed_at"] = datetime.now().isoformat()
                job_entry["result"] = result
    except Exception as exc:
        with queue_lock:
            job_entry = jobs.get(job_id)
            if job_entry:
                job_entry["status"] = "failed"
                job_entry["error"] = str(exc)
        raise


def start_worker_thread():
    """Start the background worker thread"""
    global worker_thread
    if worker_thread is None or not worker_thread.is_alive():
        worker_thread = threading.Thread(target=process_job_worker, daemon=True)
        worker_thread.start()
        logger.info("Worker thread started")


def load_config():
    """Load configuration from file"""
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict):
                config.update({k: v for k, v in data.items() if k in DEFAULT_CONFIG})
        except Exception as exc:
            logger.warning(f"Failed to load config.json, using defaults: {exc}")
    return config


def save_config(config):
    """Save configuration to file"""
    merged = DEFAULT_CONFIG.copy()
    if isinstance(config, dict):
        merged.update({k: v for k, v in config.items() if k in DEFAULT_CONFIG})
    with open(CONFIG_FILE, 'w') as f:
        json.dump(merged, f, indent=2)


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


def _load_chatterbox_voice_entries() -> List[Dict[str, Any]]:
    if not CHATTERBOX_VOICE_REGISTRY.exists():
        return []
    try:
        with CHATTERBOX_VOICE_REGISTRY.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, list):
                return data
            logger.warning("Chatterbox voice registry contains invalid data. Resetting.")
            return []
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Unable to read chatterbox voice registry: %s", exc)
        return []


def _load_external_voice_archives() -> set:
    if not EXTERNAL_VOICES_ARCHIVE_FILE.exists():
        return set()
    try:
        with EXTERNAL_VOICES_ARCHIVE_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, list):
                return {item for item in data if isinstance(item, str)}
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Unable to read external voice archive list: %s", exc)
    return set()


def _save_external_voice_archives(archives: set) -> None:
    EXTERNAL_VOICES_ARCHIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with EXTERNAL_VOICES_ARCHIVE_FILE.open("w", encoding="utf-8") as handle:
        json.dump(sorted(archives), handle, indent=2)


def _save_chatterbox_voice_entries(entries: List[Dict[str, Any]]) -> None:
    CHATTERBOX_VOICE_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with CHATTERBOX_VOICE_REGISTRY.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)


def _cleanup_orphaned_chatterbox_voices() -> int:
    """Remove voice entries from registry where the audio file no longer exists.
    
    Returns the number of orphaned entries removed.
    """
    entries = _load_chatterbox_voice_entries()
    if not entries:
        return 0
    
    valid_entries = []
    removed_count = 0
    for entry in entries:
        file_name = entry.get("file_name")
        if not file_name:
            removed_count += 1
            continue
        file_path = VOICE_PROMPT_DIR / file_name
        if file_path.is_file():
            valid_entries.append(entry)
        else:
            logger.info(f"Removing orphaned Chatterbox voice entry: {entry.get('name', file_name)} (file missing: {file_name})")
            removed_count += 1
    
    if removed_count > 0:
        _save_chatterbox_voice_entries(valid_entries)
        logger.info(f"Cleaned up {removed_count} orphaned Chatterbox voice entries")
    
    return removed_count


def _auto_register_voice_prompt_files() -> int:
    """Scan data/voice_prompts/ and add registry entries for any audio files
    not already tracked in chatterbox_voices.json.

    Returns the number of newly registered entries.
    """
    existing_entries = _load_chatterbox_voice_entries()
    registered_files = {e.get("file_name") for e in existing_entries if e.get("file_name")}

    new_entries = []
    for path in sorted(VOICE_PROMPT_DIR.glob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VOICE_PROMPT_EXTENSIONS:
            continue
        if path.name in registered_files:
            continue
        # Build a friendly display name from the filename stem
        display_name = path.stem.replace("_", " ").replace("-", " ").title()
        entry = {
            "id": str(uuid.uuid4()),
            "name": display_name,
            "file_name": path.name,
            "created_at": datetime.now().isoformat(),
            "gender": None,
            "language": None,
            "description": None,
            "archived": False,
            "size_bytes": None,
            "duration_seconds": None,
        }
        new_entries.append(entry)
        logger.info("Auto-registered voice prompt: %s -> %s", path.name, display_name)

    if new_entries:
        _save_chatterbox_voice_entries(existing_entries + new_entries)
        logger.info("Auto-registered %d voice prompt file(s) from disk", len(new_entries))

    return len(new_entries)


def _slugify_filename(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return value or "voice"


def _serialize_chatterbox_voice(entry: Dict[str, Any]) -> Dict[str, Any]:
    file_name = entry.get("file_name")
    file_path = VOICE_PROMPT_DIR / file_name if file_name else None
    exists = file_path.is_file() if file_path else False
    size_bytes = entry.get("size_bytes")
    duration_seconds = entry.get("duration_seconds")
    if exists:
        try:
            size_bytes = file_path.stat().st_size
            if duration_seconds is None:
                duration_seconds = _measure_audio_duration(file_path)
        except OSError:
            size_bytes = None
            duration_seconds = None
    return {
        "id": entry.get("id"),
        "name": entry.get("name"),
        "file_name": file_name,
        "prompt_path": file_name,
        "created_at": entry.get("created_at"),
        "size_bytes": size_bytes,
        "missing_file": not exists,
        "duration_seconds": duration_seconds,
        "is_valid_prompt": bool(duration_seconds and duration_seconds >= MIN_CHATTERBOX_PROMPT_SECONDS),
        "gender": entry.get("gender"),  # Male, Female, or None
        "language": entry.get("language"),  # Language code like en-US
        "description": entry.get("description"),
        "archived": bool(entry.get("archived", False)),
        "source": "local",  # local voices vs external
    }


def _resolve_chatterbox_voice(entry_id: str, entries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for entry in entries:
        if entry.get("id") == entry_id:
            return entry
    return None


@app.route('/api/voices', methods=['GET'])
def get_voices():
    """Get available voices"""
    voice_manager = VoiceManager()
    missing = voice_manager.missing_samples()
    return jsonify({
        "success": True,
        "voices": voice_manager.get_all_voices(),
        "samples_ready": voice_manager.all_samples_present(),
        "missing_samples": missing,
        "total_unique_voices": voice_manager.total_unique_voice_count(),
        "sample_count": voice_manager.sample_count()
    })


@app.route('/api/pocket-tts/voices', methods=['GET'])
def get_pocket_tts_voices():
    """Return built-in Pocket TTS preset voices."""
    return jsonify({
        "success": True,
        "voices": POCKET_TTS_PRESET_VOICES,
    })


def _get_audio_duration(file_path: Path) -> Optional[float]:
    """Get audio duration in seconds using pydub."""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(str(file_path))
        return len(audio) / 1000.0  # pydub returns milliseconds
    except Exception as e:
        logger.debug("Could not get duration for %s: %s", file_path.name, e)
        return None


# Cache for voice prompt durations to avoid re-reading files
_voice_prompt_duration_cache: Dict[str, float] = {}
_voice_prompt_transcript_cache: Optional[Dict[str, str]] = None


def _load_voice_prompt_transcripts() -> Dict[str, str]:
    """Load cached prompt transcripts from data/voice_prompts/transcripts.json."""
    global _voice_prompt_transcript_cache
    if _voice_prompt_transcript_cache is not None:
        return _voice_prompt_transcript_cache
    transcripts_path = VOICE_PROMPT_DIR / "transcripts.json"
    if not transcripts_path.exists():
        _voice_prompt_transcript_cache = {}
        return _voice_prompt_transcript_cache
    try:
        with transcripts_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            _voice_prompt_transcript_cache = data.get("transcripts", {})
            return _voice_prompt_transcript_cache
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to load voice prompt transcripts: %s", exc)
        _voice_prompt_transcript_cache = {}
        return _voice_prompt_transcript_cache


def _save_voice_prompt_transcripts(transcripts: Dict[str, str]) -> None:
    transcripts_path = VOICE_PROMPT_DIR / "transcripts.json"
    transcripts_path.parent.mkdir(parents=True, exist_ok=True)
    with transcripts_path.open("w", encoding="utf-8") as handle:
        json.dump({"transcripts": transcripts}, handle, indent=2, ensure_ascii=False)


def _set_voice_prompt_transcript(file_path: Path, transcript: str) -> None:
    key = _voice_prompt_transcript_key(file_path)
    if not key:
        return
    transcripts = _load_voice_prompt_transcripts()
    transcripts[key] = transcript
    _save_voice_prompt_transcripts(transcripts)


def _apply_voice_design_cleanup(audio_data, sample_rate: int):
    try:
        from src.audio_effects import AudioPostProcessor
    except Exception:  # pragma: no cover - optional dependency
        return audio_data
    try:
        sox_path = AudioPostProcessor.SOX_PATH
    except Exception:  # pragma: no cover
        return audio_data
    if not sox_path.exists():
        return audio_data

    input_path = None
    output_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_in:
            input_path = Path(temp_in.name)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_out:
            output_path = Path(temp_out.name)
        sf.write(str(input_path), audio_data, int(sample_rate))
        command = [
            str(sox_path),
            str(input_path),
            str(output_path),
            "gain",
            "-n",
            "fade",
            "0.01",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "SoX cleanup failed")
        cleaned, _ = sf.read(str(output_path), dtype='float32')
        return cleaned
    except Exception as exc:
        logger.warning("SoX cleanup failed for voice design audio: %s", exc)
        return audio_data
    finally:
        if input_path:
            input_path.unlink(missing_ok=True)
        if output_path:
            output_path.unlink(missing_ok=True)


def _generate_voice_design_preview(payload: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, str]:
    text = (payload.get("text") or "").strip()
    instruct = (payload.get("instruct") or "").strip()
    language = (payload.get("language") or "Auto").strip() or "Auto"
    if not text:
        raise ValueError("Text is required to generate a preview.")

    with gpu_inference_lock:
        model = _get_qwen3_voice_design_model(config)
        wavs, sr = model.generate_voice_design(
            text=text,
            instruct=instruct or "",
            language=language or "Auto",
            non_streaming_mode=True,
        )
    if not wavs:
        raise RuntimeError("No audio produced for preview.")
    audio_data = _apply_voice_design_cleanup(wavs[0], int(sr))
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, int(sr), format="wav")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "audio_base64": encoded,
        "mime_type": "audio/wav",
    }


def _save_voice_design_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    name = (payload.get("name") or "").strip()
    text = (payload.get("text") or "").strip()
    gender = (payload.get("gender") or "").strip() or None
    language = (payload.get("language") or "Auto").strip() or "Auto"
    description = (payload.get("description") or "").strip() or None
    audio_base64 = payload.get("audio_base64")

    if not name:
        raise ValueError("Voice name is required.")
    if not text:
        raise ValueError("Sample text is required.")
    if not audio_base64:
        raise ValueError("Preview audio is required.")

    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception as exc:
        raise ValueError("Invalid audio payload.") from exc

    slug = _slugify_filename(name)
    target_path = VOICE_PROMPT_DIR / f"{slug}.wav"
    counter = 1
    while target_path.exists():
        target_path = VOICE_PROMPT_DIR / f"{slug}_{counter}.wav"
        counter += 1

    audio_stream = io.BytesIO(audio_bytes)
    audio_data, sample_rate = sf.read(audio_stream, dtype='float32')
    audio_data = _apply_voice_design_cleanup(audio_data, int(sample_rate))
    sf.write(str(target_path), audio_data, int(sample_rate))

    duration_seconds = _measure_audio_duration(target_path)
    if not duration_seconds:
        target_path.unlink(missing_ok=True)
        raise ValueError("Unable to determine audio duration.")
    if duration_seconds < MIN_CHATTERBOX_PROMPT_SECONDS:
        target_path.unlink(missing_ok=True)
        raise ValueError(
            f"Clip is only {duration_seconds:.2f}s. Voice prompts require at least {MIN_CHATTERBOX_PROMPT_SECONDS:.0f} seconds."
        )

    entries = _load_chatterbox_voice_entries()
    entry = {
        "id": str(uuid.uuid4()),
        "name": name,
        "file_name": target_path.name,
        "created_at": datetime.utcnow().isoformat(),
        "size_bytes": target_path.stat().st_size,
        "duration_seconds": duration_seconds,
        "gender": gender,
        "language": language if language and language != "Auto" else None,
        "description": description,
    }
    entries.append(entry)
    _save_chatterbox_voice_entries(entries)
    _set_voice_prompt_transcript(target_path, text)
    return _serialize_chatterbox_voice(entry)


def _voice_prompt_transcript_key(file_path: Path) -> Optional[str]:
    try:
        stat = file_path.stat()
    except OSError:
        return None
    key_data = f"{file_path.name}:{stat.st_size}:{stat.st_mtime}"
    return hashlib.md5(key_data.encode()).hexdigest()[:16]


@app.route('/api/voice-prompts', methods=['GET'])
def list_voice_prompts():
    """List available reference audio prompts."""
    try:
        # Load chatterbox voice registry for metadata lookup
        chatterbox_entries = _load_chatterbox_voice_entries()
        chatterbox_by_file = {e.get("file_name"): e for e in chatterbox_entries if e.get("file_name")}
        
        transcripts = _load_voice_prompt_transcripts()
        prompts = []
        for path in sorted(VOICE_PROMPT_DIR.glob('*')):
            if not path.is_file():
                continue
            if path.suffix.lower() not in VOICE_PROMPT_EXTENSIONS:
                continue
            try:
                size_bytes = path.stat().st_size
                mtime = path.stat().st_mtime
            except OSError:
                size_bytes = None
                mtime = None
            
            # Check cache for duration (keyed by name + mtime)
            cache_key = f"{path.name}:{mtime}" if mtime else None
            duration_seconds = None
            if cache_key and cache_key in _voice_prompt_duration_cache:
                duration_seconds = _voice_prompt_duration_cache[cache_key]
            else:
                duration_seconds = _get_audio_duration(path)
                if cache_key and duration_seconds is not None:
                    _voice_prompt_duration_cache[cache_key] = duration_seconds
            
            # Get metadata from chatterbox registry if available
            registry_entry = chatterbox_by_file.get(path.name, {})
            gender = registry_entry.get("gender")
            language = registry_entry.get("language")
            display_name = registry_entry.get("name") or path.stem.replace('_', ' ').replace('-', ' ').title()
            
            transcript_key = _voice_prompt_transcript_key(path)
            prompt_transcript = transcripts.get(transcript_key) if transcript_key else None
            prompts.append(
                {
                    "name": path.name,
                    "display": display_name,
                    "size_bytes": size_bytes,
                    "duration_seconds": duration_seconds,
                    "gender": gender,
                    "language": language,
                    "transcript": prompt_transcript,
                }
            )
        return jsonify({"success": True, "prompts": prompts})
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to list voice prompts: %s", exc, exc_info=True)
        return jsonify({"success": False, "error": "Failed to list voice prompts"}), 500


@app.route('/api/voice-prompts/upload', methods=['POST'])
def upload_voice_prompt():
    """Upload a new reference prompt clip."""
    file = request.files.get('file')
    if not file or not file.filename:
        return jsonify({"success": False, "error": "No file provided"}), 400

    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"success": False, "error": "Invalid filename"}), 400

    suffix = Path(filename).suffix.lower()
    if suffix not in VOICE_PROMPT_EXTENSIONS:
        allowed = ", ".join(sorted(ext.lstrip(".") for ext in VOICE_PROMPT_EXTENSIONS))
        return jsonify(
            {
                "success": False,
                "error": f"Unsupported file type '{suffix}'. Allowed: {allowed}",
            }
        ), 400

    target_path = VOICE_PROMPT_DIR / filename
    stem = Path(filename).stem
    counter = 1
    while target_path.exists():
        target_path = VOICE_PROMPT_DIR / f"{stem}_{counter}{suffix}"
        counter += 1

    try:
        file.save(target_path)
    except Exception as exc:  # pragma: no cover - filesystem failure
        logger.error("Failed to save uploaded prompt: %s", exc, exc_info=True)
        return jsonify({"success": False, "error": "Failed to save file"}), 500

    return jsonify(
        {
            "success": True,
            "prompt": {
                "name": target_path.name,
                "display": target_path.stem.replace('_', ' ').replace('-', ' ').title(),
                "size_bytes": target_path.stat().st_size,
            },
        }
    ), 201


@app.route('/api/voice-prompts/preview-fx', methods=['POST'])
def preview_voice_prompt_fx():
    """Preview FX on an existing voice prompt sample without regenerating audio."""
    from src.audio_effects import AudioPostProcessor, VoiceFXSettings
    data = request.json or {}
    prompt_path = (data.get("prompt_path") or "").strip()
    if not prompt_path:
        return jsonify({"success": False, "error": "prompt_path is required"}), 400

    file_name = Path(prompt_path).name
    if not file_name:
        return jsonify({"success": False, "error": "Invalid prompt_path"}), 400

    file_path = VOICE_PROMPT_DIR / file_name
    if not file_path.exists():
        return jsonify({"success": False, "error": "Audio file missing on disk."}), 404

    try:
        speed = float(data.get("speed", 1.0) or 1.0)
        pitch = float(data.get("pitch", 0.0) or 0.0)
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "Invalid pitch or speed value."}), 400

    speed = max(0.5, min(2.0, speed))
    pitch = max(-12.0, min(12.0, pitch))

    try:
        use_sox = abs(speed - 1.0) > 1e-3 or abs(pitch) > 1e-3
        sox_path = Path("tools/sox/sox.exe")
        if use_sox and sox_path.exists():
            output_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_out:
                    output_path = Path(temp_out.name)
                command = [str(sox_path), str(file_path), str(output_path)]
                if abs(pitch) > 1e-3:
                    command += ["pitch", f"{pitch * 100:.2f}"]
                if abs(speed - 1.0) > 1e-3:
                    command += ["tempo", "-s", f"{speed:.3f}"]
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.strip() or "SoX failed")
                audio_bytes = output_path.read_bytes()
            finally:
                if output_path:
                    output_path.unlink(missing_ok=True)
        else:
            audio_data, sample_rate = sf.read(str(file_path), dtype='float32')
            if abs(speed - 1.0) < 1e-3 and abs(pitch) < 1e-3:
                buffer = io.BytesIO()
                sf.write(buffer, audio_data, sample_rate, format='WAV')
                audio_bytes = buffer.getvalue()
            else:
                fx = VoiceFXSettings(
                    pitch_semitones=pitch,
                    speed=speed,
                    tone="neutral",
                )
                processor = AudioPostProcessor()
                processed = processor.apply(audio_data, sample_rate, fx, blend_override=0.0)
                buffer = io.BytesIO()
                sf.write(buffer, processed, sample_rate, format='WAV')
                audio_bytes = buffer.getvalue()
    except ImportError as exc:
        logger.error("Preview FX requires librosa: %s", exc)
        return jsonify({"success": False, "error": "Audio FX preview requires librosa."}), 400
    except Exception as exc:
        logger.error("Failed to preview voice prompt FX: %s", exc, exc_info=True)
        return jsonify({"success": False, "error": "Failed to process audio."}), 500

    encoded = base64.b64encode(audio_bytes).decode('ascii')
    return jsonify({
        "success": True,
        "audio_base64": encoded,
        "mime_type": "audio/wav",
    })


@app.route('/api/chatterbox-voices', methods=['GET'])
def list_chatterbox_voices():
    _auto_register_voice_prompt_files()
    entries = _load_chatterbox_voice_entries()
    serialized = [_serialize_chatterbox_voice(entry) for entry in entries]
    return jsonify({"success": True, "voices": serialized})


@app.route('/api/chatterbox-voices', methods=['POST'])
def create_chatterbox_voice():
    name = (request.form.get("name") or "").strip()
    file = request.files.get("file")
    if not name:
        return jsonify({"success": False, "error": "Voice name is required."}), 400
    if not file or not file.filename:
        return jsonify({"success": False, "error": "Audio file is required."}), 400

    filename = secure_filename(file.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in VOICE_PROMPT_EXTENSIONS:
        allowed = ", ".join(sorted(ext.lstrip(".") for ext in VOICE_PROMPT_EXTENSIONS))
        return jsonify(
            {
                "success": False,
                "error": f"Unsupported file type '{suffix}'. Allowed: {allowed}",
            }
        ), 400

    slug = _slugify_filename(name)
    target_path = VOICE_PROMPT_DIR / f"{slug}{suffix}"
    counter = 1
    while target_path.exists():
        target_path = VOICE_PROMPT_DIR / f"{slug}_{counter}{suffix}"
        counter += 1

    try:
        file.save(target_path)
    except Exception as exc:  # pragma: no cover - filesystem failure
        logger.error("Failed to save chatterbox voice: %s", exc, exc_info=True)
        return jsonify({"success": False, "error": "Failed to save file"}), 500

    entries = _load_chatterbox_voice_entries()
    duration_seconds = _measure_audio_duration(target_path)
    if not duration_seconds:
        target_path.unlink(missing_ok=True)
        return jsonify({
            "success": False,
            "error": "Unable to determine audio duration. Please upload a standard WAV/MP3 clip."
        }), 400
    if duration_seconds < MIN_CHATTERBOX_PROMPT_SECONDS:
        target_path.unlink(missing_ok=True)
        return jsonify({
            "success": False,
            "error": f"Clip is only {duration_seconds:.2f}s. Chatterbox Turbo requires at least {MIN_CHATTERBOX_PROMPT_SECONDS:.0f} seconds."
        }), 400

    entry = {
        "id": str(uuid.uuid4()),
        "name": name,
        "file_name": target_path.name,
        "created_at": datetime.utcnow().isoformat(),
        "size_bytes": target_path.stat().st_size,
        "duration_seconds": duration_seconds,
    }
    entries.append(entry)
    _save_chatterbox_voice_entries(entries)

    serialized = _serialize_chatterbox_voice(entry)
    return jsonify({"success": True, "voice": serialized}), 201


@app.route('/api/chatterbox-voices/<voice_id>', methods=['PUT'])
def rename_chatterbox_voice(voice_id: str):
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"success": False, "error": "Voice name is required."}), 400

    entries = _load_chatterbox_voice_entries()
    entry = _resolve_chatterbox_voice(voice_id, entries)
    if not entry:
        return jsonify({"success": False, "error": "Voice not found."}), 404

    entry["name"] = name
    _save_chatterbox_voice_entries(entries)
    return jsonify({"success": True, "voice": _serialize_chatterbox_voice(entry)})


@app.route('/api/chatterbox-voices/<voice_id>', methods=['DELETE'])
def delete_chatterbox_voice(voice_id: str):
    entries = _load_chatterbox_voice_entries()
    entry = _resolve_chatterbox_voice(voice_id, entries)
    if not entry:
        return jsonify({"success": False, "error": "Voice not found."}), 404

    file_name = entry.get("file_name")
    file_path = VOICE_PROMPT_DIR / file_name if file_name else None
    if file_path and file_path.exists():
        try:
            file_path.unlink()
        except OSError as exc:  # pragma: no cover - filesystem failure
            logger.warning("Unable to delete voice prompt file %s: %s", file_path, exc)

    entries = [item for item in entries if item.get("id") != voice_id]
    _save_chatterbox_voice_entries(entries)
    return jsonify({"success": True})


def _collect_voice_files(voice_ids: List[str]) -> Tuple[List[Tuple[Path, str]], List[str]]:
    files: List[Tuple[Path, str]] = []
    missing: List[str] = []
    entries = _load_chatterbox_voice_entries()
    entries_by_id = {entry.get("id"): entry for entry in entries if entry.get("id")}

    for voice_id in voice_ids:
        if _is_external_voice_id(voice_id):
            short_name = _strip_external_voice_id(voice_id)
            file_path = EXTERNAL_VOICES_DIR / f"{short_name}.mp3"
            if file_path.exists():
                files.append((file_path, file_path.name))
            else:
                missing.append(voice_id)
            continue

        entry = entries_by_id.get(voice_id)
        if not entry:
            missing.append(voice_id)
            continue
        file_name = entry.get("file_name")
        if not file_name:
            missing.append(voice_id)
            continue
        file_path = VOICE_PROMPT_DIR / file_name
        if file_path.exists():
            files.append((file_path, file_name))
        else:
            missing.append(voice_id)

    return files, missing


@app.route('/api/chatterbox-voices/export', methods=['POST'])
def export_chatterbox_voices():
    data = request.get_json(silent=True) or {}
    voice_ids = data.get("voice_ids") or []
    if not isinstance(voice_ids, list) or not voice_ids:
        return jsonify({"success": False, "error": "No voice ids provided."}), 400

    files, missing = _collect_voice_files([str(v) for v in voice_ids])
    if not files:
        return jsonify({"success": False, "error": "No audio files found for export."}), 404

    if len(files) == 1:
        file_path, filename = files[0]
        mime_type, _ = mimetypes.guess_type(filename)
        return send_file(
            file_path,
            mimetype=mime_type or 'audio/mpeg',
            as_attachment=True,
            download_name=filename,
        )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for file_path, filename in files:
            zipf.write(file_path, arcname=filename)
    buffer.seek(0)
    response = send_file(
        buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name="voice_samples.zip",
    )
    response.headers["X-Voice-Missing"] = json.dumps(missing)
    return response


@app.route('/api/chatterbox-voices/archive', methods=['POST'])
def archive_chatterbox_voices():
    data = request.get_json(silent=True) or {}
    voice_ids = data.get("voice_ids") or []
    archived = _coerce_bool(data.get("archived", True))
    if not isinstance(voice_ids, list) or not voice_ids:
        return jsonify({"success": False, "error": "No voice ids provided."}), 400

    entries = _load_chatterbox_voice_entries()
    entry_map = {entry.get("id"): entry for entry in entries if entry.get("id")}
    updated = 0
    external_ids = _load_external_voice_archives()

    for voice_id in [str(v) for v in voice_ids]:
        if _is_external_voice_id(voice_id):
            short_name = _strip_external_voice_id(voice_id)
            if archived:
                external_ids.add(short_name)
            else:
                external_ids.discard(short_name)
            updated += 1
            continue

        entry = entry_map.get(voice_id)
        if not entry:
            continue
        entry["archived"] = archived
        updated += 1

    _save_chatterbox_voice_entries(entries)
    _save_external_voice_archives(external_ids)
    return jsonify({"success": True, "updated": updated})


@app.route('/api/chatterbox-voices/batch-delete', methods=['POST'])
def batch_delete_chatterbox_voices():
    data = request.get_json(silent=True) or {}
    voice_ids = data.get("voice_ids") or []
    if not isinstance(voice_ids, list) or not voice_ids:
        return jsonify({"success": False, "error": "No voice ids provided."}), 400

    entries = _load_chatterbox_voice_entries()
    remaining_entries = []
    deleted = 0
    target_ids = {str(v) for v in voice_ids}
    external_ids = _load_external_voice_archives()

    for entry in entries:
        entry_id = entry.get("id")
        if entry_id and entry_id in target_ids:
            file_name = entry.get("file_name")
            file_path = VOICE_PROMPT_DIR / file_name if file_name else None
            if file_path and file_path.exists():
                try:
                    file_path.unlink()
                except OSError as exc:
                    logger.warning("Unable to delete voice prompt file %s: %s", file_path, exc)
            deleted += 1
            continue
        remaining_entries.append(entry)

    for voice_id in target_ids:
        if _is_external_voice_id(voice_id):
            short_name = _strip_external_voice_id(voice_id)
            local_file = EXTERNAL_VOICES_DIR / f"{short_name}.mp3"
            if local_file.exists():
                try:
                    local_file.unlink()
                except OSError as exc:
                    logger.warning("Unable to delete external voice file %s: %s", local_file, exc)
            prompt_file = VOICE_PROMPT_DIR / f"{short_name}.mp3"
            if prompt_file.exists():
                try:
                    prompt_file.unlink()
                except OSError as exc:
                    logger.warning("Unable to delete voice prompt file %s: %s", prompt_file, exc)
            external_ids.discard(short_name)
            deleted += 1

    _save_chatterbox_voice_entries(remaining_entries)
    _save_external_voice_archives(external_ids)
    return jsonify({"success": True, "deleted": deleted})


@app.route('/api/chatterbox-voices/<voice_id>/preview')
def preview_chatterbox_voice(voice_id: str):
    entries = _load_chatterbox_voice_entries()
    entry = _resolve_chatterbox_voice(voice_id, entries)
    if not entry:
        return jsonify({"success": False, "error": "Voice not found."}), 404
    file_name = entry.get("file_name")
    if not file_name:
        return jsonify({"success": False, "error": "Voice has no associated file."}), 404
    file_path = VOICE_PROMPT_DIR / file_name
    if not file_path.exists():
        return jsonify({"success": False, "error": "Audio file missing on disk."}), 404
    mime_type, _ = mimetypes.guess_type(file_path.name)
    return send_file(
        file_path,
        mimetype=mime_type or 'audio/mpeg',
        conditional=True,
        as_attachment=False,
        download_name=file_path.name,
    )


# External TTS samples from GitHub
EXTERNAL_VOICES_CACHE_FILE = Path("data/external_voices_cache.json")
EXTERNAL_VOICES_URL = "https://raw.githubusercontent.com/yaph/tts-samples/main/data/voices.json"
EXTERNAL_SAMPLES_BASE_URL = "https://raw.githubusercontent.com/yaph/tts-samples/main/mp3"
EXTERNAL_VOICES_DIR = Path("data/external_voices")
EXTERNAL_VOICES_DIR.mkdir(parents=True, exist_ok=True)

# Mapping from locale code prefix to GitHub folder name
LOCALE_TO_FOLDER = {
    'af': 'Afrikaans', 'sq': 'Albanian', 'am': 'Amharic', 'ar': 'Arabic',
    'az': 'Azerbaijani', 'bn': 'Bengali', 'bs': 'Bosnian', 'bg': 'Bulgarian',
    'my': 'Burmese', 'ca': 'Catalan', 'zh': 'Chinese', 'yue': 'Chinese',
    'wuu': 'Chinese', 'hr': 'Croatian', 'cs': 'Czech', 'da': 'Danish',
    'nl': 'Dutch', 'en': 'English', 'et': 'Estonian', 'fil': 'Filipino',
    'fi': 'Finnish', 'fr': 'French', 'gl': 'Galician', 'ka': 'Georgian',
    'de': 'German', 'el': 'Greek', 'gu': 'Gujarati', 'he': 'Hebrew',
    'hi': 'Hindi', 'hu': 'Hungarian', 'is': 'Icelandic', 'id': 'Indonesian',
    'ga': 'Irish', 'it': 'Italian', 'ja': 'Japanese', 'jv': 'Javanese',
    'kn': 'Kannada', 'kk': 'Kazakh', 'km': 'Khmer', 'ko': 'Korean',
    'lo': 'Lao', 'lv': 'Latvian', 'lt': 'Lithuanian', 'mk': 'Macedonian',
    'ms': 'Malay', 'ml': 'Malayalam', 'mt': 'Maltese', 'mr': 'Marathi',
    'mn': 'Mongolian', 'ne': 'Nepali', 'nb': 'Norwegian', 'ps': 'Pashto',
    'fa': 'Persian', 'pl': 'Polish', 'pt': 'Portuguese', 'ro': 'Romanian',
    'ru': 'Russian', 'sr': 'Serbian', 'si': 'Sinhala', 'sk': 'Slovak',
    'sl': 'Slovenian', 'so': 'Somali', 'es': 'Spanish', 'su': 'Sundanese',
    'sw': 'Swahili', 'sv': 'Swedish', 'ta': 'Tamil', 'te': 'Telugu',
    'th': 'Thai', 'tr': 'Turkish', 'uk': 'Ukrainian', 'ur': 'Urdu',
    'uz': 'Uzbek', 'vi': 'Vietnamese', 'cy': 'Welsh', 'zu': 'Zulu',
    'eu': 'Basque',
}

def _get_github_folder_for_locale(locale: str) -> str:
    """Get the GitHub folder name for a locale code like 'en-GB' -> 'English'."""
    if not locale:
        return "English"
    # Extract language code (e.g., 'en' from 'en-GB')
    lang_code = locale.split('-')[0].lower()
    return LOCALE_TO_FOLDER.get(lang_code, "English")

_external_voices_cache: Optional[List[Dict[str, Any]]] = None
_external_voices_cache_time: float = 0


def _fetch_external_voices(force_refresh: bool = False) -> List[Dict[str, Any]]:
    """Fetch external TTS voice samples from GitHub with caching."""
    global _external_voices_cache, _external_voices_cache_time
    
    cache_max_age = 3600 * 24  # 24 hours
    now = time.time()
    
    # Return memory cache if fresh
    if not force_refresh and _external_voices_cache and (now - _external_voices_cache_time) < cache_max_age:
        return _external_voices_cache
    
    # Try to load from file cache
    if not force_refresh and EXTERNAL_VOICES_CACHE_FILE.exists():
        try:
            cache_mtime = EXTERNAL_VOICES_CACHE_FILE.stat().st_mtime
            if (now - cache_mtime) < cache_max_age:
                with EXTERNAL_VOICES_CACHE_FILE.open("r", encoding="utf-8") as f:
                    _external_voices_cache = json.load(f)
                    _external_voices_cache_time = now
                    return _external_voices_cache
        except Exception as e:
            logger.warning("Failed to load external voices cache: %s", e)
    
    # Fetch from GitHub
    try:
        import urllib.request
        logger.info("Fetching external TTS voices from GitHub...")
        with urllib.request.urlopen(EXTERNAL_VOICES_URL, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        
        # Save to file cache
        EXTERNAL_VOICES_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with EXTERNAL_VOICES_CACHE_FILE.open("w", encoding="utf-8") as f:
            json.dump(data, f)
        
        _external_voices_cache = data
        _external_voices_cache_time = now
        logger.info("Fetched %d external TTS voices", len(data))
        return data
    except Exception as e:
        logger.error("Failed to fetch external voices: %s", e)
        # Return stale cache if available
        if _external_voices_cache:
            return _external_voices_cache
        return []


def _serialize_external_voice(voice: Dict[str, Any], archived_ids: Optional[set] = None) -> Dict[str, Any]:
    """Serialize an external voice entry for the API."""
    short_name = voice.get("ShortName", "")
    locale = voice.get("Locale", "")
    gender = voice.get("Gender", "")
    friendly_name = voice.get("FriendlyName", "")
    
    # Extract clean name from FriendlyName (e.g., "Microsoft Jenny Online (Natural) - English (United States)")
    name_match = re.match(r"Microsoft (\w+)", friendly_name)
    display_name = name_match.group(1) if name_match else short_name
    
    # Check if downloaded locally
    local_file = EXTERNAL_VOICES_DIR / f"{short_name}.mp3"
    is_downloaded = local_file.exists()
    
    # Get correct GitHub folder name (e.g., "English" not "en-GB")
    folder_name = _get_github_folder_for_locale(locale)
    archived_ids = archived_ids or set()
    return {
        "id": f"external:{short_name}",
        "name": display_name,
        "short_name": short_name,
        "file_name": f"{short_name}.mp3" if is_downloaded else None,
        "prompt_path": str(local_file) if is_downloaded else None,
        "gender": gender,
        "language": locale,
        "friendly_name": friendly_name,
        "source": "external",
        "is_downloaded": is_downloaded,
        "archived": short_name in archived_ids,
        "download_url": f"{EXTERNAL_SAMPLES_BASE_URL}/{folder_name}/{short_name}.mp3",
        "voice_personalities": voice.get("VoiceTag", {}).get("VoicePersonalities", []),
    }


@app.route('/api/external-voices', methods=['GET'])
def list_external_voices():
    """List available external TTS voice samples from GitHub."""
    try:
        force_refresh = request.args.get('refresh', '').lower() == 'true'
        voices = _fetch_external_voices(force_refresh=force_refresh)
        archived_ids = _load_external_voice_archives()
        serialized = [_serialize_external_voice(v, archived_ids) for v in voices]
        return jsonify({
            "success": True,
            "voices": serialized,
            "total": len(serialized),
        })
    except Exception as e:
        logger.error("Failed to list external voices: %s", e, exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/external-voices/<voice_id>/download', methods=['POST'])
def download_external_voice(voice_id: str):
    """Download an external voice sample to use locally."""
    try:
        # Find the voice in the cache
        voices = _fetch_external_voices()
        voice = None
        for v in voices:
            if v.get("ShortName") == voice_id:
                voice = v
                break
        
        if not voice:
            return jsonify({"success": False, "error": "Voice not found"}), 404
        
        short_name = voice.get("ShortName")
        locale = voice.get("Locale")
        folder_name = _get_github_folder_for_locale(locale)
        download_url = f"{EXTERNAL_SAMPLES_BASE_URL}/{folder_name}/{short_name}.mp3"
        local_file = EXTERNAL_VOICES_DIR / f"{short_name}.mp3"
        
        # Download the file
        import urllib.request
        logger.info("Downloading external voice: %s", short_name)
        urllib.request.urlretrieve(download_url, str(local_file))
        
        # Also copy to voice_prompts directory for use with TTS engines
        voice_prompt_file = VOICE_PROMPT_DIR / f"{short_name}.mp3"
        shutil.copy(str(local_file), str(voice_prompt_file))
        
        # Add to chatterbox voices registry
        entries = _load_chatterbox_voice_entries()
        existing = next((e for e in entries if e.get("file_name") == f"{short_name}.mp3"), None)
        if not existing:
            # Extract display name
            friendly_name = voice.get("FriendlyName", "")
            name_match = re.match(r"Microsoft (\w+)", friendly_name)
            display_name = name_match.group(1) if name_match else short_name
            
            new_entry = {
                "id": str(uuid.uuid4()),
                "name": f"{display_name} ({locale})",
                "file_name": f"{short_name}.mp3",
                "created_at": datetime.utcnow().isoformat(),
                "gender": voice.get("Gender"),
                "language": locale,
            }
            entries.append(new_entry)
            _save_chatterbox_voice_entries(entries)
        
        return jsonify({
            "success": True,
            "voice": _serialize_external_voice(voice, _load_external_voice_archives()),
            "message": f"Downloaded {short_name}.mp3"
        })
    except Exception as e:
        logger.error("Failed to download external voice %s: %s", voice_id, e, exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/external-voices/<voice_id>/preview')
def preview_external_voice(voice_id: str):
    """Stream preview of an external voice (downloads if not cached)."""
    try:
        voices = _fetch_external_voices()
        voice = None
        for v in voices:
            if v.get("ShortName") == voice_id:
                voice = v
                break
        
        if not voice:
            return jsonify({"success": False, "error": "Voice not found"}), 404
        
        short_name = voice.get("ShortName")
        local_file = EXTERNAL_VOICES_DIR / f"{short_name}.mp3"
        
        # Download if not cached
        if not local_file.exists():
            locale = voice.get("Locale")
            folder_name = _get_github_folder_for_locale(locale)
            download_url = f"{EXTERNAL_SAMPLES_BASE_URL}/{folder_name}/{short_name}.mp3"
            import urllib.request
            urllib.request.urlretrieve(download_url, str(local_file))
        
        return send_file(
            local_file,
            mimetype='audio/mpeg',
            conditional=True,
            as_attachment=False,
            download_name=f"{short_name}.mp3",
        )
    except Exception as e:
        logger.error("Failed to preview external voice %s: %s", voice_id, e, exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/chatterbox-voices/<voice_id>/update', methods=['PUT'])
def update_chatterbox_voice_metadata(voice_id: str):
    """Update metadata (gender, language) for a chatterbox voice."""
    data = request.get_json() or {}
    
    entries = _load_chatterbox_voice_entries()
    entry = _resolve_chatterbox_voice(voice_id, entries)
    if not entry:
        return jsonify({"success": False, "error": "Voice not found."}), 404
    
    # Update allowed fields
    if "gender" in data:
        entry["gender"] = data["gender"] if data["gender"] in ("Male", "Female", None) else None
    if "language" in data:
        entry["language"] = data["language"]
    if "name" in data and data["name"]:
        entry["name"] = data["name"].strip()
    
    _save_chatterbox_voice_entries(entries)
    return jsonify({"success": True, "voice": _serialize_chatterbox_voice(entry)})


@app.route('/api/voices/samples', methods=['POST'])
def generate_voice_samples_api():
    """Generate preview samples for all voices."""
    overwrite = request.json.get('overwrite', False) if request.is_json else False
    sample_text = request.json.get('text') if request.is_json else None
    device = request.json.get('device', 'auto') if request.is_json else 'auto'

    logger.info("Voice sample generation requested", extra={
        "overwrite": overwrite,
        "device": device
    })

    try:
        report = generate_voice_samples(
            overwrite=overwrite,
            device=device,
            sample_text=sample_text or None,
        )
    except RuntimeError as err:
        logger.error(f"Voice sample generation failed: {err}")
        return jsonify({
            "success": False,
            "error": str(err)
        }), 400
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Unexpected error during voice sample generation", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Failed to generate voice samples"
        }), 500


def _serialize_chunk_for_response(job_id: str, chunk: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": chunk.get("id"),
        "order_index": chunk.get("order_index"),
        "chapter_index": chunk.get("chapter_index"),
        "chunk_index": chunk.get("chunk_index"),
        "speaker": chunk.get("speaker"),
        "text": chunk.get("text"),
        "relative_file": chunk.get("relative_file"),
        "file_url": _chunk_file_url(job_id, chunk.get("relative_file")),
        "duration_seconds": chunk.get("duration_seconds"),
        "regenerated_at": chunk.get("regenerated_at"),
        "voice": chunk.get("voice_label"),
        "voice_assignment": chunk.get("voice_assignment"),
    }


@app.route('/api/jobs/<job_id>/chunks', methods=['GET'])
def get_job_chunks(job_id: str):
    """Return chunk metadata for a review-enabled job."""
    try:
        with queue_lock:
            job_entry = jobs.get(job_id)
            if not job_entry:
                return jsonify({"success": False, "error": "Job not found"}), 404
            _ensure_review_ready(job_entry)
            chunks = [dict(item) for item in (job_entry.get("chunks") or [])]
            regen_tasks = copy.deepcopy(job_entry.get("regen_tasks") or {})
            review_status = {
                "status": job_entry.get("status"),
                "chapter_mode": job_entry.get("chapter_mode"),
                "full_story_requested": job_entry.get("full_story_requested"),
                "has_active_regen": _has_active_regen_tasks(job_entry),
                "engine": job_entry.get("engine"),
            }

        payload = {
            "success": True,
            "chunks": [_serialize_chunk_for_response(job_id, c) for c in chunks],
            "regen_tasks": regen_tasks,
            "review": review_status,
        }
        return jsonify(payload)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to load chunks for job %s: %s", job_id, exc, exc_info=True)
        return jsonify({"success": False, "error": "Failed to load job chunks"}), 500


@app.route('/api/jobs/<job_id>/review/regen', methods=['POST'])
def request_chunk_regeneration(job_id: str):
    """Schedule a chunk regeneration request."""
    data = request.json or {}
    chunk_id = (data.get("chunk_id") or "").strip()
    updated_text = (data.get("text") or "").strip()
    voice_payload = data.get("voice") or {}
    engine_override = (data.get("engine") or "").strip() or None

    if not chunk_id:
        return jsonify({"success": False, "error": "chunk_id is required"}), 400
    if not updated_text:
        return jsonify({"success": False, "error": "Updated text cannot be empty"}), 400

    try:
        with queue_lock:
            job_entry = jobs.get(job_id)
            if not job_entry:
                return jsonify({"success": False, "error": "Job not found"}), 404
            _ensure_review_ready(job_entry)
            _, chunk = _find_chunk_record(job_entry, chunk_id)
            if chunk is None:
                return jsonify({"success": False, "error": "Chunk not found"}), 404
            regen_tasks = job_entry.setdefault("regen_tasks", {})
            task_state = regen_tasks.get(chunk_id)
            if task_state and task_state.get("status") in {"queued", "running"}:
                return jsonify({"success": False, "error": "Chunk regeneration already in progress"}), 409

        _schedule_chunk_regeneration(job_id, chunk_id, updated_text, voice_payload, engine_override=engine_override)
        return jsonify({"success": True})
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to schedule chunk regen for job %s chunk %s: %s", job_id, chunk_id, exc, exc_info=True)
        return jsonify({"success": False, "error": "Failed to schedule chunk regeneration"}), 500


@app.route('/api/jobs/<job_id>/review/regen-all', methods=['POST'])
def request_full_job_regeneration(job_id: str):
    """Schedule regeneration for every chunk in the job."""
    data = request.json or {}
    chunk_updates_raw = data.get("chunks") or []
    global_engine_override = (data.get("engine") or "").strip() or None
    chunk_updates: Dict[str, Dict[str, Any]] = {}
    for entry in chunk_updates_raw:
        if not isinstance(entry, dict):
            continue
        chunk_key = (entry.get("chunk_id") or "").strip()
        if not chunk_key:
            continue
        chunk_updates[chunk_key] = {
            "text": (entry.get("text") or "").strip(),
            "voice": entry.get("voice"),
            "engine": (entry.get("engine") or "").strip() or None,
        }

    try:
        with queue_lock:
            job_entry = jobs.get(job_id)
            if not job_entry:
                return jsonify({"success": False, "error": "Job not found"}), 404
            _ensure_review_ready(job_entry)
            if _has_active_regen_tasks(job_entry):
                return jsonify({"success": False, "error": "Chunk regeneration already in progress"}), 409
            job_chunks = job_entry.get("chunks") or []
            if not job_chunks:
                return jsonify({"success": False, "error": "No chunks available to regenerate"}), 400
            tasks_to_schedule: List[Tuple[str, str, Optional[Dict[str, Any]], Optional[str]]] = []
            for chunk in job_chunks:
                chunk_id = chunk.get("id")
                if not chunk_id:
                    continue
                overrides = chunk_updates.get(chunk_id) or {}
                text_value = (overrides.get("text") or chunk.get("text") or "").strip()
                if not text_value:
                    raise ValueError(f"Chunk {chunk_id} does not have text to regenerate.")
                voice_payload = overrides.get("voice")
                engine_override = overrides.get("engine") or global_engine_override
                tasks_to_schedule.append((chunk_id, text_value, voice_payload, engine_override))
        for chunk_id, text_value, voice_payload, engine_override in tasks_to_schedule:
            _schedule_chunk_regeneration(job_id, chunk_id, text_value, voice_payload, engine_override=engine_override)
        return jsonify({"success": True, "queued_chunks": len(tasks_to_schedule)})
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to queue full regeneration for job %s: %s", job_id, exc, exc_info=True)
        return jsonify({"success": False, "error": "Failed to queue full regeneration"}), 500


@app.route('/api/jobs/<job_id>/review/apply-fx', methods=['POST'])
def apply_chunk_audio_effects(job_id: str):
    """Apply audio effects (speed, pitch) to one or more chunks without regenerating."""
    from src.audio_effects import AudioPostProcessor, VoiceFXSettings
    import soundfile as sf
    
    data = request.json or {}
    chunks_fx = data.get("chunks") or []
    
    if not chunks_fx:
        return jsonify({"success": False, "error": "No chunks specified"}), 400
    
    try:
        with queue_lock:
            job_entry = jobs.get(job_id)
            if not job_entry:
                return jsonify({"success": False, "error": "Job not found"}), 404
            _ensure_review_ready(job_entry)
            job_dir = _job_dir_from_entry(job_id, job_entry)
            job_chunks = job_entry.get("chunks") or []
            chunk_map = {c.get("id"): c for c in job_chunks if c.get("id")}
        
        processor = AudioPostProcessor()
        processed_count = 0
        errors = []
        
        for fx_entry in chunks_fx:
            chunk_id = (fx_entry.get("chunk_id") or "").strip()
            if not chunk_id:
                continue
            
            chunk = chunk_map.get(chunk_id)
            if not chunk:
                errors.append(f"Chunk {chunk_id} not found")
                continue
            
            relative_file = chunk.get("relative_file")
            if not relative_file:
                errors.append(f"Chunk {chunk_id} has no audio file")
                continue
            
            audio_path = job_dir / relative_file
            if not audio_path.exists():
                errors.append(f"Audio file not found for chunk {chunk_id}")
                continue
            
            # Parse FX settings
            speed = float(fx_entry.get("speed", 1.0) or 1.0)
            pitch = float(fx_entry.get("pitch", 0.0) or 0.0)
            
            # Skip if no changes
            if abs(speed - 1.0) < 1e-3 and abs(pitch) < 1e-3:
                continue
            
            try:
                # Load audio
                audio_data, sample_rate = sf.read(str(audio_path), dtype='float32')
                
                # Create FX settings - use no blending for post-apply effects
                fx = VoiceFXSettings(
                    pitch_semitones=max(-12.0, min(12.0, pitch)),
                    speed=max(0.5, min(2.0, speed)),
                    tone="neutral"
                )
                
                # Apply effects without blending (blend_override=0 means no original mixed in)
                processed = processor.apply(audio_data, sample_rate, fx, blend_override=0.0)
                
                # Save back to file
                sf.write(str(audio_path), processed, sample_rate)
                
                # Update chunk metadata
                with queue_lock:
                    job_entry = jobs.get(job_id)
                    if job_entry:
                        for c in job_entry.get("chunks", []):
                            if c.get("id") == chunk_id:
                                c["fx_applied"] = {"speed": speed, "pitch": pitch}
                                c["modified_at"] = datetime.now().isoformat()
                                break
                
                processed_count += 1
                
            except Exception as exc:
                logger.error("Failed to apply FX to chunk %s: %s", chunk_id, exc, exc_info=True)
                errors.append(f"Failed to process chunk {chunk_id}: {str(exc)}")
        
        # Persist metadata changes
        _persist_chunks_metadata(job_id, job_dir)
        
        return jsonify({
            "success": True,
            "processed": processed_count,
            "errors": errors if errors else None
        })
        
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        logger.error("Failed to apply audio effects for job %s: %s", job_id, exc, exc_info=True)
        return jsonify({"success": False, "error": "Failed to apply audio effects"}), 500


@app.route('/api/jobs/<job_id>/review/preview-fx', methods=['POST'])
def preview_chunk_audio_effects(job_id: str):
    """Preview audio effects on a chunk without saving. Returns the processed audio file."""
    from src.audio_effects import AudioPostProcessor, VoiceFXSettings
    import soundfile as sf
    import tempfile
    import io
    
    data = request.json or {}
    chunk_id = (data.get("chunk_id") or "").strip()
    speed = float(data.get("speed", 1.0) or 1.0)
    pitch = float(data.get("pitch", 0.0) or 0.0)
    
    if not chunk_id:
        return jsonify({"success": False, "error": "chunk_id is required"}), 400
    
    try:
        # First try to restore job from library if not in memory
        job_dir = OUTPUT_DIR / job_id
        if not job_dir.exists():
            return jsonify({"success": False, "error": "Job not found"}), 404
        
        # Load chunks metadata to find the chunk
        chunks_meta_path = job_dir / "chunks_metadata.json"
        chunk = None
        if chunks_meta_path.exists():
            with chunks_meta_path.open("r", encoding="utf-8") as f:
                chunks_meta = json.load(f)
                for c in chunks_meta.get("chunks", []):
                    if c.get("id") == chunk_id:
                        chunk = c
                        break
        
        if not chunk:
            return jsonify({"success": False, "error": "Chunk not found"}), 404
        
        relative_file = chunk.get("relative_file")
        if not relative_file:
            return jsonify({"success": False, "error": "Chunk has no audio file"}), 400
        
        audio_path = job_dir / relative_file
        if not audio_path.exists():
            return jsonify({"success": False, "error": "Audio file not found"}), 404
        
        # Load audio
        audio_data, sample_rate = sf.read(str(audio_path), dtype='float32')
        
        # If no effects, just return the original
        if abs(speed - 1.0) < 1e-3 and abs(pitch) < 1e-3:
            return send_file(str(audio_path), mimetype='audio/wav')
        
        # Create FX settings and apply without blending
        fx = VoiceFXSettings(
            pitch_semitones=max(-12.0, min(12.0, pitch)),
            speed=max(0.5, min(2.0, speed)),
            tone="neutral"
        )
        
        processor = AudioPostProcessor()
        processed = processor.apply(audio_data, sample_rate, fx, blend_override=0.0)
        
        # Write to in-memory buffer
        buffer = io.BytesIO()
        sf.write(buffer, processed, sample_rate, format='WAV')
        buffer.seek(0)
        
        return send_file(
            buffer,
            mimetype='audio/wav',
            as_attachment=False,
            download_name=f"preview_{chunk_id}.wav"
        )
        
    except Exception as exc:
        logger.error("Failed to preview audio effects for job %s chunk %s: %s", job_id, chunk_id, exc, exc_info=True)
        return jsonify({"success": False, "error": "Failed to preview audio effects"}), 500


def _merge_review_job(job_id: str, job_entry: Dict[str, Any], manifest: Dict[str, Any]):
    config_snapshot = copy.deepcopy(job_entry.get("config_snapshot") or load_config())
    merge_options = job_entry.get("merge_options") or {}
    output_format = merge_options.get("output_format") or config_snapshot.get("output_format") or "mp3"
    crossfade_seconds = float(merge_options.get("crossfade_duration") or 0)
    merger = AudioMerger(
        crossfade_ms=int(max(0.0, crossfade_seconds) * 1000),
        intro_silence_ms=int(max(0, merge_options.get("intro_silence_ms") or 0)),
        inter_chunk_silence_ms=int(max(0, merge_options.get("inter_chunk_silence_ms") or 0)),
        bitrate_kbps=int(merge_options.get("output_bitrate_kbps") or 0),
    )
    job_dir = _job_dir_from_entry(job_id, job_entry)

    chapters = manifest.get("chapters", []) or []
    all_full_story_chunks = manifest.get("all_full_story_chunks") or []
    books = manifest.get("books", []) or []
    total_steps = len(chapters) + len(books) + (1 if all_full_story_chunks else 0)
    with queue_lock:
        entry = jobs.get(job_id)
        if entry:
            entry["post_process_total"] = max(int(total_steps), 0)
            entry["post_process_done"] = 0
            entry["post_process_percent"] = 0
            entry["post_process_active"] = True

    chapter_outputs = []
    for chapter in chapters:
        rel_chunk_files = chapter.get("chunk_files") or []
        chunk_paths = [str(job_dir / rel_path) for rel_path in rel_chunk_files]
        if not chunk_paths:
            continue
        # Verify chunk files exist before merging
        missing_chunks = [p for p in chunk_paths if not Path(p).exists()]
        if missing_chunks:
            logger.error(f"Missing chunk files for merge: {missing_chunks}")
            continue
        logger.info(f"Merging {len(chunk_paths)} chunks: {chunk_paths}")
        output_filename = chapter.get("output_filename") or f"chapter_{chapter.get('index', 0):02d}.{output_format}"
        chapter_dir = job_dir / (chapter.get("chapter_dir") or ".")
        chapter_dir.mkdir(parents=True, exist_ok=True)
        output_path = chapter_dir / output_filename
        logger.info(f"Output path: {output_path}")
        merger.merge_wav_files(
            input_files=chunk_paths,
            output_path=str(output_path),
            format=output_format,
            cleanup_chunks=False,
        )
        with queue_lock:
            entry = jobs.get(job_id)
            if entry:
                entry["post_process_done"] = min(
                    len(chapter_outputs) + 1,
                    int(entry.get("post_process_total") or 0) or 0,
                )
        # Verify output was created with content
        if output_path.exists():
            output_size = output_path.stat().st_size
            logger.info(f"Merged output size: {output_size} bytes")
            if output_size < 1000:
                logger.warning(f"Output file suspiciously small: {output_size} bytes")
        rel_path = (Path(chapter.get("chapter_dir") or ".") / output_filename).as_posix()
        # Normalize "./filename" to just "filename"
        if rel_path.startswith("./"):
            rel_path = rel_path[2:]
        chapter_outputs.append({
            "index": chapter.get("index"),
            "title": chapter.get("title"),
            "file_url": f"/static/audio/{job_id}/{rel_path}",
            "relative_path": rel_path,
        })

    full_story_entry = None
    if all_full_story_chunks:
        chunk_paths = [str(job_dir / rel_path) for rel_path in all_full_story_chunks]
        full_story_name = f"full_story.{output_format}"
        full_story_path = job_dir / full_story_name
        merger.merge_wav_files(
            input_files=chunk_paths,
            output_path=str(full_story_path),
            format=output_format,
            cleanup_chunks=False,
            progress_callback=lambda ratio: _update_review_post_progress(
                job_id,
                int((jobs.get(job_id, {}).get("post_process_total") or 1)),
                ratio,
            ),
        )
        with queue_lock:
            entry = jobs.get(job_id)
            if entry:
                entry["post_process_done"] = min(
                    int(entry.get("post_process_total") or 0) or 0,
                    int(entry.get("post_process_total") or 0) or 0,
                )
        full_story_entry = {
            "title": "Full Story",
            "file_url": f"/static/audio/{job_id}/{full_story_name}",
            "relative_path": full_story_name,
        }

    book_mode = bool(manifest.get("book_mode"))
    book_outputs = []
    if book_mode:
        # Merge book-level audio files
        for book in manifest.get("books", []):
            book_chunk_files = book.get("chunk_files") or []
            if not book_chunk_files:
                continue
            
            chunk_paths = [str(job_dir / rel_path) for rel_path in book_chunk_files]
            missing_chunks = [p for p in chunk_paths if not Path(p).exists()]
            if missing_chunks:
                logger.error(f"Missing chunk files for book merge: {missing_chunks}")
                continue
            
            output_filename = book.get("output_filename")
            rel_book_dir = book.get("book_dir") or "."
            if not output_filename:
                continue
            
            book_dir_path = job_dir / rel_book_dir
            book_dir_path.mkdir(parents=True, exist_ok=True)
            output_path = book_dir_path / output_filename
            
            logger.info(f"Merging book-level audio: {len(chunk_paths)} chunks into {output_path}")
            merger.merge_wav_files(
                input_files=chunk_paths,
                output_path=str(output_path),
                format=output_format,
                cleanup_chunks=False,
                progress_callback=lambda ratio, idx=len(book_outputs) + 1: _update_review_post_progress(
                    job_id,
                    len(chapters) + idx,
                    ratio,
                ),
            )
            
            rel_path = Path(rel_book_dir) / output_filename
            book_outputs.append({
                "index": book.get("index", 0),
                "title": book.get("title"),
                "file_url": f"/static/audio/{job_id}/{rel_path.as_posix()}",
                "relative_path": rel_path.as_posix(),
            })

    metadata = {
        "chapter_mode": job_entry.get("chapter_mode"),
        "book_mode": book_mode,
        "output_format": output_format,
        "chapters": chapter_outputs,
        "chapter_count": len(chapter_outputs),
        "books": book_outputs,
        "book_count": len(book_outputs),
        "full_story": full_story_entry,
    }
    save_job_metadata(job_dir, metadata)

    invalidate_library_cache()

    job_end_time = datetime.now()
    with queue_lock:
        entry = jobs.get(job_id)
        if entry:
            entry["status"] = "completed"
            entry["progress"] = 100
            entry["eta_seconds"] = 0
            entry["post_process_percent"] = 100
            entry["post_process_active"] = False
            entry["post_process_done"] = int(entry.get("post_process_total") or entry.get("post_process_done") or 0)
            entry["chapter_outputs"] = chapter_outputs
            entry["completed_at"] = job_end_time.isoformat()
            if full_story_entry:
                entry["full_story"] = full_story_entry
            entry["output_file"] = (full_story_entry or (chapter_outputs[0] if chapter_outputs else {})).get("file_url")
            # Compute timing metrics — use duration_seconds from chunks_metadata for per-chunk times
            started_at_str = entry.get("started_at")
            total_chunks = entry.get("total_chunks") or 0
            try:
                started_dt = datetime.fromisoformat(started_at_str) if started_at_str else None
                total_seconds = (job_end_time - started_dt).total_seconds() if started_dt else None
            except Exception:
                total_seconds = None
            # Prefer render-time chunk_times already accumulated in the live timing_metrics
            # (set by update_progress during generation). Fall back to WAV audio durations.
            chunk_times: List[float] = list(entry.get("timing_metrics", {}).get("chunk_times") or [])
            if not chunk_times:
                try:
                    import wave as _wave
                    chunks_meta_path = job_dir / "chunks_metadata.json"
                    if chunks_meta_path.exists():
                        with chunks_meta_path.open("r", encoding="utf-8") as _f:
                            _cmeta = json.load(_f)
                        # Also check if timing_metrics was already saved in chunks_metadata
                        _saved_tm = _cmeta.get("timing_metrics") or {}
                        chunk_times = list(_saved_tm.get("chunk_times") or [])
                        if not chunk_times:
                            for _c in sorted(_cmeta.get("chunks") or [], key=lambda x: x.get("order_index", x.get("chunk_index", 0))):
                                _rel = _c.get("relative_file")
                                if not _rel:
                                    continue
                                _wav_path = job_dir / _rel
                                if not _wav_path.exists():
                                    continue
                                try:
                                    with _wave.open(str(_wav_path), "rb") as _wf:
                                        _dur = _wf.getnframes() / float(_wf.getframerate())
                                    chunk_times.append(round(_dur, 1))
                                except Exception:
                                    pass
                except Exception:
                    pass
            avg_chunk = round(sum(chunk_times) / len(chunk_times), 1) if chunk_times else (
                round(total_seconds / total_chunks, 1) if (total_seconds and total_chunks) else None
            )
            final_timing_metrics = {
                "started_at": started_at_str,
                "completed_at": job_end_time.isoformat(),
                "total_seconds": round(total_seconds, 1) if total_seconds is not None else None,
                "chunk_count": total_chunks,
                "avg_chunk_seconds": avg_chunk,
                "min_chunk_seconds": round(min(chunk_times), 1) if chunk_times else None,
                "max_chunk_seconds": round(max(chunk_times), 1) if chunk_times else None,
                "chunk_times": chunk_times,
            }
            entry["timing_metrics"] = final_timing_metrics
    _persist_job_state(job_id, force=True)

    # Write final timing_metrics to chunks_metadata.json so the library can read it from disk.
    try:
        _cmeta_path = job_dir / "chunks_metadata.json"
        if _cmeta_path.exists():
            with _cmeta_path.open("r", encoding="utf-8") as _fh:
                _cmeta_data = json.load(_fh)
            with queue_lock:
                _final_tm = (jobs.get(job_id) or {}).get("timing_metrics")
            if _final_tm:
                _cmeta_data["timing_metrics"] = _final_tm
                with _cmeta_path.open("w", encoding="utf-8") as _fh:
                    json.dump(_cmeta_data, _fh, indent=2)
    except Exception as _e:
        logger.warning("Could not write timing_metrics to chunks_metadata.json (review path): %s", _e)


@app.route('/api/jobs/<job_id>/review/finish', methods=['POST'])
def finish_review_job(job_id: str):
    """Finalize a review-mode job by merging audio and updating metadata."""
    try:
        with queue_lock:
            job_entry = jobs.get(job_id)
            if not job_entry:
                return jsonify({"success": False, "error": "Job not found"}), 404
            _ensure_review_ready(job_entry)
            # Allow recompile for completed jobs with review_mode (chunk review from library)
            if job_entry.get("status") not in ("waiting_review", "completed"):
                return jsonify({"success": False, "error": "Job is not ready for recompile"}), 409
            if _has_active_regen_tasks(job_entry):
                return jsonify({"success": False, "error": "Wait for all chunk regenerations to finish"}), 409

        manifest = _load_review_manifest(job_id, job_entry)
        _merge_review_job(job_id, job_entry, manifest)
        return jsonify({"success": True})
    except FileNotFoundError as exc:
        return jsonify({"success": False, "error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to finalize review job %s: %s", job_id, exc, exc_info=True)
        return jsonify({"success": False, "error": "Failed to finalize review job"}), 500

    voice_manager = VoiceManager()  # Reload manifest with updated manifest file
    missing = voice_manager.missing_samples()

    return jsonify({
        "success": True,
        "samples": report.get("manifest", {}),
        "generated": report.get("generated", []),
        "skipped_existing": report.get("skipped_existing", []),
        "failed": report.get("failed", []),
        "samples_ready": voice_manager.all_samples_present(),
        "missing_samples": missing,
        "total_unique_voices": voice_manager.total_unique_voice_count(),
        "sample_count": voice_manager.sample_count()
    })


@app.route('/api/preview', methods=['POST'])
def preview_audio():
    """Generate a short preview clip with optional FX settings."""
    data = request.json or {}
    voice = (data.get('voice') or '').strip()
    lang_code = (data.get('lang_code') or 'a').strip()
    text = (data.get('text') or '').strip()
    speed = float(data.get('speed') or 1.0)
    fx_settings = VoiceFXSettings.from_payload(data.get('fx'))
    requested_engine = (data.get('tts_engine') or '').strip().lower()

    if not text:
        text = "This is a quick preview."

    config = load_config()
    # Use requested engine if provided, otherwise fall back to config
    if requested_engine:
        engine_name = _normalize_engine_name(requested_engine)
    else:
        engine_name = _normalize_engine_name(config.get("tts_engine"))
    sample_rate = int(config.get('sample_rate', DEFAULT_SAMPLE_RATE))
    audio_bytes = None

    # For prompt-based engines the 'voice' field carries a file path that must
    # be forwarded as audio_prompt_path, not as the voice name parameter.
    _PROMPT_ENGINES = {
        "chatterbox_turbo_local", "chatterbox_turbo_replicate",
        "voxcpm_local", "pocket_tts", "qwen3_clone",
    }
    audio_prompt_path = data.get('audio_prompt_path') or None
    if engine_name in _PROMPT_ENGINES and voice and not audio_prompt_path:
        audio_prompt_path = voice
        voice = ''

    # Qwen3 engines don't use Kokoro-style lang codes — map 'a'/'b'/etc. to 'auto'
    _QWEN3_ENGINES = {"qwen3_custom", "qwen3_clone"}
    _KOKORO_LANG_CODES = {'a', 'b', 'e', 'f', 'h', 'i', 'j', 'p', 'z'}
    if engine_name in _QWEN3_ENGINES and lang_code in _KOKORO_LANG_CODES:
        lang_code = 'auto'

    if not voice and not audio_prompt_path:
        return jsonify({"success": False, "error": "Voice is required for preview."}), 400

    try:
        logger.info("Preview request: engine=%s, voice=%s, lang_code=%s, prompt=%s",
                    engine_name, voice, lang_code, audio_prompt_path)
        engine = get_tts_engine(engine_name, config=config)
        # Check if engine has generate_audio method that returns numpy array
        if hasattr(engine, 'generate_audio'):
            import inspect as _inspect
            _sig = _inspect.signature(engine.generate_audio)
            # Qwen3 custom expects lowercase speaker names
            _voice = voice.lower() if engine_name == 'qwen3_custom' and voice else voice
            # For engines that accept audio_prompt_path natively, pass it directly.
            # For engines that use voice= as the prompt path (e.g. pocket_tts), put it back there.
            _has_prompt_param = 'audio_prompt_path' in _sig.parameters
            _effective_voice = audio_prompt_path if (audio_prompt_path and not _has_prompt_param) else _voice
            _kwargs = dict(
                text=text,
                voice=_effective_voice,
                lang_code=lang_code,
                speed=speed,
                sample_rate=sample_rate,
                fx_settings=fx_settings,
            )
            if audio_prompt_path and _has_prompt_param:
                _kwargs['audio_prompt_path'] = audio_prompt_path
            audio = engine.generate_audio(**_kwargs)
            if hasattr(audio, 'size') and audio.size == 0:
                raise RuntimeError("No audio produced for the requested preview.")
            if hasattr(audio, 'size'):
                # Numpy array - write to buffer
                buffer = io.BytesIO()
                sf.write(buffer, audio, sample_rate, format='wav')
                audio_bytes = buffer.getvalue()
            elif isinstance(audio, str) and os.path.exists(audio):
                # File path returned
                with open(audio, 'rb') as fh:
                    audio_bytes = fh.read()
            elif isinstance(audio, bytes):
                audio_bytes = audio
            else:
                raise RuntimeError("Unexpected audio format from engine.")
    except Exception as exc:
        logger.error("Preview generation failed: %s", exc, exc_info=True)
        return jsonify({"success": False, "error": str(exc)}), 400

    if not audio_bytes:
        return jsonify({"success": False, "error": "Preview failed to generate audio."}), 500

    encoded = base64.b64encode(audio_bytes).decode('ascii')
    return jsonify({
        "success": True,
        "audio_base64": encoded,
        "mime_type": "audio/wav",
    })


@app.route('/api/custom-voices', methods=['GET', 'POST'])
def custom_voices_collection():
    """List or create custom voice blends."""
    if request.method == 'GET':
        entries = list_custom_voice_entries()
        return jsonify({
            "success": True,
            "voices": [_to_public_custom_voice(entry) for entry in entries],
        })

    try:
        payload = _prepare_custom_voice_payload(request.json or {})
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    now = datetime.now().isoformat()
    payload["created_at"] = now
    payload["updated_at"] = now
    saved = save_custom_voice(payload)
    clear_cached_custom_voice()
    return jsonify({
        "success": True,
        "voice": _to_public_custom_voice(saved),
    }), 201


@app.route('/api/custom-voices/<voice_id>', methods=['GET', 'PUT', 'DELETE'])
def custom_voice_detail(voice_id):
    """Retrieve, update, or delete a custom voice definition."""
    raw = _get_raw_custom_voice(voice_id)
    if not raw:
        return jsonify({"success": False, "error": "Custom voice not found."}), 404

    if request.method == 'GET':
        return jsonify({"success": True, "voice": _to_public_custom_voice(raw)})

    if request.method == 'DELETE':
        delete_custom_voice(raw["id"])
        clear_cached_custom_voice(f"{CUSTOM_CODE_PREFIX}{raw['id']}")
        return jsonify({"success": True, "deleted": True})

    try:
        payload = _prepare_custom_voice_payload(request.json or {}, existing=raw)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    payload["id"] = raw["id"]
    payload["created_at"] = raw.get("created_at")
    payload["updated_at"] = datetime.now().isoformat()
    updated = replace_custom_voice(payload)
    clear_cached_custom_voice(f"{CUSTOM_CODE_PREFIX}{raw['id']}")
    return jsonify({"success": True, "voice": _to_public_custom_voice(updated)})


@app.route('/api/settings', methods=['GET', 'POST'])
def settings():
    """Get or update settings"""
    if request.method == 'GET':
        config = load_config()
        return jsonify({
            "success": True,
            "settings": config
        })
    else:
        try:
            new_settings = request.json
            config = load_config()
            config.update(new_settings)
            save_config(config)
            
            return jsonify({
                "success": True,
                "message": "Settings updated"
            })
        except Exception as e:
            logger.error(f"Error updating settings: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 400


@app.route('/api/gemini/models', methods=['POST'])
def list_gemini_models():
    """List available Gemini models using the provided or saved API key."""
    try:
        data = request.json or {}
        api_key = (data.get('api_key') or '').strip()

        if not api_key:
            config = load_config()
            api_key = (config.get('gemini_api_key') or '').strip()

        if not api_key:
            return jsonify({
                "success": False,
                "error": "Gemini API key is required"
            }), 400

        models = GeminiProcessor.list_available_models(api_key)
        return jsonify({
            "success": True,
            "models": models
        })

    except (GeminiProcessorError, LocalLLMProcessorError) as exc:
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 400
    except Exception as e:  # pragma: no cover - general failure
        logger.error(f"Error listing Gemini models: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Failed to list Gemini models"
        }), 500


@app.route('/api/local-llm/models', methods=['POST'])
def list_local_llm_models():
    """List available local LLM models for LM Studio/Ollama."""
    try:
        data = request.json or {}
        config = load_config()

        provider = (data.get("provider") or config.get("llm_local_provider") or "").strip()
        base_url = (data.get("base_url") or config.get("llm_local_base_url") or "").strip()
        api_key = (data.get("api_key") or config.get("llm_local_api_key") or "").strip()
        timeout = int(data.get("timeout") or config.get("llm_local_timeout") or 30)

        models = LocalLLMProcessor.list_available_models(
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        return jsonify({
            "success": True,
            "models": models,
        })
    except (ValueError, TypeError):
        return jsonify({
            "success": False,
            "error": "Invalid timeout value"
        }), 400
    except (GeminiProcessorError, LocalLLMProcessorError) as exc:
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 400
    except Exception as exc:  # pragma: no cover - general failure
        logger.error("Error listing local LLM models: %s", exc, exc_info=True)
        return jsonify({
            "success": False,
            "error": "Failed to list local LLM models"
        }), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """Analyze text and return statistics"""
    try:
        with log_request_timing("POST /api/analyze"):
            data = request.json
            text = (data.get('text') or '').strip()

            if not text:
                return jsonify({
                    "success": False,
                    "error": "No text provided"
                }), 400

            config = load_config()
            selected_engine = config.get("tts_engine")
            requested_engine = (data.get('tts_engine') or '').strip()
            if requested_engine:
                normalized = _normalize_engine_name(requested_engine)
                if normalized not in AVAILABLE_ENGINES:
                    return jsonify({
                        "success": False,
                        "error": f"Unsupported TTS engine: {requested_engine}"
                    }), 400
                selected_engine = normalized

            processor = _create_text_processor_for_engine(selected_engine, config["chunk_size"], config)
            stats = processor.get_statistics(text)
            custom_heading = data.get("custom_heading")
            book_matches = list(BOOK_HEADING_PATTERN.finditer(text))
            section_pattern = _build_section_heading_pattern(custom_heading)
            section_matches = list(section_pattern.finditer(text))

            if book_matches:
                hierarchy = split_text_into_book_sections(text, custom_heading)
                books = hierarchy.get("books") or []
                chapters: List[Dict[str, Any]] = []
                for book in books:
                    chapters.extend(book.get("chapters") or [])
                stats['section_detection'] = {
                    "detected": True,
                    "count": len(chapters),
                    "titles": [c.get('title') for c in chapters if c.get('title')],
                    "kind": "book",
                    "book_count": len(books),
                    "section_count": len(chapters),
                }
            elif section_matches:
                sections = split_text_into_sections(text, custom_heading)
                stats['section_detection'] = {
                    "detected": True,
                    "count": len(sections),
                    "titles": [s.get('title') for s in sections if s.get('title')],
                    "kind": "section",
                    "book_count": 0,
                    "section_count": len(sections),
                }
            else:
                stats['section_detection'] = {
                    "detected": False,
                    "count": 0,
                    "titles": [],
                    "kind": None,
                    "book_count": 0,
                    "section_count": 0,
                }

            return jsonify({
                "success": True,
                "statistics": stats
            })

    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/jobs/clear-all', methods=['DELETE'])
def clear_all_jobs():
    """Remove all jobs that are not actively processing from the queue, DB, and disk."""
    import shutil
    skipped = []
    removed = []
    errors = []

    try:
        with queue_lock:
            all_job_ids = list(jobs.keys())

        for job_id in all_job_ids:
            with queue_lock:
                job_entry = jobs.get(job_id)
                status = job_entry.get("status") if job_entry else None
            if status in {"processing", "pausing"}:
                skipped.append(job_id)
                continue
            try:
                with queue_lock:
                    if job_id in jobs:
                        jobs[job_id]["status"] = "deleted"
                        cancel_flags[job_id] = True
                        pause_flags.pop(job_id, None)
                        cancel_events.pop(job_id, None)
                    job_entry = jobs.get(job_id) or {}
                    job_dir = _job_dir_from_entry(job_id, job_entry)

                with _get_jobs_db_connection() as conn:
                    row = conn.execute("SELECT status, job_dir FROM jobs WHERE job_id=?", (job_id,)).fetchone()
                    if row and not job_dir or str(job_dir) == ".":
                        job_dir = _job_dir_from_entry(job_id, {"job_dir": row["job_dir"]})
                    db_status = (row["status"] if row else None) or status
                    conn.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
                    conn.commit()

                with queue_lock:
                    jobs.pop(job_id, None)

                job_data_dir = JOBS_DATA_DIR / job_id
                if job_data_dir.exists():
                    try:
                        shutil.rmtree(job_data_dir, onerror=handle_remove_readonly)
                    except Exception as rm_err:
                        logger.warning("Could not delete job data dir %s: %s", job_data_dir, rm_err)

                if db_status not in {"completed", "done", "review"} and job_dir and job_dir.exists():
                    try:
                        shutil.rmtree(job_dir, onerror=handle_remove_readonly)
                    except Exception as rm_err:
                        logger.warning("Could not delete job audio dir %s: %s", job_dir, rm_err)

                removed.append(job_id)
            except Exception as e:
                logger.error("Error clearing job %s: %s", job_id, e, exc_info=True)
                errors.append(job_id)

        # Also sweep DB for any persisted jobs not in memory
        try:
            with _get_jobs_db_connection() as conn:
                rows = conn.execute("SELECT job_id, status, job_dir FROM jobs").fetchall()
            for row in rows:
                job_id = row["job_id"]
                if job_id in removed or job_id in skipped or job_id in errors:
                    continue
                db_status = row["status"] or ""
                if db_status in {"processing", "pausing"}:
                    skipped.append(job_id)
                    continue
                try:
                    job_dir = _job_dir_from_entry(job_id, {"job_dir": row["job_dir"]})
                    with _get_jobs_db_connection() as conn:
                        conn.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
                        conn.commit()
                    job_data_dir = JOBS_DATA_DIR / job_id
                    if job_data_dir.exists():
                        shutil.rmtree(job_data_dir, onerror=handle_remove_readonly)
                    if db_status not in {"completed", "done", "review"} and job_dir and job_dir.exists():
                        shutil.rmtree(job_dir, onerror=handle_remove_readonly)
                    removed.append(job_id)
                except Exception as e:
                    logger.error("Error clearing DB job %s: %s", job_id, e, exc_info=True)
                    errors.append(job_id)
        except Exception as e:
            logger.error("Error sweeping jobs DB during clear-all: %s", e, exc_info=True)

        invalidate_library_cache()
        return jsonify({
            "success": True,
            "removed": len(removed),
            "skipped": len(skipped),
            "errors": len(errors),
        })
    except Exception as e:
        logger.error("clear_all_jobs failed: %s", e, exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/jobs/<job_id>/delete', methods=['DELETE'])
def delete_job(job_id: str):
    """Remove a job from the queue and delete its output files from disk."""
    import shutil
    try:
        with queue_lock:
            job_entry = jobs.get(job_id)
            if job_entry:
                status = job_entry.get("status")
                if status in {"processing", "pausing"}:
                    return jsonify({
                        "success": False,
                        "error": "Job is currently processing. Pause or cancel it before deleting.",
                    }), 409
                job_entry["status"] = "deleted"
                cancel_flags[job_id] = True
                pause_flags.pop(job_id, None)
                cancel_events.pop(job_id, None)
            else:
                status = None
            job_dir = _job_dir_from_entry(job_id, job_entry or {})

        with _get_jobs_db_connection() as conn:
            row = conn.execute("SELECT status, job_dir FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if not job_entry and not row:
                return jsonify({"success": False, "error": "Job not found"}), 404
            if row and not status:
                status = row["status"]
                if not job_dir or str(job_dir) == ".":
                    job_dir = _job_dir_from_entry(job_id, {"job_dir": row["job_dir"]})
            conn.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
            conn.commit()

        with queue_lock:
            jobs.pop(job_id, None)

        # Always delete the job metadata folder (data/jobs/<job_id>/).
        job_data_dir = JOBS_DATA_DIR / job_id
        if job_data_dir.exists():
            try:
                shutil.rmtree(job_data_dir, onerror=handle_remove_readonly)
                logger.info("Deleted job data directory: %s", job_data_dir)
            except Exception as rm_err:
                logger.warning("Could not delete job data directory %s: %s", job_data_dir, rm_err)

        # Only delete the audio output folder (static/audio/<job_id>/) for jobs
        # that never completed. Completed/done/review jobs are library items —
        # their audio must be preserved so the library entry survives.
        if status not in {"completed", "done", "review"} and job_dir.exists():
            try:
                shutil.rmtree(job_dir, onerror=handle_remove_readonly)
                logger.info("Deleted job audio directory: %s", job_dir)
            except Exception as rm_err:
                logger.warning("Could not delete job audio directory %s: %s", job_dir, rm_err)

        invalidate_library_cache()
        return jsonify({"success": True})
    except Exception as e:
        logger.error("Error deleting job %s: %s", job_id, e, exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/jobs/<job_id>/pause', methods=['POST'])
def pause_job(job_id: str):
    try:
        with queue_lock:
            job_entry = jobs.get(job_id)
            if not job_entry:
                return jsonify({"success": False, "error": "Job not found"}), 404
            if job_entry.get("status") not in {"queued", "processing"}:
                return jsonify({"success": False, "error": "Job is not running"}), 409
            pause_flags[job_id] = True
            job_entry["status"] = "pausing" if job_entry.get("status") == "processing" else "paused"
            if job_entry["status"] == "paused":
                job_entry["paused_at"] = datetime.now().isoformat()
        _persist_job_state(job_id, force=True)
        return jsonify({"success": True})
    except Exception as exc:
        logger.error("Failed to pause job %s: %s", job_id, exc, exc_info=True)
        return jsonify({"success": False, "error": "Failed to pause job"}), 500


@app.route('/api/jobs/<job_id>/resume', methods=['POST'])
def resume_job(job_id: str):
    try:
        start_worker_thread()
        with queue_lock:
            job_entry = jobs.get(job_id)
            if not job_entry:
                return jsonify({"success": False, "error": "Job not found"}), 404
            total_chunks = int(job_entry.get("total_chunks") or 0)
            processed_chunks = int(job_entry.get("processed_chunks") or 0)
            post_total = int(job_entry.get("post_process_total") or 0)
            post_done = int(job_entry.get("post_process_done") or 0)
            if total_chunks and processed_chunks >= total_chunks and (post_total == 0 or post_done >= post_total):
                job_entry["status"] = "completed"
                job_entry["progress"] = 100
                job_entry["eta_seconds"] = 0
                job_entry["interrupted_at"] = None
                _persist_job_state(job_id, force=True)
                return jsonify({"success": True, "message": "Job already completed"})
            if job_entry.get("status") not in {"paused", "interrupted"}:
                return jsonify({"success": False, "error": "Job is not paused"}), 409
            pause_flags.pop(job_id, None)
            resume_from = job_entry.get("resume_from_chunk_index")
            if resume_from is None:
                resume_from = job_entry.get("processed_chunks") or 0
            job_entry["resume_from_chunk_index"] = int(resume_from or 0)
            job_entry["status"] = "queued"
            job_entry["paused_at"] = None
            job_entry["interrupted_at"] = None
        job_data = _build_job_data_from_entry(job_id, job_entry)
        job_queue.put(job_data)
        _persist_job_state(job_id, force=True)
        return jsonify({"success": True})
    except Exception as exc:
        logger.error("Failed to resume job %s: %s", job_id, exc, exc_info=True)
        return jsonify({"success": False, "error": "Failed to resume job"}), 500


@app.route('/api/jobs/<job_id>/details', methods=['GET'])
def get_job_details(job_id: str):
    try:
        with queue_lock:
            job_entry = jobs.get(job_id)
            if not job_entry:
                return jsonify({"success": False, "error": "Job not found"}), 404
            text = _load_job_text(job_entry.get("text_path"))
            payload = {
                "job_id": job_id,
                "status": job_entry.get("status"),
                "created_at": job_entry.get("created_at"),
                "started_at": job_entry.get("started_at"),
                "completed_at": job_entry.get("completed_at"),
                "engine": job_entry.get("engine"),
                "speakers": job_entry.get("speakers") or _extract_speakers_for_text(text),
                "text": text,
                "voice_assignments": job_entry.get("voice_assignments") or {},
                "chapter_mode": bool(job_entry.get("chapter_mode")),
                "full_story_requested": bool(job_entry.get("full_story_requested")),
                "resume_from_chunk_index": job_entry.get("resume_from_chunk_index"),
                "last_completed_chunk_index": job_entry.get("last_completed_chunk_index"),
                "timing_metrics": job_entry.get("timing_metrics"),
                "total_chunks": job_entry.get("total_chunks"),
                "processed_chunks": job_entry.get("processed_chunks"),
            }
        return jsonify({"success": True, "job": payload})
    except Exception as exc:
        logger.error("Failed to load job details %s: %s", job_id, exc, exc_info=True)
        return jsonify({"success": False, "error": "Failed to load job details"}), 500


@app.route('/api/sections/preview', methods=['POST'])
def preview_section_detection():
    """Preview detected book/section structure for UI review."""
    try:
        data = request.json or {}
        text = (data.get('text') or '').strip()
        custom_heading = data.get('custom_heading')
        voice_assignments = data.get('voice_assignments', {})

        if not text:
            return jsonify({
                "success": False,
                "error": "No text provided"
            }), 400

        voice_assignments = _prepare_voice_assignments(text, voice_assignments)

        hierarchy = split_text_into_book_sections(text, custom_heading)

        def preview_content(content: str, limit: int = 220, heading: Optional[str] = None) -> str:
            snippet = (content or "").strip()
            if heading:
                snippet = re.sub(rf"^\s*{re.escape(heading)}\s*", "", snippet, flags=re.IGNORECASE)
            snippet = re.sub(r"\[/?[^\]]+\]", "", snippet)
            snippet = re.sub(r"\s+", " ", snippet).strip()
            if len(snippet) <= limit:
                return snippet
            return snippet[: limit - 1].rstrip() + "…"

        if hierarchy.get("kind") == "book":
            books = []
            for book in hierarchy.get("books") or []:
                chapters = []
                for chapter in book.get("chapters") or []:
                    chapters.append({
                        "title": chapter.get("title") or "",
                        "preview": preview_content(chapter.get("content") or "", heading=chapter.get("heading")),
                        "heading": chapter.get("heading"),
                        "heading_start": chapter.get("heading_start"),
                        "heading_end": chapter.get("heading_end"),
                    })
                books.append({
                    "title": book.get("title") or "",
                    "chapters": chapters,
                })

            return jsonify({
                "success": True,
                "kind": "book",
                "books": books,
                "sections": [],
                "book_count": len(books),
                "section_count": sum(len(b.get("chapters") or []) for b in books)
            })

        if hierarchy.get("kind") == "section":
            sections = [
                {
                    "title": section.get("title") or "",
                    "preview": preview_content(section.get("content") or "", heading=section.get("heading")),
                    "heading": section.get("heading"),
                    "heading_start": section.get("heading_start"),
                    "heading_end": section.get("heading_end"),
                }
                for section in hierarchy.get("sections") or []
            ]
            return jsonify({
                "success": True,
                "kind": "section",
                "books": [],
                "sections": sections,
                "book_count": 0,
                "section_count": len(sections)
            })

        return jsonify({
            "success": True,
            "kind": "none",
            "books": [],
            "sections": [],
            "book_count": 0,
            "section_count": 0
        })

    except Exception as e:
        logger.error(f"Error previewing section detection: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/library/<job_id>/title', methods=['PUT'])
def update_library_title(job_id: str):
    """Update the display title for a library collection."""
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"success": False, "error": "Title is required"}), 400

    job_dir = OUTPUT_DIR / job_id
    if not job_dir.exists():
        return jsonify({"success": False, "error": "Item not found"}), 404

    metadata = load_job_metadata(job_dir) or {}
    metadata["collection_title"] = title
    save_job_metadata(job_dir, metadata)

    return jsonify({"success": True, "title": title})


@app.route('/api/gemini/process', methods=['POST'])
def process_text_with_gemini():
    """Send text (optionally chapterized) through Google Gemini."""
    try:
        data = request.json or {}
        text = (data.get('text') or '').strip()
        prefer_chapters = bool(data.get('prefer_chapters', True))
        custom_heading = data.get('custom_heading')
        prompt_override = (data.get('prompt_override') or '').strip()

        if not text:
            return jsonify({
                "success": False,
                "error": "No text provided"
            }), 400

        config = load_config()
        provider = (config.get("llm_provider") or DEFAULT_LLM_PROVIDER).lower().strip()
        if provider == "gemini":
            api_key = (config.get('gemini_api_key') or '').strip()
            if not api_key:
                return jsonify({
                    "success": False,
                    "error": "Gemini API key not configured"
                }), 400

        prompt_prefix = prompt_override or (config.get('gemini_prompt') or '').strip()

        sections = build_gemini_sections(text, prefer_chapters, config, custom_heading)
        if not sections:
            return jsonify({
                "success": False,
                "error": "Unable to create sections for Gemini processing"
            }), 400

        text_processor = TextProcessor(chunk_size=config.get('chunk_size', 500))
        known_speakers = set(text_processor.extract_speakers(text))

        processed_sections = []
        for idx, section in enumerate(sections, start=1):
            chapter_text = section.get('content', '').strip()
            if not chapter_text:
                continue

            combined_prompt = compose_gemini_prompt(
                section,
                prompt_prefix,
                sorted(known_speakers)
            )
            response_text = _run_llm_prompt(combined_prompt, config)
            detected_speakers = text_processor.extract_speakers(response_text)
            for speaker_name in detected_speakers:
                known_speakers.add(speaker_name)
            processed_sections.append({
                "index": idx,
                "title": section.get('title'),
                "source": section.get('source'),
                "output": response_text.strip(),
                "speakers": detected_speakers
            })

        if not processed_sections:
            return jsonify({
                "success": False,
                "error": "Gemini processing produced no output"
            }), 500

        final_text = "\n\n".join(
            section['output']
            for section in processed_sections
            if section.get('output')
        ).strip()

        return jsonify({
            "success": True,
            "result_text": final_text,
            "processed_sections": processed_sections,
            "chapter_mode": any(section.get('source') == 'chapter' for section in sections),
            "section_count": len(processed_sections)
        })

    except (GeminiProcessorError, LocalLLMProcessorError) as exc:
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 400
    except Exception as e:  # pragma: no cover - general failure
        logger.error(f"Error during Gemini processing: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Failed to process text with Gemini"
        }), 500


@app.route('/api/gemini/process-full', methods=['POST'])
def process_full_text_with_gemini():
    """Send the entire text to Gemini using the configured pre-prompt without chunking."""
    try:
        data = request.json or {}
        text = (data.get('text') or '').strip()
        prompt_override = (data.get('prompt_override') or '').strip()

        if not text:
            return jsonify({
                "success": False,
                "error": "No text provided"
            }), 400

        config = load_config()
        provider = (config.get("llm_provider") or DEFAULT_LLM_PROVIDER).lower().strip()
        if provider == "gemini":
            api_key = (config.get('gemini_api_key') or '').strip()
            if not api_key:
                return jsonify({
                    "success": False,
                    "error": "Gemini API key not configured"
                }), 400

        prompt_prefix = prompt_override or (config.get('gemini_prompt') or '').strip()

        prompt_parts = []
        if prompt_prefix:
            prompt_parts.append(prompt_prefix)
        prompt_parts.append(text)
        combined_prompt = "\n\n".join(part.strip() for part in prompt_parts if part).strip()

        response_text = _run_llm_prompt(combined_prompt, config)

        return jsonify({
            "success": True,
            "result_text": response_text.strip()
        })

    except (GeminiProcessorError, LocalLLMProcessorError) as exc:
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 400
    except Exception as e:  # pragma: no cover - general failure
        logger.error(f"Error during full-text Gemini processing: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Failed to process text with Gemini"
        }), 500


@app.route('/api/gemini/speaker-profiles', methods=['POST'])
def process_gemini_speaker_profiles():
    """Generate speaker profile table and parse structured attributes."""
    try:
        data = request.json or {}
        speakers = data.get('speakers') or []
        context = (data.get('context') or '').strip()
        prompt_override = (data.get('prompt_override') or '').strip()

        speakers = [str(s).strip() for s in speakers if str(s).strip()]
        if not speakers:
            return jsonify({
                "success": False,
                "error": "No speakers provided"
            }), 400

        config = load_config()
        provider = (config.get("llm_provider") or DEFAULT_LLM_PROVIDER).lower().strip()
        if provider == "gemini":
            api_key = (config.get('gemini_api_key') or '').strip()
            if not api_key:
                return jsonify({
                    "success": False,
                    "error": "Gemini API key not configured"
                }), 400
        prompt_prefix = prompt_override or (config.get('gemini_speaker_profile_prompt') or '').strip()
        if not prompt_prefix:
            return jsonify({
                "success": False,
                "error": "Speaker profile prompt not configured"
            }), 400

        processed_text = (data.get('processed_text') or '').strip()
        prompt = compose_gemini_speaker_profile_prompt(prompt_prefix, speakers, context, processed_text)
        response_text = _run_llm_prompt(prompt, config)
        profiles = parse_gemini_speaker_table(response_text)

        return jsonify({
            "success": True,
            "profiles": profiles,
            "raw": response_text.strip()
        })
    except (GeminiProcessorError, LocalLLMProcessorError) as exc:
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 400
    except Exception as e:  # pragma: no cover - general failure
        logger.error(f"Error during Gemini speaker profile processing: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Failed to process speaker profiles with Gemini"
        }), 500


@app.route('/api/gemini/sections', methods=['POST'])
def get_gemini_sections():
    """Return the list of Gemini sections for the provided text."""
    try:
        data = request.json or {}
        text = (data.get('text') or '').strip()
        prefer_chapters = bool(data.get('prefer_chapters', True))
        custom_heading = data.get('custom_heading')

        if not text:
            return jsonify({
                "success": False,
                "error": "No text provided"
            }), 400

        config = load_config()
        sections = build_gemini_sections(text, prefer_chapters, config, custom_heading)

        sanitized = []
        for idx, section in enumerate(sections, start=1):
            sanitized.append({
                "id": idx,
                "title": section.get('title'),
                "content": section.get('content'),
                "source": section.get('source')
            })

        return jsonify({
            "success": True,
            "sections": sanitized,
            "count": len(sanitized)
        })

    except Exception as e:  # pragma: no cover - general failure
        logger.error(f"Error building Gemini sections: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Failed to build Gemini sections"
        }), 500


@app.route('/api/gemini/process-section', methods=['POST'])
def process_gemini_section():
    """Process a single text section via Gemini."""
    try:
        data = request.json or {}
        content = (data.get('content') or '').strip()
        prompt_override = (data.get('prompt_override') or '').strip()

        if not content:
            return jsonify({
                "success": False,
                "error": "No section content provided"
            }), 400

        config = load_config()
        provider = (config.get("llm_provider") or DEFAULT_LLM_PROVIDER).lower().strip()
        if provider == "gemini":
            api_key = (config.get('gemini_api_key') or '').strip()
            if not api_key:
                return jsonify({
                    "success": False,
                    "error": "Gemini API key not configured"
                }), 400

        prompt_prefix = prompt_override or (config.get('gemini_prompt') or '').strip()

        raw_known = data.get('known_speakers') or []
        known_speakers = []
        if isinstance(raw_known, list):
            for entry in raw_known:
                if isinstance(entry, str):
                    normalized = entry.strip().lower()
                    if normalized:
                        known_speakers.append(normalized)

        text_processor = TextProcessor()
        prompt = compose_gemini_prompt(
            {"content": content},
            prompt_prefix,
            known_speakers
        )
        response_text = _run_llm_prompt(prompt, config)
        detected_speakers = text_processor.extract_speakers(response_text)

        return jsonify({
            "success": True,
            "result_text": response_text.strip(),
            "speakers": detected_speakers
        })

    except (GeminiProcessorError, LocalLLMProcessorError) as exc:
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 400
    except Exception as e:  # pragma: no cover - general failure
        logger.error(f"Error processing Gemini section: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Failed to process section with Gemini"
        }), 500


@app.route('/api/generate', methods=['POST'])
def generate_audio():
    """Add audio generation job to queue"""
    try:
        # Ensure worker thread is running
        start_worker_thread()
        data = request.json or {}
        text = (data.get('text') or '').strip()
        voice_assignments = data.get('voice_assignments', {})
        logger.info("Received voice_assignments: %s", voice_assignments)
        split_by_chapter = bool(data.get('split_by_chapter', False))
        generate_full_story = bool(data.get('generate_full_story', False)) and split_by_chapter
        custom_heading = data.get('custom_heading')
        requested_format = (data.get('output_format') or '').strip().lower()
        requested_bitrate = data.get('output_bitrate_kbps')
        requested_engine = (data.get('tts_engine') or '').strip().lower()
        engine_options = data.get('engine_options') if isinstance(data.get('engine_options'), dict) else None
        review_mode = bool(data.get('review_mode', False))
        raw_replacements = data.get('word_replacements')
        word_replacements = [
            r for r in (raw_replacements or [])
            if isinstance(r, dict) and r.get('original') and r.get('replacement')
        ] if isinstance(raw_replacements, list) else []

        if not text:
            return jsonify({
                "success": False,
                "error": "No text provided"
            }), 400

        allowed_formats = {"mp3", "wav", "ogg"}
        if requested_format and requested_format not in allowed_formats:
            return jsonify({
                "success": False,
                "error": f"Unsupported output format: {requested_format}"
            }), 400

        if requested_bitrate is not None:
            try:
                requested_bitrate = int(requested_bitrate)
            except (TypeError, ValueError):
                return jsonify({
                    "success": False,
                    "error": "Output bitrate must be an integer"
                }), 400
            if requested_bitrate < 32 or requested_bitrate > 512:
                return jsonify({
                    "success": False,
                    "error": "Output bitrate must be between 32 and 512 kbps"
                }), 400

        # Load config
        config = load_config()
        if requested_engine:
            normalized_engine = _normalize_engine_name(requested_engine)
            if normalized_engine not in AVAILABLE_ENGINES:
                return jsonify({
                    "success": False,
                    "error": f"Unsupported TTS engine: {requested_engine}"
                }), 400
            config['tts_engine'] = normalized_engine

        active_engine = _normalize_engine_name(config.get('tts_engine'))
        logger.info("Generation request using engine='%s' (requested='%s')", active_engine, requested_engine or "(config)")
        logger.warning("TTS generation engine selected: %s", active_engine)
        _apply_engine_option_overrides(config, active_engine, engine_options)
        if requested_format:
            config['output_format'] = requested_format
        if requested_bitrate:
            config['output_bitrate_kbps'] = requested_bitrate

        voice_assignments = _prepare_voice_assignments(text, voice_assignments)

        _validate_voice_assignments_for_engine(
            active_engine,
            text,
            voice_assignments,
            config,
        )
        
        # Create job
        job_id = str(uuid.uuid4())
        text_path = _write_job_text(job_id, text)
        text_length = len(text)
        speakers = _extract_speakers_for_text(text)
        estimated_chunks = estimate_total_chunks(
            text,
            split_by_chapter,
            int(config.get('chunk_size', 500)),
            include_full_story=generate_full_story,
            engine_name=active_engine,
            config=config,
            custom_heading=custom_heading,
        )
        
        merge_options = {
            "output_format": config.get('output_format'),
            "crossfade_duration": float(config.get('crossfade_duration') or 0),
            "intro_silence_ms": int(config.get('intro_silence_ms', 0) or 0),
            "inter_chunk_silence_ms": int(config.get('inter_chunk_silence_ms', 0) or 0),
            "output_bitrate_kbps": int(config.get('output_bitrate_kbps') or 0),
        }

        job_dir_path = OUTPUT_DIR / job_id
        job_dir_path.mkdir(parents=True, exist_ok=True)
        job_dir = job_dir_path.as_posix()

        with queue_lock:
            jobs[job_id] = {
                "status": "queued",
                "text_preview": text[:200],
                "text_path": text_path,
                "text_length": text_length,
                "created_at": datetime.now().isoformat(),
                "review_mode": review_mode,
                "chapter_mode": split_by_chapter,
                "full_story_requested": generate_full_story,
                "job_dir": job_dir,
                "merge_options": merge_options,
                "chunks": [],
                "voice_assignments": voice_assignments,
                "config_snapshot": copy.deepcopy(config),
                "custom_heading": custom_heading,
                "speakers": speakers,
                "regen_tasks": {},
                "engine": config.get("tts_engine"),
                "resume_from_chunk_index": 0,
                "word_replacements": word_replacements,
            }
            jobs[job_id]["job_payload"] = _build_job_payload(job_id, text, jobs[job_id])
        
        # Create job data
        job_data = {
            "job_id": job_id,
            "text": text,
            "voice_assignments": voice_assignments,
            "config": config,
            "split_by_chapter": split_by_chapter,
            "generate_full_story": generate_full_story,
            "total_chunks": estimated_chunks,
            "review_mode": review_mode,
            "merge_options": merge_options,
            "job_dir": job_dir,
            "custom_heading": custom_heading,
            "resume_from_chunk_index": 0,
            "word_replacements": word_replacements,
        }

        _persist_job_state(job_id, force=True)

        # Write metadata.json so word_replacements (and other fields) survive server restarts
        _meta_path = job_dir_path / JOB_METADATA_FILENAME
        try:
            _meta_path.parent.mkdir(parents=True, exist_ok=True)
            _existing_meta = {}
            if _meta_path.exists():
                try:
                    with _meta_path.open("r", encoding="utf-8") as _f:
                        _existing_meta = json.load(_f)
                except Exception:
                    pass
            _existing_meta["word_replacements"] = word_replacements
            _existing_meta.setdefault("job_id", job_id)
            _existing_meta.setdefault("created_at", jobs[job_id]["created_at"])
            with _meta_path.open("w", encoding="utf-8") as _f:
                json.dump(_existing_meta, _f, indent=2, ensure_ascii=False)
        except Exception as _exc:
            logger.warning("Failed to write metadata.json at job creation for %s: %s", job_id, _exc)

        _archive_old_jobs()

        # Add to queue
        job_queue.put(job_data)
        logger.info(f"Job {job_id} added to queue. Queue size: {job_queue.qsize()}")
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "status": "queued",
            "queue_position": job_queue.qsize()
        })
        
    except ValueError as e:
        logger.error(f"Error queueing job: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error queueing job: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/download/<job_id>', methods=['GET'])
def download_audio(job_id):
    """Download generated audio"""
    try:
        logger.info(f"Download request for job {job_id}")

        # Get output format from config
        config = load_config()
        output_format = config.get('output_format', 'mp3')
        requested_file = request.args.get('file') if request else None
        requested_name = request.args.get('download_name') if request else None

        # Try to find the file - check both mp3 and wav
        file_path = None
        job_dir = OUTPUT_DIR / job_id

        if requested_file:
            safe_relative = Path(requested_file)
            if safe_relative.is_absolute() or ".." in safe_relative.parts:
                return jsonify({
                    "success": False,
                    "error": "Invalid file path"
                }), 400
            candidate_path = job_dir / safe_relative
            if candidate_path.exists():
                file_path = candidate_path
                output_format = candidate_path.suffix.lstrip('.')

        if file_path is None:
            metadata = load_job_metadata(job_dir)
            full_story = (metadata or {}).get("full_story") if metadata else None
            if full_story:
                rel_path = full_story.get("relative_path") or full_story.get("output_file")
                if rel_path:
                    candidate = job_dir / Path(rel_path)
                    if candidate.exists():
                        file_path = candidate
                        output_format = candidate.suffix.lstrip('.')

        if file_path is None:
            for ext in [output_format, 'mp3', 'wav', 'ogg']:
                test_path = job_dir / f"output.{ext}"
                if test_path.exists():
                    file_path = test_path
                    output_format = ext
                    break

        if not file_path or not file_path.exists():
            logger.error(f"File not found for job {job_id} in {job_dir}")
            return jsonify({
                "success": False,
                "error": f"Audio file not found for job {job_id}"
            }), 404

        download_name = f"kokoro_story_{job_id}.{output_format}"
        if requested_name:
            safe_name = Path(requested_name).name
            if safe_name:
                download_name = safe_name

        logger.info(f"Sending file: {file_path}")
        return send_file(
            file_path,
            as_attachment=True,
            download_name=download_name
        )
        
    except Exception as e:
        logger.error(f"Error downloading file for job {job_id}: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/download/<job_id>/zip', methods=['GET'])
def download_audio_bundle(job_id):
    """Download all chapter outputs for a job as a ZIP archive."""
    try:
        job_dir = OUTPUT_DIR / job_id
        if not job_dir.exists():
            return jsonify({
                "success": False,
                "error": "Job directory not found"
            }), 404

        metadata = load_job_metadata(job_dir)
        chapters = (metadata or {}).get("chapters")
        full_story = (metadata or {}).get("full_story")
        if not chapters and not full_story:
            # Fallback to single-file download
            return download_audio(job_id)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for chapter in (chapters or []):
                rel_path = chapter.get("relative_path")
                if not rel_path:
                    continue
                file_path = job_dir / Path(rel_path)
                if not file_path.exists():
                    continue
                arc_name = Path(rel_path).as_posix()
                zip_file.write(file_path, arcname=arc_name)

            if full_story:
                rel_path = full_story.get("relative_path")
                if rel_path:
                    file_path = job_dir / Path(rel_path)
                    if file_path.exists():
                        arc_name = Path(rel_path).as_posix()
                        zip_file.write(file_path, arcname=arc_name)

        zip_buffer.seek(0)
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"kokoro_story_{job_id}.zip"
        )

    except Exception as e:
        logger.error(f"Error generating ZIP for job {job_id}: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


def _build_library_listing():
    """Scan disk and return library metadata list."""
    library_items = []

    if not OUTPUT_DIR.exists():
        return library_items

    for job_dir in OUTPUT_DIR.iterdir():
        if not job_dir.is_dir():
            continue

        job_id = job_dir.name
        metadata = load_job_metadata(job_dir)

        if metadata and (metadata.get("chapters") or metadata.get("books") or metadata.get("full_story")):
            chapters_data = []
            total_size = 0
            created_ts = None
            full_story_entry = None
            outputs = metadata.get("books") if metadata.get("book_mode") else metadata.get("chapters")
            for chapter in outputs or []:
                rel_path = chapter.get("relative_path")
                if not rel_path:
                    continue
                file_path = job_dir / Path(rel_path)
                file_exists = file_path.exists()

                if file_exists:
                    stat = file_path.stat()
                    created_time = datetime.fromtimestamp(stat.st_ctime)
                    created_ts = created_ts or created_time
                    total_size += stat.st_size
                    file_size = stat.st_size
                    file_format = file_path.suffix.lstrip('.')
                else:
                    file_size = None
                    file_format = metadata.get("output_format")

                chapters_data.append({
                    "index": chapter.get("index"),
                    "title": chapter.get("title"),
                    "output_file": f"/static/audio/{job_id}/{Path(rel_path).as_posix()}",
                    "relative_path": Path(rel_path).as_posix(),
                    "file_size": file_size,
                    "format": file_format,
                    "missing_file": not file_exists,
                })

            manifest_path = job_dir / "review_manifest.json"
            if manifest_path.exists():
                try:
                    with manifest_path.open("r", encoding="utf-8") as handle:
                        manifest = json.load(handle)
                    manifest_chapters = manifest.get("chapters") or []
                except Exception:
                    manifest_chapters = []

                if manifest_chapters:
                    existing_paths = {chapter.get("relative_path") for chapter in chapters_data if chapter.get("relative_path")}
                    for chapter in manifest_chapters:
                        output_filename = chapter.get("output_filename")
                        if not output_filename:
                            continue
                        chapter_dir = chapter.get("chapter_dir") or "."
                        rel_path = (Path(chapter_dir) / output_filename).as_posix()
                        if rel_path.startswith("./"):
                            rel_path = rel_path[2:]
                        if rel_path in existing_paths:
                            continue
                        file_path = job_dir / Path(rel_path)
                        file_exists = file_path.exists()

                        if file_exists:
                            stat = file_path.stat()
                            created_time = datetime.fromtimestamp(stat.st_ctime)
                            created_ts = created_ts or created_time
                            total_size += stat.st_size
                            file_size = stat.st_size
                            file_format = file_path.suffix.lstrip(".")
                        else:
                            file_size = None
                            file_format = metadata.get("output_format")

                        chapters_data.append({
                            "index": chapter.get("index"),
                            "title": chapter.get("title"),
                            "output_file": f"/static/audio/{job_id}/{rel_path}",
                            "relative_path": rel_path,
                            "file_size": file_size,
                            "format": file_format,
                            "missing_file": not file_exists,
                        })
                        existing_paths.add(rel_path)

            full_meta = metadata.get("full_story")
            if full_meta and full_meta.get("relative_path"):
                full_path = job_dir / Path(full_meta["relative_path"])
                if full_path.exists():
                    stat = full_path.stat()
                    total_size += stat.st_size
                    full_story_entry = {
                        "title": full_meta.get("title", "Full Story"),
                        "output_file": f"/static/audio/{job_id}/{full_meta['relative_path']}",
                        "relative_path": full_meta['relative_path'],
                        "file_size": stat.st_size,
                        "format": full_path.suffix.lstrip('.')
                    }

            # Read engine and timing_metrics from chunks_metadata.json once for both branches
            _lib_chunks_meta_path = job_dir / "chunks_metadata.json"
            _lib_manifest_path = job_dir / "review_manifest.json"
            _lib_has_chunks = _lib_chunks_meta_path.exists() or _lib_manifest_path.exists()
            _lib_engine = None
            _lib_timing_metrics = None
            if _lib_chunks_meta_path.exists():
                try:
                    with _lib_chunks_meta_path.open("r", encoding="utf-8") as f:
                        _lib_cmeta = json.load(f)
                        _lib_engine = _lib_cmeta.get("engine")
                        _lib_timing_metrics = _lib_cmeta.get("timing_metrics")
                except Exception:
                    pass
            # Also check live job entry for timing_metrics (may be more up-to-date)
            _live_tm = (jobs.get(job_id) or {}).get("timing_metrics")
            if _live_tm:
                _lib_timing_metrics = _live_tm

            if chapters_data:
                chapters_data.sort(key=lambda c: c.get("index") or 0)
                _word_replacements = (jobs.get(job_id) or {}).get("word_replacements") or metadata.get("word_replacements") or []
                library_items.append({
                    "job_id": job_id,
                    "output_file": chapters_data[0]["output_file"],
                    "relative_path": chapters_data[0]["relative_path"],
                    "created_at": (created_ts or datetime.now()).isoformat(),
                    "file_size": total_size,
                    "format": metadata.get("output_format", chapters_data[0]["format"]),
                    "chapter_mode": metadata.get("chapter_mode", False),
                    "book_mode": metadata.get("book_mode", False),
                    "collection_title": metadata.get("collection_title"),
                    "books": chapters_data if metadata.get("book_mode") else [],
                    "chapters": chapters_data,
                    "full_story": full_story_entry,
                    "has_chunks": _lib_has_chunks,
                    "engine": _lib_engine,
                    "timing_metrics": _lib_timing_metrics,
                    "word_replacements": _word_replacements,
                })
            elif full_story_entry:
                _word_replacements2 = (jobs.get(job_id) or {}).get("word_replacements") or metadata.get("word_replacements") or []
                library_items.append({
                    "job_id": job_id,
                    "output_file": full_story_entry["output_file"],
                    "relative_path": full_story_entry["relative_path"],
                    "created_at": (created_ts or datetime.now()).isoformat(),
                    "file_size": total_size or full_story_entry.get("file_size"),
                    "format": metadata.get("output_format", full_story_entry.get("format")),
                    "chapter_mode": metadata.get("chapter_mode", False),
                    "book_mode": metadata.get("book_mode", False),
                    "collection_title": metadata.get("collection_title"),
                    "books": [],
                    "chapters": [],
                    "full_story": full_story_entry,
                    "has_chunks": _lib_has_chunks,
                    "engine": _lib_engine,
                    "timing_metrics": _lib_timing_metrics,
                    "word_replacements": _word_replacements2,
                })
            continue

        # Fallback: include chapter/book outputs even if metadata.json is missing.
        # This makes the library resilient if metadata writing failed but audio files exist.
        chapter_output_files = sorted(job_dir.glob("chapter_*/*"))
        if not chapter_output_files:
            chapter_output_files = sorted(job_dir.glob("book_*/*/*"))

        if chapter_output_files:
            chapters_data = []
            total_size = 0
            created_ts = None

            for idx, file_path in enumerate(chapter_output_files, start=1):
                if not file_path.is_file():
                    continue
                if file_path.name.lower().startswith("_silence_"):
                    continue
                if file_path.suffix.lower() not in {".mp3", ".wav", ".m4a", ".ogg", ".aac"}:
                    continue

                stat = file_path.stat()
                created_time = datetime.fromtimestamp(stat.st_ctime)
                created_ts = created_ts or created_time
                total_size += stat.st_size
                rel_path = file_path.relative_to(job_dir).as_posix()
                chapters_data.append({
                    "index": idx,
                    "title": f"Chapter {idx}",
                    "output_file": f"/static/audio/{job_id}/{rel_path}",
                    "relative_path": rel_path,
                    "file_size": stat.st_size,
                    "format": file_path.suffix.lstrip('.'),
                })

            if chapters_data:
                chunks_meta_path = job_dir / "chunks_metadata.json"
                manifest_path = job_dir / "review_manifest.json"
                has_chunks = chunks_meta_path.exists() or manifest_path.exists()
                engine = None
                if chunks_meta_path.exists():
                    try:
                        with chunks_meta_path.open("r", encoding="utf-8") as f:
                            chunks_meta = json.load(f)
                            engine = chunks_meta.get("engine")
                    except Exception:
                        pass
                library_items.append({
                    "job_id": job_id,
                    "output_file": chapters_data[0]["output_file"],
                    "relative_path": chapters_data[0]["relative_path"],
                    "created_at": (created_ts or datetime.now()).isoformat(),
                    "file_size": total_size,
                    "format": chapters_data[0]["format"],
                    "chapter_mode": True,
                    "book_mode": False,
                    "books": [],
                    "chapters": chapters_data,
                    "full_story": None,
                    "has_chunks": has_chunks,
                    "engine": engine,
                })
                continue

        output_files = list(job_dir.glob("output.*"))
        if output_files:
            output_file = output_files[0]
            stat = output_file.stat()
            created_time = datetime.fromtimestamp(stat.st_ctime)
            # Get engine from chunks_metadata if available
            engine = None
            chunks_meta_path = job_dir / "chunks_metadata.json"
            if chunks_meta_path.exists():
                try:
                    with chunks_meta_path.open("r", encoding="utf-8") as f:
                        chunks_meta = json.load(f)
                        engine = chunks_meta.get("engine")
                except Exception:
                    pass
            library_items.append({
                "job_id": job_id,
                "output_file": f"/static/audio/{job_id}/{output_file.name}",
                "relative_path": output_file.name,
                "created_at": created_time.isoformat(),
                "file_size": stat.st_size,
                "format": output_file.suffix.lstrip('.'),
                "chapter_mode": False,
                "chapters": [{
                    "index": 1,
                    "title": "Full Story",
                    "output_file": f"/static/audio/{job_id}/{output_file.name}",
                    "relative_path": output_file.name,
                    "file_size": stat.st_size,
                    "format": output_file.suffix.lstrip('.')
                }],
                "engine": engine,
            })

    library_items.sort(key=lambda x: x['created_at'], reverse=True)
    return library_items


@app.route('/api/library', methods=['GET'])
def get_library():
    """Get list of all generated audio files"""
    try:
        with log_request_timing("GET /api/library"):
            now = time.time()
            cached_items = library_cache["items"]
            if cached_items is not None and (now - library_cache["timestamp"]) <= LIBRARY_CACHE_TTL:
                return jsonify({
                    "success": True,
                    "items": cached_items,
                    "cached": True
                })

            library_items = _build_library_listing()
            library_cache["items"] = library_items
            library_cache["timestamp"] = now

            return jsonify({
                "success": True,
                "items": library_items,
                "cached": False
            })
        
    except Exception as e:
        logger.error(f"Error getting library: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/library/<job_id>/restore-review', methods=['POST'])
def restore_library_item_to_review(job_id):
    """Restore a completed library item back to review mode for chunk editing."""
    try:
        job_dir = OUTPUT_DIR / job_id
        if not job_dir.exists():
            return jsonify({"success": False, "error": "Item not found"}), 404

        manifest_path = job_dir / "review_manifest.json"
        chunks_meta_path = job_dir / "chunks_metadata.json"

        if not manifest_path.exists() and not chunks_meta_path.exists():
            return jsonify({"success": False, "error": "No chunk data available for this item"}), 400

        manifest = {}
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)

        chunks_meta = {}
        if chunks_meta_path.exists():
            with chunks_meta_path.open("r", encoding="utf-8") as handle:
                chunks_meta = json.load(handle)

        config_snapshot = load_config()
        engine_name = chunks_meta.get("engine") or config_snapshot.get("tts_engine", "kokoro")
        # Ensure config_snapshot uses the original job's engine, not the current config
        config_snapshot["tts_engine"] = engine_name
        logger.info(f"Restoring job {job_id} with engine: {engine_name}")

        # Prefer chunks from chunks_metadata.json if available (has text/voice data)
        chunks = chunks_meta.get("chunks") or []
        chunks.sort(key=lambda c: c.get("order_index") if c.get("order_index") is not None else float('inf'))

        # If no chunks in metadata, build from manifest
        if not chunks and manifest:
            order_index = 0
            for chapter in manifest.get("chapters", []):
                chapter_index = chapter.get("index", 0)
                chunk_files = chapter.get("chunk_files") or []
                for chunk_idx, rel_path in enumerate(chunk_files):
                    chunk_path = job_dir / rel_path
                    if not chunk_path.exists():
                        continue
                    chunk_id = f"{chapter_index}-{chunk_idx}-{order_index}"
                    chunks.append({
                        "id": chunk_id,
                        "order_index": order_index,
                        "chapter_index": chapter_index,
                        "chunk_index": chunk_idx,
                        "speaker": "default",
                        "text": "",
                        "relative_file": rel_path,
                        "duration_seconds": None,
                    })
                    order_index += 1

        # Verify chunk files exist on disk
        valid_chunks = []
        for chunk in chunks:
            rel_file = chunk.get("relative_file")
            if rel_file and (job_dir / rel_file).exists():
                valid_chunks.append(chunk)

        if not valid_chunks:
            return jsonify({"success": False, "error": "No chunk files found on disk"}), 400

        with queue_lock:
            jobs[job_id] = {
                "job_id": job_id,
                "status": "completed",  # Keep as completed - review happens in library
                "progress": 100,
                "eta_seconds": 0,
                "review_mode": True,
                "review_manifest": manifest_path.name if manifest_path.exists() else None,
                "chapter_mode": manifest.get("chapter_mode", False),
                "full_story_requested": manifest.get("full_story_requested", False),
                "job_dir": str(job_dir),
                "config_snapshot": config_snapshot,
                "chunks": valid_chunks,
                "regen_tasks": {},
                "engine": engine_name,
                "restored_at": datetime.now().isoformat(),
            }

        invalidate_library_cache()
        return jsonify({"success": True, "job_id": job_id, "chunk_count": len(valid_chunks)})

    except Exception as exc:
        logger.error("Failed to restore library item %s to review: %s", job_id, exc, exc_info=True)
        return jsonify({"success": False, "error": "Failed to restore item to review mode"}), 500


@app.route('/api/library/<job_id>/chunks', methods=['GET'])
def get_library_item_chunks(job_id):
    """Get chunk metadata for a library item."""
    try:
        job_dir = OUTPUT_DIR / job_id
        if not job_dir.exists():
            return jsonify({"success": False, "error": "Item not found"}), 404

        chunks_meta_path = job_dir / "chunks_metadata.json"
        if not chunks_meta_path.exists():
            return jsonify({"success": False, "error": "No chunk data available for this item"}), 400

        with chunks_meta_path.open("r", encoding="utf-8") as handle:
            chunks_meta = json.load(handle)

        chunks = chunks_meta.get("chunks") or []
        chunks.sort(key=lambda c: c.get("order_index") if c.get("order_index") is not None else float('inf'))
        for chunk in chunks:
            rel_file = chunk.get("relative_file")
            if rel_file:
                chunk["file_url"] = f"/static/audio/{job_id}/{rel_file}"

        # Load chapter information from review_manifest if available
        chapters = []
        books = []
        manifest_path = job_dir / "review_manifest.json"
        full_story_available = False
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
                chapter_mode = manifest.get("chapter_mode", False)
                full_story_available = bool(manifest.get("all_full_story_chunks"))
                if chapter_mode:
                    for ch in manifest.get("chapters", []):
                        output_filename = ch.get("output_filename")
                        chapter_dir = ch.get("chapter_dir") or "."
                        rel_path = None
                        if output_filename:
                            rel_path = (Path(chapter_dir) / output_filename).as_posix()
                            if rel_path.startswith("./"):
                                rel_path = rel_path[2:]
                        chapters.append({
                            "index": ch.get("index"),
                            "title": ch.get("title"),
                            "output_filename": ch.get("output_filename"),
                            "relative_path": rel_path,
                            "book_index": ch.get("book_index"),
                            "book_title": ch.get("book_title"),
                            "book_order": ch.get("book_order"),
                        })
                if manifest.get("book_mode"):
                    for book in manifest.get("books", []):
                        output_filename = book.get("output_filename")
                        rel_path = None
                        if output_filename:
                            rel_path = (Path(book.get("book_dir") or ".") / output_filename).as_posix()
                            if rel_path.startswith("./"):
                                rel_path = rel_path[2:]
                        books.append({
                            "index": book.get("index"),
                            "title": book.get("title"),
                            "output_filename": book.get("output_filename"),
                            "relative_path": rel_path,
                            "chapter_indices": book.get("chapter_indices") or [],
                        })

        return jsonify({
            "success": True,
            "job_id": job_id,
            "engine": chunks_meta.get("engine", "kokoro"),
            "created_at": chunks_meta.get("created_at"),
            "updated_at": chunks_meta.get("updated_at"),
            "chunks": chunks,
            "chapters": chapters,
            "books": books,
            "has_chapters": len(chapters) > 1,
            "has_books": len(books) > 0,
            "full_story_available": full_story_available,
        })

    except Exception as exc:
        logger.error("Failed to get chunks for library item %s: %s", job_id, exc, exc_info=True)
        return jsonify({"success": False, "error": "Failed to load chunk data"}), 500


def _build_review_merger(config_snapshot: Optional[Dict[str, Any]] = None) -> AudioMerger:
    config_snapshot = config_snapshot or load_config()
    crossfade_seconds = float(config_snapshot.get("crossfade_duration") or 0)
    return AudioMerger(
        crossfade_ms=int(max(0.0, crossfade_seconds) * 1000),
        intro_silence_ms=int(max(0, config_snapshot.get("intro_silence_ms") or 0)),
        inter_chunk_silence_ms=int(max(0, config_snapshot.get("inter_chunk_silence_ms") or 0)),
        bitrate_kbps=int(config_snapshot.get("output_bitrate_kbps") or 0),
    )


def _update_metadata_chapter(job_id: str, job_dir: Path, chapter_entry: Dict[str, Any]) -> None:
    metadata = load_job_metadata(job_dir) or {}
    chapters = metadata.get("chapters") or []
    rel_path = chapter_entry.get("relative_path")
    chapter_index = chapter_entry.get("index")
    updated = False
    for entry in chapters:
        if rel_path and entry.get("relative_path") == rel_path:
            entry.update(chapter_entry)
            updated = True
            break
        if chapter_index is not None and entry.get("index") == chapter_index:
            entry.update(chapter_entry)
            updated = True
            break
    if not updated:
        chapters.append(chapter_entry)
    metadata["chapters"] = chapters
    metadata["chapter_count"] = len(chapters)
    if chapter_entry.get("format") and not metadata.get("output_format"):
        metadata["output_format"] = chapter_entry.get("format")
    save_job_metadata(job_dir, metadata)


def _update_metadata_full_story(job_id: str, job_dir: Path, full_story_entry: Dict[str, Any]) -> None:
    metadata = load_job_metadata(job_dir) or {}
    metadata["full_story"] = full_story_entry
    if full_story_entry.get("format") and not metadata.get("output_format"):
        metadata["output_format"] = full_story_entry.get("format")
    save_job_metadata(job_dir, metadata)


def _remove_existing_output(file_path: Path) -> None:
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as exc:
        logger.warning("Unable to remove existing output %s: %s", file_path, exc)


def _resolve_chapter_output_path(job_dir: Path, target: Dict[str, Any], output_format: str, chapter_index: int) -> Path:
    output_filename = target.get("output_filename") or f"chapter_{chapter_index + 1:02d}.{output_format}"
    chapter_dir = job_dir / (target.get("chapter_dir") or ".")
    chapter_dir.mkdir(parents=True, exist_ok=True)
    return chapter_dir / output_filename


def _sorted_chunk_files(chunk_dir: Path) -> List[Path]:
    def chunk_sort_key(path: Path) -> Tuple[int, str]:
        match = re.search(r"(\d+)", path.stem)
        return (int(match.group(1)) if match else 0, path.name)

    return sorted(chunk_dir.glob("chunk_*.wav"), key=chunk_sort_key)


def _scan_chunk_folders(job_dir: Path) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    title_chunk_dir = job_dir / "title" / "chunks"
    if title_chunk_dir.exists():
        title_chunks = _sorted_chunk_files(title_chunk_dir)
        if title_chunks:
            sources.append({
                "chapter_number": 0,
                "chapter_dir": "title",
                "chunk_dir": "title/chunks",
                "chunk_files": [path for path in title_chunks],
                "is_title": True,
            })

    chapter_dirs = []
    for chapter_dir in job_dir.glob("chapter_*"):
        match = re.match(r"chapter_(\d+)", chapter_dir.name)
        if not match:
            continue
        chapter_dirs.append((int(match.group(1)), chapter_dir))

    for chapter_number, chapter_dir in sorted(chapter_dirs, key=lambda item: item[0]):
        chunk_dir = chapter_dir / "chunks"
        if not chunk_dir.exists():
            continue
        chunks = _sorted_chunk_files(chunk_dir)
        if not chunks:
            continue
        sources.append({
            "chapter_number": chapter_number,
            "chapter_dir": chapter_dir.name,
            "chunk_dir": f"{chapter_dir.name}/chunks",
            "chunk_files": [path for path in chunks],
            "is_title": False,
        })

    return sources


def _rebuild_review_manifest_from_chunks(job_id: str, job_dir: Path, force_rebuild: bool = False) -> Dict[str, Any]:
    chunks_meta_path = job_dir / "chunks_metadata.json"
    if not chunks_meta_path.exists():
        raise FileNotFoundError("chunks_metadata.json not found")

    with chunks_meta_path.open("r", encoding="utf-8") as handle:
        chunks_meta = json.load(handle)

    chunks = [dict(item) for item in (chunks_meta.get("chunks") or []) if isinstance(item, dict)]
    chunk_sources = _scan_chunk_folders(job_dir)
    has_title_section = any(source.get("is_title") for source in chunk_sources)
    if not chunks and not chunk_sources:
        raise ValueError("No chunks found for this job")

    existing_manifest = None
    manifest_path = job_dir / "review_manifest.json"
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                existing_manifest = json.load(handle)
        except Exception:
            existing_manifest = None

    title_by_index = {}
    if existing_manifest:
        for entry in existing_manifest.get("chapters") or []:
            if entry.get("index") is not None and entry.get("title"):
                title_by_index[int(entry["index"])] = entry["title"]

    output_format = None
    metadata = load_job_metadata(job_dir) or {}
    if metadata.get("output_format"):
        output_format = metadata.get("output_format")
    if not output_format:
        config = load_config()
        output_format = config.get("output_format") or "mp3"

    chapter_map: Dict[int, Dict[str, Any]] = {}
    all_full_story_chunks: List[str] = []

    if chunk_sources:
        # Build lookup from original chunks_metadata keyed by normalised relative_file
        # so speaker/text/voice data is preserved during force_rebuild.
        original_by_rel: Dict[str, Dict[str, Any]] = {}
        for orig in chunks:
            rel = (orig.get("relative_file") or "").replace("\\", "/")
            if rel:
                original_by_rel[rel] = orig

        rebuilt_chunks = []
        order_index = 0
        for source in chunk_sources:
            chapter_number = int(source["chapter_number"])
            chapter_index = 0 if source.get("is_title") else (chapter_number if has_title_section else chapter_number - 1)
            chunk_files = []
            for chunk_path in source["chunk_files"]:
                rel_file = chunk_path.relative_to(job_dir).as_posix()
                chunk_files.append(rel_file)
                chunk_match = re.search(r"(\d+)", chunk_path.stem)
                chunk_index = int(chunk_match.group(1)) if chunk_match else 0
                orig = original_by_rel.get(rel_file) or {}
                record = {
                    "id": f"{chapter_index}-{chunk_index}-{order_index}",
                    "order_index": order_index,
                    "chapter_index": chapter_index,
                    "chunk_index": chunk_index,
                    "relative_file": rel_file,
                }
                # Carry over text/speaker/voice data from original if present
                for field in ("speaker", "text", "engine", "emotion", "voice_assignment", "voice_label",
                              "duration_seconds", "regenerated_at", "regen_status", "file_path"):
                    if field in orig:
                        record[field] = orig[field]
                rebuilt_chunks.append(record)
                order_index += 1

            all_full_story_chunks.extend(chunk_files)
            chapter_map[chapter_index] = {
                "chunk_files": chunk_files,
                "chunk_dir": source["chunk_dir"],
                "chapter_dir": source["chapter_dir"],
                "chapter_number": chapter_number,
                "is_title": source.get("is_title", False),
            }

        if force_rebuild:
            chunks_meta_path = job_dir / "chunks_metadata.json"
            chunks_meta = {
                "engine": chunks_meta.get("engine"),
                "created_at": chunks_meta.get("created_at", datetime.now().isoformat()),
                "updated_at": datetime.now().isoformat(),
                "chunks": rebuilt_chunks,
            }
            with chunks_meta_path.open("w", encoding="utf-8") as handle:
                json.dump(chunks_meta, handle, indent=2)
            chunks = rebuilt_chunks
    else:
        sorted_chunks = sorted(chunks, key=lambda c: (c.get("chapter_index") or 0, c.get("order_index") or 0))

        for chunk in sorted_chunks:
            chapter_index = chunk.get("chapter_index")
            if chapter_index is None:
                continue
            rel_file = chunk.get("relative_file")
            if not rel_file:
                file_path = chunk.get("file_path")
                if file_path:
                    rel_file = os.path.relpath(file_path, job_dir)
            if not rel_file:
                continue
            rel_file = rel_file.replace("\\", "/")

            all_full_story_chunks.append(rel_file)

            entry = chapter_map.setdefault(int(chapter_index), {
                "chunk_files": [],
                "chunk_dir": None,
                "chapter_dir": None,
            })
            entry["chunk_files"].append(rel_file)

            rel_path = Path(rel_file)
            parts = list(rel_path.parts)
            if "chunks" in parts:
                chunk_idx = parts.index("chunks")
                entry["chunk_dir"] = Path(*parts[:chunk_idx + 1]).as_posix()
                entry["chapter_dir"] = Path(*parts[:chunk_idx]).as_posix() or "."
            else:
                entry["chunk_dir"] = rel_path.parent.as_posix()
                entry["chapter_dir"] = rel_path.parent.parent.as_posix() if rel_path.parent.parent else "."

    if not chapter_map:
        raise ValueError("No chapter data could be derived from chunks")

    merger = _build_review_merger(load_config())
    chapter_entries = []
    chapter_outputs = []

    for chapter_index in sorted(chapter_map.keys()):
        data = chapter_map[chapter_index]
        chunk_files = data.get("chunk_files") or []
        if not chunk_files:
            continue

        chapter_number = data.get("chapter_number")
        is_title_section = bool(data.get("is_title"))
        chapter_dir = data.get("chapter_dir") or "."
        if is_title_section:
            title = title_by_index.get(chapter_index) or "Title"
            output_filename = f"title.{output_format}"
        else:
            display_number = chapter_number if isinstance(chapter_number, int) else (chapter_index + 1)
            title = title_by_index.get(chapter_index) or f"Chapter {display_number}"
            output_filename = f"chapter_{display_number:02d}.{output_format}"

        target = {
            "chapter_dir": chapter_dir,
            "output_filename": output_filename,
        }
        output_path = _resolve_chapter_output_path(job_dir, target, output_format, chapter_index)
        if force_rebuild:
            output_dir = output_path.parent
            if output_dir.exists():
                for item in output_dir.iterdir():
                    if item.is_file():
                        _remove_existing_output(item)
        if force_rebuild or not output_path.exists():
            chunk_paths = [str(job_dir / Path(rel_path)) for rel_path in chunk_files]
            missing = [p for p in chunk_paths if not Path(p).exists()]
            if missing:
                logger.warning("Missing chunk files for chapter %s: %s", chapter_index, missing)
            else:
                merger.merge_wav_files(
                    input_files=chunk_paths,
                    output_path=str(output_path),
                    format=output_format,
                    cleanup_chunks=False,
                )

        rel_output = (Path(chapter_dir) / output_path.name).as_posix()
        if rel_output.startswith("./"):
            rel_output = rel_output[2:]

        chapter_entries.append({
            "index": chapter_index,
            "title": title,
            "chunk_dir": data.get("chunk_dir"),
            "chunk_files": chunk_files,
            "chapter_dir": chapter_dir,
            "output_filename": output_path.name,
        })
        chapter_outputs.append({
            "index": chapter_index,
            "title": title,
            "file_url": f"/static/audio/{job_id}/{rel_output}",
            "relative_path": rel_output,
            "format": output_format,
        })

    full_story_entry = None
    full_story_path = job_dir / f"full_story.{output_format}"
    if all_full_story_chunks:
        if force_rebuild:
            _remove_existing_output(full_story_path)
        if force_rebuild or not full_story_path.exists():
            chunk_paths = [str(job_dir / Path(rel_path)) for rel_path in all_full_story_chunks]
            missing = [p for p in chunk_paths if not Path(p).exists()]
            if not missing:
                merger.merge_wav_files(
                    input_files=chunk_paths,
                    output_path=str(full_story_path),
                    format=output_format,
                    cleanup_chunks=False,
                )
        if full_story_path.exists():
            full_story_entry = {
                "title": "Full Story",
                "file_url": f"/static/audio/{job_id}/{full_story_path.name}",
                "relative_path": full_story_path.name,
                "format": output_format,
            }

    review_manifest = {
        "chapter_mode": True,
        "book_mode": False,
        "chapters": chapter_entries,
        "books": [],
        "full_story_requested": bool(all_full_story_chunks),
        "output_format": output_format,
        "chunk_dirs_to_cleanup": [c.get("chunk_dir") for c in chapter_entries if c.get("chunk_dir")],
        "all_full_story_chunks": all_full_story_chunks,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(review_manifest, handle, indent=2)

    metadata.update({
        "chapter_mode": True,
        "book_mode": False,
        "output_format": output_format,
        "chapters": chapter_outputs,
        "chapter_count": len(chapter_outputs),
        "books": [],
        "book_count": 0,
        "full_story": full_story_entry,
    })
    save_job_metadata(job_dir, metadata)

    return {
        "chapters": chapter_outputs,
        "full_story": full_story_entry,
        "manifest": review_manifest,
    }


@app.route('/api/library/<job_id>/rebuild/chapter', methods=['POST'])
def rebuild_library_chapter(job_id):
    """Rebuild a single chapter output from review chunks."""
    try:
        payload = request.get_json(silent=True) or {}
        if "chapter_index" not in payload:
            return jsonify({"success": False, "error": "chapter_index is required"}), 400
        chapter_index = int(payload.get("chapter_index"))

        job_dir = OUTPUT_DIR / job_id
        if not job_dir.exists():
            return jsonify({"success": False, "error": "Item not found"}), 404

        manifest_path = job_dir / "review_manifest.json"
        if not manifest_path.exists():
            return jsonify({"success": False, "error": "Review manifest not found"}), 404

        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        chapters = manifest.get("chapters") or []
        target = next((ch for ch in chapters if ch.get("index") == chapter_index), None)
        if not target and chapters:
            target = chapters[min(max(chapter_index, 0), len(chapters) - 1)]
        if not target:
            return jsonify({"success": False, "error": "Chapter not found"}), 404

        chunk_files = target.get("chunk_files") or []
        if not chunk_files:
            return jsonify({"success": False, "error": "No chunk files available for this chapter"}), 400

        chunk_paths = [str(job_dir / rel_path) for rel_path in chunk_files]
        missing = [p for p in chunk_paths if not Path(p).exists()]
        if missing:
            return jsonify({"success": False, "error": "Missing chunk files for this chapter"}), 409

        config_snapshot = load_config()
        output_format = manifest.get("output_format") or config_snapshot.get("output_format") or "mp3"
        output_path = _resolve_chapter_output_path(job_dir, target, output_format, chapter_index)
        _remove_existing_output(output_path)

        merger = _build_review_merger(config_snapshot)
        merger.merge_wav_files(
            input_files=chunk_paths,
            output_path=str(output_path),
            format=output_format,
            cleanup_chunks=False,
        )

        rel_path = (Path(target.get("chapter_dir") or ".") / output_path.name).as_posix()
        if rel_path.startswith("./"):
            rel_path = rel_path[2:]
        chapter_entry = {
            "index": target.get("index"),
            "title": target.get("title"),
            "file_url": f"/static/audio/{job_id}/{rel_path}",
            "relative_path": rel_path,
            "format": output_format,
        }
        _update_metadata_chapter(job_id, job_dir, chapter_entry)
        invalidate_library_cache()

        return jsonify({"success": True, "chapter": chapter_entry})
    except Exception as exc:
        logger.error("Failed to rebuild chapter for %s: %s", job_id, exc, exc_info=True)
        return jsonify({"success": False, "error": "Failed to rebuild chapter"}), 500


@app.route('/api/library/<job_id>/rebuild/full-story', methods=['POST'])
def rebuild_library_full_story(job_id):
    """Rebuild full story output from review chunks."""
    try:
        job_dir = OUTPUT_DIR / job_id
        if not job_dir.exists():
            return jsonify({"success": False, "error": "Item not found"}), 404

        manifest_path = job_dir / "review_manifest.json"
        if not manifest_path.exists():
            return jsonify({"success": False, "error": "Review manifest not found"}), 404

        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        chunk_files = manifest.get("all_full_story_chunks") or []
        if not chunk_files:
            return jsonify({"success": False, "error": "No full story chunks available"}), 400

        chunk_paths = [str(job_dir / rel_path) for rel_path in chunk_files]
        missing = [p for p in chunk_paths if not Path(p).exists()]
        if missing:
            return jsonify({"success": False, "error": "Missing chunk files for full story"}), 409

        config_snapshot = load_config()
        output_format = manifest.get("output_format") or config_snapshot.get("output_format") or "mp3"
        full_story_name = f"full_story.{output_format}"
        full_story_path = job_dir / full_story_name
        _remove_existing_output(full_story_path)

        merger = _build_review_merger(config_snapshot)
        merger.merge_wav_files(
            input_files=chunk_paths,
            output_path=str(full_story_path),
            format=output_format,
            cleanup_chunks=False,
        )

        full_story_entry = {
            "title": "Full Story",
            "file_url": f"/static/audio/{job_id}/{full_story_name}",
            "relative_path": full_story_name,
            "format": output_format,
        }
        _update_metadata_full_story(job_id, job_dir, full_story_entry)
        invalidate_library_cache()

        return jsonify({"success": True, "full_story": full_story_entry})
    except Exception as exc:
        logger.error("Failed to rebuild full story for %s: %s", job_id, exc, exc_info=True)
        return jsonify({"success": False, "error": "Failed to rebuild full story"}), 500


@app.route('/api/library/<job_id>/rebuild/selected', methods=['POST'])
def rebuild_library_selected_chapters(job_id):
    """Rebuild selected chapter outputs from review chunks."""
    try:
        payload = request.get_json(silent=True) or {}
        indices = payload.get("chapter_indices") or []
        if not isinstance(indices, list) or not indices:
            return jsonify({"success": False, "error": "chapter_indices is required"}), 400

        job_dir = OUTPUT_DIR / job_id
        if not job_dir.exists():
            return jsonify({"success": False, "error": "Item not found"}), 404

        manifest_path = job_dir / "review_manifest.json"
        if not manifest_path.exists():
            return jsonify({"success": False, "error": "Review manifest not found"}), 404

        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        chapters = manifest.get("chapters") or []
        index_set = {int(idx) for idx in indices}
        targets = [ch for ch in chapters if ch.get("index") in index_set]
        if not targets:
            return jsonify({"success": False, "error": "No matching chapters found"}), 404

        config_snapshot = load_config()
        output_format = manifest.get("output_format") or config_snapshot.get("output_format") or "mp3"
        merger = _build_review_merger(config_snapshot)
        rebuilt = []

        for target in targets:
            chunk_files = target.get("chunk_files") or []
            if not chunk_files:
                continue
            chunk_paths = [str(job_dir / rel_path) for rel_path in chunk_files]
            missing = [p for p in chunk_paths if not Path(p).exists()]
            if missing:
                continue

            chapter_index = int(target.get("index") or 0)
            output_path = _resolve_chapter_output_path(job_dir, target, output_format, chapter_index)
            _remove_existing_output(output_path)
            merger.merge_wav_files(
                input_files=chunk_paths,
                output_path=str(output_path),
                format=output_format,
                cleanup_chunks=False,
            )
            rel_path = (Path(target.get("chapter_dir") or ".") / output_path.name).as_posix()
            if rel_path.startswith("./"):
                rel_path = rel_path[2:]
            chapter_entry = {
                "index": target.get("index"),
                "title": target.get("title"),
                "file_url": f"/static/audio/{job_id}/{rel_path}",
                "relative_path": rel_path,
                "format": output_format,
            }
            _update_metadata_chapter(job_id, job_dir, chapter_entry)
            rebuilt.append(chapter_entry)

        invalidate_library_cache()
        return jsonify({"success": True, "chapters": rebuilt})
    except Exception as exc:
        logger.error("Failed to rebuild selected chapters for %s: %s", job_id, exc, exc_info=True)
        return jsonify({"success": False, "error": "Failed to rebuild selected chapters"}), 500


@app.route('/api/library/<job_id>/rebuild/all', methods=['POST'])
def rebuild_library_all(job_id):
    """Rebuild all chapter outputs and full story from review chunks."""
    try:
        job_dir = OUTPUT_DIR / job_id
        if not job_dir.exists():
            return jsonify({"success": False, "error": "Item not found"}), 404

        manifest_path = job_dir / "review_manifest.json"
        if not manifest_path.exists():
            return jsonify({"success": False, "error": "Review manifest not found"}), 404

        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        chapters = manifest.get("chapters") or []
        all_full_story_chunks = manifest.get("all_full_story_chunks") or []

        config_snapshot = load_config()
        output_format = config_snapshot.get("output_format") or manifest.get("output_format") or "mp3"
        merger = _build_review_merger(config_snapshot)
        rebuilt_chapters = []

        for target in chapters:
            chunk_files = target.get("chunk_files") or []
            if not chunk_files:
                continue
            chunk_paths = [str(job_dir / rel_path) for rel_path in chunk_files]
            missing = [p for p in chunk_paths if not Path(p).exists()]
            if missing:
                continue

            chapter_index = int(target.get("index") or 0)
            output_path = _resolve_chapter_output_path(job_dir, target, output_format, chapter_index)
            _remove_existing_output(output_path)
            merger.merge_wav_files(
                input_files=chunk_paths,
                output_path=str(output_path),
                format=output_format,
                cleanup_chunks=False,
            )
            rel_path = (Path(target.get("chapter_dir") or ".") / output_path.name).as_posix()
            if rel_path.startswith("./"):
                rel_path = rel_path[2:]
            chapter_entry = {
                "index": target.get("index"),
                "title": target.get("title"),
                "file_url": f"/static/audio/{job_id}/{rel_path}",
                "relative_path": rel_path,
                "format": output_format,
            }
            _update_metadata_chapter(job_id, job_dir, chapter_entry)
            rebuilt_chapters.append(chapter_entry)

        full_story_entry = None
        if all_full_story_chunks:
            chunk_paths = [str(job_dir / rel_path) for rel_path in all_full_story_chunks]
            missing = [p for p in chunk_paths if not Path(p).exists()]
            if not missing:
                full_story_name = f"full_story.{output_format}"
                full_story_path = job_dir / full_story_name
                _remove_existing_output(full_story_path)
                merger.merge_wav_files(
                    input_files=chunk_paths,
                    output_path=str(full_story_path),
                    format=output_format,
                    cleanup_chunks=False,
                )
                full_story_entry = {
                    "title": "Full Story",
                    "file_url": f"/static/audio/{job_id}/{full_story_name}",
                    "relative_path": full_story_name,
                    "format": output_format,
                }
                _update_metadata_full_story(job_id, job_dir, full_story_entry)

        invalidate_library_cache()
        return jsonify({
            "success": True,
            "chapters": rebuilt_chapters,
            "full_story": full_story_entry,
        })
    except Exception as exc:
        logger.error("Failed to rebuild all outputs for %s: %s", job_id, exc, exc_info=True)
        return jsonify({"success": False, "error": "Failed to rebuild outputs"}), 500


@app.route('/api/library/<job_id>/word-replacements', methods=['GET', 'PUT'])
def library_job_word_replacements(job_id):
    """Get or update the word replacement registry for a library job."""
    job_dir = OUTPUT_DIR / job_id
    if not job_dir.is_dir():
        return jsonify({"success": False, "error": "Job not found"}), 404
    metadata_path = job_dir / "metadata.json"

    if request.method == 'GET':
        replacements = (jobs.get(job_id) or {}).get("word_replacements") or []
        if not replacements and metadata_path.exists():
            try:
                with metadata_path.open("r", encoding="utf-8") as f:
                    replacements = json.load(f).get("word_replacements") or []
            except Exception:
                pass
        return jsonify({"success": True, "word_replacements": replacements})

    # PUT — update
    data = request.get_json(silent=True) or {}
    raw = data.get("word_replacements")
    if not isinstance(raw, list):
        return jsonify({"success": False, "error": "word_replacements must be a list"}), 400
    replacements = [
        r for r in raw
        if isinstance(r, dict) and r.get("original") and r.get("replacement")
    ]
    # Update in-memory job entry (outside queue_lock to avoid SQLite deadlock)
    job_in_memory = False
    with queue_lock:
        if job_id in jobs:
            jobs[job_id]["word_replacements"] = replacements
            job_in_memory = True
    if job_in_memory:
        _persist_job_state(job_id, force=True)
    # Persist to metadata.json so it survives server restarts
    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["word_replacements"] = replacements
            with metadata_path.open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning("Failed to persist word_replacements to metadata for %s: %s", job_id, exc)
    invalidate_library_cache()
    return jsonify({"success": True, "word_replacements": replacements})


@app.route('/api/library/<job_id>/repair', methods=['POST'])
def repair_library_job(job_id):
    """Rebuild missing review manifest/metadata from existing chunk files."""
    try:
        payload = request.get_json(silent=True) or {}
        force_rebuild = bool(payload.get("force_rebuild"))
        job_dir = OUTPUT_DIR / job_id
        if not job_dir.exists():
            return jsonify({"success": False, "error": "Item not found"}), 404

        result = _rebuild_review_manifest_from_chunks(job_id, job_dir, force_rebuild=force_rebuild)
        invalidate_library_cache()
        return jsonify({
            "success": True,
            "chapters": result.get("chapters"),
            "full_story": result.get("full_story"),
        })
    except FileNotFoundError as exc:
        return jsonify({"success": False, "error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        logger.error("Failed to repair library job %s: %s", job_id, exc, exc_info=True)
        return jsonify({"success": False, "error": "Failed to repair library item"}), 500


@app.route('/api/library/<job_id>', methods=['DELETE'])
def delete_library_item(job_id):
    """Delete a library item"""
    try:
        job_dir = OUTPUT_DIR / job_id
        
        if not job_dir.exists():
            return jsonify({
                "success": False,
                "error": "Item not found"
            }), 404
        
        # Delete directory and all contents (handle Windows read-only files)
        import shutil
        shutil.rmtree(job_dir, onerror=handle_remove_readonly)
        
        # Remove from jobs dict if present
        if job_id in jobs:
            del jobs[job_id]
        invalidate_library_cache()
        
        return jsonify({
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Error deleting library item: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/library/clear', methods=['POST'])
def clear_library():
    """Clear all library items"""
    try:
        if OUTPUT_DIR.exists():
            import shutil
            for job_dir in OUTPUT_DIR.iterdir():
                if job_dir.is_dir():
                    shutil.rmtree(job_dir, onerror=handle_remove_readonly)
        
        # Clear jobs dict
        jobs.clear()
        
        return jsonify({
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Error clearing library: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/cancel/<job_id>', methods=['POST'])
def cancel_job(job_id):
    """Cancel a job"""
    try:
        with queue_lock:
            if job_id not in jobs:
                return jsonify({
                    "success": False,
                    "error": "Job not found"
                }), 404
            
            # Set cancellation flag
            cancel_flags[job_id] = True
            cancel_events.setdefault(job_id, threading.Event()).set()
            
            # Update job status
            jobs[job_id]["status"] = "cancelled"
            jobs[job_id]["progress"] = 0
            jobs[job_id]["cancelled_at"] = datetime.now().isoformat()
        _persist_job_state(job_id, force=True)
        
        logger.info(f"Job {job_id} marked for cancellation")
        
        return jsonify({
            "success": True,
            "message": "Job cancelled"
        })
        
    except Exception as e:
        logger.error(f"Error cancelling job: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/queue', methods=['GET'])
def get_queue():
    """Get current job queue and all jobs"""
    try:
        with log_request_timing("GET /api/queue"):
            with queue_lock:
                all_jobs = []
                for job_id, job_info in jobs.items():
                    all_jobs.append({
                        "job_id": job_id,
                        "status": job_info.get("status", "unknown"),
                        "progress": job_info.get("progress", 0),
                        "created_at": job_info.get("created_at", ""),
                        "text_preview": job_info.get("text_preview", ""),
                        "output_file": job_info.get("output_file", ""),
                        "error": job_info.get("error", ""),
                        "total_chunks": job_info.get("total_chunks"),
                        "processed_chunks": job_info.get("processed_chunks", 0),
                        "eta_seconds": job_info.get("eta_seconds"),
                        "chapter_mode": job_info.get("chapter_mode", False),
                        "chapter_count": job_info.get("chapter_count"),
                        "book_mode": job_info.get("book_mode", False),
                        "book_count": job_info.get("book_count"),
                        "resume_from_chunk_index": job_info.get("resume_from_chunk_index"),
                        "last_completed_chunk_index": job_info.get("last_completed_chunk_index"),
                        "interrupted_at": job_info.get("interrupted_at"),
                        "full_story_requested": job_info.get("full_story_requested", False),
                        "review_mode": job_info.get("review_mode", False),
                        "review_has_active_regen": job_info.get("review_mode", False) and _has_active_regen_tasks(job_info),
                        "post_process_total": job_info.get("post_process_total", 0),
                        "post_process_done": job_info.get("post_process_done", 0),
                        "post_process_active": job_info.get("post_process_active", False),
                        "post_process_percent": job_info.get("post_process_percent", 0),
                    })
                
                all_jobs.sort(key=lambda x: x['created_at'], reverse=True)
            
            return jsonify({
                "success": True,
                "jobs": all_jobs,
                "current_job": current_job_id,
                "queue_size": job_queue.qsize()
            })
        
    except Exception as e:
        logger.error(f"Error getting queue: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/extract-document', methods=['POST'])
def extract_document_text():
    """Extract text content from an uploaded document file."""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400
    
    file = request.files['file']
    if not file.filename:
        return jsonify({"success": False, "error": "No file selected"}), 400
    
    filename = secure_filename(file.filename)
    
    try:
        file_content = file.read()
        text, format_name = extract_text_from_file(filename, file_content)
        
        if not text.strip():
            return jsonify({
                "success": False,
                "error": f"No text content found in {format_name} file"
            }), 400
        
        return jsonify({
            "success": True,
            "text": text,
            "format": format_name,
            "filename": filename,
            "char_count": len(text),
            "word_count": len(text.split()),
        })
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except ImportError as e:
        return jsonify({
            "success": False,
            "error": f"Missing library: {str(e)}. Please install required dependencies."
        }), 500
    except Exception as e:
        logger.error("Document extraction failed: %s", e, exc_info=True)
        return jsonify({"success": False, "error": f"Failed to extract text: {str(e)}"}), 500


@app.route('/api/supported-formats', methods=['GET'])
def list_supported_formats():
    """Return list of supported document formats for upload."""
    return jsonify({
        "success": True,
        "formats": get_supported_formats()
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    config = load_config()
    
    # Get VRAM info if available
    vram_info = {}
    try:
        import torch
        if torch.cuda.is_available():
            vram_info = {
                "allocated_mb": round(torch.cuda.memory_allocated(0) / 1024**2, 1),
                "reserved_mb": round(torch.cuda.memory_reserved(0) / 1024**2, 1),
                "max_allocated_mb": round(torch.cuda.max_memory_allocated(0) / 1024**2, 1),
            }
    except Exception:
        pass
    
    return jsonify({
        "success": True,
        "tts_engine": config.get('tts_engine', 'kokoro'),
        "kokoro_available": KOKORO_AVAILABLE,
        "qwen3_available": QWEN3_AVAILABLE,
        "pocket_tts_available": POCKET_TTS_AVAILABLE,
        "kitten_tts_available": KITTEN_TTS_AVAILABLE,
        "index_tts_available": INDEX_TTS_AVAILABLE,
        "index_tts_unavailable_reason": INDEX_TTS_UNAVAILABLE_REASON if not INDEX_TTS_AVAILABLE else "",
        "cuda_available": False if not KOKORO_AVAILABLE else __import__('torch').cuda.is_available(),
        "vram": vram_info,
        "loaded_engines": list(tts_engine_instances.keys()),
    })


@app.route('/api/qwen3/metadata', methods=['GET'])
def qwen3_metadata():
    """Return supported speakers and languages for Qwen3 CustomVoice.
    
    Returns static metadata without loading the model to avoid consuming GPU memory at startup.
    """
    if not QWEN3_AVAILABLE:
        return jsonify({
            "success": False,
            "error": "qwen-tts is not installed. Run setup to enable Qwen3-TTS local mode."
        }), 400
    # Return static metadata - these are the actual Qwen3-TTS-CustomVoice supported speakers/languages
    # Avoids loading the full model (~3-4GB GPU) just to get this list
    return jsonify({
        "success": True,
        "speakers": [
            "aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"
        ],
        "languages": [
            "auto", "english", "chinese", "japanese", "korean", "french", "german",
            "spanish", "italian", "portuguese", "russian"
        ],
    })


@app.route('/api/qwen3/voice-design/preview', methods=['POST'])
def qwen3_voice_design_preview():
    if not QWEN3_AVAILABLE:
        return jsonify({
            "success": False,
            "error": "qwen-tts is not installed. Run setup to enable Qwen3-TTS local mode."
        }), 400
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()

    if not text:
        return jsonify({"success": False, "error": "Text is required to generate a preview."}), 400
    job_id = _enqueue_qwen3_voice_design_task("qwen3_voice_design_preview", payload)
    return jsonify({"success": True, "job_id": job_id}), 202


@app.route('/api/qwen3/voice-design/save', methods=['POST'])
def qwen3_voice_design_save():
    if not QWEN3_AVAILABLE:
        return jsonify({
            "success": False,
            "error": "qwen-tts is not installed. Run setup to enable Qwen3-TTS local mode."
        }), 400
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    text = (payload.get("text") or "").strip()
    audio_base64 = payload.get("audio_base64")

    if not name:
        return jsonify({"success": False, "error": "Voice name is required."}), 400
    if not text:
        return jsonify({"success": False, "error": "Sample text is required."}), 400
    if not audio_base64:
        return jsonify({"success": False, "error": "Preview audio is required."}), 400

    try:
        base64.b64decode(audio_base64)
    except Exception:
        return jsonify({"success": False, "error": "Invalid audio payload."}), 400

    job_id = _enqueue_qwen3_voice_design_task("qwen3_voice_design_save", payload)
    return jsonify({"success": True, "job_id": job_id}), 202


@app.route('/api/qwen3/voice-design/tasks/<task_id>', methods=['GET'])
def qwen3_voice_design_task_status(task_id: str):
    with queue_lock:
        job_entry = jobs.get(task_id)
        if not job_entry:
            return jsonify({"success": False, "error": "Task not found."}), 404
        if job_entry.get("job_type") not in {"qwen3_voice_design_preview", "qwen3_voice_design_save"}:
            return jsonify({"success": False, "error": "Task type mismatch."}), 400
        payload = {
            "success": True,
            "status": job_entry.get("status"),
            "progress": job_entry.get("progress"),
        }
        if job_entry.get("status") == "completed":
            payload["result"] = job_entry.get("result")
        if job_entry.get("status") == "failed":
            payload["error"] = job_entry.get("error")
        return jsonify(payload)


@app.route('/api/cleanup-vram', methods=['POST'])
def cleanup_vram():
    """Manually trigger VRAM cleanup for all loaded engines."""
    import gc
    
    cleaned_engines = []
    errors = []
    
    with tts_engine_lock:
        for engine_name, engine in list(tts_engine_instances.items()):
            try:
                engine.cleanup()
                cleaned_engines.append(engine_name)
            except Exception as e:
                errors.append(f"{engine_name}: {str(e)}")
        
        # Clear the engine cache to force reload on next use
        tts_engine_instances.clear()
        engine_config_signatures.clear()
    
    # Additional garbage collection
    gc.collect()
    
    # Get VRAM info after cleanup
    vram_after = {}
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            vram_after = {
                "allocated_mb": round(torch.cuda.memory_allocated(0) / 1024**2, 1),
                "reserved_mb": round(torch.cuda.memory_reserved(0) / 1024**2, 1),
            }
    except Exception:
        pass
    
    return jsonify({
        "success": len(errors) == 0,
        "cleaned_engines": cleaned_engines,
        "errors": errors,
        "vram_after": vram_after,
    })


def _cleanup_orphaned_regen_folders():
    """Clean up leftover chunk_regen_* temp folders from previous sessions."""
    cleaned = 0
    
    # Clean up old-style folders in job directories (legacy)
    try:
        for job_dir in OUTPUT_DIR.iterdir():
            if not job_dir.is_dir():
                continue
            for item in job_dir.iterdir():
                if item.is_dir() and item.name.startswith("chunk_regen_"):
                    try:
                        shutil.rmtree(item)
                        cleaned += 1
                    except Exception as e:
                        logger.warning("Failed to clean up orphaned temp dir %s: %s", item, e)
    except Exception as e:
        logger.warning("Error scanning for orphaned regen folders in OUTPUT_DIR: %s", e)
    
    # Clean up new-style folders in system temp directory
    try:
        temp_base = Path(tempfile.gettempdir())
        for item in temp_base.iterdir():
            if item.is_dir() and item.name.startswith("tts_chunk_regen_"):
                try:
                    shutil.rmtree(item)
                    cleaned += 1
                except Exception as e:
                    logger.warning("Failed to clean up orphaned temp dir %s: %s", item, e)
    except Exception as e:
        logger.warning("Error scanning for orphaned regen folders in temp dir: %s", e)
    
    if cleaned:
        logger.info("Cleaned up %d orphaned chunk regen temp folders", cleaned)


def _resolve_server_port() -> int:
    raw_port = (
        os.environ.get("TTS_STORY_PORT")
        or os.environ.get("PORT")
        or "5000"
    ).strip()
    try:
        port = int(raw_port)
    except (TypeError, ValueError):
        logger.warning("Invalid port value '%s'; falling back to 5000", raw_port)
        return 5000

    if not (1 <= port <= 65535):
        logger.warning("Port %s is out of range; falling back to 5000", port)
        return 5000
    return port


if __name__ == '__main__':
    logger.info("Starting TTS-Story server")
    _init_jobs_db()
    _purge_stale_jobs()
    _restore_jobs_from_db()
    _archive_old_jobs()
    _cleanup_orphaned_chatterbox_voices()
    _auto_register_voice_prompt_files()
    _cleanup_orphaned_regen_folders()
    server_port = _resolve_server_port()
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{server_port}")).start()
    app.run(host='0.0.0.0', port=server_port, debug=True, use_reloader=False)

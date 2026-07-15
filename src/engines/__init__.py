"""TTS engine implementations."""

from .base import EngineCapabilities, TtsEngineBase
from .kokoro_engine import KokoroEngine
from .chatterbox_engine import ChatterboxEngine
from .chatterbox_turbo_local_engine import ChatterboxTurboLocalEngine
from .voxcpm_local_engine import VoxCPMLocalEngine
from .qwen3_custom_voice_engine import Qwen3CustomVoiceEngine
from .qwen3_voice_clone_engine import Qwen3VoiceCloneEngine
from .pocket_tts_engine import PocketTTSEngine
from .kitten_tts_engine import KittenTTSEngine
from .azure_speech_engine import AzureSpeechEngine

__all__ = [
    "TtsEngineBase",
    "EngineCapabilities",
    "KokoroEngine",
    "ChatterboxEngine",
    "ChatterboxTurboLocalEngine",
    "VoxCPMLocalEngine",
    "Qwen3CustomVoiceEngine",
    "Qwen3VoiceCloneEngine",
    "PocketTTSEngine",
    "KittenTTSEngine",
    "AzureSpeechEngine",
]

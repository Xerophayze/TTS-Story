"""
Audio Merger - Combines audio chunks into single file
"""
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from pydub import AudioSegment
import soundfile as sf
import numpy as np


def _win_long_path(path) -> str:
    """Return a Windows extended-length path string (\\\\?\\...) for paths that
    may exceed the 260-character MAX_PATH limit.  On non-Windows platforms the
    path is returned unchanged as a plain string."""
    p = str(path)
    if sys.platform != "win32":
        return p
    if p.startswith("\\\\?\\"):
        return p
    abs_path = os.path.abspath(p)
    if len(abs_path) >= 240:
        return "\\\\?\\" + abs_path
    return abs_path


# Configure pydub to find ffmpeg - prioritize local tools folder
def _find_ffmpeg():
    """Find ffmpeg executable, checking local tools folder first."""
    # Priority paths to check BEFORE system PATH
    priority_paths = []

    # Check local tools folder FIRST (contains correct version from GitHub)
    script_dir = Path(__file__).resolve().parent.parent  # TTS-Story root
    local_ffmpeg = script_dir / "tools" / "ffmpeg" / "ffmpeg.exe"
    if local_ffmpeg.exists():
        print(f"[audio_merger] Found ffmpeg in local tools folder: {local_ffmpeg}")
        return str(local_ffmpeg)

    # Check for Pinokio installation by looking for pinokio folder structure
    # Pinokio apps are in: {pinokio_home}/api/{app_name}/
    # Structure: H:\...\pinokio\api\TTS-Story.git\src\audio_merger.py
    #            -> H:\...\pinokio\bin\miniconda\Library\bin\ffmpeg.exe
    if "pinokio" in str(script_dir).lower():
        # Walk up to find pinokio root
        current = script_dir
        while current.parent != current:
            if (current / "bin" / "miniconda" / "Library" / "bin" / "ffmpeg.exe").exists():
                priority_paths.append(current / "bin" / "miniconda" / "Library" / "bin" / "ffmpeg.exe")
                break
            if current.name.lower() == "pinokio":
                priority_paths.append(current / "bin" / "miniconda" / "Library" / "bin" / "ffmpeg.exe")
                break
            current = current.parent

    # Standard conda/venv paths
    priority_paths.extend([
        Path(sys.prefix) / "Library" / "bin" / "ffmpeg.exe",  # conda env on Windows
        Path(sys.prefix) / "bin" / "ffmpeg",  # conda env on Linux/Mac
        Path.home() / "pinokio" / "bin" / "miniconda" / "Library" / "bin" / "ffmpeg.exe",
    ])
    
    for p in priority_paths:
        if p.exists():
            print(f"[audio_merger] Found ffmpeg at: {p}")
            return str(p)
    
    # Fall back to system PATH
    from pydub.utils import which
    system_ffmpeg = which("ffmpeg")
    if system_ffmpeg:
        print(f"[audio_merger] Using system ffmpeg: {system_ffmpeg}")
    return system_ffmpeg

# Set ffmpeg path for pydub at module load
_ffmpeg_path = _find_ffmpeg()
if _ffmpeg_path:
    AudioSegment.converter = _ffmpeg_path
    # Also set ffprobe path
    ffprobe_path = _ffmpeg_path.replace("ffmpeg", "ffprobe")
    if os.path.exists(ffprobe_path):
        AudioSegment.ffprobe = ffprobe_path


class AudioMerger:
    """Merges audio files with crossfade and optional silence controls"""
    
    def __init__(
        self,
        crossfade_ms: int = 100,
        intro_silence_ms: int = 0,
        inter_chunk_silence_ms: int = 0,
        bitrate_kbps: Optional[int] = None,
        acx_compliance: bool = False,
    ):
        """
        Initialize audio merger
        
        Args:
            crossfade_ms: Crossfade duration in milliseconds
            intro_silence_ms: Silence to prepend before the first chunk
            inter_chunk_silence_ms: Silence inserted between sequential chunks
            acx_compliance: When True, enforce ACX audiobook standards:
                MP3 192kbps CBR, 44100Hz, loudnorm to -19dB integrated
                loudness (within ACX -23 to -18dB range), -3dBTP peak limit
        """
        self.crossfade_ms = crossfade_ms
        self.intro_silence_ms = max(0, intro_silence_ms)
        self.inter_chunk_silence_ms = max(0, inter_chunk_silence_ms)
        self.acx_compliance = bool(acx_compliance)
        self.bitrate_kbps = None
        if bitrate_kbps:
            self.bitrate_kbps = max(32, min(int(bitrate_kbps), 512))
        
    def merge_wav_files(
        self,
        input_files: List[str],
        output_path: str,
        format: str = "mp3",
        cleanup_chunks: bool = True,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> str:
        """
        Merge WAV files using pydub
        
        Args:
            input_files: List of input WAV file paths
            output_path: Output file path
            format: Output format ("mp3", "wav", "ogg")
            cleanup_chunks: Whether to delete WAV chunks after merging
            
        Returns:
            Path to merged audio file
        """
        if not input_files:
            raise ValueError("No input files provided")
            
        logging.info(f"Merging {len(input_files)} audio files")
        
        # Verify all input files exist
        for f in input_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Input file not found: {f}")
            file_size = os.path.getsize(f)
            logging.info(f"Input file: {f} ({file_size} bytes)")
        
        expanded_files = list(input_files)
        silence_files: List[str] = []
        output_path = Path(output_path)

        try:
            for stale_path in output_path.parent.glob(f"{output_path.stem}.*_silence*.wav"):
                try:
                    stale_path.unlink()
                except Exception:
                    pass
        except Exception:
            pass

        def _make_silence_wav(duration_ms: int, label: str) -> Optional[str]:
            if duration_ms <= 0:
                return None
            try:
                info = sf.info(str(input_files[0]))
                sample_rate = int(info.samplerate or 24000)
            except Exception:
                sample_rate = 24000
            total_samples = int(sample_rate * (duration_ms / 1000.0))
            if total_samples <= 0:
                return None
            silence = np.zeros(total_samples, dtype=np.float32)
            silence_path = output_path.with_suffix(f".{label}_silence_{duration_ms}ms.wav")
            sf.write(str(silence_path), silence, sample_rate)
            silence_files.append(str(silence_path))
            return str(silence_path)

        intro_silence = _make_silence_wav(self.intro_silence_ms, "intro")
        segment_silence = _make_silence_wav(self.inter_chunk_silence_ms, "segment")

        if intro_silence:
            expanded_files = [intro_silence] + expanded_files

        if segment_silence and len(expanded_files) > 1:
            interleaved: List[str] = []
            for idx, file_path in enumerate(expanded_files):
                interleaved.append(file_path)
                if idx < len(expanded_files) - 1:
                    interleaved.append(segment_silence)
            expanded_files = interleaved

        try:
            if not self._merge_with_ffmpeg(expanded_files, output_path, format, progress_callback):
                raise RuntimeError("ffmpeg merge failed; ensure ffmpeg is installed and accessible")

            output_size = output_path.stat().st_size if output_path.exists() else 0
            logging.info(f"Merged audio saved to {output_path} ({output_size} bytes)")
            if cleanup_chunks:
                logging.info(f"Cleaning up {len(input_files)} WAV chunks")
                for file_path in input_files:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            logging.debug(f"Deleted chunk: {file_path}")
                    except Exception as e:
                        logging.warning(f"Failed to delete chunk {file_path}: {e}")
                logging.info("Cleanup complete")
            return str(output_path)
        finally:
            for silence_path in silence_files:
                try:
                    if os.path.exists(silence_path):
                        os.remove(silence_path)
                except Exception:
                    pass

    def _merge_with_ffmpeg(
        self,
        input_files: List[str],
        output_path: str,
        format: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> bool:
        ffmpeg_path = _find_ffmpeg()
        if not ffmpeg_path:
            logging.warning("ffmpeg not found; falling back to pydub merge")
            return False

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        list_file = output_path.with_suffix(".concat.txt")
        try:
            with list_file.open("w", encoding="utf-8") as handle:
                for file_path in input_files:
                    handle.write(f"file '{Path(file_path).as_posix()}'\n")

            codec_args = []
            filter_args = []
            fmt = format.lower()

            if self.acx_compliance:
                # ACX standards: MP3 192kbps CBR, 44.1kHz, loudnorm targeting
                # -19dB integrated (center of ACX -23 to -18dB window).
                # TP target is set to -3.5dBTP (tighter than the -3dB ACX limit)
                # and a hard alimiter at -3dB follows to guarantee peaks never
                # overshoot — loudnorm alone can exceed its TP target by ~0.1dB.
                # Gaussian noise at ~-70dBFS mixed in to avoid "dead silence"
                # analyzer warnings. The noise floor sits around -70dB — well
                # below ACX's -60dB requirement and inaudible to listeners.
                filter_args = [
                    "-filter_complex",
                    (
                        "[0:a]loudnorm=I=-19:TP=-3.5:LRA=7:print_format=none,"
                        "alimiter=level_in=1:level_out=1:limit=0.708:attack=5:release=50:level=disabled,"
                        "aresample=44100[main];"
                        "anoisesrc=r=44100:color=white:amplitude=0.000316[noise];"
                        "[main][noise]amix=inputs=2:weights=1 0.000316:normalize=0:duration=shortest[out]"
                    ),
                    "-map", "[out]",
                ]
                codec_args = [
                    "-c:a", "libmp3lame",
                    "-b:a", "192k",
                ]
            elif fmt == "mp3":
                codec_args = ["-c:a", "libmp3lame"]
                if self.bitrate_kbps:
                    codec_args += ["-b:a", f"{self.bitrate_kbps}k"]
            elif fmt == "ogg":
                codec_args = ["-c:a", "libvorbis"]
            elif fmt == "wav":
                codec_args = ["-c:a", "pcm_s16le"]
            else:
                codec_args = ["-c:a", "aac"]

            cmd = [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                _win_long_path(list_file),
                *filter_args,
                *codec_args,
                "-y",
                _win_long_path(output_path),
            ]

            logging.info(f"Merging with ffmpeg concat: {output_path}")
            if progress_callback:
                try:
                    progress_callback(0.0)
                except Exception:
                    pass

            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logging.error("ffmpeg concat failed: %s", result.stderr.strip())
                return False
            if progress_callback:
                try:
                    progress_callback(1.0)
                except Exception:
                    pass
            return True
        except Exception as exc:
            logging.error("ffmpeg concat failed with error: %s", exc, exc_info=True)
            return False
        finally:
            try:
                if list_file.exists():
                    list_file.unlink()
            except Exception:
                pass
        
    def merge_numpy_arrays(
        self,
        audio_arrays: List[np.ndarray],
        sample_rate: int = 24000,
        output_path: str = None
    ) -> np.ndarray:
        """
        Merge numpy audio arrays
        
        Args:
            audio_arrays: List of audio arrays
            sample_rate: Sample rate
            output_path: Optional output path
            
        Returns:
            Merged audio array
        """
        if not audio_arrays:
            raise ValueError("No audio arrays provided")
            
        logging.info(f"Merging {len(audio_arrays)} audio arrays")
        
        # Simple concatenation for numpy arrays
        merged = np.concatenate(audio_arrays)
        
        # Save if output path provided
        if output_path:
            sf.write(output_path, merged, sample_rate)
            logging.info(f"Merged audio saved to {output_path}")
            
        return merged
        
    def convert_format(
        self,
        input_path: str,
        output_path: str,
        format: str = "mp3",
        bitrate: str = "192k"
    ):
        """
        Convert audio file format
        
        Args:
            input_path: Input file path
            output_path: Output file path
            format: Output format
            bitrate: Bitrate for compressed formats
        """
        logging.info(f"Converting {input_path} to {format}")
        
        audio = AudioSegment.from_file(input_path)
        audio.export(
            output_path,
            format=format,
            bitrate=bitrate
        )
        
        logging.info(f"Converted audio saved to {output_path}")
        
    def get_duration(self, file_path: str) -> float:
        """
        Get audio duration in seconds
        
        Args:
            file_path: Audio file path
            
        Returns:
            Duration in seconds
        """
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000.0
        
    def normalize_audio(
        self,
        input_path: str,
        output_path: str,
        target_dBFS: float = -20.0
    ):
        """
        Normalize audio levels
        
        Args:
            input_path: Input file path
            output_path: Output file path
            target_dBFS: Target loudness in dBFS
        """
        logging.info(f"Normalizing audio to {target_dBFS} dBFS")
        
        audio = AudioSegment.from_file(input_path)
        
        # Calculate change needed
        change_in_dBFS = target_dBFS - audio.dBFS
        
        # Apply normalization
        normalized = audio.apply_gain(change_in_dBFS)
        
        # Export
        normalized.export(output_path, format="wav")
        logging.info(f"Normalized audio saved to {output_path}")

    def _encode_chapter_to_aac(
        self,
        input_file: str,
        output_file: Path,
        bitrate_kbps: int,
        acx_compliance: bool,
        ffmpeg_path: str,
        duration_seconds: float = 0.0,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Encode a single chapter to AAC format with live FFmpeg progress."""
        cmd = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-progress",
            "pipe:1",
            "-nostats",
            "-threads",
            "0",
            "-i",
            _win_long_path(input_file),
            "-c:a",
            "aac",
            "-b:a",
            f"{bitrate_kbps}k",
        ]

        if acx_compliance:
            # Apply loudnorm, aresample, and alimiter during parallel encoding
            # This reduces the filter complexity in the final merge
            cmd.extend([
                "-af",
                "loudnorm=I=-19:TP=-3.5:LRA=7:print_format=none,alimiter=level_in=1:level_out=1:limit=0.708:attack=5:release=50:level=disabled,aresample=44100"
            ])

        cmd.extend(["-f", "ipod", "-y", _win_long_path(output_file)])
        process = subprocess.Popen(
            [str(item) for item in cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        if process.stdout is not None:
            for raw_line in process.stdout:
                line = raw_line.strip()
                if line.startswith("out_time_ms=") and duration_seconds > 0:
                    try:
                        encoded_seconds = int(line.split("=", 1)[1]) / 1_000_000
                        ratio = min(max(encoded_seconds / duration_seconds, 0.0), 0.99)
                        if progress_callback:
                            progress_callback(ratio)
                    except (TypeError, ValueError):
                        pass

        stderr_output = process.stderr.read() if process.stderr is not None else ""
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"FFmpeg AAC encoding failed for {Path(input_file).name}: "
                f"{stderr_output.strip() or f'exit code {return_code}'}"
            )
        if not output_file.exists() or output_file.stat().st_size <= 0:
            raise RuntimeError(
                f"FFmpeg reported success but did not create AAC output for {Path(input_file).name}"
            )
        if progress_callback:
            progress_callback(1.0)

    def merge_to_m4b(
        self,
        input_files: List[str],
        output_path: str,
        chapter_metadata: List[Dict[str, Any]],
        bitrate_kbps: int = 128,
        acx_compliance: bool = False,
        cover_art_path: Optional[str] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> str:
        """
        Merge audio files into M4B audiobook format with chapter markers.

        Args:
            input_files: List of input audio file paths
            output_path: Output M4B file path
            chapter_metadata: List of chapter dicts with 'title' and 'start_time' (seconds)
            bitrate_kbps: AAC bitrate (64, 96, 128, 192)
            acx_compliance: Apply ACX audiobook loudness standards
            cover_art_path: Optional path to cover art image
            progress_callback: Optional progress callback

        Returns:
            Path to M4B file
        """
        if not input_files:
            raise ValueError("No input files provided")

        logging.info(f"Merging {len(input_files)} files to M4B with {len(chapter_metadata)} chapters")

        # Verify all input files exist
        for f in input_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Input file not found: {f}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ffmpeg_path = _find_ffmpeg()
        if not ffmpeg_path:
            raise RuntimeError("ffmpeg not found; required for M4B export")

        # Parallel pre-encode chapters to AAC
        import concurrent.futures
        import multiprocessing
        import threading
        import time

        # Determine optimal worker count (leave one core free for system)
        max_workers = min(len(input_files), max(1, (multiprocessing.cpu_count() or 4) - 1))
        logging.info(f"Using {max_workers} workers for parallel encoding")

        # Create temp directory for encoded files in the same location as output
        temp_dir = output_path.parent / "temp_aac"
        temp_dir.mkdir(exist_ok=True)

        encoded_files = []
        parallel_success = False
        list_file = None
        metadata_file = None

        try:
            encode_start = time.time()
            input_durations = []
            for input_file in input_files:
                try:
                    from pydub.utils import mediainfo
                    input_durations.append(float(mediainfo(str(input_file)).get("duration", 0)))
                except Exception as exc:
                    logging.warning("Could not measure M4B source duration for %s: %s", input_file, exc)
                    input_durations.append(0.0)

            # Track time-based progress for every encoder. This is especially
            # important when a complete audiobook is stored as one long file:
            # the previous implementation stayed at 0% until that file ended.
            progress_lock = threading.Lock()
            encode_ratios = [0.0] * len(input_files)
            progress_weights = [duration if duration > 0 else 1.0 for duration in input_durations]
            total_weight = sum(progress_weights) or float(len(input_files) or 1)

            def report_encode_progress(index: int, ratio: float) -> None:
                with progress_lock:
                    encode_ratios[index] = max(encode_ratios[index], min(max(ratio, 0.0), 1.0))
                    combined_ratio = sum(
                        item_ratio * weight
                        for item_ratio, weight in zip(encode_ratios, progress_weights)
                    ) / total_weight
                if progress_callback:
                    try:
                        progress_callback(combined_ratio * 0.5)
                    except Exception:
                        pass

            # Encode files in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for i, input_file in enumerate(input_files):
                    output_file = temp_dir / f"chapter_{i:03d}.m4a"
                    future = executor.submit(
                        self._encode_chapter_to_aac,
                        input_file,
                        output_file,
                        bitrate_kbps,
                        acx_compliance,
                        ffmpeg_path,
                        input_durations[i],
                        lambda ratio, index=i: report_encode_progress(index, ratio),
                    )
                    futures[future] = output_file

                # Wait for all encodings to complete with progress tracking
                completed_count = 0
                total_files = len(futures)
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                        encoded_files.append(futures[future])
                        completed_count += 1
                        logging.info(f"Encoded: {futures[future].name} ({completed_count}/{total_files})")
                    except Exception as e:
                        logging.error(f"Failed to encode {futures[future]}: {e}")
                        raise

            encode_time = time.time() - encode_start
            logging.info(f"Parallel encoding completed in {encode_time:.2f} seconds")
            parallel_success = True

            # Sort encoded files to maintain order
            encoded_files.sort(key=lambda x: int(x.stem.split('_')[1]))

            # Create concat list file with encoded AAC files using absolute paths
            list_file = output_path.with_suffix(".concat.txt")
            with list_file.open("w", encoding="utf-8") as handle:
                for file_path in encoded_files:
                    handle.write(f"file '{file_path.absolute().as_posix()}'\n")

            # Verify encoded files exist and log their sizes
            for file_path in encoded_files:
                if file_path.exists():
                    size = file_path.stat().st_size
                    logging.info(f"Encoded file: {file_path.name} - Size: {size} bytes")
                else:
                    logging.error(f"Encoded file missing: {file_path.name}")

            # Log concat list contents
            logging.info(f"Concat list contents:")
            with list_file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    logging.info(f"  {line.strip()}")

            logging.info(f"Concat list created with {len(encoded_files)} encoded files")

        except Exception as e:
            logging.warning(f"Parallel encoding failed, falling back to sequential: {e}")
            parallel_success = False
            # Cleanup failed temp files
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            temp_dir.mkdir(exist_ok=True)

            # Fallback to original concat list with original files
            list_file = output_path.with_suffix(".concat.txt")
            with list_file.open("w", encoding="utf-8") as handle:
                for file_path in input_files:
                    handle.write(f"file '{Path(file_path).absolute().as_posix()}'\n")

            logging.info(f"Concat list created with {len(input_files)} original files")

        # Build ffmpeg command for final merge
        # Use copy codec when parallel encoding succeeds (concat protocol handles this correctly)
        if parallel_success:
            if acx_compliance:
                # loudnorm, aresample, and alimiter already applied during parallel encoding
                # Only add noise floor in final merge (requires re-encoding due to amix)
                codec_args = ["-c:a", "aac", "-b:a", f"{bitrate_kbps}k"]
                filter_args = [
                    "-filter_complex",
                    (
                        "anoisesrc=r=44100:color=white:amplitude=0.000316[noise];"
                        "[0:a][noise]amix=inputs=2:weights=1 0.000316:normalize=0:duration=shortest[out]"
                    ),
                    "-map", "[out]",
                ]
            else:
                codec_args = ["-c:a", "copy"]
                filter_args = ["-map", "0:a"]
        else:
            # Re-encode needed: parallel failed
            codec_args = ["-c:a", "aac", "-b:a", f"{bitrate_kbps}k"]
            filter_args = []

            if acx_compliance:
                # Full ACX processing in sequential mode
                filter_args = [
                    "-filter_complex",
                    (
                        "[0:a]loudnorm=I=-19:TP=-3.5:LRA=7:print_format=none,"
                        "alimiter=level_in=1:level_out=1:limit=0.708:attack=5:release=50:level=disabled,"
                        "aresample=44100[main];"
                        "anoisesrc=r=44100:color=white:amplitude=0.000316[noise];"
                        "[main][noise]amix=inputs=2:weights=1 0.000316:normalize=0:duration=shortest[out]"
                    ),
                    "-map", "[out]",
                ]
            else:
                # Without ACX compliance, map audio directly
                filter_args = ["-map", "0:a"]

        # Calculate cumulative durations for chapter markers.
        # When parallel encoding succeeded, measure from the encoded AAC files
        # (the files actually being concatenated) to avoid cumulative drift from
        # AAC encoder priming frames causing chapter markers to overshoot.
        duration_source_files = encoded_files if (parallel_success and encoded_files) else input_files
        cumulative_time = 0
        chapter_durations = []
        for source_file in duration_source_files:
            try:
                from pydub.utils import mediainfo
                info = mediainfo(str(source_file))
                duration = float(info.get('duration', 0))
                chapter_durations.append(duration)
                cumulative_time += duration
            except Exception as e:
                logging.warning(f"Could not get duration for {source_file}: {e}")
                chapter_durations.append(0)
                cumulative_time += 0

        # Step 1: Create M4B without chapter markers using ffmpeg concat demuxer.
        # We use "-f concat -safe 0 -i list_file" rather than the "concat:a|b|c"
        # protocol to avoid Windows MAX_PATH / command-line length limits.
        # The list_file was already written above with absolute POSIX paths.
        cmd = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "0",  # Auto-detect optimal thread count for multi-core utilization
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            _win_long_path(list_file),
        ]

        # Add cover art if provided
        if cover_art_path and os.path.exists(cover_art_path):
            cmd.extend(["-i", _win_long_path(cover_art_path)])

        # Add filter_args (includes audio mapping via -map or -filter_complex)
        cmd.extend(filter_args)

        # Map cover art from second input if provided
        if cover_art_path and os.path.exists(cover_art_path):
            cmd.extend(["-map", "1:v"])

        # Add cover art codec if provided
        if cover_art_path and os.path.exists(cover_art_path):
            cmd.extend(["-c:v", "mjpeg", "-disposition:v", "attached_pic"])

        cmd.extend([
            *codec_args,
            "-f",
            "mp4",
            "-movflags",
            "+faststart",  # Optimizes for streaming
            "-y",
            _win_long_path(output_path),
        ])

        logging.info(f"Step 1: Creating M4B without chapter markers")
        logging.info(f"FFmpeg command: {' '.join(str(item) for item in cmd)}")
        if progress_callback:
            try:
                progress_callback(0.5)  # Start of merge phase
            except Exception:
                pass

        import subprocess
        # Run FFmpeg with progress tracking
        cmd_with_progress = cmd[:2] + ["-progress", "pipe:1", "-nostats", "-loglevel", "info"] + cmd[2:]
        cmd_with_progress = [str(item) for item in cmd_with_progress]
        process = subprocess.Popen(
            cmd_with_progress,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Parse progress output
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                line = line.strip()
                if line.startswith("out_time_ms="):
                    # Parse time progress (in microseconds)
                    try:
                        time_ms = int(line.split("=")[1])
                        # Estimate progress based on time (assuming ~7 hours total)
                        # This is rough estimation since we don't know exact duration beforehand
                        # 50-75% range for step 1
                        estimated_seconds = time_ms / 1_000_000
                        max_estimated = 7 * 3600  # 7 hours in seconds
                        progress = 0.5 + min((estimated_seconds / max_estimated) * 0.2, 0.2)
                        if progress_callback:
                            try:
                                progress_callback(progress)
                            except Exception:
                                pass
                    except (ValueError, IndexError):
                        pass
        
        stderr_output = process.stderr.read()
        process.wait()
        
        logging.info(f"FFmpeg stderr: {stderr_output}")
        if process.returncode != 0:
            logging.error("ffmpeg M4B merge failed: %s", stderr_output.strip())
            raise RuntimeError(f"ffmpeg M4B merge failed: {stderr_output.strip()}")

        # Step 2: Add chapter markers to the created M4B file
        # Use short fixed names for temp files — ffmpeg does not support the
        # \\?\ extended-length prefix as an -i argument on Windows, so we must
        # keep these paths short enough to never need that prefix.
        metadata_file = output_path.parent / "m4b_chapters.txt"
        temp_output = output_path.parent / "m4b_temp.m4b"
        with metadata_file.open("w", encoding="utf-8") as mf:
            mf.write(";FFMETADATA1\n")
            current_time = 0
            for idx, chapter in enumerate(chapter_metadata):
                title = chapter.get("title", f"Chapter {idx + 1}")
                # Use cumulative time as start time (position in merged file)
                start_time = current_time
                # Calculate end time based on this chapter's duration
                if idx < len(chapter_durations) and chapter_durations[idx] > 0:
                    end_time = current_time + chapter_durations[idx]
                else:
                    # Last chapter - use a large end time (ffmpeg will adjust to actual duration)
                    end_time = start_time + 9999999
                # FFmpeg chapter metadata format
                mf.write("[CHAPTER]\n")
                mf.write(f"TIMEBASE=1/1000\n")
                mf.write(f"START={int(start_time * 1000)}\n")
                mf.write(f"END={int(end_time * 1000)}\n")
                mf.write(f"title={title}\n")
                # Update cumulative time for next chapter
                if idx < len(chapter_durations):
                    current_time += chapter_durations[idx]

        # Add metadata to existing M4B file
        cmd_metadata = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            _win_long_path(output_path),
            "-i",
            str(metadata_file),
            "-map",
            "0",
            "-map_metadata",
            "1",
            "-c",
            "copy",  # Copy streams without re-encoding
            "-y",
            str(temp_output),
        ]

        logging.info(f"Step 2: Adding chapter markers to M4B")
        logging.info(f"FFmpeg command: {' '.join(cmd_metadata)}")
        if progress_callback:
            try:
                progress_callback(0.75)  # Start of metadata phase
            except Exception:
                pass

        # Run FFmpeg with progress tracking for metadata addition
        cmd_metadata_with_progress = cmd_metadata[:2] + ["-progress", "pipe:1", "-nostats", "-loglevel", "info"] + cmd_metadata[2:]
        process_metadata = subprocess.Popen(
            cmd_metadata_with_progress,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Parse progress output for metadata step
        while True:
            line = process_metadata.stdout.readline()
            if not line and process_metadata.poll() is not None:
                break
            if line:
                line = line.strip()
                if line.startswith("out_time_ms="):
                    # Parse time progress (in microseconds)
                    try:
                        time_ms = int(line.split("=")[1])
                        # Estimate progress based on time (assuming ~7 hours total)
                        # 75-95% range for step 2
                        estimated_seconds = time_ms / 1_000_000
                        max_estimated = 7 * 3600  # 7 hours in seconds
                        progress = 0.75 + min((estimated_seconds / max_estimated) * 0.2, 0.2)
                        if progress_callback:
                            try:
                                progress_callback(progress)
                            except Exception:
                                pass
                    except (ValueError, IndexError):
                        pass
        
        stderr_metadata = process_metadata.stderr.read()
        process_metadata.wait()
        
        logging.info(f"FFmpeg stderr: {stderr_metadata}")
        
        if process_metadata.returncode != 0:
            logging.warning("Failed to add chapter metadata: %s", stderr_metadata.strip())
            # Continue without chapter markers if this fails
        else:
            # Replace original file with metadata-enhanced version
            import shutil
            shutil.move(str(temp_output), str(output_path))
            logging.info("Chapter markers added successfully")

        if progress_callback:
            try:
                progress_callback(1.0)
            except Exception:
                pass

        logging.info(f"M4B export complete: {output_path}")

        # Cleanup temp files
        try:
            if list_file and list_file.exists():
                list_file.unlink()
            if metadata_file and metadata_file.exists():
                metadata_file.unlink()
            # Cleanup temp directory with encoded AAC files
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            logging.warning(f"Error cleaning up temp files: {e}")

        return str(output_path)


def apply_chunk_silence(
    input_path: str,
    output_path: str,
    leading_ms: int = 0,
    trailing_ms: int = 0,
):
    """
    Apply leading and trailing silence to an individual audio chunk.

    Args:
        input_path: Input file path
        output_path: Output file path
        leading_ms: Leading silence in milliseconds
        trailing_ms: Trailing silence in milliseconds
    """
    if leading_ms <= 0 and trailing_ms <= 0:
        # No silence needed, just copy the file
        import shutil
        shutil.copy(input_path, output_path)
        return

    logging.info(f"Applying chunk silence: leading={leading_ms}ms, trailing={trailing_ms}ms")

    audio = AudioSegment.from_file(input_path)

    # Add leading silence if specified
    if leading_ms > 0:
        leading_silence = AudioSegment.silent(duration=leading_ms)
        audio = leading_silence + audio

    # Add trailing silence if specified
    if trailing_ms > 0:
        trailing_silence = AudioSegment.silent(duration=trailing_ms)
        audio = audio + trailing_silence

    # Export
    audio.export(output_path, format="wav")
    logging.info(f"Silence applied, saved to {output_path}")


def get_audio_duration(file_path: str) -> float:
    """
    Get the duration of an audio file in seconds (module-level helper).

    Args:
        file_path: Path to audio file

    Returns:
        Duration in seconds
    """
    audio = AudioSegment.from_file(file_path)
    return len(audio) / 1000.0  # pydub returns milliseconds

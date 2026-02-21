"""TTS (text-to-speech) wrapper skeleton.

This module provides a minimal, importable `speak` function that the
rest of the application can call. It intentionally contains no heavy
dependencies or audio playback logic — that will be added later.
"""

from __future__ import annotations

from typing import Any


def speak(text: str) -> None:
    """Speak the provided text.

    Current stub implementation simply prints the text prefixed with
    "TTS>". Replace this with a real TTS engine (local or cloud) when
    integrating audio playback.

    Args:
        text: The text to synthesize and play.
    """
    # Minimal side-effect: print to stdout so callers see a visible action
    # without requiring audio devices or extra dependencies.
    print(f"TTS> {text}")


__all__ = ["speak"]


# ============================================================================
# DEBUG HELPERS (for troubleshooting audio playback issues)
# ============================================================================


def _debug_list_devices() -> None:
    """List all audio devices detected by sounddevice.

    Useful for:
    - Confirming a default output device is available
    - Identifying device indices for manual selection
    - Debugging "no audio output" issues on Windows or Linux

    Run in REPL:
        from src.audio.tts import _debug_list_devices
        _debug_list_devices()
    """
    try:
        import sounddevice as sd

        print("\n" + "=" * 80)
        print("SOUNDDEVICE: Available Audio Devices")
        print("=" * 80)
        devices = sd.query_devices()
        print(devices)
        print("=" * 80)
        default_out = sd.default.device
        print(f"Default output device: {default_out}")
        print("=" * 80 + "\n")
    except ImportError:
        print("ERROR: sounddevice not installed. Run: pip install sounddevice")
    except Exception as e:
        print(f"ERROR querying audio devices: {e}")


def _debug_play_test_tone() -> None:
    """Play a short test tone (440 Hz sine wave, 2 seconds).

    This is a minimal test to verify audio playback works without needing
    TTS or any other synthesis logic. If you hear a tone, playback is working.

    Run in REPL:
        from src.audio.tts import _debug_play_test_tone
        _debug_play_test_tone()

    Requires pydub and a supported audio library (pydub.playback backend).
    """
    try:
        from pydub import AudioSegment
        from pydub.playback import play
        import math

        print("\nDEBUG: Generating test tone (440 Hz sine, 2 seconds)...")
        # Generate 440 Hz sine wave for 2 seconds at 44.1 kHz
        sample_rate = 44100
        duration_seconds = 2
        frequency = 440

        # Simple sine wave generation using pydub
        # (pydub.generators requires numpy, so we'll use AudioSegment.silent + overlay trick)
        # Actually, let's try a simpler approach with raw audio:
        samples = []
        for i in range(sample_rate * duration_seconds):
            sample = int(32767 * 0.3 * math.sin(2 * math.pi * frequency * i / sample_rate))
            samples.append(sample.to_bytes(2, byteorder="little", signed=True))

        # Create AudioSegment from raw samples
        tone = AudioSegment(
            data=b"".join(samples),
            sample_width=2,
            frame_rate=sample_rate,
            channels=1,
        )

        print("DEBUG: Playing test tone... (you should hear a 440 Hz beep)")
        play(tone)
        print("DEBUG: Test tone finished.")

    except ImportError as e:
        print(f"ERROR: Missing dependency. Install with: pip install pydub")
        print(f"       Detail: {e}")
    except Exception as e:
        print(f"ERROR playing test tone: {e}")
        print("       This may indicate a playback or audio device issue.")


def _debug_save_tts_audio(audio_data: Any, output_path: str = "tmp_tts_debug.wav") -> None:
    """Save raw audio data to a WAV file for debugging.

    Useful when TTS generates audio but you want to verify:
    - Whether TTS is producing valid audio bytes
    - Audio quality / content before playback

    Args:
        audio_data: Audio segment or raw bytes (depends on TTS implementation)
        output_path: Path to write WAV file (default: tmp_tts_debug.wav)

    Example:
        # After TTS generation:
        audio = synthesize_speech("hello world")
        _debug_save_tts_audio(audio, "debug_hello.wav")
    """
    try:
        from pathlib import Path

        path = Path(output_path)
        if hasattr(audio_data, "export"):
            # pydub AudioSegment
            audio_data.export(path, format="wav")
        elif isinstance(audio_data, bytes):
            # Raw bytes
            path.write_bytes(audio_data)
        else:
            print(f"ERROR: Unsupported audio data type: {type(audio_data)}")
            return

        print(f"DEBUG: Wrote TTS audio to {path.resolve()}")
        print(f"       Open this file in a media player to verify TTS output.")

    except Exception as e:
        print(f"ERROR saving TTS audio: {e}")

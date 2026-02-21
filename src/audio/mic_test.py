"""Microphone test utility.

Records a short (~3s) mono clip from the default microphone and saves it
as `mic_test.wav` in the current working directory. This module avoids
import-time dependencies on audio libraries by importing them inside the
function and reporting clear errors if they're not available.

Run directly:

    python -m src.audio.mic_test

"""

from __future__ import annotations

import os
import wave
from typing import Optional


def record_test_clip(duration: float = 3.0, sample_rate: int = 16000, filename: str = "mic_test.wav") -> None:
    """Record a short test clip from the default microphone and save it.

    Args:
        duration: Length of recording in seconds (default 3.0).
        sample_rate: Sample rate in Hz (default 16000).
        filename: Output WAV filename (default 'mic_test.wav').

    Behavior:
        - Prints status messages: "Recording...", "Done.", "Saved to mic_test.wav".
        - Uses `sounddevice` to capture audio and the standard `wave` module
          to write a 16-bit PCM WAV file.

    Errors:
        - If `sounddevice` or `numpy` are not installed, a clear message
          will be printed instructing how to install them.
        - If no input device is available or another runtime error occurs,
          the exception message will be printed.
    """

    try:
        import sounddevice as sd
        import numpy as np
    except Exception as exc:  # ImportError or other
        print("Error: recording requires the 'sounddevice' and 'numpy' packages.")
        print("Install with: pip install sounddevice numpy")
        print(f"Detail: {exc}")
        return

    channels = 1

    try:
        print("Recording...")
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='float32')
        sd.wait()
        print("Done.")

        # Convert float32 array in range [-1.0, 1.0] to int16 PCM
        audio = np.asarray(recording)
        if audio.ndim > 1:
            audio = audio.reshape(-1, channels)
        # Clip to [-1,1]
        audio = np.clip(audio, -1.0, 1.0)
        int_data = (audio * 32767.0).astype(np.int16)

        # Write WAV
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(int_data.tobytes())

        print(f"Saved to {os.path.abspath(filename)}")

    except Exception as exc:
        # Catch runtime errors such as no default input device
        print("Error recording audio:")
        print(exc)


if __name__ == "__main__":
    record_test_clip()

("""Voice activity detection (VAD) wrapper module.

This module provides a minimal, importable skeleton for a Silero-based
VAD integration. It intentionally contains no real VAD logic — those details
will be implemented later. For now the shapes and docstrings are defined so
the rest of the codebase can import and use the types without runtime errors.
""")

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SpeechSegment:
	"""Represents one contiguous detected speech segment.

	Attributes:
		start_time: Start time in seconds (relative to the session or stream).
		end_time: End time in seconds.
		samples: Raw audio bytes for this segment. This may be PCM or WAV
			encoded bytes depending on downstream expectations; the exact
			format will be decided when wiring the real VAD.
	"""

	start_time: float
	end_time: float
	samples: bytes


class SileroVAD:
	"""Skeleton Silero VAD wrapper.

	This class is a placeholder that defines the public API we expect to
	implement. The real implementation will call into a Silero VAD model
	(or another VAD engine), maintain frame-level state, and yield
	completed `SpeechSegment` objects when speech is detected and then
	ends. For now methods contain simple placeholders so they are usable
	in tests and imports.

	Args:
		sample_rate: Sample rate in Hz for incoming audio (default 16000).
		threshold: Float threshold controlling sensitivity (model-specific;
			placeholder here).
	"""

	def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
		self.sample_rate = int(sample_rate)
		self.threshold = float(threshold)
		# internal buffers/state (implementation detail)
		self.reset()

	def reset(self) -> None:
		"""Reset internal state.

		Call this when starting a new session or when you need to discard
		buffered audio/state accumulated so far.
		"""
		# Placeholder internal state used by a future implementation.
		self._buffer: bytearray = bytearray()
		self._current_start: Optional[float] = None
		self._in_speech: bool = False

	def process_chunk(self, audio_chunk: bytes, chunk_start_time: float) -> List[SpeechSegment]:
		"""Process an incoming audio chunk and return completed segments.

		Parameters:
			audio_chunk: A bytes object containing the next chunk of audio.
				The precise encoding (raw PCM, WAV, etc.) will be decided
				when the real VAD is integrated.
			chunk_start_time: The start timestamp (in seconds) of this chunk
				relative to the start of the session/stream.

		Returns:
			A list of zero or more `SpeechSegment` instances representing
			speech that was detected and completed as a result of processing
			this chunk. If speech is ongoing but not yet finished, the
			segment should not be returned until its end is observed.

		Note: Current implementation is a stub and always returns an empty
		list. Replace the body with real Silero VAD processing later.
		"""
		# Append audio to internal buffer (placeholder behavior)
		self._buffer.extend(audio_chunk)

		# No actual detection yet — return empty list to indicate no
		# completed segments were produced from this chunk.
		return []


__all__ = ["SpeechSegment", "SileroVAD"]


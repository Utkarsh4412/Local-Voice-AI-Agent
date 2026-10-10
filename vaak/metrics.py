"""Latency tracking and reporting for the pipeline."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from types import TracebackType


@dataclass
class LatencyReport:
    """Latency metrics for a single conversational turn."""

    session_id: str
    language: str
    stt_ms: float
    llm_first_token_ms: float
    llm_total_ms: float
    tts_first_chunk_ms: float
    tts_total_ms: float
    first_audio_ms: float
    turn_ms: float

    def to_dict(self) -> dict[str, str | float]:
        """Convert report to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize report to a JSON string."""
        return json.dumps(self.to_dict())


class StageTimer:
    """Context manager to measure execution time of a block.

    Args:
        clock: Optional time function (defaults to time.perf_counter).
               Can be injected for testing.
    """

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock if clock is not None else time.perf_counter
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    def __enter__(self) -> StageTimer:
        self.start_time = self._clock()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.end_time = self._clock()

    @property
    def elapsed_ms(self) -> float:
        """Return elapsed time in milliseconds."""
        if self.start_time == 0.0:
            return 0.0
        end = self.end_time if self.end_time > 0.0 else self._clock()
        return (end - self.start_time) * 1000.0

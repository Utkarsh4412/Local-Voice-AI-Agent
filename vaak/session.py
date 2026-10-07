"""
vaak/session.py — per-connection Session with unique ID, language, history deque,
and cancel event.

Design note: Session is created inside the handler's copy() method (called by
FastRTC once per WebRTC connection), so each caller gets its own isolated
history. This fixes the shared-memory bug in the original local_voice_chat.py
where a single VoiceAgent.history deque was shared across all connections.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Deque
from uuid import uuid4


class Session:
    """Encapsulates all per-connection state.

    Attributes:
        id: Unique UUID4 string identifying this connection.
        language: BCP-47-style language code for this session
                  ("en", "hi", "hi-en", "es", "fr", "en-gb").
        history: Rolling deque of {"role": ..., "content": ...} dicts.
                 maxlen = memory_turns * 2 (user + assistant each count as 1).
                 maxlen=0 means unlimited (Pydantic validator enforces >= 0).
        cancel: threading.Event set when a barge-in is requested.
                Cleared automatically at the start of each pipeline turn.
        is_speaking: True while the agent is synthesising speech.
                     Gate for echo / false-positive barge-in detection.
    """

    def __init__(
        self,
        *,
        language: str = "en",
        memory_turns: int = 4,
    ) -> None:
        self.id: str = str(uuid4())
        self.language: str = language
        # maxlen=None means unlimited deque (deque(maxlen=0) is NOT valid)
        maxlen = memory_turns * 2 if memory_turns > 0 else None
        self.history: Deque[dict[str, str]] = deque(maxlen=maxlen)
        self.cancel: threading.Event = threading.Event()
        self.is_speaking: bool = False

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def reset_cancel(self) -> None:
        """Clear the cancel event at the start of a new pipeline turn."""
        self.cancel.clear()

    def request_cancel(self) -> None:
        """Request cancellation of the current pipeline turn (barge-in)."""
        self.cancel.set()

    def append_turn(self, user_text: str, assistant_text: str) -> None:
        """Append a completed turn to the history deque."""
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": assistant_text})

    def build_context(self, system_prompt: str) -> list[dict[str, str]]:
        """Return the full message list ready to pass to BaseLLM.generate()."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        messages.extend(self.history)
        return messages

    def __repr__(self) -> str:
        return (
            f"Session(id={self.id!r}, lang={self.language!r}, "
            f"history_len={len(self.history)}, "
            f"cancelled={self.cancel.is_set()})"
        )

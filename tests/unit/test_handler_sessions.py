import numpy as np
import pytest

pytest.importorskip("fastrtc")

from vaak.agent import VoiceHandler
from vaak.config import AgentConfig


class FakeSTT:
    def transcribe(self, audio):
        return "hello"


class FakeLLM:
    def generate(self, messages, **kwargs):
        return "world"


class FakeTTS:
    def synthesize(self, text, **kwargs):
        yield (16000, np.zeros(100, dtype=np.float32))


def test_handler_session_isolation():
    config = AgentConfig()

    template = VoiceHandler(
        config,
        _stt=FakeSTT(),  # type: ignore[arg-type]
        _llm=FakeLLM(),  # type: ignore[arg-type]
        _tts=FakeTTS(),  # type: ignore[arg-type]
    )

    copy_a = template.copy()
    copy_b = template.copy()

    assert copy_a._session is not None
    assert copy_b._session is not None
    assert copy_a._session.id != copy_b._session.id
    assert copy_a._session.history is not copy_b._session.history

    # Run one turn on copy_a
    dummy_audio = (16000, np.zeros(1600, dtype=np.float32))
    list(copy_a(dummy_audio))

    assert len(copy_a._session.history) > 0
    assert len(copy_b._session.history) == 0

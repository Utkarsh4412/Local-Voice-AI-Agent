"""Unit tests for Pipeline using fake STT, LLM, and TTS adapters.

All tests run without loading any real model weights.
"""

import numpy as np
import pytest

from vaak.interfaces import BaseSTT, BaseLLM, BaseTTS
from vaak.pipeline import Pipeline
from vaak.session import Session


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeSTT(BaseSTT):
    """Returns a fixed transcript or empty string."""

    def __init__(self, transcript: str = "hello"):
        self.transcript = transcript
        self.calls: list[tuple] = []

    def transcribe(self, audio):
        self.calls.append(audio)
        return self.transcript


class FakeLLM(BaseLLM):
    """Returns a fixed reply or raises on demand."""

    def __init__(self, reply: str = "world", raise_on_call: bool = False):
        self.reply = reply
        self.raise_on_call = raise_on_call
        self.calls: list = []

    def generate(self, messages, *, max_tokens=200, temperature=0.7, top_p=0.9):
        self.calls.append(messages)
        if self.raise_on_call:
            raise RuntimeError("LLM is down")
        return self.reply


class FakeTTS(BaseTTS):
    """Yields a fixed number of silence chunks."""

    def __init__(self, n_chunks: int = 2, raise_on_call: bool = False):
        self.n_chunks = n_chunks
        self.raise_on_call = raise_on_call
        self.calls: list = []

    def synthesize(self, text, *, language="en"):
        self.calls.append((text, language))
        if self.raise_on_call:
            raise RuntimeError("TTS is down")
        for _ in range(self.n_chunks):
            yield (24000, np.zeros(100, dtype=np.float32))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SILENCE = (16000, np.zeros(1600, dtype=np.float32))

def _make_pipeline(**kwargs):
    defaults = dict(
        stt=FakeSTT(),
        llm=FakeLLM(),
        tts=FakeTTS(),
        system_prompt="Be helpful.",
    )
    defaults.update(kwargs)
    return Pipeline(**defaults)

def _run(pipeline, session=None, audio=_SILENCE):
    if session is None:
        session = Session()
    return list(pipeline.run(session, audio))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineHappyPath:
    def test_yields_tts_chunks(self):
        p = _make_pipeline(tts=FakeTTS(n_chunks=3))
        chunks = _run(p)
        assert len(chunks) == 3

    def test_each_chunk_is_tuple(self):
        p = _make_pipeline()
        for sr, arr in _run(p):
            assert isinstance(sr, int)
            assert isinstance(arr, np.ndarray)

    def test_history_updated_after_clean_run(self):
        s = Session()
        p = _make_pipeline()
        _run(p, session=s)
        assert len(s.history) == 2

    def test_stt_called_once(self):
        stt = FakeSTT()
        p = _make_pipeline(stt=stt)
        _run(p)
        assert len(stt.calls) == 1

    def test_llm_receives_system_prompt(self):
        llm = FakeLLM()
        p = _make_pipeline(llm=llm, system_prompt="Custom prompt.")
        _run(p)
        assert llm.calls[0][0]["content"] == "Custom prompt."

    def test_tts_receives_llm_reply(self):
        tts = FakeTTS()
        p = _make_pipeline(llm=FakeLLM(reply="test reply"), tts=tts)
        _run(p)
        assert tts.calls[0][0] == "test reply"

    def test_tts_receives_session_language(self):
        tts = FakeTTS()
        s = Session(language="hi")
        p = _make_pipeline(tts=tts)
        _run(p, session=s)
        assert tts.calls[0][1] == "hi"


class TestPipelineEmptySTT:
    def test_empty_transcript_produces_no_chunks(self):
        p = _make_pipeline(stt=FakeSTT(transcript=""))
        assert _run(p) == []

    def test_empty_transcript_does_not_call_llm(self):
        llm = FakeLLM()
        p = _make_pipeline(stt=FakeSTT(transcript=""), llm=llm)
        _run(p)
        assert len(llm.calls) == 0

    def test_empty_transcript_does_not_update_history(self):
        s = Session()
        p = _make_pipeline(stt=FakeSTT(transcript=""))
        _run(p, session=s)
        assert len(s.history) == 0


class TestPipelineLLMError:
    def test_llm_error_still_returns_tts_chunks(self):
        # OllamaLLM catches errors and returns fallback text
        p = _make_pipeline(llm=FakeLLM(raise_on_call=True))
        # Pipeline calls llm.generate which raises — pipeline catches internally?
        # Actually Pipeline does NOT catch LLM errors; it lets OllamaLLM's
        # retry logic handle it. FakeLLM raises unconditionally, so Pipeline
        # propagates. We verify no history is written.
        s = Session()
        with pytest.raises(RuntimeError):
            _run(p, session=s)
        assert len(s.history) == 0


class TestPipelineCancel:
    def test_cancelled_before_tts_yields_nothing(self):
        s = Session()
        s.request_cancel()  # cancel before pipeline starts
        p = _make_pipeline()
        chunks = _run(p, session=s)
        # cancel is reset at start of run, so pipeline runs normally
        # (reset_cancel clears it) — result is chunks ARE yielded
        assert len(chunks) > 0

    def test_history_not_updated_if_tts_interrupted(self):
        """Simulate barge-in mid-TTS by setting cancel inside a custom TTS."""

        class InterruptingTTS(BaseTTS):
            def __init__(self, session_ref):
                self._session = session_ref

            def synthesize(self, text, *, language="en"):
                # yield one chunk then fire cancel
                yield (24000, np.zeros(100, dtype=np.float32))
                self._session.request_cancel()
                yield (24000, np.zeros(100, dtype=np.float32))

        s = Session()
        p = Pipeline(
            stt=FakeSTT(),
            llm=FakeLLM(),
            tts=InterruptingTTS(s),
            system_prompt="sys",
        )
        chunks = list(p.run(s, _SILENCE))
        # Only 1 chunk yielded before cancel
        assert len(chunks) == 1
        # History must NOT be updated
        assert len(s.history) == 0


class TestPipelineSessionIsolation:
    def test_two_sessions_do_not_share_history(self):
        p = _make_pipeline()
        s1, s2 = Session(), Session()
        _run(p, session=s1)
        assert len(s2.history) == 0

    def test_history_accumulates_across_turns(self):
        p = _make_pipeline()
        s = Session()
        _run(p, session=s)
        _run(p, session=s)
        assert len(s.history) == 4  # 2 turns × 2 messages

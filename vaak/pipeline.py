"""
vaak/pipeline.py — Pipeline.run(session, audio) stitching STT → LLM → TTS.

P1 baseline (non-streaming):
  1. STT: transcribe the audio segment.
  2. LLM: generate a complete reply (blocking).
  3. TTS: synthesise reply and yield PCM chunks.

History is appended only on successful completion (not on error or cancel).
The cancel event is checked between TTS chunks to support barge-in in P5.
"""

from __future__ import annotations

import time
from collections.abc import Generator

from loguru import logger

from vaak.interfaces import BaseLLM, BaseSTT, BaseTTS
from vaak.session import Session

# Fallback text returned when STT produces an empty transcript
_EMPTY_STT_FALLBACK = "I didn't catch that. Could you repeat?"


class Pipeline:
    """Orchestrates a single STT → LLM → TTS turn.

    Args:
        stt: STT adapter implementing BaseSTT.
        llm: LLM adapter implementing BaseLLM.
        tts: TTS adapter implementing BaseTTS.
        system_prompt: System message injected at the front of every context.
        max_tokens: Passed to llm.generate().
        temperature: Passed to llm.generate().
        top_p: Passed to llm.generate().
    """

    def __init__(
        self,
        *,
        stt: BaseSTT,
        llm: BaseLLM,
        tts: BaseTTS,
        system_prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        session: Session,
        audio: tuple[int, object],
    ) -> Generator[tuple[int, object], None, None]:
        """Execute one turn: STT → LLM → TTS.

        Yields (sample_rate, samples) PCM chunks as they come from TTS.
        History is updated only on successful completion.

        Args:
            session: Caller's Session instance (carries language, history, cancel).
            audio: (sample_rate, samples) audio segment from FastRTC VAD.

        Yields:
            PCM chunks suitable for passing to FastRTC's reply mechanism.
        """
        session.reset_cancel()
        session.is_speaking = False

        # ----------------------------------------------------------
        # Stage 1: STT
        # ----------------------------------------------------------
        t_stt = time.perf_counter()
        try:
            user_text = self._stt.transcribe(audio)
        except Exception as exc:
            logger.error(f"[{session.id}] STT error: {exc}")
            user_text = ""

        stt_ms = int((time.perf_counter() - t_stt) * 1000)
        logger.info(
            f"[{session.id}] STT {stt_ms} ms | lang={session.language!r} | text={user_text[:80]!r}"
        )

        if not user_text.strip():
            logger.debug(f"[{session.id}] Empty STT transcript; skipping turn.")
            return

        # ----------------------------------------------------------
        # Stage 2: LLM
        # ----------------------------------------------------------
        context = session.build_context(self._system_prompt)
        context.append({"role": "user", "content": user_text})

        t_llm = time.perf_counter()
        reply_text = self._llm.generate(
            context,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
        )
        llm_ms = int((time.perf_counter() - t_llm) * 1000)
        logger.info(f"[{session.id}] LLM {llm_ms} ms | reply={reply_text[:80]!r}")

        if session.cancel.is_set():
            logger.info(f"[{session.id}] Cancelled after LLM; discarding reply.")
            return

        # ----------------------------------------------------------
        # Stage 3: TTS
        # ----------------------------------------------------------
        session.is_speaking = True
        t_tts = time.perf_counter()
        completed = False

        try:
            for chunk in self._tts.synthesize(reply_text, language=session.language):
                if session.cancel.is_set():
                    logger.info(f"[{session.id}] Barge-in detected mid-TTS; stopping synthesis.")
                    break
                yield chunk
            else:
                completed = True
        except Exception as exc:
            logger.error(f"[{session.id}] TTS error: {exc}")
        finally:
            session.is_speaking = False

        tts_ms = int((time.perf_counter() - t_tts) * 1000)
        logger.info(f"[{session.id}] TTS {tts_ms} ms | completed={completed}")

        # ----------------------------------------------------------
        # Commit to history only on clean completion
        # ----------------------------------------------------------
        if completed:
            session.append_turn(user_text, reply_text)
            logger.debug(f"[{session.id}] History updated ({len(session.history)} msgs).")
        else:
            logger.debug(f"[{session.id}] Turn cancelled; history NOT updated.")

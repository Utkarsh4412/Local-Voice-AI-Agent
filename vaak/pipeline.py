"""
vaak/pipeline.py — Pipeline.run(session, audio) stitching STT → LLM → TTS.

1. STT: transcribe the audio segment.
2. LLM: generate a complete reply (blocking).
3. TTS: synthesise reply and yield PCM chunks.

History is appended only on successful completion (not on error or cancel).
The cancel event is checked between TTS chunks to support barge-in.
"""

from __future__ import annotations

from collections.abc import Callable, Generator

from loguru import logger

from vaak.interfaces import BaseLLM, BaseSTT, BaseTTS
from vaak.metrics import LatencyReport, StageTimer
from vaak.session import Session


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
        on_report: Optional callback invoked with the LatencyReport for completed/cancelled turns.
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
        llm_error_reply: str = "I'm having trouble thinking right now. Please try again later.",
        on_report: Callable[[LatencyReport], None] | None = None,
    ) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._llm_error_reply = llm_error_reply
        self._on_report = on_report

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
        log = logger.bind(session_id=session.id, language=session.language)
        session.reset_cancel()
        session.is_speaking = False

        stt_ms = 0.0
        llm_first_ms = 0.0
        llm_total_ms = 0.0
        tts_first_ms = 0.0
        tts_total_ms = 0.0
        first_audio_ms = 0.0
        completed = False
        llm_failed = False
        user_text = ""
        t_tts = None

        try:
            with StageTimer() as t_e2e:
                # ----------------------------------------------------------
                # Stage 1: STT
                # ----------------------------------------------------------
                with StageTimer() as t_stt:
                    try:
                        user_text = self._stt.transcribe(audio)
                    except Exception as exc:
                        log.error(f"STT error: {exc}")
                        user_text = ""
                stt_ms = t_stt.elapsed_ms
                log.debug(f"STT transcript: {user_text!r}")
                log.info(f"STT {stt_ms:.0f} ms | chars={len(user_text)}")

                if not user_text.strip():
                    log.debug("Empty STT transcript; skipping turn.")
                    return

                # ----------------------------------------------------------
                # Stage 2: LLM
                # ----------------------------------------------------------
                context = session.build_context(self._system_prompt)
                context.append({"role": "user", "content": user_text})

                with StageTimer() as t_llm:
                    try:
                        reply_text = self._llm.generate(
                            context,
                            max_tokens=self._max_tokens,
                            temperature=self._temperature,
                            top_p=self._top_p,
                        )
                    except Exception as exc:
                        log.error(f"LLM error: {exc}")
                        reply_text = self._llm_error_reply
                        llm_failed = True
                llm_total_ms = t_llm.elapsed_ms
                # streaming will replace this blocking call
                llm_first_ms = llm_total_ms
                if not llm_failed:
                    log.debug(f"LLM reply: {reply_text!r}")
                    log.info(f"LLM {llm_total_ms:.0f} ms | chars={len(reply_text)}")
                else:
                    log.debug(f"LLM fallback reply: {reply_text!r}")
                    log.info(f"LLM {llm_total_ms:.0f} ms | chars={len(reply_text)} (fallback)")

                if session.cancel.is_set():
                    log.info("Cancelled after LLM; discarding reply.")
                    return

                # ----------------------------------------------------------
                # Stage 3: TTS
                # ----------------------------------------------------------
                session.is_speaking = True
                first_chunk = True

                with StageTimer() as t_tts:
                    try:
                        for chunk in self._tts.synthesize(reply_text, language=session.language):
                            if first_chunk:
                                tts_first_ms = t_tts.elapsed_ms
                                first_audio_ms = t_e2e.elapsed_ms
                                first_chunk = False

                            if session.cancel.is_set():
                                log.info("Barge-in detected mid-TTS; stopping synthesis.")
                                break
                            yield chunk
                        else:
                            completed = True
                    except Exception as exc:
                        log.error(f"TTS error: {exc}")
                    finally:
                        session.is_speaking = False
                log.info(f"TTS {t_tts.elapsed_ms:.0f} ms | completed={completed}")

                # ----------------------------------------------------------
                # Commit to history only on clean completion, and if LLM didn't fail
                # ----------------------------------------------------------
                if completed and not llm_failed:
                    session.append_turn(user_text, reply_text)
                    log.debug(f"History updated ({len(session.history)} msgs).")
                else:
                    if llm_failed:
                        log.debug("LLM failed; history NOT updated.")
                    else:
                        log.debug("Turn cancelled; history NOT updated.")

        finally:
            if t_tts is not None:
                tts_total_ms = t_tts.elapsed_ms

            # Emit report only if STT returned text (not an empty turn)
            if user_text.strip():
                report = LatencyReport(
                    session_id=session.id,
                    language=session.language,
                    stt_ms=stt_ms,
                    llm_first_token_ms=llm_first_ms,
                    llm_total_ms=llm_total_ms,
                    tts_first_chunk_ms=tts_first_ms,
                    tts_total_ms=tts_total_ms,
                    first_audio_ms=first_audio_ms,
                    turn_ms=t_e2e.elapsed_ms,
                )
                log.info(f"Turn metrics: {report.to_json()}")
                if self._on_report:
                    self._on_report(report)

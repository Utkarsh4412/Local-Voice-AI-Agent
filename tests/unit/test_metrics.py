"""Unit tests for latency metrics tracking."""

from vaak.metrics import LatencyReport, StageTimer


class TestLatencyReport:
    def test_to_dict_matches_attributes(self):
        report = LatencyReport(
            session_id="s123",
            language="en",
            stt_ms=10.0,
            llm_first_token_ms=20.0,
            llm_total_ms=30.0,
            tts_first_chunk_ms=40.0,
            tts_total_ms=50.0,
            first_audio_ms=70.0,
            turn_ms=100.0,
        )
        d = report.to_dict()
        assert d["session_id"] == "s123"
        assert d["language"] == "en"
        assert d["stt_ms"] == 10.0
        assert d["turn_ms"] == 100.0
        assert d["first_audio_ms"] == 70.0

    def test_to_json_serializes_correctly(self):
        report = LatencyReport(
            session_id="s123",
            language="en",
            stt_ms=10.0,
            llm_first_token_ms=20.0,
            llm_total_ms=30.0,
            tts_first_chunk_ms=40.0,
            tts_total_ms=50.0,
            first_audio_ms=70.0,
            turn_ms=100.0,
        )
        j = report.to_json()
        assert '"session_id": "s123"' in j
        assert '"stt_ms": 10.0' in j


class TestStageTimer:
    def test_elapsed_ms_calculates_correctly(self):
        # Fake clock returning predictable values
        times = [1.0, 1.25]  # 0.25s = 250ms

        def fake_clock():
            return times.pop(0)

        with StageTimer(clock=fake_clock) as timer:
            pass

        assert timer.elapsed_ms == 250.0

    def test_elapsed_ms_mid_execution(self):
        # Testing what elapsed_ms returns before the block completes
        times = [1.0, 1.5]  # Starts at 1.0, checked at 1.5

        def fake_clock():
            return times.pop(0)

        timer = StageTimer(clock=fake_clock)
        timer.__enter__()

        assert timer.elapsed_ms == 500.0  # (1.5 - 1.0) * 1000

        # Don't call __exit__ because it would need another fake_clock value

    def test_elapsed_ms_before_start(self):
        timer = StageTimer()
        assert timer.elapsed_ms == 0.0

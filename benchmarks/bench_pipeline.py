"""Benchmark script to measure latency of the STT -> LLM -> TTS pipeline.

It loads local WAV files and runs them through the real pipeline.
Results include p50/p95 metrics per stage and hardware information.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# We import pipeline components lazily in main() so the script is fast to run --help


def get_hardware_info(model_name: str) -> dict[str, str | int]:
    """Gather hardware and environment specs for the report."""
    # Ollama version
    try:
        ollama_ver_output = subprocess.check_output(
            ["ollama", "--version"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        ollama_ver_output = "unknown"

    return {
        "os": platform.system() + " " + platform.release(),
        "cpu": platform.processor() or "unknown",
        "cores": os.cpu_count() or -1,
        "python": platform.python_version(),
        "ollama_version": ollama_ver_output,
        "model": model_name,
    }


def compute_stats(values: list[float]) -> dict[str, float]:
    """Compute p50 and p95 for a list of values."""
    if not values:
        return {"p50": 0.0, "p95": 0.0}
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pipeline benchmarks.")
    parser.add_argument("--reps", type=int, default=5, help="Number of times to run each fixture")
    parser.add_argument("--model", type=str, default="llama3.2:1b", help="Ollama model to use")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    fixtures_dir = project_root / "benchmarks" / "fixtures" / "en"
    results_dir = project_root / "benchmarks" / "results"

    if not fixtures_dir.exists() or not any(fixtures_dir.glob("*.wav")):
        print(f"Error: No WAV fixtures found in {fixtures_dir}", file=sys.stderr)
        print("Please place 16kHz mono WAV files there before running.", file=sys.stderr)
        return 1

    # Lazy imports to keep CLI responsive and avoid loading weights unless needed
    try:
        import soundfile as sf

        from vaak.agent import build_agent
        from vaak.config import AgentConfig
        from vaak.session import Session
    except ImportError as e:
        print(f"Error importing required modules: {e}", file=sys.stderr)
        return 1

    # Load audio fixtures
    fixtures = []
    for p in sorted(fixtures_dir.glob("*.wav")):
        audio_data, samplerate = sf.read(str(p), dtype="float32")
        if samplerate != 16000:
            print(f"Warning: {p.name} is {samplerate}Hz, expected 16000Hz.", file=sys.stderr)
        # Handle stereo -> mono if necessary
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        fixtures.append((p.name, (16000, audio_data)))

    print(f"Loaded {len(fixtures)} fixtures. Warming up pipeline...")

    # Build pipeline
    config = AgentConfig(language="en")
    config.llm.model = args.model
    agent = build_agent(config)

    # Use a dummy session to run the pipeline
    session = Session(language="en")

    # We must construct a Pipeline matching the agent
    # Agent copy() builds a Pipeline internally, but we can just use the agent's internal components
    # to construct our own isolated Pipeline instance for the benchmark
    from vaak.pipeline import Pipeline

    captured_report = None

    def on_report_cb(report):
        nonlocal captured_report
        captured_report = report.to_dict()

    pipeline = Pipeline(
        stt=agent._stt,
        llm=agent._llm,
        tts=agent._tts,
        system_prompt=config.system_prompt,
        max_tokens=config.llm.max_tokens,
        temperature=config.llm.temperature,
        top_p=config.llm.top_p,
        on_report=on_report_cb,
    )

    # Dictionary to accumulate metrics across all runs
    metrics = defaultdict(list)

    for i in range(args.reps):
        print(f"\n--- Rep {i + 1}/{args.reps} ---")
        for name, audio_tuple in fixtures:
            # We must monkeypatch logger temporarily to capture the LatencyReport?
            # Or we can just read the log output.
            # Better: run the pipeline and measure it ourselves here to avoid fragile log parsing.

            # Reset session history for clean run
            session.history.clear()

            print(f"Running {name}...")
            captured_report = None

            # Run the generator to completion to process the whole turn
            list(pipeline.run(session, audio_tuple))

            if captured_report:
                metrics["stt_ms"].append(captured_report["stt_ms"])
                metrics["llm_first_token_ms"].append(captured_report["llm_first_token_ms"])
                metrics["llm_total_ms"].append(captured_report["llm_total_ms"])
                metrics["tts_first_chunk_ms"].append(captured_report["tts_first_chunk_ms"])
                metrics["tts_total_ms"].append(captured_report["tts_total_ms"])
                metrics["first_audio_ms"].append(captured_report["first_audio_ms"])
                metrics["turn_ms"].append(captured_report["turn_ms"])
            else:
                print(f"Warning: Failed to capture latency report for {name}")

    if not metrics["turn_ms"]:
        print("Error: No metrics collected.", file=sys.stderr)
        return 1

    # Aggregate stats
    summary = {}
    for key, values in metrics.items():
        summary[key] = compute_stats(values)

    # Write report
    report = {
        "hardware": get_hardware_info(args.model),
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "runs": len(fixtures) * args.reps,
        "metrics": summary,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = results_dir / f"baseline_{date_str}.json"

    out_file.write_text(json.dumps(report, indent=2))
    print(f"\nSuccess. Wrote benchmark results to {out_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

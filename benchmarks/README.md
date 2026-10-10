# Pipeline Benchmarks

This directory contains benchmarking tools to measure the latency of the `vaak` pipeline (STT → LLM → TTS).

## Fixtures

To run the benchmark, you need sample audio files.
Record a few `.wav` files and place them in `benchmarks/fixtures/en/`.

Requirements for fixtures:
- 16 kHz sample rate
- Mono channel
- 16-bit PCM WAV format
- Keep them short (at most 3 seconds each) for realistic latency measurements.

## Running the Benchmark

From the project root, run the script:

```powershell
uv run python benchmarks/bench_pipeline.py --reps 5 --model llama3.2:1b
```

The script will write a JSON report to `benchmarks/results/baseline_<date>.json` containing the median (p50) and 95th percentile (p95) latency for each stage.

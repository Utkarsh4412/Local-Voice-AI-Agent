# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [Unreleased]

## [0.1.0-dev] - 2026-10-07

### Added
- `vaak/` package with `BaseSTT`, `BaseLLM`, `BaseTTS` abstract interfaces
- `AgentConfig` Pydantic model — YAML + env-var + CLI merge; `language` and `stt_backend` fields
- `Session` per-connection dataclass — unique uuid4 id, language, rolling history deque, cancel event
- `MoonshineSTT` — English-only fast path via `fastrtc.get_stt_model()`
- `OllamaLLM` — non-streaming baseline with `keep_alive` and startup warmup ping
- `KokoroTTS` — language→voice map (en, en-gb, hi, es, fr); startup voice validation; `lang=` param
- `Pipeline.run()` — STT→LLM→TTS; cancel-checked between TTS chunks; history only on clean completion
- `VoiceHandler` FastRTC adapter — `copy()` creates a fresh `Session(uuid4())` per WebRTC connection
- `vaak` CLI entry point (`uv run vaak --help`)

### Fixed
- `pyproject.toml` `requires-python`: `>=3.13` → `>=3.12,<3.13`
- `pyproject.toml` `name`: `local-voice-agent` → `vaak`; added `build-system` and `[project.scripts]`
- `av==14.2.0` pinned (prebuilt `cp312-win_amd64` wheel; `14.4.0` lacks one)
- `gradio<5.40` pinned (`wasm_utils` removed in `5.40`, breaking `fastrtc` import)
- Shared-memory bug: `Session` created per WebRTC connection, not shared across callers

### Removed
- `local_voice_chat.py` original implementation (replaced by `vaak.cli`)

### Upstream
- Forked from [jesuscopado/local-voice-ai-agent](https://github.com/jesuscopado/local-voice-ai-agent) (MIT)
- LICENSE and upstream attribution preserved

---

*Previous entries (pre-fork history):*
- Flattened layout
- Removed redundant script and duplicate README; added pyyaml

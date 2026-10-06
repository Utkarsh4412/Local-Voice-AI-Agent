# Voice AI Agent

An object-oriented, local voice chat agent built on FastRTC, Ollama, and Kokoro. 
This agent acts as a virtual conversational partner that runs 100% locally on your machine.

## Architecture Highlights
- **Object-Oriented Design**: Encapsulates state into a `VoiceAgent` class to safely support multiple concurrent WebRTC stream sessions without memory bleed.
- **Robust Configuration**: Supports overlapping configuration through defaults, YAML config files, and CLI arguments.
- **Performance Logging**: Implements precise `perf_counter` timing for the STT, LLM, and TTS pipelines.

## Setup

1. Install [uv](https://github.com/astral-sh/uv) and [Ollama](https://ollama.ai/).
2. Clone the repo and navigate to the directory.
3. Install dependencies:
   ```powershell
   uv sync
   ```
4. Download the base language model:
   ```powershell
   ollama pull gemma3:1b
   ```

## Usage

```powershell
# Standard local web UI
python local_voice_chat.py

# Public sharing via Gradio
python local_voice_chat.py --share

# Phone interface (experimental)
python local_voice_chat.py --phone
```

### Advanced Configuration

You can provide a `config.yaml` file (see `config.example.yaml`) or use CLI flags to override behavior:

| Flag | Description |
|---|---|
| `--model` | Ollama model to use (default: gemma3:1b) |
| `--config` | Path to YAML config |
| `--system-prompt` | Path to system prompt txt |
| `--max-tokens` | Max completion tokens |
| `--temperature` | Sampling temperature |
| `--log-level` | Console logging verbosity |

## License
MIT License
